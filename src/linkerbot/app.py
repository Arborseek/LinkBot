from __future__ import annotations

import time
from enum import Enum
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
import yaml

from linkerbot.capture.camera import CameraCapture
from linkerbot.config.session import (
    HandProfile,
    HandSide,
    HardwareMode,
    SessionConfig,
    load_hand_profiles,
    load_init_config,
)
from linkerbot.hardware import create_drivers
from linkerbot.sim.mujoco_sim import MujocoSimRegistry
from linkerbot.models import HandRuntimeState, HandTracking, PipelineState, TrackingFrame
from linkerbot.retarget.factory import _has_spread_calibration, create_retargeter
from linkerbot.setup.init_calibrator import InitCalibrator
from linkerbot.setup.wizard import SetupWizard
from linkerbot.tracking.hand_tracker import HandTracker
from linkerbot.voice import GestureLibrary, GestureRecorder, VoiceController
from linkerbot.dance import DancePlayer, load_or_generate
from linkerbot.dance.music_analyzer import analyze_beats
from linkerbot.viz.composite import compose_split
from linkerbot.viz.overlay import draw_init_overlay, draw_teleop_overlay
from linkerbot.viz.text import render_texts
from linkerbot.viz.window import setup_window


class AppPhase(str, Enum):
    SETUP = "setup"
    INIT = "init"
    TELEOP = "teleop"


def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _physical_side(handedness: str, mirror: bool) -> str:
    side = handedness.lower()
    if mirror:
        return "left" if side == "right" else "right"
    return side


def _remap_frame(frame: TrackingFrame, mirror: bool) -> Dict[str, HandTracking]:
    mapped: Dict[str, HandTracking] = {}
    for label, tr in frame.hands.items():
        side = _physical_side(tr.handedness, mirror)
        mapped[side] = tr
    return mapped


def _assign_hand_tracking(
    state_hands: Dict[str, HandRuntimeState],
    mapped: Dict[str, HandTracking],
    active_sides: list,
) -> None:
    """单手模式下任意检测到的手都映射到配置侧，避免镜像导致左右对不上"""
    if len(active_sides) == 1:
        side = active_sides[0].value
        tr = mapped.get(side, HandTracking())
        if not tr.detected:
            for m in mapped.values():
                if m.detected:
                    tr = m
                    break
        state_hands[side].tracking = tr
        return
    for side, hs in state_hands.items():
        hs.tracking = mapped.get(side, HandTracking())


class LinkerBotApp:
    def __init__(self, base_config: Dict[str, Any], profiles_path: Path):
        self.base_config = base_config
        self.profiles = load_hand_profiles(profiles_path)
        self.init_cfg = load_init_config(profiles_path)
        self.session: SessionConfig | None = None
        self.profile: HandProfile | None = None
        self.phase = AppPhase.SETUP
        self.window_name = base_config.get("viz", {}).get("window_name", "LinkerBot")

        self.camera: CameraCapture | None = None
        self.tracker: HandTracker | None = None
        self.calibrator: InitCalibrator | None = None
        self.retargeters: Dict[str, Any] = {}
        self.drivers = None
        self.state = PipelineState()
        self._enabled = False
        self._last_send = 0.0
        self._frame_count = 0
        self._fps_timer = time.time()
        self._mirror = True
        self._camera_error = ""
        self._voice_mode = False
        self._voice_controller: VoiceController | None = None
        self._gesture_library: GestureLibrary | None = None
        self._gesture_recorder: GestureRecorder | None = None
        self._dance_mode = False
        self._dance_player: DancePlayer | None = None
        self._key_states: Dict[int, bool] = {}  # 按键防抖：记录上一帧状态

    def run(self) -> None:
        viz = self.base_config.get("viz", {})
        wizard = SetupWizard(
            self.window_name + " - Setup",
            window_width=viz.get("setup_width", 960),
            window_height=viz.get("setup_height", 680),
        )
        self.session = wizard.run()
        if self.session is None:
            return

        self.profile = self.profiles[self.session.hand_model]
        self.state.session_summary = self.session.summary_zh()
        self._mirror = self.base_config["camera"].get("mirror", True)

        self._setup_runtime()
        self.phase = AppPhase.INIT
        self.state.phase = "init"
        viz = self.base_config.get("viz", {})
        setup_window(
            self.window_name,
            viz.get("window_width", 1920),
            viz.get("window_height", 900),
        )

        try:
            while True:
                if self.phase == AppPhase.INIT:
                    if not self._step_init():
                        break
                elif self.phase == AppPhase.TELEOP:
                    if not self._step_teleop():
                        break

                key = cv2.waitKey(1) & 0xFF
                pressed = (key != 255)  # 255 = 无按键
                was_pressed = self._key_states.get(key, False)
                self._key_states = {k: (k == key and pressed) for k in self._key_states}
                if key not in self._key_states:
                    self._key_states[key] = pressed
                edge = pressed and not was_pressed  # 仅上升沿触发

                if key == ord("q"):
                    break
                if edge and key == ord(" ") and self.phase == AppPhase.TELEOP:
                    self._toggle_enabled()
                if edge and key == ord("r") and self.phase == AppPhase.TELEOP:
                    self._handle_r_key()
                if edge and key == ord("v") and self.phase == AppPhase.TELEOP:
                    self._toggle_voice_mode()
                if edge and key == ord("a") and self.phase == AppPhase.TELEOP:
                    self._toggle_dance_mode()
                if edge and key == ord("i"):
                    self._restart_init()
        finally:
            cv2.destroyAllWindows()
            self._shutdown()

    def _setup_runtime(self) -> None:
        assert self.session and self.profile
        cam = self.base_config["camera"]
        track = self.base_config["tracking"]
        hw = dict(self.base_config["hardware"])
        hw["mode"] = self.session.hardware_mode.value
        hw["simulation"] = self.base_config.get("simulation", {})
        sim_model = hw["simulation"].get("mujoco_model")
        if sim_model in (None, "null", ""):
            hw["simulation"] = {**hw["simulation"], "mujoco_model": None}

        self.session.camera_id = cam.get("device_id", 0)
        self.camera = CameraCapture(
            device_id=self.session.camera_id,
            width=cam.get("width", 1280),
            height=cam.get("height", 720),
            mirror=self._mirror,
            auto_detect=cam.get("auto_detect", True),
            warmup_frames=cam.get("warmup_frames", 5),
            read_retries=cam.get("read_retries", 5),
            probe_timeout=cam.get("probe_timeout", 12.0),
            strict_probe=cam.get("strict_probe", False),
        )
        self.tracker = HandTracker(
            model_path=track.get("model_path", "assets/hand_landmarker.task"),
            max_hands=self.session.max_hands,
            min_detection_confidence=track.get("min_detection_confidence", 0.7),
            min_tracking_confidence=track.get("min_tracking_confidence", 0.6),
        )
        retarget_cfg = self.base_config.get("retarget", {})
        self.retargeters = {
            s.value: create_retargeter(self.profile, retarget_cfg, s.value)
            for s in self.session.active_sides
        }
        self.drivers = create_drivers(self.session, self.profile, hw)
        self.calibrator = InitCalibrator(
            rois=self.init_cfg.get("rois", {}),
            active_sides=self.session.active_sides,
            hold_frames=self.init_cfg.get("hold_frames", 45),
            min_mcp_angle=self.init_cfg.get("min_mcp_angle", 2.2),
        )
        self.state.hands = {
            s.value: HandRuntimeState(side=s.value) for s in self.session.active_sides
        }

        self.camera.open()
        self.drivers.connect()
        self.drivers.send_open_pose()

        # ---- 语音控制初始化 ----
        voice_cfg = self.base_config.get("voice", {})
        gestures_path = Path(self.base_config.get("voice", {}).get(
            "gestures_path", "config/gestures.yaml"
        ))
        self._gesture_library = GestureLibrary(gestures_path)
        self._gesture_library.load()
        self._voice_controller = VoiceController(self._gesture_library, voice_cfg)
        self._gesture_recorder = GestureRecorder(
            save_callback=self._gesture_library.add,
            get_current_pose=self._get_current_hardware_pose,
        )

    def _get_current_hardware_pose(self) -> list[int] | None:
        """获取当前手的 hardware_pose（用于姿态录制）"""
        if not self.session:
            return None
        for side in self.session.active_sides:
            hs = self.state.hands.get(side.value)
            if hs and hs.hardware_pose:
                return list(hs.hardware_pose)
        return None

    def _toggle_dance_mode(self) -> None:
        """A 键：切换手势舞模式（与语音/跟手互斥）"""
        self._dance_mode = not self._dance_mode
        if self._dance_mode:
            # 互斥：关语音、关跟手
            if self._voice_mode:
                self._toggle_voice_mode()
            self._enabled = False
            self.state.enabled = False
            dance_cfg = self.base_config.get("dance", {})

            try:
                audio_path = str(dance_cfg.get("audio_path", "assets/music/qicai_yangguang.ogg"))

                # Step 1: 根据 choreography_source 决定是否需要节拍分析
                source = dance_cfg.get("choreography_source")
                if source is None:
                    source = "llm" if dance_cfg.get("use_llm", True) else "fallback"

                beat_times = None
                if source in ("llm", "fallback"):
                    bpm, beat_times = analyze_beats(audio_path)
                    dance_cfg["bpm"] = bpm
                # ASR 模式不需要 librosa 节拍分析

                # Step 2: 生成或加载舞谱
                cache_path = dance_cfg.get("cache_path", "config/dance/qicai_yangguang.yaml")
                choreography = load_or_generate(cache_path, beat_times, config=dance_cfg)

                # Step 3: 先把手摆到第一个手势位
                first_event = choreography["events"][0] if choreography.get("events") else None
                first_pose = self._gesture_library.lookup(first_event["gesture"]) if first_event and self._gesture_library else None
                if first_pose:
                    for side in self.session.active_sides:
                        self.drivers.send(side.value, first_pose)
                    time.sleep(0.3)

                # Step 4: 启动播放
                self._dance_player = DancePlayer()
                self._dance_player.start(
                    audio_path=audio_path,
                    choreography=choreography,
                    gesture_library=self._gesture_library,
                )
            except Exception as e:
                print(f"[Dance] ❌ 启动失败: {e}")
                self._dance_mode = False
                self._dance_player = None
                self.drivers.send_open_pose()
        else:
            if self._dance_player:
                self._dance_player.stop()
            self._dance_player = None
            self.drivers.send_open_pose()

    def _overlay_dance_status(self, frame: np.ndarray) -> np.ndarray:
        """在画面上叠加手势舞状态条"""
        from linkerbot.viz.text import put_text

        h, w = frame.shape[:2]
        bar_h = 36
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - bar_h), (w, h), (20, 20, 50), -1)
        status = self._dance_player.status if self._dance_player else ""
        overlay = put_text(overlay, status, (10, h - bar_h + 6), font_size=20, color=(255, 200, 50))
        return overlay

    def _toggle_voice_mode(self) -> None:
        self._voice_mode = not self._voice_mode
        if self._voice_mode:
            self._voice_controller.start()
            self._enabled = True
            self.state.enabled = True
        else:
            self._voice_controller.stop()
            self._gesture_recorder.toggle()  # 确保退出录制
            self.drivers.send_open_pose()

    def _shutdown(self) -> None:
        if self._dance_player:
            self._dance_player.stop()
        if self._voice_controller:
            self._voice_controller.stop()
        if self.tracker:
            self.tracker.close()
        if self.camera:
            self.camera.release()
        if self.drivers:
            self.drivers.disconnect()

    def _read_track(self) -> tuple[bool, Any, TrackingFrame | None]:
        assert self.camera and self.tracker
        ok, frame = self.camera.read()
        if not ok or frame is None:
            return False, None, None
        ts = int(time.time() * 1000)
        tracking = self.tracker.process(frame, timestamp_ms=ts)
        mapped = _remap_frame(tracking, self._mirror)
        assert self.session
        _assign_hand_tracking(self.state.hands, mapped, self.session.active_sides)
        return True, frame, tracking

    def _with_sim_panel(self, frame: np.ndarray) -> np.ndarray:
        if not self.session or self.session.hardware_mode != HardwareMode.MOCK:
            return frame
        if not MujocoSimRegistry.is_embedded():
            return frame
        sim_cfg = self.base_config.get("simulation", {})
        renders = MujocoSimRegistry.render_all()
        if not renders:
            return frame
        return compose_split(frame, renders, int(sim_cfg.get("panel_width", 720)))

    def _update_fps(self) -> None:
        self._frame_count += 1
        elapsed = time.time() - self._fps_timer
        if elapsed >= 1.0:
            self.state.fps = self._frame_count / elapsed
            self._frame_count = 0
            self._fps_timer = time.time()

    def _show_camera_error(self) -> None:
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        msg = self._camera_error or "摄像头读取失败，正在重试..."
        hint = f"设备: /dev/video{self.camera.device_id if self.camera else '?'} | 按 Q 退出"
        display = render_texts(blank, [
            ("摄像头异常", (40, 80), 28, (80, 80, 255)),
            (msg, (40, 140), 20, (200, 200, 200)),
            (hint, (40, 200), 18, (150, 150, 150)),
            ("可尝试: python main.py --camera 1", (40, 240), 18, (150, 150, 150)),
        ])
        cv2.imshow(self.window_name, display)

    def _step_init(self) -> bool:
        assert self.calibrator and self.session
        ok, frame, _ = self._read_track()
        if not ok:
            self._camera_error = "无法从摄像头读取画面，请检查连接或尝试 --camera 1"
            self._show_camera_error()
            cv2.waitKey(1)
            return True
        self._camera_error = ""

        result = self.calibrator.update(
            TrackingFrame(hands={s: self.state.hands[s].tracking for s in self.state.hands}),
            mirror=self._mirror,
        )
        self.state.init_progress = result.progress
        self.state.init_message = result.message

        for side in self.session.active_sides:
            s = side.value
            hs = self.state.hands.get(s)
            rt = self.retargeters.get(s)
            if (
                _has_spread_calibration(rt)
                and hs
                and hs.tracking.detected
                and result.side_status.get(s, "").startswith("保持")
            ):
                rt.accumulate_spread_sample(hs.tracking)

        display = draw_init_overlay(frame, self.state, self.calibrator, self.session.active_sides, self._mirror)
        cv2.imshow(self.window_name, self._with_sim_panel(display))
        self._update_fps()

        if result.ready:
            for r in self.retargeters.values():
                r.reset()
            for side, hs in self.state.hands.items():
                rt = self.retargeters.get(side)
                if _has_spread_calibration(rt):
                    if not rt.finalize_spread_calibration():
                        if hs.tracking.detected:
                            rt.calibrate_spread(hs.tracking)
            self.phase = AppPhase.TELEOP
            self.state.phase = "teleop"
            auto_start = self.base_config.get("viz", {}).get("auto_start_teleop", True)
            self._enabled = auto_start
            self.state.enabled = auto_start
            if auto_start:
                print("初始化完成，遥操作已自动开始")
            else:
                print("初始化完成，按 Space 开始遥操作")
        return True

    def _step_teleop(self) -> bool:
        assert self.drivers and self.profile and self.session
        ok, frame, _ = self._read_track()
        if not ok:
            self._camera_error = "摄像头读取中断，正在自动重连..."
            self._show_camera_error()
            cv2.waitKey(1)
            return True
        self._camera_error = ""

        # ---- 手势舞模式 ----
        if self._dance_mode and self._dance_player:
            pose = self._dance_player.update()
            if pose is None:
                # 播放结束，自动退出
                self._dance_mode = False
                self._dance_player = None
                self.drivers.send_open_pose()
            else:
                for side in self.session.active_sides:
                    self.drivers.send(side.value, pose)
            viz = self.base_config.get("viz", {})
            display = draw_teleop_overlay(
                frame,
                self.state,
                viz.get("show_landmarks", True),
                viz.get("show_joint_panel", True),
                self.session.hand_model,
            )
            display = self._overlay_dance_status(display)
            cv2.imshow(self.window_name, self._with_sim_panel(display))
            self._update_fps()
            return True

        # ---- 语音模式 ----
        if self._voice_mode:
            now = time.time()
            gesture_name = self._voice_controller.update()
            if gesture_name and self._gesture_library:
                pose = self._gesture_library.lookup(gesture_name)
                if pose:
                    for side in self.session.active_sides:
                        self.drivers.send(side.value, pose)
                    self._last_send = now
            # 录制模式
            if self._gesture_recorder and self._gesture_recorder.recording:
                self._handle_recording_input()

            viz = self.base_config.get("viz", {})
            display = draw_teleop_overlay(
                frame,
                self.state,
                viz.get("show_landmarks", True),
                viz.get("show_joint_panel", True),
                self.session.hand_model,
            )
            display = self._overlay_voice_status(display)
            cv2.imshow(self.window_name, self._with_sim_panel(display))
            self._update_fps()
            return True

        # ---- 跟手模式 ----
        hw = self.base_config["hardware"]
        interval = 1.0 / hw.get("send_rate_hz", 30)
        now = time.time()
        should_send = now - self._last_send >= interval

        for side, hs in self.state.hands.items():
            rt = self.retargeters[side]
            joints = rt.retarget(hs.tracking)
            if joints is None:
                continue
            hs.joints = joints
            hs.hardware_pose = joints.to_hardware_pose(side)
            hs.pinch_raw = float(getattr(rt, "last_pinch_raw", 0.0))
            hs.pinch_strength = float(getattr(rt, "last_pinch", 0.0))
            hs.spread_im = float(getattr(rt, "last_spread_im", 0.0))
            if self._enabled and should_send:
                self.drivers.send(side, hs.hardware_pose)

        if self._enabled and should_send:
            self._last_send = now

        viz = self.base_config.get("viz", {})
        display = draw_teleop_overlay(
            frame,
            self.state,
            viz.get("show_landmarks", True),
            viz.get("show_joint_panel", True),
            self.session.hand_model,
        )
        cv2.imshow(self.window_name, self._with_sim_panel(display))
        self._update_fps()
        return True

    def _handle_recording_input(self) -> None:
        """录制模式下从终端读取姿态名"""
        # 用非阻塞方式检测终端输入（仅提示，实际录制在终端操作）
        print(
            f"\r🎥 当前姿态: {self._get_current_hardware_pose() or 'N/A'} | "
            "输入姿态名保存，留空跳过 | 按 R 退出录制",
            end="",
            flush=True,
        )

    def _overlay_voice_status(self, frame: np.ndarray) -> np.ndarray:
        """在画面上叠加语音模式状态条"""
        from linkerbot.viz.text import put_text

        h, w = frame.shape[:2]
        bar_h = 36
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - bar_h), (w, h), (30, 30, 35), -1)
        status = self._voice_controller.status if self._voice_controller else ""
        if self._gesture_recorder and self._gesture_recorder.recording:
            status = self._gesture_recorder.status
        overlay = put_text(overlay, status, (10, h - bar_h + 6), font_size=20, color=(0, 220, 130))
        return overlay

    def _handle_r_key(self) -> None:
        """R 键：语音模式下录制姿态，跟手模式下重置滤波器"""
        if self._voice_mode:
            status = self._gesture_recorder.toggle()
            print(status if status else "已退出录制")
        else:
            for r in self.retargeters.values():
                r.reset()

    def _toggle_enabled(self) -> None:
        self._enabled = not self._enabled
        self.state.enabled = self._enabled
        if not self._enabled:
            for r in self.retargeters.values():
                r.reset()
            if self.drivers:
                self.drivers.send_open_pose()

    def _restart_init(self) -> None:
        if self.calibrator:
            self.calibrator.reset()
        for r in self.retargeters.values():
            r.reset()
            if _has_spread_calibration(r):
                r.clear_spread_accumulator()
        self._enabled = False
        self.state.enabled = False
        self.phase = AppPhase.INIT
        self.state.phase = "init"
        self.state.init_progress = 0.0
        if self.drivers:
            self.drivers.send_open_pose()
