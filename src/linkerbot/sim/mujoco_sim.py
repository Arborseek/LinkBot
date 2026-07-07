from __future__ import annotations

import subprocess
import threading
import time
from pathlib import Path
from typing import Dict, List, Literal

import cv2
import mujoco
import numpy as np

from linkerbot.sim.l20_kinematics import is_l20_urdf_model, pose_to_l20_qpos
from linkerbot.sim.pose_mapping import open_ctrl, pose_to_mujoco_ctrl

ROOT = Path(__file__).resolve().parents[3]
DisplayMode = Literal["embedded", "window"]

L10_MODEL = "assets/mujoco/linker_hand_l10/linker_hand_l10_left.xml"

MODEL_TABLE: Dict[str, str] = {
    "L10": L10_MODEL,
    "L7": L10_MODEL,
    "O6": L10_MODEL,
    "L6": L10_MODEL,
}


def resolve_model_path(hand_model: str, side: str = "left", configured: str | None = None) -> Path:
    if configured:
        path = Path(configured)
        if not path.is_absolute():
            path = ROOT / path
    elif hand_model == "L20":
        s = side.lower()
        path = ROOT / f"assets/mujoco/linker_hand_l20/{s}/linkerhand_l20_{s}.urdf"
    else:
        path = ROOT / MODEL_TABLE.get(hand_model, L10_MODEL)
    if not path.exists():
        if hand_model == "L20":
            raise FileNotFoundError(
                f"L20 MuJoCo 模型不存在: {path}\n"
                "请先运行: bash scripts/setup_linker_urdf.sh"
            )
        raise FileNotFoundError(f"MuJoCo 模型不存在: {path}")
    return path


def _try_resize_native_window(width: int, height: int) -> None:
    time.sleep(0.6)
    for cmd in [
        ["xdotool", "search", "--name", "MuJoCo", "windowmove", "0", "0", "windowsize", str(width), str(height)],
        ["wmctrl", "-r", "MuJoCo", "-e", f"0,0,0,{width},{height}"],
    ]:
        try:
            subprocess.run(cmd, capture_output=True, timeout=2, check=False)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue


def _ensure_offscreen_buffer(model: mujoco.MjModel, width: int, height: int) -> None:
    """URDF 默认 framebuffer 较小，渲染前需扩大 offwidth/offheight"""
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), width)
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), height)


class MujocoHandSim:
    """MuJoCo 仿真：L20 用官方 URDF 运动学；其他型号用 L10 MJCF 执行器"""

    def __init__(
        self,
        model_path: Path,
        side: str = "right",
        display: DisplayMode = "embedded",
        render_width: int = 720,
        render_height: int = 720,
        window_width: int = 1400,
        window_height: int = 900,
        hand_model: str = "L20",
    ):
        self.side = side
        self.hand_model = hand_model
        self.display = display
        self.render_width = render_width
        self.render_height = render_height
        self.window_width = window_width
        self.window_height = window_height

        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self._l20_native = is_l20_urdf_model(self.model)
        self.ctrl_ranges = self.model.actuator_ctrlrange.copy()
        self.dof = self.model.nu

        self._target_ctrl = np.zeros(max(self.dof, 1), dtype=np.float64)
        self._target_pose: List[int] = [255] * 20

        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.azimuth = 135
        self.camera.elevation = -18
        self.camera.distance = 0.35 if self._l20_native else 0.42
        self.camera.lookat[:] = [0.0, 0.0, 0.13]

        self.renderer: mujoco.Renderer | None = None
        if display == "embedded":
            _ensure_offscreen_buffer(self.model, render_width, render_height)
            self.renderer = mujoco.Renderer(self.model, height=render_height, width=render_width)

        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._needs_update = True
        self._last_render: np.ndarray | None = None
        self._display_blend = 0.32
        self._shown_qpos: np.ndarray | None = None

        self.set_open_pose()

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        if self.display == "window":
            self._thread = threading.Thread(target=self._viewer_loop, name=f"mujoco-{self.side}", daemon=True)
            self._thread.start()
            threading.Thread(
                target=_try_resize_native_window,
                args=(self.window_width, self.window_height),
                daemon=True,
            ).start()
        time.sleep(0.15)

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self.renderer:
            self.renderer.close()
            self.renderer = None

    def set_pose(self, pose: List[int], hand_model: str | None = None) -> None:
        model = hand_model or self.hand_model
        with self._lock:
            if self._l20_native and model == "L20":
                new_pose = list(pose[:20]) if len(pose) >= 20 else list(pose) + [255] * (20 - len(pose))
                if self._target_pose == new_pose:
                    return
                self._target_pose = new_pose
            else:
                ctrl = pose_to_mujoco_ctrl(pose, model, self.side, self.ctrl_ranges, self.dof)
                if np.allclose(self._target_ctrl, ctrl):
                    return
                self._target_ctrl = ctrl
            self._needs_update = True
            self._last_render = None

    def set_open_pose(self) -> None:
        with self._lock:
            if self._l20_native:
                self._target_pose = [255] * 20
            else:
                self._target_ctrl = open_ctrl(self.ctrl_ranges, self.dof, self.hand_model, self.side)
            self._needs_update = True
            self._last_render = None
        self._apply_kinematic()

    def _apply_kinematic(self) -> None:
        with self._lock:
            if self._l20_native:
                target = pose_to_l20_qpos(self._target_pose, self.model, self.side)
                if self._display_blend > 0.0 and self._shown_qpos is not None:
                    err = float(np.max(np.abs(target - self._shown_qpos)))
                    blend = self._display_blend
                    if err > 0.12:
                        blend = min(0.78, blend * 2.4)
                    if err > 0.35:
                        blend = min(0.92, blend * 1.15)
                    self.data.qpos[:] = self._shown_qpos + blend * (target - self._shown_qpos)
                else:
                    self.data.qpos[:] = target
                self._shown_qpos = self.data.qpos.copy()
            else:
                self.data.ctrl[:] = self._target_ctrl
                ctrl = self._target_ctrl.copy()
                for i in range(self.model.nu):
                    if self.model.actuator_trntype[i] != mujoco.mjtTrn.mjTRN_JOINT:
                        continue
                    joint_id = int(self.model.actuator_trnid[i, 0])
                    qposadr = int(self.model.jnt_qposadr[joint_id])
                    self.data.qpos[qposadr] = ctrl[i]
        mujoco.mj_forward(self.model, self.data)
        self._needs_update = False

    def render_bgr(self) -> np.ndarray:
        if self.renderer is None:
            raise RuntimeError("embedded 模式才支持 render_bgr")
        if self._needs_update or self._last_render is None:
            self._apply_kinematic()
            self.renderer.update_scene(self.data, self.camera)
            rgb = self.renderer.render()
            self._last_render = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return self._last_render

    def _viewer_loop(self) -> None:
        import mujoco.viewer as mjv

        with mjv.launch_passive(
            self.model,
            self.data,
            show_left_ui=False,
            show_right_ui=False,
        ) as viewer:
            viewer.cam.azimuth = self.camera.azimuth
            viewer.cam.elevation = self.camera.elevation
            viewer.cam.distance = self.camera.distance
            viewer.cam.lookat[:] = self.camera.lookat

            while self._running and viewer.is_running():
                if self._needs_update:
                    self._apply_kinematic()
                viewer.sync()
                time.sleep(0.016)


class MujocoSimRegistry:
    _sims: Dict[str, MujocoHandSim] = {}
    _cfg: dict = {}
    _lock = threading.Lock()

    @classmethod
    def configure(cls, sim_cfg: dict) -> None:
        cls._cfg = dict(sim_cfg)

    @classmethod
    def _display_blend(cls) -> float:
        return float(cls._cfg.get("display_smooth", 0.32))

    @classmethod
    def acquire(cls, side: str, hand_model: str, model_path: str | None = None) -> MujocoHandSim:
        with cls._lock:
            if side in cls._sims:
                return cls._sims[side]
            cfg_path = model_path or cls._cfg.get("mujoco_model")
            if hand_model == "L20" and cfg_path and "linker_hand_l10" in str(cfg_path):
                cfg_path = None
            path = resolve_model_path(hand_model, side, cfg_path)
            display: DisplayMode = cls._cfg.get("display", "embedded")
            sim = MujocoHandSim(
                path,
                side=side,
                hand_model=hand_model,
                display=display,
                render_width=int(cls._cfg.get("render_width", 720)),
                render_height=int(cls._cfg.get("render_height", 720)),
                window_width=int(cls._cfg.get("window_width", 1400)),
                window_height=int(cls._cfg.get("window_height", 900)),
            )
            sim._display_blend = cls._display_blend()
            sim.start()
            cls._sims[side] = sim
            kind = "L20 官方 URDF" if sim._l20_native else "L10 MJCF"
            mode = "嵌入主窗口" if display == "embedded" else "独立窗口"
            print(f"[MuJoCo] 仿真已启动 side={side} model={hand_model} ({kind}, {mode})")
            return sim

    @classmethod
    def render_all(cls) -> Dict[str, np.ndarray]:
        with cls._lock:
            sims = list(cls._sims.items())
        out: Dict[str, np.ndarray] = {}
        for side, sim in sims:
            if sim.renderer is not None:
                out[side] = sim.render_bgr()
        return out

    @classmethod
    def is_embedded(cls) -> bool:
        return cls._cfg.get("display", "embedded") == "embedded"

    @classmethod
    def release(cls, side: str) -> None:
        with cls._lock:
            sim = cls._sims.pop(side, None)
            if sim:
                sim.stop()
                print(f"[MuJoCo] 仿真已关闭 side={side}")

    @classmethod
    def release_all(cls) -> None:
        with cls._lock:
            sides = list(cls._sims.keys())
        for side in sides:
            cls.release(side)
