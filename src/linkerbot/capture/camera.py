from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from pathlib import Path

import cv2

_PROBE_EXECUTOR = ThreadPoolExecutor(max_workers=1)

# 按优先级尝试的分辨率 / 编码
_PROBE_PROFILES: list[tuple[int, int, str | None]] = [
    (1280, 720, "MJPG"),
    (1280, 720, "YUYV"),
    (1280, 720, None),
    (640, 480, "MJPG"),
    (640, 480, "YUYV"),
    (640, 480, None),
]


def list_video_devices(max_id: int = 8) -> list[int]:
    return [i for i in range(max_id) if Path(f"/dev/video{i}").exists()]


def list_capture_devices(max_id: int = 8) -> list[int]:
    """只返回 index=0 的真实采集节点，跳过 Metadata 等虚拟节点"""
    out: list[int] = []
    for dev in list_video_devices(max_id):
        if _is_capture_device(dev):
            out.append(dev)
    return out


def _device_name(device_id: int) -> str:
    name_path = Path(f"/sys/class/video4linux/video{device_id}/name")
    if name_path.exists():
        return name_path.read_text(encoding="utf-8", errors="ignore").strip()
    return ""


def _device_index(device_id: int) -> int | None:
    index_path = Path(f"/sys/class/video4linux/video{device_id}/index")
    if not index_path.exists():
        return None
    try:
        return int(index_path.read_text(encoding="utf-8").strip())
    except ValueError:
        return None


def _is_capture_device(device_id: int) -> bool:
    idx = _device_index(device_id)
    if idx is not None and idx != 0:
        return False
    name = _device_name(device_id).lower()
    if not name:
        return True
    skip_keys = ("metadata", "ir", "depth", "front", "rear")
    return not any(k in name for k in skip_keys)


def _apply_fourcc(cap: cv2.VideoCapture, fourcc: str | None) -> None:
    if fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))


def _set_timeouts(cap: cv2.VideoCapture, open_ms: int = 5000, read_ms: int = 3000) -> None:
    if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, open_ms)
    if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, read_ms)


def _open_capture(device_id: int) -> cv2.VideoCapture | None:
    if sys.platform == "linux":
        cap = cv2.VideoCapture(device_id, cv2.CAP_V4L2)
    else:
        cap = cv2.VideoCapture(device_id)
    if not cap.isOpened():
        cap.release()
        return None
    _set_timeouts(cap)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def _try_read_frames(cap: cv2.VideoCapture, attempts: int = 8, delay: float = 0.08) -> bool:
    for _ in range(attempts):
        ok, frame = cap.read()
        if ok and frame is not None and frame.size > 0:
            return True
        time.sleep(delay)
    return False


def _probe_profile(
    device_id: int,
    width: int,
    height: int,
    fourcc: str | None,
    warmup: int,
) -> tuple[bool, int, int, str | None]:
    cap = _open_capture(device_id)
    if cap is None:
        return False, width, height, fourcc
    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        _apply_fourcc(cap, fourcc)
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if _try_read_frames(cap, attempts=max(warmup, 3)):
            return True, actual_w, actual_h, fourcc
        return False, width, height, fourcc
    finally:
        cap.release()


def _probe_sync(
    device_id: int,
    width: int,
    height: int,
    warmup: int,
) -> tuple[bool, int, int, str | None]:
    profiles: list[tuple[int, int, str | None]] = [(width, height, "MJPG"), (width, height, None)]
    for p in _PROBE_PROFILES:
        if p not in profiles:
            profiles.append(p)

    for w, h, fc in profiles:
        ok, aw, ah, used_fc = _probe_profile(device_id, w, h, fc, warmup)
        if ok:
            return True, aw, ah, used_fc
    return False, width, height, None


def probe_camera(
    device_id: int,
    width: int,
    height: int,
    warmup: int = 3,
    timeout: float = 12.0,
) -> tuple[bool, int, int, str | None]:
    if not _is_capture_device(device_id):
        return False, width, height, None
    future = _PROBE_EXECUTOR.submit(_probe_sync, device_id, width, height, warmup)
    try:
        return future.result(timeout=timeout)
    except FuturesTimeout:
        return False, width, height, None


def find_working_camera(
    preferred: int = 0,
    width: int = 1280,
    height: int = 720,
    warmup: int = 3,
    timeout: float = 12.0,
) -> tuple[int, int, int, str | None] | None:
    devices = list_capture_devices()
    candidates: list[int] = []
    if preferred in devices:
        candidates.append(preferred)
    candidates.extend(d for d in devices if d != preferred)

    for dev in candidates:
        ok, aw, ah, fc = probe_camera(dev, width, height, warmup, timeout)
        if ok:
            return dev, aw, ah, fc
    return None


class CameraCapture:
    def __init__(
        self,
        device_id: int = 0,
        width: int = 1280,
        height: int = 720,
        mirror: bool = True,
        auto_detect: bool = True,
        warmup_frames: int = 5,
        read_retries: int = 3,
        probe_timeout: float = 12.0,
        strict_probe: bool = False,
    ):
        self.device_id = device_id
        self.width = width
        self.height = height
        self.mirror = mirror
        self.auto_detect = auto_detect
        self.warmup_frames = warmup_frames
        self.read_retries = read_retries
        self.probe_timeout = probe_timeout
        self.strict_probe = strict_probe
        self._fourcc: str | None = "MJPG"
        self._cap: cv2.VideoCapture | None = None
        self._fail_count = 0

    def open(self) -> None:
        dev = self.device_id
        target_w, target_h = self.width, self.height
        chosen_fc: str | None = "MJPG"

        if self.auto_detect:
            ok, aw, ah, fc = probe_camera(
                dev, self.width, self.height, 3, self.probe_timeout
            )
            if ok:
                target_w, target_h, chosen_fc = aw, ah, fc
            else:
                found = find_working_camera(
                    dev, self.width, self.height, 3, self.probe_timeout
                )
                if found is not None:
                    dev, target_w, target_h, chosen_fc = found
                    if dev != self.device_id:
                        print(f"摄像头 /dev/video{self.device_id} 不可用，已切换到 /dev/video{dev}")
                elif self.strict_probe:
                    self._raise_not_found()
                else:
                    capture_devs = list_capture_devices()
                    if dev not in capture_devs and capture_devs:
                        dev = capture_devs[0]
                        print(f"将尝试直接打开采集设备 /dev/video{dev}（跳过 probe）")
                    elif dev not in capture_devs:
                        self._raise_not_found()
                    else:
                        print(
                            f"摄像头 probe 未通过，仍尝试打开 /dev/video{dev}。"
                            "若失败请关闭占用摄像头的程序后重试。"
                        )
                    target_w, target_h, chosen_fc = 640, 480, None

        self.device_id = dev
        self.width = target_w
        self.height = target_h
        self._fourcc = chosen_fc
        self._cap = _open_capture(dev)
        if self._cap is None:
            raise RuntimeError(f"无法打开 /dev/video{dev}")

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        _apply_fourcc(self._cap, self._fourcc)

        warmed = 0
        for _ in range(self.warmup_frames):
            if self._cap.read()[0]:
                warmed += 1
            time.sleep(0.05)

        if warmed == 0 and not _try_read_frames(self._cap, attempts=10, delay=0.1):
            self.release()
            names = {d: _device_name(d) for d in list_video_devices()}
            capture = list_capture_devices()
            raise RuntimeError(
                "摄像头已打开但无法读取画面。\n"
                f"  当前设备: /dev/video{dev} ({_device_name(dev)})\n"
                f"  全部节点: {names}\n"
                f"  采集节点: {capture}\n"
                "  请确认:\n"
                "  1. 没有其他程序占用摄像头（浏览器、Cheese 等）\n"
                "  2. 用户已加入 video 组: sudo usermod -aG video $USER 后重新登录\n"
                "  3. 尝试: python main.py --camera 0"
            )

        self._fail_count = 0
        name = _device_name(dev)
        fc = self._fourcc or "auto"
        extra = f" ({name})" if name else ""
        print(
            f"摄像头已打开: /dev/video{dev}{extra} "
            f"{self.width}x{self.height} fourcc={fc} warmup={warmed}"
        )

    def _raise_not_found(self) -> None:
        names = {d: _device_name(d) for d in list_video_devices()}
        capture = list_capture_devices()
        raise RuntimeError(
            "未找到可用摄像头。\n"
            f"  全部节点: {names}\n"
            f"  采集节点( index=0 ): {capture}\n"
            "  说明: 同名 /dev/video0 与 /dev/video1 通常只有 index=0 才是画面。\n"
            "  请确认:\n"
            "  1. 摄像头未被其他程序占用\n"
            "  2. sudo usermod -aG video $USER 后重新登录\n"
            "  3. 尝试: python main.py --camera 0"
        )

    def read(self) -> tuple[bool, "cv2.Mat | None"]:
        if self._cap is None:
            raise RuntimeError("摄像头未打开")

        for _ in range(self.read_retries):
            ok, frame = self._cap.read()
            if ok and frame is not None:
                self._fail_count = 0
                if self.mirror:
                    frame = cv2.flip(frame, 1)
                return True, frame
            time.sleep(0.05)

        self._fail_count += 1
        if self._fail_count >= 8:
            self.reopen()
        return False, None

    def reopen(self) -> bool:
        print("正在重新连接摄像头...")
        self.release()
        try:
            self.open()
            return True
        except RuntimeError as exc:
            print(exc)
            return False

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> CameraCapture:
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


def ensure_model(model_path: str) -> Path:
    path = Path(model_path)
    if path.exists():
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    url = (
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    )
    import urllib.request

    print(f"下载 MediaPipe 手部模型到 {path} ...")
    urllib.request.urlretrieve(url, path)
    return path
