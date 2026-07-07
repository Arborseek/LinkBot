from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from linkerbot.config.session import HandSide
from linkerbot.models import HandTracking, TrackingFrame

WRIST = 0
FINGER_MCPS = [5, 9, 13, 17]
FINGER_PIPS = [6, 10, 14, 18]


def _angle(a, b, c) -> float:
    v1, v2 = a - b, c - b
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    return float(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)))


@dataclass
class InitCheckResult:
    ready: bool
    progress: float
    message: str
    side_status: Dict[str, str]


class InitCalibrator:
    """要求用户在指定 ROI 内摆出张开手掌姿态，稳定后完成初始化"""

    def __init__(
        self,
        rois: Dict[str, List[float]],
        active_sides: List[HandSide],
        hold_frames: int = 45,
        min_mcp_angle: float = 2.2,
    ):
        self.rois = rois
        self.active_sides = [s.value for s in active_sides]
        self.hold_frames = hold_frames
        self.min_mcp_angle = min_mcp_angle
        self._stable: Dict[str, int] = {s: 0 for s in self.active_sides}

    def reset(self) -> None:
        self._stable = {s: 0 for s in self.active_sides}

    def _in_roi(self, pt: np.ndarray, roi: List[float]) -> bool:
        x1, y1, x2, y2 = roi
        return x1 <= pt[0] <= x2 and y1 <= pt[1] <= y2

    def _is_open_palm(self, tr: HandTracking) -> Tuple[bool, str]:
        if not tr.detected or tr.landmarks is None:
            return False, "未检测到手"
        pts = tr.landmarks
        wrist = pts[WRIST]
        curls = [_angle(pts[WRIST], pts[mcp], pts[pip]) for mcp, pip in zip(FINGER_MCPS, FINGER_PIPS)]
        avg = sum(curls) / len(curls)
        if avg < self.min_mcp_angle:
            return False, f"请张开手掌 (弯曲度 {avg:.2f})"
        # 拇指也应较伸展
        thumb = _angle(pts[1], pts[2], pts[3])
        if thumb < 1.5:
            return False, "请伸开拇指"
        return True, "姿态正确"

    def _roi_key(self, side: str, mirror: bool) -> str:
        """镜像画面下左右与真人一致；未镜像时摄像头视角左右相反"""
        if mirror:
            return side
        return "left" if side == "right" else "right"

    def update(self, frame: TrackingFrame, mirror: bool) -> InitCheckResult:
        side_status: Dict[str, str] = {}
        all_ok = True

        for side in self.active_sides:
            roi = self.rois.get(
                self._roi_key(side, mirror),
                self.rois.get(side, [0.2, 0.2, 0.8, 0.8]),
            )
            tr = frame.hands.get(side, HandTracking())

            if not tr.detected or tr.landmarks is None:
                self._stable[side] = 0
                side_status[side] = "等待检测"
                all_ok = False
                continue

            if not self._in_roi(tr.landmarks[WRIST], roi):
                self._stable[side] = 0
                side_status[side] = "请将手移入绿色区域"
                all_ok = False
                continue

            ok, msg = self._is_open_palm(tr)
            if not ok:
                self._stable[side] = 0
                side_status[side] = msg
                all_ok = False
                continue

            self._stable[side] = min(self._stable[side] + 1, self.hold_frames)
            pct = int(100 * self._stable[side] / self.hold_frames)
            side_status[side] = f"保持姿态 {pct}%"

        min_stable = min(self._stable.values()) if self._stable else 0
        progress = min_stable / self.hold_frames
        ready = all_ok and min_stable >= self.hold_frames

        if ready:
            msg = "初始化完成，即将进入遥操作"
        elif len(self.active_sides) > 1:
            msg = " | ".join(f"{s}: {side_status[s]}" for s in self.active_sides)
        else:
            msg = side_status.get(self.active_sides[0], "准备初始化")

        return InitCheckResult(ready=ready, progress=progress, message=msg, side_status=side_status)

    def roi_rect_px(self, side: str, w: int, h: int, mirror: bool) -> Tuple[int, int, int, int]:
        roi = self.rois.get(
            self._roi_key(side, mirror),
            [0.2, 0.2, 0.8, 0.8],
        )
        x1, y1, x2, y2 = roi
        return int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
