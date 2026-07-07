from __future__ import annotations

from typing import List

import cv2
import numpy as np

from linkerbot.config.session import HandSide
from linkerbot.models import PipelineState
from linkerbot.setup.init_calibrator import InitCalibrator
from linkerbot.tracking.hand_tracker import HAND_CONNECTIONS
from linkerbot.viz.text import render_texts


def _to_pixel(lm: np.ndarray, w: int, h: int) -> tuple[int, int]:
    return int(lm[0] * w), int(lm[1] * h)


def draw_init_overlay(
    frame: np.ndarray,
    state: PipelineState,
    calibrator: InitCalibrator,
    active_sides: List[HandSide],
    mirror: bool,
) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]

    cv2.rectangle(out, (10, 10), (min(w - 10, 680), 120), (0, 0, 0), -1)

    labels: list[tuple[str, tuple[int, int], int, tuple[int, int, int]]] = [
        ("初始化校准", (20, 10), 28, (100, 220, 255)),
        (state.init_message, (20, 48), 20, (220, 220, 220)),
        ("请将手放入绿色区域，保持张开手掌", (20, 78), 18, (180, 180, 180)),
    ]

    for side in active_sides:
        s = side.value
        x1, y1, x2, y2 = calibrator.roi_rect_px(s, w, h, mirror)
        color = (0, 220, 80) if state.init_progress >= 1.0 else (0, 180, 255)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = "左" if s == "left" else "右"
        labels.append((f"{label}手区域", (x1 + 8, y1 + 4), 22, color))

        hs = state.hands.get(s)
        if hs and hs.tracking.detected and hs.tracking.landmarks is not None:
            pts = hs.tracking.landmarks
            for i, j in HAND_CONNECTIONS:
                cv2.line(out, _to_pixel(pts[i], w, h), _to_pixel(pts[j], w, h), (0, 255, 128), 2)
            px, py = _to_pixel(pts[0], w, h)
            cv2.circle(out, (px, py), 8, (0, 255, 255), -1)

    bar_w = int((w - 80) * state.init_progress)
    cv2.rectangle(out, (40, h - 50), (w - 40, h - 22), (50, 50, 50), -1)
    cv2.rectangle(out, (40, h - 50), (40 + bar_w, h - 22), (0, 200, 80), -1)
    labels.append((f"{int(state.init_progress * 100)}%", (w // 2 - 20, h - 46), 18, (255, 255, 255)))

    return render_texts(out, labels)


def draw_teleop_overlay(
    frame: np.ndarray,
    state: PipelineState,
    show_landmarks: bool,
    show_panel: bool,
    hand_model: str,
) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]

    for side, hs in state.hands.items():
        if show_landmarks and hs.tracking.detected and hs.tracking.landmarks is not None:
            pts = hs.tracking.landmarks
            color_line = (0, 255, 180) if side == "right" else (255, 180, 0)
            for i, j in HAND_CONNECTIONS:
                cv2.line(out, _to_pixel(pts[i], w, h), _to_pixel(pts[j], w, h), color_line, 2)

    status = "遥操作中" if state.enabled else "待命 (按 Space 开始)"
    sc = (0, 220, 0) if state.enabled else (0, 200, 255)
    cv2.rectangle(out, (10, 10), (520, 130), (0, 0, 0), -1)

    labels: list[tuple[str, tuple[int, int], int, tuple[int, int, int]]] = [
        (status, (20, 10), 28, sc),
        (state.session_summary, (20, 48), 18, (200, 200, 200)),
        (f"FPS: {state.fps:.1f}", (20, 78), 18, (255, 255, 255)),
        ("Space: 启停 | R: 重置 | I: 重新初始化 | Q: 退出", (20, h - 28), 16, (180, 180, 180)),
    ]

    if show_panel:
        _collect_panels(out, state, hand_model, labels)

    return render_texts(out, labels)


def _collect_panels(
    frame: np.ndarray,
    state: PipelineState,
    hand_model: str,
    labels: list,
) -> None:
    x0 = frame.shape[1] - 340
    y = 20
    for side, hs in state.hands.items():
        if hs.joints is None:
            continue
        label = "左" if side == "left" else "右"
        vals = hs.joints.as_array()
        pose = hs.hardware_pose
        show = [
            (0, "拇根"),
            (1, "拇摆"),
            (9, "拇横"),
            (6, "食摆"),
            (7, "无摆"),
            (8, "小摆"),
            (2, "食根"),
            (5, "小根"),
        ] if hand_model == "L10" else [
            (0, "拇根"),
            (15, "拇尖"),
            (10, "拇横"),
            (5, "拇摆"),
            (6, "食摆"),
            (7, "中摆"),
            (1, "食根"),
            (2, "中根"),
        ]
        panel_h = 24 + len(show) * 16 + 36
        cv2.rectangle(frame, (x0, y), (frame.shape[1] - 10, y + panel_h), (0, 0, 0), -1)
        labels.append((f"{label} {hand_model}", (x0 + 10, y + 2), 18, (255, 255, 255)))
        pinch_c = (0, 255, 120) if hs.pinch_strength > 0.12 else (140, 140, 140)
        labels.append(
            (
                f"捏合 raw={hs.pinch_raw:.2f} act={hs.pinch_strength:.2f} | 叉开 im={hs.spread_im:.2f}",
                (x0 + 10, y + 20),
                13,
                pinch_c,
            )
        )
        for row, (i, tag) in enumerate(show):
            if i >= len(vals):
                continue
            p = pose[i] if i < len(pose) else 0
            labels.append(
                (f"J{i} {tag}: {vals[i]:.2f} [{p:3d}]", (x0 + 10, y + 36 + row * 16), 14, (180, 255, 180))
            )
        y += panel_h + 10
