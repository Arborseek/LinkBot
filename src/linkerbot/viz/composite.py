from __future__ import annotations

from typing import Dict

import cv2
import numpy as np


def compose_split(
    camera: np.ndarray,
    sim_frames: Dict[str, np.ndarray],
    panel_width: int,
) -> np.ndarray:
    """左侧摄像头 + 右侧 MuJoCo 仿真"""
    if not sim_frames:
        return camera

    h = camera.shape[0]
    panels = []
    for side in sorted(sim_frames.keys()):
        img = sim_frames[side]
        img = cv2.resize(img, (panel_width, h // len(sim_frames) if len(sim_frames) > 1 else h))
        panels.append(img)

    sim_col = np.vstack(panels) if len(panels) > 1 else panels[0]
    if sim_col.shape[0] != h:
        sim_col = cv2.resize(sim_col, (panel_width, h))

    cv2.line(sim_col, (0, 0), (0, h - 1), (80, 80, 80), 2)
    return np.hstack([camera, sim_col])
