from __future__ import annotations

import cv2


def setup_window(name: str, width: int, height: int) -> None:
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, width, height)
