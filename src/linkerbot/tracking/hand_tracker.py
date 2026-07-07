from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from linkerbot.models import HandTracking, TrackingFrame

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]


class HandTracker:
    def __init__(
        self,
        model_path: str,
        max_hands: int = 1,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.6,
    ):
        model_file = Path(model_path)
        if not model_file.exists():
            from linkerbot.capture.camera import ensure_model
            model_file = ensure_model(model_path)

        base_options = python.BaseOptions(model_asset_path=str(model_file))
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_tracking_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._detector = vision.HandLandmarker.create_from_options(options)
        self._timestamp_ms = 0

    def process(self, frame_bgr: np.ndarray, timestamp_ms: int | None = None) -> TrackingFrame:
        if timestamp_ms is None:
            self._timestamp_ms += 33
            timestamp_ms = self._timestamp_ms
        else:
            self._timestamp_ms = timestamp_ms

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._detector.detect_for_video(mp_image, timestamp_ms)

        frame = TrackingFrame()
        if not result.hand_landmarks:
            return frame

        for i, hand in enumerate(result.hand_landmarks):
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand], dtype=np.float64)
            world = None
            if result.hand_world_landmarks and i < len(result.hand_world_landmarks):
                world = np.array(
                    [[lm.x, lm.y, lm.z] for lm in result.hand_world_landmarks[i]],
                    dtype=np.float64,
                )
            handedness = "Right"
            if result.handedness and i < len(result.handedness):
                handedness = result.handedness[i][0].category_name
            side = handedness.lower()
            frame.hands[side] = HandTracking(
                landmarks=landmarks,
                world_landmarks=world,
                handedness=handedness,
                detected=True,
            )
        return frame

    def close(self) -> None:
        self._detector.close()
