from __future__ import annotations

from typing import Callable, List, Tuple

import cv2
import numpy as np

from linkerbot.config.session import HandMode, HandSide, HardwareMode, SessionConfig
from linkerbot.viz.text import render_texts
from linkerbot.viz.window import setup_window

Option = Tuple[str, str, Callable[[SessionConfig], SessionConfig]]


def _set_model(model: str) -> Callable[[SessionConfig], SessionConfig]:
    def apply(s: SessionConfig) -> SessionConfig:
        s.hand_model = model
        return s
    return apply


def _set_mode(mode: HandMode) -> Callable[[SessionConfig], SessionConfig]:
    def apply(s: SessionConfig) -> SessionConfig:
        s.hand_mode = mode
        if mode == HandMode.DUAL:
            s.active_sides = [HandSide.LEFT, HandSide.RIGHT]
        elif len(s.active_sides) != 1:
            s.active_sides = [HandSide.RIGHT]
        return s
    return apply


def _set_side(side: HandSide) -> Callable[[SessionConfig], SessionConfig]:
    def apply(s: SessionConfig) -> SessionConfig:
        if s.hand_mode == HandMode.SINGLE:
            s.active_sides = [side]
        return s
    return apply


def _set_hw(mode: HardwareMode) -> Callable[[SessionConfig], SessionConfig]:
    def apply(s: SessionConfig) -> SessionConfig:
        s.hardware_mode = mode
        return s
    return apply


class SetupWizard:
    STEPS = [
        ("选择灵巧手型号", [
            ("1 - O6", "O6", _set_model("O6")),
            ("2 - L7", "L7", _set_model("L7")),
            ("3 - L10 (推荐)", "L10", _set_model("L10")),
            ("4 - L20", "L20", _set_model("L20")),
        ]),
        ("选择控制模式", [
            ("1 - 单手", "single", _set_mode(HandMode.SINGLE)),
            ("2 - 双手", "dual", _set_mode(HandMode.DUAL)),
        ]),
        ("选择手别 (单手)", [
            ("1 - 左手", "left", _set_side(HandSide.LEFT)),
            ("2 - 右手", "right", _set_side(HandSide.RIGHT)),
        ]),
        ("选择硬件连接", [
            ("1 - MuJoCo 仿真 (Mock)", "mock", _set_hw(HardwareMode.MOCK)),
            ("2 - 真机 SDK", "sdk", _set_hw(HardwareMode.LINKER_SDK)),
        ]),
    ]

    def __init__(
        self,
        window_name: str = "LinkerBot Setup",
        window_width: int = 960,
        window_height: int = 680,
    ):
        self.window_name = window_name
        self.window_width = window_width
        self.window_height = window_height
        self.session = SessionConfig()
        self.step = 0
        self._hover = -1

    def _skip_side_step(self) -> bool:
        return self.session.hand_mode == HandMode.DUAL

    def _effective_steps(self) -> List[tuple]:
        steps = []
        for title, opts in self.STEPS:
            if title.startswith("选择手别") and self._skip_side_step():
                continue
            steps.append((title, opts))
        return steps

    def _draw(self) -> np.ndarray:
        steps = self._effective_steps()
        title, options = steps[self.step]
        w, h = 900, 620
        canvas = np.full((h, w, 3), 28, dtype=np.uint8)

        labels: list[tuple[str, tuple[int, int], int, tuple[int, int, int]]] = [
            ("LinkerBot 灵巧手遥操作", (40, 20), 32, (240, 240, 240)),
            (f"步骤 {self.step + 1}/{len(steps)}: {title}", (40, 68), 24, (100, 200, 255)),
            (f"当前: {self.session.summary_zh()}", (40, 108), 18, (180, 180, 180)),
        ]

        y0 = 190
        self._hitboxes: List[Tuple[int, int, int, int, int]] = []
        for i, (label, _, _) in enumerate(options):
            y1, y2 = y0 + i * 70, y0 + i * 70 + 52
            color = (60, 140, 60) if i == self._hover else (45, 45, 45)
            cv2.rectangle(canvas, (60, y1), (840, y2), color, -1)
            cv2.rectangle(canvas, (60, y1), (840, y2), (90, 90, 90), 1)
            labels.append((label, (85, y1 + 8), 24, (230, 230, 230)))
            self._hitboxes.append((60, y1, 840, y2, i))

        labels.append(("鼠标点击选项 | B: 上一步 | Enter: 开始初始化 | Q: 退出", (40, h - 36), 18, (140, 140, 140)))
        return render_texts(canvas, labels)

    def _on_mouse(self, event, x, y, _flags, _param) -> None:
        if event == cv2.EVENT_MOUSEMOVE:
            self._hover = -1
            for x1, y1, x2, y2, idx in getattr(self, "_hitboxes", []):
                if x1 <= x <= x2 and y1 <= y <= y2:
                    self._hover = idx
                    break
            return
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        for x1, y1, x2, y2, idx in getattr(self, "_hitboxes", []):
            if x1 <= x <= x2 and y1 <= y <= y2:
                self._select(idx)
                break

    def _select(self, idx: int) -> None:
        steps = self._effective_steps()
        _, options = steps[self.step]
        self.session = options[idx][2](self.session)
        if self.step < len(steps) - 1:
            self.step += 1

    def run(self) -> SessionConfig | None:
        setup_window(self.window_name, self.window_width, self.window_height)
        cv2.setMouseCallback(self.window_name, self._on_mouse)
        try:
            while True:
                canvas = self._draw()
                cv2.imshow(self.window_name, canvas)
                key = cv2.waitKey(30) & 0xFF
                if key == ord("q"):
                    return None
                if key == ord("b") and self.step > 0:
                    self.step -= 1
                if key in (ord("1"), ord("2"), ord("3"), ord("4")):
                    idx = key - ord("1")
                    steps = self._effective_steps()
                    if idx < len(steps[self.step][1]):
                        self._select(idx)
                if key in (13, 10):
                    return self.session
        finally:
            cv2.destroyWindow(self.window_name)
