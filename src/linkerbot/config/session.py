from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List

import yaml


class HandMode(str, Enum):
    SINGLE = "single"
    DUAL = "dual"


class HandSide(str, Enum):
    LEFT = "left"
    RIGHT = "right"


class HardwareMode(str, Enum):
    MOCK = "mock"
    LINKER_SDK = "linker_sdk"


@dataclass
class HandProfile:
    model: str
    dof: int
    joint_names: List[str]
    joint_limits: Dict[str, List[float]]
    open_pose: List[int]
    hardware_open_pose: List[int] | None = None
    speed_len: int = 5
    torque_len: int = 5

    def effective_open_pose(self, *, hardware: bool = False) -> List[int]:
        if hardware and self.hardware_open_pose:
            return list(self.hardware_open_pose)
        return list(self.open_pose)


@dataclass
class SessionConfig:
    hand_model: str = "L10"
    hand_mode: HandMode = HandMode.SINGLE
    active_sides: List[HandSide] = field(default_factory=lambda: [HandSide.RIGHT])
    hardware_mode: HardwareMode = HardwareMode.MOCK
    camera_id: int = 0

    @property
    def is_dual(self) -> bool:
        return self.hand_mode == HandMode.DUAL

    @property
    def max_hands(self) -> int:
        return 2 if self.is_dual else 1

    def summary(self) -> str:
        sides = "+".join(s.value for s in self.active_sides)
        return f"{self.hand_model} | {self.hand_mode.value} | {sides} | {self.hardware_mode.value}"

    def summary_zh(self) -> str:
        mode = "双手" if self.is_dual else "单手"
        side_map = {"left": "左", "right": "右"}
        sides = "+".join(side_map.get(s.value, s.value) for s in self.active_sides)
        hw = "MuJoCo仿真" if self.hardware_mode == HardwareMode.MOCK else "真机"
        return f"{self.hand_model} | {mode} | {sides} | {hw}"


def load_hand_profiles(path: str | Path) -> Dict[str, HandProfile]:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    profiles = {}
    for model, cfg in raw["models"].items():
        profiles[model] = HandProfile(
            model=model,
            dof=cfg["dof"],
            joint_names=cfg["joint_names"],
            joint_limits=cfg["joint_limits"],
            open_pose=cfg["open_pose"],
            hardware_open_pose=cfg.get("hardware_open_pose"),
            speed_len=cfg.get("speed_len", 5),
            torque_len=cfg.get("torque_len", 5),
        )
    return profiles


def load_init_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return raw.get("init", {})
