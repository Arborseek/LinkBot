from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from linkerbot.config.session import HandProfile


@dataclass
class HandJoints:
    """灵巧手关节角度 (rad)"""

    values: np.ndarray
    profile: HandProfile

    def __post_init__(self) -> None:
        self.values = np.asarray(self.values, dtype=np.float64)
        if self.values.shape != (self.profile.dof,):
            raise ValueError(
                f"{self.profile.model} 需要 {self.profile.dof} 个关节，收到 {self.values.shape}"
            )

    @classmethod
    def zeros(cls, profile: HandProfile) -> HandJoints:
        return cls(values=np.zeros(profile.dof, dtype=np.float64), profile=profile)

    def as_dict(self) -> Dict[str, float]:
        return {
            name: float(self.values[i])
            for i, name in enumerate(self.profile.joint_names)
        }

    def as_array(self) -> np.ndarray:
        return self.values.copy()

    def to_hardware_pose(self, side: str = "left") -> List[int]:
        from linkerbot.sim.pose_mapping import joints_to_sdk_pose

        return joints_to_sdk_pose(self, side)


@dataclass
class HandTracking:
    landmarks: np.ndarray | None = None
    world_landmarks: np.ndarray | None = None
    handedness: str = "Right"
    detected: bool = False


@dataclass
class TrackingFrame:
    hands: Dict[str, HandTracking] = field(default_factory=dict)

    @property
    def detected_sides(self) -> List[str]:
        return [side for side, h in self.hands.items() if h.detected]


@dataclass
class HandRuntimeState:
    side: str
    tracking: HandTracking = field(default_factory=HandTracking)
    joints: HandJoints | None = None
    hardware_pose: List[int] = field(default_factory=list)
    pinch_raw: float = 0.0
    pinch_strength: float = 0.0
    spread_im: float = 0.0


@dataclass
class PipelineState:
    session_summary: str = ""
    phase: str = "init"
    hands: Dict[str, HandRuntimeState] = field(default_factory=dict)
    fps: float = 0.0
    enabled: bool = False
    init_progress: float = 0.0
    init_message: str = ""
