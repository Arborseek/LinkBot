from __future__ import annotations

from typing import Dict, List

from linkerbot.config.session import HandProfile, SessionConfig
from linkerbot.hardware.base import HandDriver
from linkerbot.sim.mujoco_sim import MujocoSimRegistry


class MujocoSimDriver(HandDriver):
    """Mock：与真机相同，只接收 SDK pose"""

    def __init__(self, side: str, profile: HandProfile, sim_cfg: dict):
        self.side = side
        self.profile = profile
        self._sim_cfg = sim_cfg
        self._sim = None
        self._connected = False

    def connect(self) -> None:
        MujocoSimRegistry.configure(self._sim_cfg)
        self._sim = MujocoSimRegistry.acquire(
            side=self.side,
            hand_model=self.profile.model,
            model_path=self._sim_cfg.get("mujoco_model"),
        )
        self.send_open_pose()
        self._connected = True

    def disconnect(self) -> None:
        MujocoSimRegistry.release(self.side)
        self._sim = None
        self._connected = False

    def send_pose(self, pose: List[int]) -> None:
        if self._sim:
            self._sim.set_pose(pose, self.profile.model)

    def send_open_pose(self) -> None:
        if self._sim:
            self._sim.set_open_pose()

    def is_connected(self) -> bool:
        return self._connected


class LinkerSdkDriver(HandDriver):
    def __init__(self, side: str, profile: HandProfile, hw_cfg: dict):
        self.side = side
        self.profile = profile
        self.hw_cfg = dict(hw_cfg)
        self.hw_cfg["hand_type"] = side
        self.hw_cfg["hand_model"] = profile.model
        self.speed = hw_cfg.get("speed") or [120] * profile.speed_len
        self.torque = hw_cfg.get("torque") or [200] * profile.torque_len
        self.open_on_disable = hw_cfg.get("open_on_disable", True)
        self._api = None
        self._connected = False

    def connect(self) -> None:
        from linkerbot.hardware.linker_sdk import create_linker_hand_api

        self._api = create_linker_hand_api(self.hw_cfg)
        self._api.set_speed(speed=self.speed[: self.profile.speed_len])
        self._api.set_torque(torque=self.torque[: self.profile.torque_len])
        self.send_open_pose()
        self._connected = True
        print(f"[LinkerSDK] {self.side} {self.profile.model} @ {self.hw_cfg.get('can', 'can0')}")

    def disconnect(self) -> None:
        if self._api and self.open_on_disable:
            try:
                self.send_open_pose()
            except Exception:
                pass
        self._api = None
        self._connected = False

    def send_pose(self, pose: List[int]) -> None:
        if self._api:
            self._api.finger_move(pose=pose)

    def send_open_pose(self) -> None:
        if self._api:
            self._api.finger_move(pose=self.profile.open_pose)

    def is_connected(self) -> bool:
        return self._connected


class HandDriverSet:
    def __init__(self, drivers: Dict[str, HandDriver]):
        self.drivers = drivers

    def connect(self) -> None:
        for d in self.drivers.values():
            d.connect()

    def disconnect(self) -> None:
        for d in self.drivers.values():
            d.disconnect()

    def send_open_pose(self) -> None:
        for d in self.drivers.values():
            d.send_open_pose()

    def send(self, side: str, pose: List[int]) -> None:
        if side in self.drivers:
            self.drivers[side].send_pose(pose)


def _can_for_side(session: SessionConfig, hw_cfg: dict, side: str) -> str:
    """单手始终用 can；双手时左/右可分别指定 can_left / can_right"""
    default = hw_cfg.get("can", "can0")
    if not session.is_dual:
        return default
    if side == "left":
        return hw_cfg.get("can_left", default)
    return hw_cfg.get("can_right", default)


def create_driver_set(session: SessionConfig, profile: HandProfile, hw_cfg: dict) -> HandDriverSet:
    drivers: Dict[str, HandDriver] = {}
    sim_cfg = hw_cfg.get("simulation", {})
    MujocoSimRegistry.configure(sim_cfg)
    for side in session.active_sides:
        s = side.value
        if session.hardware_mode.value == "linker_sdk":
            cfg = dict(hw_cfg)
            cfg["can"] = _can_for_side(session, hw_cfg, s)
            drivers[s] = LinkerSdkDriver(s, profile, cfg)
        else:
            drivers[s] = MujocoSimDriver(s, profile, sim_cfg)
    return HandDriverSet(drivers)
