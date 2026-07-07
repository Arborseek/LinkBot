from __future__ import annotations

from typing import Dict

from linkerbot.config.session import HandProfile, SessionConfig
from linkerbot.hardware.driver import HandDriverSet, create_driver_set


def create_drivers(session: SessionConfig, profile: HandProfile, hw_cfg: dict) -> HandDriverSet:
    return create_driver_set(session, profile, hw_cfg)
