from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import yaml

DEFAULT_SDK_PATH = Path(__file__).resolve().parents[3] / "vendor" / "linkerhand-python-sdk"


def resolve_sdk_path(configured: str | None) -> Path:
    if configured:
        path = Path(configured).expanduser().resolve()
    else:
        path = DEFAULT_SDK_PATH.resolve()
    if not path.exists():
        raise RuntimeError(
            f"LinkerHand SDK 未找到: {path}\n"
            "请执行: git clone https://github.com/linker-bot/linkerhand-python-sdk.git "
            f"{DEFAULT_SDK_PATH}"
        )
    linker_pkg = path / "LinkerHand"
    if not linker_pkg.exists():
        raise RuntimeError(f"SDK 目录无效，缺少 LinkerHand 包: {linker_pkg}")
    return path


def ensure_sdk_import_path(sdk_root: Path) -> None:
    sdk_root_str = str(sdk_root)
    if sdk_root_str not in sys.path:
        sys.path.insert(0, sdk_root_str)


def sync_sdk_setting(sdk_root: Path, hw_cfg: Dict[str, Any]) -> Path:
    """将 linkerbot 硬件配置同步到 SDK 的 setting.yaml"""
    setting_path = sdk_root / "LinkerHand" / "config" / "setting.yaml"
    if not setting_path.exists():
        raise RuntimeError(f"SDK setting.yaml 不存在: {setting_path}")

    with open(setting_path, "r", encoding="utf-8") as f:
        setting = yaml.safe_load(f)

    hand_type = hw_cfg.get("hand_type", "right").lower()
    hand_key = "LEFT_HAND" if hand_type == "left" else "RIGHT_HAND"
    other_key = "RIGHT_HAND" if hand_key == "LEFT_HAND" else "LEFT_HAND"

    setting["LINKER_HAND"][hand_key]["EXISTS"] = True
    setting["LINKER_HAND"][hand_key]["JOINT"] = hw_cfg.get("hand_model", "L20")
    setting["LINKER_HAND"][hand_key]["CAN"] = hw_cfg.get("can", "can0")
    setting["LINKER_HAND"][hand_key]["MODBUS"] = "None"
    setting["LINKER_HAND"][hand_key]["TOUCH"] = hw_cfg.get("touch_sensor", True)
    setting["LINKER_HAND"][other_key]["EXISTS"] = False

    if hw_cfg.get("sudo_password"):
        setting["PASSWORD"] = str(hw_cfg["sudo_password"])

    with open(setting_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(setting, f, allow_unicode=True, sort_keys=False)

    return setting_path


def create_linker_hand_api(hw_cfg: Dict[str, Any]):
    sdk_root = resolve_sdk_path(hw_cfg.get("sdk_path"))
    ensure_sdk_import_path(sdk_root)
    sync_sdk_setting(sdk_root, hw_cfg)

    from LinkerHand.linker_hand_api import LinkerHandApi

    hand_type = hw_cfg.get("hand_type", "right")
    hand_model = hw_cfg.get("hand_model", "L20")
    can = hw_cfg.get("can", "can0")
    return LinkerHandApi(hand_type=hand_type, hand_joint=hand_model, can=can)
