#!/usr/bin/env python3
"""LinkerBot 灵巧手遥操作"""

from __future__ import annotations

import os

# 须在 import cv2 之前设置，抑制 Qt 字体目录警告
os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts")
os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.fonts.warning=false")

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from linkerbot.app import LinkerBotApp, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="LinkerHand 灵巧手摄像头遥操作")
    parser.add_argument("-c", "--config", default=str(ROOT / "config" / "default.yaml"))
    parser.add_argument("-p", "--profiles", default=str(ROOT / "config" / "hand_profiles.yaml"))
    parser.add_argument("--camera", type=int, default=None, help="摄像头 device_id，如 0 或 1")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.camera is not None:
        config.setdefault("camera", {})["device_id"] = args.camera

    app = LinkerBotApp(config, Path(args.profiles))
    app.run()


if __name__ == "__main__":
    main()
