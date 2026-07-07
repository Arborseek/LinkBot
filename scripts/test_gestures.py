"""依次测试手势 1-8，每个保持 2 秒，验证机械手响应"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor" / "linkerhand-python-sdk"))

from LinkerHand.linker_hand_api import LinkerHandApi
from linkerbot.config.session import load_hand_profiles
from linkerbot.sim.pose_mapping import joints_to_sdk_pose, sdk_pose_to_rad

# 手势 1-8（壹到捌）+ 张开手掌 + 握拳
GESTURES = {
    "壹(1)": [55, 0, 255, 0, 0, 0, 128, 67, 89, 124],
    "贰(2)": [55, 0, 255, 255, 0, 0, 128, 67, 89, 124],
    "叁(3)": [116, 255, 255, 255, 255, 0, 128, 67, 89, 255],
    "肆(4)": [0, 0, 255, 255, 255, 255, 128, 67, 89, 255],
    "伍(5)": [255, 255, 255, 255, 255, 255, 128, 67, 89, 255],
    "陆(6)": [255, 255, 0, 0, 0, 255, 128, 67, 89, 255],
    "柒(7)": [255, 37, 119, 112, 0, 0, 128, 67, 89, 211],
    "捌(8)": [255, 255, 255, 0, 0, 0, 128, 67, 89, 255],
    "张开手掌": [255, 70, 255, 255, 255, 255, 255, 255, 255, 255],
    "握拳": [80, 0, 80, 80, 80, 80, 255, 255, 255, 197],
}

HAND_TYPE = "left"     # 改成 "right" 如果是右手
HAND_MODEL = "L10"     # 改成 "L20" 如果是 L20
CAN = "can0"
WAIT_SEC = 2.0

def main():
    print(f"连接 {HAND_MODEL} {HAND_TYPE} @ {CAN} ...")
    api = LinkerHandApi(hand_type=HAND_TYPE, hand_joint=HAND_MODEL, can=CAN)
    print("已连接\n")

    # 先张开
    print("→ 张开手掌 (初始位)")
    api.finger_move(pose=GESTURES["张开手掌"])
    time.sleep(1.5)

    for name, pose in GESTURES.items():
        print(f"→ {name}: {pose}")
        api.finger_move(pose=pose)
        time.sleep(WAIT_SEC)

    # 回到张开
    print("\n→ 张开手掌 (结束)")
    api.finger_move(pose=GESTURES["张开手掌"])
    print("完成！")

if __name__ == "__main__":
    main()
