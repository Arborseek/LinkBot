#!/bin/bash
# LinkerHand HTTP API 启动脚本

set -e

# ---- CAN 接口自动拉起 ----
CAN_IF="${CAN_IF:-can0}"
CAN_BITRATE="${CAN_BITRATE:-1000000}"
SUDO_PASS="${SUDO_PASS:-0}"

if ip link show "$CAN_IF" 2>/dev/null | grep -q "state UP"; then
    echo "[CAN] $CAN_IF 已就绪"
else
    echo "[CAN] 正在拉起 $CAN_IF (bitrate=$CAN_BITRATE)..."
    echo "$SUDO_PASS" | sudo -S ip link set "$CAN_IF" down 2>/dev/null || true
    echo "$SUDO_PASS" | sudo -S ip link set "$CAN_IF" up type can bitrate "$CAN_BITRATE"
    echo "[CAN] $CAN_IF 已就绪"
fi

# 清掉 socks 代理（httpx 不支持）
unset ALL_PROXY
unset all_proxy

cd /home/robot/linkerbot
PYTHONPATH=src /home/robot/miniconda3/envs/linkerbot/bin/uvicorn \
  linkerbot.api.server:app --host 0.0.0.0 --port 8765
