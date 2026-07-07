#!/usr/bin/env bash
# 将 vendor/linkerhand-urdf-main 链到 assets/mujoco/linker_hand_l20
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${ROOT}/vendor/linkerhand-urdf-main"
DST="${ROOT}/assets/mujoco/linker_hand_l20"

if [[ ! -d "$SRC/l20" ]]; then
  echo "未找到 ${SRC}/l20"
  echo "请从 https://github.com/linker-bot/linkerhand-urdf 下载并解压到 vendor/linkerhand-urdf-main"
  exit 1
fi

mkdir -p "${ROOT}/assets/mujoco"
rm -rf "$DST"
mkdir -p "$DST"

ln -sfn "$(realpath "$SRC/l20/left")" "$DST/left"
ln -sfn "$(realpath "$SRC/l20/right")" "$DST/right"

echo "L20 URDF 已链接:"
echo "  $DST/left  -> linkerhand_l20_left.urdf"
echo "  $DST/right -> linkerhand_l20_right.urdf"
