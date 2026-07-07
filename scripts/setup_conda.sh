#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if ! command -v conda >/dev/null 2>&1; then
  echo "未找到 conda，请先安装 Miniconda/Anaconda"
  exit 1
fi

eval "$(conda shell.bash hook)"

if conda env list | awk '{print $1}' | grep -qx linkerbot; then
  conda env update -f environment.yml --prune
else
  conda env create -f environment.yml
fi

conda activate linkerbot
echo "环境已就绪: linkerbot"
echo "运行: conda activate linkerbot && python main.py"
