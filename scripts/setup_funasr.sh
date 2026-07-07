#!/bin/bash
# 安装 FunASR + 下载模型
echo "[1/2] 安装 funasr..."
~/miniconda3/envs/linkerbot/bin/pip install funasr -q
echo "[2/2] 下载 paraformer 中文模型 (~800MB，首次)..."
cd ~/linkerbot && PYTHONPATH=src ~/miniconda3/envs/linkerbot/bin/python -c "
import os
for k in ('ALL_PROXY','all_proxy','SOCKS_PROXY','socks_proxy'):
    os.environ.pop(k,None)
from funasr import AutoModel
print('下载中...')
m = AutoModel(model='iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch')
print('✅ FunASR 就绪')
"
echo "✅ 完成！启动: bash ~/linkerbot/run_voice.sh"
