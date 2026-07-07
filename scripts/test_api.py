#!/usr/bin/env python
"""测试 DeepSeek API 分类器"""
import os
import sys
import yaml
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from linkerbot.voice.classifier import GestureClassifier
from linkerbot.voice.gesture_library import GestureLibrary

glib = GestureLibrary(os.path.join(os.path.dirname(__file__), "..", "config", "gestures.yaml"))
glib.load()
names = glib.names
print(f"姿态库: {names}")

# 从 .env_test 读 key
env_path = os.path.join(os.path.dirname(__file__), "..", ".env_test")
voice_path = os.path.join(os.path.dirname(__file__), "..", "config", "voice.yaml")
with open(voice_path) as f:
    voice_cfg = yaml.safe_load(f).get("voice", {})
api_key = voice_cfg.get("api_key", "")
if os.path.exists(env_path):
    for line in open(env_path):
        if line.startswith("DEEPSEEK_API_KEY="):
            api_key = line.strip().split("=", 1)[1]
            break
api_base = voice_cfg.get("api_base", "https://api.deepseek.com")
model = voice_cfg.get("model", "deepseek-chat")

if not api_key:
    print("请在 .env_test 中设置 DEEPSEEK_API_KEY")
    sys.exit(1)

print(f"API: {api_base} 模型: {model}")
clf = GestureClassifier(api_key=api_key, api_base=api_base, model=model)

tests = ["握拳", "比个耶", "给我点赞", "OK手势", "把手指张开", "准备抓东西", "攥拳头"]
for t in tests:
    result = clf.classify(t, names)
    print(f'  "{t}" → {result}')
print("\n✅ API 测试完成!")
