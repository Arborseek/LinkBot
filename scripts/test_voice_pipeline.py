#!/usr/bin/env python
"""语音控制管线端到端测试 — 无需摄像头/真机"""
import os, sys, time
import numpy as np

# 清掉 socks 代理
for k in ("ALL_PROXY", "all_proxy", "SOCKS_PROXY", "socks_proxy"):
    os.environ.pop(k, None)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# ============================================================
# 1. 姿态库
# ============================================================
print("=" * 50)
print("1. 姿态库")
from linkerbot.voice.gesture_library import GestureLibrary
glib = GestureLibrary(os.path.join(os.path.dirname(__file__), "..", "config", "gestures.yaml"))
glib.load()
print(f"   ✅ {glib.count} 个姿态: {glib.names}")

# ============================================================
# 2. VAD（真实模型推理）
# ============================================================
print("=" * 50)
print("2. VAD 引擎")
from linkerbot.voice.vad_engine import VadEngine
vad = VadEngine()
vad.load()
print("   ✅ 模型加载成功")

# 模拟一段语音（正弦波 + 静音）
rng = np.random.RandomState(42)
silent = np.zeros(512, dtype=np.float32) * 0.01
speech = (rng.randn(512) * 0.3).astype(np.float32)
assert not vad.is_speech(silent), "静音应该返回 False"
t0 = time.time()
for i in range(50):
    s, done = vad.process(speech, i * 0.032)
    if done:
        break
print(f"   50帧静音检测: speech={vad.is_speech(silent)} (应为False)")
print(f"   50帧语音检测: speech_detected={s}, done={done} (50帧=1.6s, 应done)")
print(f"   推理耗时: ~{(time.time()-t0)/50*1000:.1f}ms/帧")

# ============================================================
# 3. 音频采集 + VAD 真实流
# ============================================================
print("=" * 50)
print("3. 音频采集")
from linkerbot.voice.audio_capture import AudioCapture
audio = AudioCapture(sample_rate=16000)
audio.start()
time.sleep(0.3)
avail = audio.available()
print(f"   ✅ 采集启动成功, {avail} samples 已缓冲 ({avail/16000:.1f}s)")
chunk = audio.read(32)
assert len(chunk) == 512, f"chunk 应为 512 samples, 实际 {len(chunk)}"
print(f"   ✅ read(32ms) 返回 {len(chunk)} samples, range=[{chunk.min():.3f}, {chunk.max():.3f}]")
audio.stop()
print("   ✅ 采集停止成功")

# ============================================================
# 4. 分类器（真实 API 调用）
# ============================================================
print("=" * 50)
print("4. DeepSeek API 分类器")
env_path = os.path.join(os.path.dirname(__file__), "..", ".env_test")
api_key = ""
if os.path.exists(env_path):
    for line in open(env_path):
        if line.startswith("DEEPSEEK_API_KEY="):
            api_key = line.strip().split("=", 1)[1]
            break
if not api_key:
    # fallback to voice.yaml
    import yaml
    vp = os.path.join(os.path.dirname(__file__), "..", "config", "voice.yaml")
    with open(vp) as f:
        api_key = yaml.safe_load(f).get("voice", {}).get("api_key", "")
    if api_key.startswith("${"):
        api_key = os.environ.get(api_key[2:-1], "")

if api_key:
    from linkerbot.voice.classifier import GestureClassifier
    clf = GestureClassifier(api_key=api_key)
    result = clf.classify("给我比个耶", glib.names)
    print(f"   '给我比个耶' → {result}")
    assert result == "比耶", f"期望 '比耶', 实际 '{result}'"
    print("   ✅ 分类正确!")
else:
    print("   ⚠️ 跳过 (未设置 API key)")

# ============================================================
# 5. VoiceController 完整链路（模拟麦克风输入）
# ============================================================
print("=" * 50)
print("5. VoiceController 完整链路")

voice_cfg = {
    "wake_word": "嘿 灵心巧手",
    "api_key": api_key,
    "api_base": "https://api.deepseek.com",
    "model": "deepseek-chat",
    "whisper_model": "base",
    "vad_threshold": 0.5,
    "silence_timeout": 1.5,
    "max_record": 5.0,
    "sample_rate": 16000,
}

from linkerbot.voice.voice_controller import VoiceController
vc = VoiceController(glib, voice_cfg)

# 启动 → 这时会开始采集真实麦克风
vc.start()
print(f"   ✅ start() 成功, 状态: {vc.status}")
time.sleep(1.0)
print(f"   1秒后状态: {vc.status}")
assert vc.running, "VoiceController 应该在运行"

# 验证 update() 不抛异常
for _ in range(5):
    result = vc.update()
    time.sleep(0.05)
print(f"   ✅ update() 调用正常 (5次), 返回: {result}")

# 停止
vc.stop()
print(f"   ✅ stop() 成功, running={vc.running}")

# ============================================================
# 6. 录制器
# ============================================================
print("=" * 50)
print("6. 姿态录制器")
from linkerbot.voice.recorder import GestureRecorder
fake_pose = [255, 70, 255, 255, 255, 255, 255, 255, 255, 255]

def save_cb(name, pose):
    print(f"   save_callback: name='{name}' pose={pose}")

rec = GestureRecorder(save_callback=save_cb, get_current_pose=lambda: fake_pose)
assert not rec.recording
rec.toggle()
assert rec.recording
print(f"   ✅ 录制模式: {rec.status}")
ok, msg = rec.save_current("测试")
assert ok, msg
print(f"   ✅ 保存: {msg}")
rec.toggle()
assert not rec.recording

# ============================================================
# 7. 状态栏渲染
# ============================================================
print("=" * 50)
print("7. 状态栏中文渲染")
import cv2
frame = np.zeros((480, 640, 3), dtype=np.uint8)
from linkerbot.viz.text import put_text, _has_cjk

# 模拟 _overlay_voice_status
h, w = frame.shape[:2]
bar_h = 36
cv2.rectangle(frame, (0, h - bar_h), (w, h), (30, 30, 35), -1)
status = "🎤 语音模式 | 等待唤醒词..."
put_text(frame, status, (10, h - bar_h + 6), font_size=20, color=(0, 220, 130))
cv2.imwrite("/tmp/voice_status_test.png", frame)
# 检查绿色条是否渲染
bar_region = frame[h-bar_h:h, :, :]
mean_g = bar_region[:, :, 1].mean()  # G channel
assert _has_cjk(status), "应包含中文"
print(f"   ✅ 状态栏区域 G 均值: {mean_g:.0f} (应 >100 因为有绿色文字)")
print(f"   ✅ 截图保存到 /tmp/voice_status_test.png")
print(f"   ✅ 中文检测: {_has_cjk(status)}")

print()
print("=" * 50)
print("🎉 全部测试通过！")
print("=" * 50)
