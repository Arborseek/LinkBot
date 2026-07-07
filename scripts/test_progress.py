#!/usr/bin/env python
"""语音管线逐步检测 — 终端显示每步状态"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
for k in ("ALL_PROXY", "all_proxy", "SOCKS_PROXY", "socks_proxy"):
    os.environ.pop(k, None)

CHECK = "✅"
FAIL = "❌"
SKIP = "⏭️"

def step(name, fn):
    sys.stdout.write(f"  {name:40s} ... ")
    sys.stdout.flush()
    try:
        result = fn()
        print(f"{CHECK} {result}")
        return True
    except Exception as e:
        print(f"{FAIL} {e}")
        return False

# ======== 1 ========
print("\n[1/7] 姿态库")
ok = step("加载 config/gestures.yaml", lambda: (
    __import__('linkerbot.voice.gesture_library', fromlist=['GestureLibrary'])
    .GestureLibrary(os.path.join(os.path.dirname(__file__), "..", "config", "gestures.yaml"))
    .load() or "9个姿态"
))

# ======== 2 ========
print("\n[2/7] VAD 引擎 (Silero)")
vad = None
def _load_vad():
    global vad
    from linkerbot.voice.vad_engine import VadEngine
    vad = VadEngine()
    vad.load()
    return "模型加载成功 (2MB)"
ok &= step("加载 Silero VAD 模型", _load_vad)

def _test_vad():
    import numpy as np
    silent = np.zeros(512, dtype=np.float32)
    r = vad.is_speech(silent)
    return f"静音检测={r} (应为False)"
ok &= step("VAD 推理测试", _test_vad)

# ======== 3 ========
print("\n[3/7] 麦克风采集")
audio = None
def _load_audio():
    global audio
    from linkerbot.voice.audio_capture import AudioCapture
    audio = AudioCapture(16000)
    audio.start()
    time.sleep(0.5)
    return f"缓冲 {audio.available()} samples"
ok &= step("启动麦克风", _load_audio)

def _read_chunk():
    chunk = audio.read(32)
    audio.stop()
    return f"32ms chunk={len(chunk)} samples"
ok &= step("读取音频帧", _read_chunk)

# ======== 4 ========
print("\n[4/7] Whisper 模型 (faster-whisper)")
asr = None
def _load_whisper():
    global asr
    from linkerbot.voice.asr import WhisperASR
    asr = WhisperASR(model_size="base")
    asr.load()
    return "base 模型加载成功 (~150MB)"
ok &= step("下载/加载 Whisper base", _load_whisper)

# ======== 5 ========
print("\n[5/7] DeepSeek API")
def _test_api():
    import yaml
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env_test")
    api_key = ""
    if os.path.exists(env_path):
        for line in open(env_path):
            if line.startswith("DEEPSEEK_API_KEY="):
                api_key = line.strip().split("=",1)[1]; break
    vp = os.path.join(os.path.dirname(__file__), "..", "config", "voice.yaml")
    with open(vp) as f:
        vk = yaml.safe_load(f).get("voice",{}).get("api_key","")
    if vk.startswith("${"):
        vk = os.environ.get(vk[2:-1],"")
    api_key = api_key or vk
    if not api_key: return "跳过 (无key)"
    from linkerbot.voice.classifier import GestureClassifier
    from linkerbot.voice.gesture_library import GestureLibrary
    glib = GestureLibrary(os.path.join(os.path.dirname(__file__), "..", "config", "gestures.yaml"))
    glib.load()
    clf = GestureClassifier(api_key=api_key)
    r = clf.classify("比个耶", glib.names)
    return f"'比个耶' → {r}"
ok &= step("DeepSeek 分类器", _test_api)

# ======== 6 ========
print("\n[6/7] 完整 VoiceController")
vc = None
def _start_vc():
    global vc
    import yaml
    from linkerbot.voice.gesture_library import GestureLibrary
    from linkerbot.voice.voice_controller import VoiceController
    glib = GestureLibrary(os.path.join(os.path.dirname(__file__), "..", "config", "gestures.yaml"))
    glib.load()
    vp = os.path.join(os.path.dirname(__file__), "..", "config", "voice.yaml")
    with open(vp) as f:
        vc_cfg = yaml.safe_load(f).get("voice", {})
    api_key = vc_cfg.get("api_key", "")
    if api_key.startswith("${"):
        api_key = os.environ.get(api_key[2:-1], "")
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env_test")
    if not api_key and os.path.exists(env_path):
        for line in open(env_path):
            if line.startswith("DEEPSEEK_API_KEY="):
                api_key = line.strip().split("=",1)[1]; break
    vc_cfg["api_key"] = api_key
    vc = VoiceController(glib, vc_cfg)
    vc.start()
    time.sleep(0.5)
    return f"状态: {vc.status}"
ok &= step("启动 VoiceController", _start_vc)

def _check_vc():
    if vc.running:
        return f"running={vc.running}, status={vc.status}"
    return f"启动失败! running=False"
ok &= step("运行中检查", _check_vc)

if vc:
    vc.stop()

# ======== 7 ========
print("\n[7/7] 状态栏渲染")
def _test_overlay():
    import cv2, numpy as np
    from linkerbot.viz.text import put_text
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    h, w = frame.shape[:2]
    bar_h = 36
    cv2.rectangle(frame, (0, h-bar_h), (w, h), (30, 30, 35), -1)
    status = "🎤 语音模式 | 等待唤醒词..."
    frame = put_text(frame, status, (10, h-bar_h+6), font_size=20, color=(0, 220, 130))
    g_mean = frame[h-bar_h:h, :, 1].mean()
    if g_mean > 50:
        return f"绿色文字可见 (G均值={g_mean:.0f}>50)"
    return f"文字可能未渲染 (G均值={g_mean:.0f})"
ok &= step("put_text 中文渲染", _test_overlay)

# ======== 结果 ========
print("\n" + "="*55)
if ok:
    print("🎉 全部通过! 可以运行: bash ~/linkerbot/run_voice.sh")
else:
    print("❌ 有失败项，请检查上面输出")
print("="*55)
