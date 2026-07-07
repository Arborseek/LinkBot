"""LinkerBot 语音控制模块。

组件:
- AudioCapture: 麦克风采集 (ring buffer)
- VadEngine: Silero VAD 语音活动检测
- WhisperASR: faster-whisper 语音转文字
- GestureClassifier: DeepSeek LLM 分类器
- GestureLibrary: 姿态库 CRUD
- GestureRecorder: 姿态录制
- VoiceController: 语音模式主控
"""

from linkerbot.voice.audio_capture import AudioCapture
from linkerbot.voice.vad_engine import VadEngine
from linkerbot.voice.asr import WhisperASR
from linkerbot.voice.classifier import GestureClassifier
from linkerbot.voice.gesture_library import GestureLibrary
from linkerbot.voice.recorder import GestureRecorder
from linkerbot.voice.voice_controller import VoiceController
