"""语音模式主控制器。

协调 VAD → Whisper → LLM 分类 → 姿态查询的完整链路。
在独立线程中处理音频，主线程通过 update() 获取结果。
"""

from __future__ import annotations

import os
import queue
import threading
import time
from typing import Optional

import numpy as np

from linkerbot.voice.audio_capture import AudioCapture
from linkerbot.voice.asr import WhisperASR
from linkerbot.voice.classifier import GestureClassifier
from linkerbot.voice.gesture_library import GestureLibrary
from linkerbot.voice.vad_engine import VadEngine


class VoiceController:
    """语音控制主控"""

    def __init__(
        self,
        gesture_library: GestureLibrary,
        voice_cfg: dict,
    ):
        cfg = voice_cfg or {}
        self._gesture_lib = gesture_library
        self._wake_word = str(cfg.get("wake_word", "嘿机器人"))

        sample_rate = int(cfg.get("sample_rate", 16000))
        self._audio = AudioCapture(sample_rate=sample_rate)
        self._vad = VadEngine(
            sample_rate=sample_rate,
            threshold=float(cfg.get("vad_threshold", 0.5)),
            silence_timeout=float(cfg.get("silence_timeout", 1.5)),
            max_record=float(cfg.get("max_record", 5.0)),
        )
        self._asr = WhisperASR(
            model_size=str(cfg.get("whisper_model", "base")),
            sample_rate=sample_rate,
        )
        api_key = str(cfg.get("api_key", ""))
        if api_key.startswith("${") and api_key.endswith("}"):
            env_var = api_key[2:-1]
            api_key = os.environ.get(env_var, "")
        self._classifier = GestureClassifier(
            api_key=api_key,
            api_base=str(cfg.get("api_base", "https://api.deepseek.com")),
            model=str(cfg.get("model", "deepseek-chat")),
        )

        self._sample_rate = sample_rate
        self._result_queue: queue.Queue = queue.Queue()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._status = "待命中"
        self._error = ""

    # ---- public ----

    @property
    def status(self) -> str:
        return self._error or self._status

    @property
    def running(self) -> bool:
        return self._running

    def start(self) -> None:
        if self._running:
            return
        # 清掉 socks 代理（httpx/huggingface_hub 不支持）
        for k in ("ALL_PROXY", "all_proxy", "SOCKS_PROXY", "socks_proxy"):
            os.environ.pop(k, None)
        self._running = True
        self._status = "🎤 启动麦克风..."
        self._audio.start()
        self._status = "⏳ 加载语音模型..."
        # 模型加载放到后台线程，避免阻塞主线程导致 UI 卡死
        threading.Thread(target=self._load_models, daemon=True).start()

    def _load_models(self) -> None:
        """后台加载 VAD + Whisper 模型，完成后启动音频循环"""
        try:
            print("[语音] 加载 VAD 模型...")
            self._vad.load()
            print("[语音] 加载 FunASR 模型...")
            self._asr.load()
        except Exception as e:
            self._error = f"语音模块加载失败: {e}"
            print(f"[语音] ❌ {self._error}")
            self._running = False
            self._status = "❌ 语音加载失败"
            return
        self._status = "🎤 语音模式 | 等待唤醒词..."
        print(f"[语音] ✅ 就绪, 唤醒词: {self._wake_word}")
        if not self._running:
            return  # 加载期间用户已关闭语音模式
        self._thread = threading.Thread(target=self._audio_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        self._audio.stop()
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._status = "待命中"
        self._error = ""

    def update(self) -> Optional[str]:
        """主线程每帧调用。返回姿态名（有结果时）或 None"""
        try:
            return self._result_queue.get_nowait()
        except queue.Empty:
            return None

    # ---- internal ----

    def _audio_loop(self) -> None:
        chunk_ms = 32  # 512 samples @16kHz，VAD 严格要求
        chunk_samples = int(self._sample_rate * chunk_ms / 1000)
        speech_buffer: list[np.ndarray] = []
        segment_start = 0.0

        while self._running:
            try:
                chunk = self._audio.read(chunk_ms)
            except Exception:
                continue

            timestamp = time.time()
            speech, segment_complete = self._vad.process(chunk, timestamp)

            if speech:
                if not speech_buffer:
                    segment_start = timestamp
                    self._status = "🔊 听到声音..."
                speech_buffer.append(chunk.astype(np.float32))

            if segment_complete and speech_buffer:
                self._status = "🧠 识别中..."
                audio = np.concatenate(speech_buffer)
                speech_buffer.clear()
                self._process_segment(audio)

            # 丢弃过长的静音缓冲
            if not speech and not speech_buffer and self._audio.available() > self._sample_rate * 3:
                self._audio.read_all()

    def _process_segment(self, audio: np.ndarray) -> None:
        """处理一段完整语音"""
        try:
            self._status = "🔄 识别语音..."
            print("[语音] Whisper 转录中...")
            text = self._asr.transcribe(audio)
        except Exception as e:
            self._error = f"Whisper 错误: {e}"
            print(f"[语音] ❌ {self._error}")
            self._status = "🎤 语音模式"
            return

        if not text:
            self._status = "🎤 语音模式 | 未识别到文字"
            return

        print(f"[语音] 转录: {text!r}")

        # 检查唤醒词
        if self._wake_word and self._wake_word not in text:
            print(f"[语音] 未检测到 '{self._wake_word}'")
            self._status = "🎤 语音模式"
            return

        # 提取唤醒词后面的指令
        idx = text.index(self._wake_word) + len(self._wake_word)
        command = text[idx:].strip().lstrip("，,。. ").strip()

        if not command:
            self._status = "🎤 语音模式 | 等待指令..."
            return

        print(f"[语音] 指令: {command!r}, LLM 分类中...")
        self._status = f"🤖 分类中: {command}"

        names = self._gesture_lib.names
        try:
            gesture_name = self._classifier.classify(command, names)
        except Exception as e:
            self._error = f"API 错误: {e}"
            print(f"[语音] ❌ {self._error}")
            self._status = "🎤 语音模式"
            return

        if gesture_name:
            print(f"[语音] ✅ 执行: {gesture_name}")
            self._status = f"✅ 执行: {gesture_name}"
            self._result_queue.put(gesture_name)
        else:
            print(f"[语音] ⏭️ 非手势指令，忽略")
            self._status = "🎤 语音模式"
