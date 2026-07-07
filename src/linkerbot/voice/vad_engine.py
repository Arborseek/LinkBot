"""Silero VAD 封装。

用 silero-vad 做轻量级语音活动检测（~2MB 模型）。
"""

from __future__ import annotations

from typing import Optional


class VadEngine:
    """Silero VAD 状态机"""

    # 状态
    SILENCE = "silence"
    SPEECH = "speech"

    def __init__(
        self,
        sample_rate: int = 16000,
        threshold: float = 0.5,
        silence_timeout: float = 1.5,
        max_record: float = 5.0,
        min_speech_ms: int = 300,
    ):
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.silence_timeout = silence_timeout
        self.max_record = max_record
        self.min_speech_samples = int(sample_rate * min_speech_ms / 1000)
        self._model = None
        self._state = self.SILENCE
        self._speech_start = 0.0
        self._last_speech = 0.0
        self._speech_samples = 0

    # ---- public ----

    @property
    def state(self) -> str:
        return self._state

    def load(self) -> None:
        if self._model is not None:
            return
        from silero_vad import load_silero_vad

        self._model = load_silero_vad()
        self.reset()

    def reset(self) -> None:
        self._state = self.SILENCE
        self._speech_start = 0.0
        self._last_speech = 0.0
        self._speech_samples = 0

    def is_speech(self, chunk: "np.ndarray") -> bool:
        """单帧判定。chunk: float32 [-1,1], 期望 512/1024 samples (30/60ms @16kHz)"""
        import numpy as np
        import torch

        if self._model is None:
            self.load()
        # silero-vad 6.x 需要 torch Tensor
        t = torch.from_numpy(np.asarray(chunk, dtype=np.float32))
        prob = self._model(t, self.sample_rate).item()
        return prob >= self.threshold

    def process(
        self, chunk: "np.ndarray", timestamp: float
    ) -> tuple[bool, bool]:
        """
        处理一帧音频。
        返回 (is_speech_now, segment_complete)。
        segment_complete=True 时表示一段完整语音结束，可以拿去转录。
        """
        import numpy as np

        speech = bool(self.is_speech(np.asarray(chunk, dtype=np.float32)))
        segment_complete = False

        if speech:
            if self._state == self.SILENCE:
                self._state = self.SPEECH
                self._speech_start = timestamp
            self._last_speech = timestamp
            self._speech_samples += len(chunk)

        if self._state == self.SPEECH:
            elapsed = timestamp - self._speech_start
            silent_gap = timestamp - self._last_speech

            if silent_gap >= self.silence_timeout:
                # 静音超时 → 语音段结束
                self._state = self.SILENCE
                if self._speech_samples >= self.min_speech_samples:
                    segment_complete = True
                self._speech_samples = 0
            elif elapsed >= self.max_record:
                # 最长录音超时
                self._state = self.SILENCE
                if self._speech_samples >= self.min_speech_samples:
                    segment_complete = True
                self._speech_samples = 0

        return speech, segment_complete
