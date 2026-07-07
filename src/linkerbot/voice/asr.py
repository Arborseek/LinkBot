"""FunASR 语音转文字。

阿里达摩院 Paraformer——中文英文识别率远超 Whisper base。
模型首次加载自动下载 (~800MB)。
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class WhisperASR:
    """FunASR Paraformer 语音识别（接口兼容原名）"""

    def __init__(self, model_size: str = "base", sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self._model = None

    def load(self) -> None:
        if self._model is not None:
            return
        from funasr import AutoModel
        self._model = AutoModel(
            model="iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
            vad_model="fsmn-vad",
            punc_model="ct-punc",
        )

    def transcribe(self, audio: np.ndarray) -> Optional[str]:
        """
        转录音频段。audio: float32 [-1,1], 16kHz mono。
        返回文字或 None。
        """
        if self._model is None:
            self.load()
        if audio.size < self.sample_rate * 0.3:
            return None
        # FunASR 需要 PCM int16
        audio_i16 = (np.asarray(audio, dtype=np.float32) * 32767).clip(-32768, 32767).astype("int16")
        result = self._model.generate(input=audio_i16.tobytes(), batch_size_s=300)
        if result and len(result) > 0:
            text = result[0].get("text", "").strip()
            return text if text else None
        return None
