"""麦克风采集模块。

后台线程持续录制，写入线程安全的 ring buffer。
主线程按需读取音频段。
"""

from __future__ import annotations

import collections
import threading
import time
from typing import Optional

import numpy as np
import sounddevice as sd


class AudioCapture:
    """16kHz mono 麦克风采集，ring buffer"""

    def __init__(self, sample_rate: int = 16000, chunk_ms: int = 32):
        self.sample_rate = sample_rate
        self.chunk_size = int(sample_rate * chunk_ms / 1000)  # 每帧采样数
        self._buffer: collections.deque = collections.deque()
        self._lock = threading.Lock()
        self._running = False
        self._stream: Optional[sd.InputStream] = None
        self._thread: Optional[threading.Thread] = None
        self._total_samples = 0

    # ---- public ----

    @property
    def running(self) -> bool:
        return self._running

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="int16",
            blocksize=self.chunk_size,
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        self._running = False
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    def read(self, duration_ms: float) -> np.ndarray:
        """读取指定时长的音频数据。阻塞直到足够数据可用。返回 float32 [-1,1]"""
        needed = int(self.sample_rate * duration_ms / 1000)
        deadline = time.time() + max(duration_ms / 1000 * 3, 0.5)
        while True:
            with self._lock:
                if self._total_samples >= needed:
                    break
            if time.time() > deadline:
                return np.zeros(needed, dtype=np.float32)
            time.sleep(0.005)
        with self._lock:
            chunks = []
            remaining = needed
            while remaining > 0 and self._buffer:
                chunk = self._buffer.popleft()
                chunks.append(chunk)
                remaining -= len(chunk)
            self._total_samples = max(0, self._total_samples - needed)
        if not chunks:
            return np.zeros(needed, dtype=np.float32)
        audio = np.concatenate(chunks).astype(np.float32)
        if len(audio) < needed:
            audio = np.pad(audio, (0, needed - len(audio)))
        return audio[:needed] / 32768.0

    def read_all(self) -> np.ndarray:
        """非阻塞读取当前缓冲区全部数据"""
        with self._lock:
            if not self._buffer:
                return np.array([], dtype=np.float32)
            chunks = list(self._buffer)
            self._buffer.clear()
            self._total_samples = 0
        if not chunks:
            return np.array([], dtype=np.float32)
        return np.concatenate(chunks).astype(np.float32) / 32768.0

    def available(self) -> int:
        """缓冲区可用采样数"""
        with self._lock:
            return self._total_samples

    # ---- internal ----

    def _callback(self, indata: np.ndarray, _frames, _time, _status) -> None:
        if not self._running:
            return
        arr = indata[:, 0].copy()  # mono
        with self._lock:
            self._buffer.append(arr)
            self._total_samples += len(arr)
