"""
音乐节拍分析：从音频文件提取 BPM 和每个节拍的时间戳。

依赖: librosa
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np


def analyze_beats(audio_path: str | Path, sr: int = 22050) -> Tuple[float, List[float]]:
    """
    分析音频节拍。

    Returns:
        (bpm, beat_times) — bpm 为估计速度，beat_times 为每个节拍的时间（秒）列表
    """
    try:
        import librosa
    except ImportError as e:
        raise ImportError("请先安装 librosa: pip install librosa") from e

    y, sr = librosa.load(str(audio_path), sr=sr, mono=True)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units="frames")
    beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()

    # librosa 返回的 tempo 可能是 array，取标量
    bpm = float(np.atleast_1d(tempo)[0])
    return bpm, beat_times


def beats_on_downbeats(beat_times: List[float], beats_per_bar: int = 4) -> List[float]:
    """返回每小节第一拍（强拍）的时间列表"""
    return [t for i, t in enumerate(beat_times) if i % beats_per_bar == 0]
