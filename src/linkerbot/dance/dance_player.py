"""手势舞播放器：拿着舞谱 + 放音乐 + 按时切换手势。

舞谱生成见 choreographer.py
"""
from __future__ import annotations

import time
from enum import Enum
from pathlib import Path
from typing import List, Optional


class PlayerState(str, Enum):
    IDLE = "idle"
    PLAYING = "playing"
    FINISHED = "finished"


class DancePlayer:
    """手势舞播放引擎"""

    def __init__(self):
        self._state = PlayerState.IDLE
        self._events: list[dict] = []    # [{"time": 0.0, "gesture": "张开"}, ...]
        self._gesture_library = None
        self._current_idx = 0
        self._last_idx = -1
        self._t0: float = 0.0
        self._status = ""
        self._current_name = ""

    # ---- public ----

    @property
    def state(self) -> PlayerState:
        return self._state

    @property
    def is_playing(self) -> bool:
        return self._state == PlayerState.PLAYING

    @property
    def status(self) -> str:
        return self._status

    @property
    def current_gesture_name(self) -> str:
        return self._current_name

    def start(self, audio_path: str, choreography: dict, gesture_library) -> None:
        """加载舞谱 + 播放音乐

        Args:
            audio_path: 音频文件路径
            choreography: choreographer 生成的舞谱 dict {"bpm": ..., "events": [...]}
            gesture_library: GestureLibrary 实例
        """
        self._gesture_library = gesture_library
        self._events = choreography.get("events", [])

        if not self._events:
            raise ValueError("舞谱为空，无法播放")

        print(f"[Dance] 加载舞谱: {len(self._events)} 个事件, "
              f"总时长≈{self._events[-1]['time']:.0f}s")

        self._start_audio(str(audio_path))

        self._t0 = time.time()
        self._current_idx = 0
        self._last_idx = -1
        self._state = PlayerState.PLAYING
        self._status = "🎵 手势舞"
        self._current_name = ""

    def update(self) -> list[int] | None:
        """每帧调用。返回应发送的 pose（snap 切换），None=结束。"""
        if self._state != PlayerState.PLAYING:
            return None

        elapsed = time.time() - self._t0

        # 音乐播完检测
        try:
            import pygame.mixer
            if not pygame.mixer.music.get_busy():
                self._state = PlayerState.FINISHED
                self._status = "✅ 手势舞结束"
                print("[Dance] 音乐结束")
                return None
        except Exception:
            pass

        # 查找当前事件
        target_idx = self._current_idx
        for i in range(self._current_idx, len(self._events)):
            if self._events[i]["time"] <= elapsed:
                target_idx = i
            else:
                break

        self._current_idx = target_idx
        event = self._events[self._current_idx]
        name = event["gesture"]
        self._current_name = name

        # 切换时打印日志
        if self._current_idx != self._last_idx:
            self._last_idx = self._current_idx
            next_event = self._events[self._current_idx + 1] if self._current_idx + 1 < len(self._events) else None
            next_info = f" → {next_event['gesture']}@{next_event['time']:.1f}s" if next_event else " → 结束"
            print(f"[Dance] {name} @ {elapsed:.1f}s{next_info}")

        # 状态栏
        mins = int(elapsed // 60)
        secs = int(elapsed % 60)
        self._status = f"🎵 手势舞 | {name} | {mins:02d}:{secs:02d}"

        # 查 pose
        pose = self._lookup_pose(name)
        if pose is None:
            return None
        return list(pose)

    def stop(self) -> None:
        try:
            import pygame.mixer
            pygame.mixer.music.stop()
        except Exception:
            pass
        self._state = PlayerState.IDLE
        self._status = ""

    # ---- internal ----

    def _lookup_pose(self, name: str) -> list[int] | None:
        if self._gesture_library is None:
            return None
        return self._gesture_library.lookup(name)

    @staticmethod
    def _start_audio(audio_path: str) -> None:
        try:
            import pygame
            pygame.mixer.init()
            pygame.mixer.music.load(audio_path)
            pygame.mixer.music.play()
            print(f"[Dance] 播放: {Path(audio_path).name}")
        except ImportError:
            raise ImportError("请先安装 pygame: pip install pygame")
        except Exception as e:
            raise RuntimeError(f"音频播放失败: {e}")
