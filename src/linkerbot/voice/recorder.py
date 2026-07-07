"""姿态录制模块。

用户掰机械手到目标位 → 按键 → 捕获当前 SDK pose → 命名保存。
"""

from __future__ import annotations

from typing import Callable, Optional


class GestureRecorder:
    """按键捕获当前手部姿态并保存到姿态库"""

    def __init__(
        self,
        save_callback: Callable[[str, list[int]], None],
        get_current_pose: Callable[[], Optional[list[int]]],
    ):
        self._save = save_callback
        self._get_pose = get_current_pose
        self._recording = False
        self._pending_input = False
        self._status = ""

    # ---- public ----

    @property
    def recording(self) -> bool:
        return self._recording

    @property
    def status(self) -> str:
        return self._status

    def toggle(self) -> str:
        """切换录制状态。返回提示文字"""
        self._recording = not self._recording
        if self._recording:
            self._status = "📷 录制模式：掰好姿势后在终端输入名称回车保存，再按 R 退出"
        else:
            self._status = ""
            self._pending_input = False
        return self._status

    def save_current(self, name: str) -> tuple[bool, str]:
        """保存当前姿态为指定名称"""
        if not name.strip():
            return False, "名称不能为空"
        pose = self._get_pose()
        if pose is None:
            return False, "无法获取当前姿态（手未检测到？）"
        self._save(name.strip(), pose)
        return True, f"已保存姿态「{name.strip()}」"
