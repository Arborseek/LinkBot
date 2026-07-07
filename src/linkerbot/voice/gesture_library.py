"""姿态库 CRUD。

从 YAML 文件加载命名姿态，支持查询、添加、保存。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import yaml


class GestureLibrary:
    """命名姿态集合"""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._gestures: Dict[str, list[int]] = {}

    # ---- public ----

    @property
    def names(self) -> List[str]:
        return list(self._gestures.keys())

    @property
    def count(self) -> int:
        return len(self._gestures)

    def load(self) -> None:
        """从 YAML 加载姿态库"""
        if not self.path.exists():
            self._gestures = {}
            return
        with open(self.path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        self._gestures = {}
        for name, entry in data.get("gestures", {}).items():
            if isinstance(entry, dict) and "pose" in entry:
                self._gestures[name] = [int(v) for v in entry["pose"]]

    def lookup(self, name: str) -> Optional[list[int]]:
        """查询姿态。返回 0-255 pose 列表或 None"""
        return self._gestures.get(name)

    def add(self, name: str, pose: list[int]) -> None:
        """添加或覆盖姿态"""
        self._gestures[name] = [int(v) for v in pose]

    def remove(self, name: str) -> bool:
        """删除姿态。返回是否成功"""
        if name in self._gestures:
            del self._gestures[name]
            return True
        return False

    def save(self) -> None:
        """保存到 YAML"""
        data = {
            "gestures": {
                name: {"pose": [int(v) for v in pose]}
                for name, pose in self._gestures.items()
            }
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, allow_unicode=True, default_flow_style=None, sort_keys=False)
