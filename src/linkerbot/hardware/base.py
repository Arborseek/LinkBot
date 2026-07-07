from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List


class HandDriver(ABC):
    @abstractmethod
    def connect(self) -> None:
        ...

    @abstractmethod
    def disconnect(self) -> None:
        ...

    @abstractmethod
    def send_pose(self, pose: List[int]) -> None:
        ...

    def send_open_pose(self) -> None:
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        ...
