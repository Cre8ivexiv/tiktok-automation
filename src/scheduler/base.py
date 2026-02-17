from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class UploadPayload:
    channel_name: str
    account_id: str
    video_path: Path
    caption: str
    scheduled_time: datetime


class SchedulerAdapter(ABC):
    @abstractmethod
    def schedule_post(self, payload: UploadPayload) -> dict[str, Any]:
        """Submit one scheduled post."""

