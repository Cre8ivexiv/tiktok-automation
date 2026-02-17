from __future__ import annotations

from datetime import datetime
from threading import Lock
from typing import Any


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, dict[str, Any]] = {}
        self._lock = Lock()

    def create(self, job_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            record = {
                "job_id": job_id,
                "state": "queued",
                "logs": [],
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            record.update(payload)
            self._jobs[job_id] = record
            return dict(record)

    def append_log(self, job_id: str, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        with self._lock:
            record = self._jobs.get(job_id)
            if not record:
                return
            record.setdefault("logs", []).append(f"[{timestamp}] {message}")
            record["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def update(self, job_id: str, **fields: Any) -> dict[str, Any]:
        with self._lock:
            record = self._jobs.get(job_id)
            if not record:
                raise KeyError(f"Job not found: {job_id}")
            record.update(fields)
            record["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            return dict(record)

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            record = self._jobs.get(job_id)
            return dict(record) if record else None

    def exists(self, job_id: str) -> bool:
        with self._lock:
            return job_id in self._jobs

