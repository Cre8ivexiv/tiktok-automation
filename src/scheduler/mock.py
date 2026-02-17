from __future__ import annotations

import json
from datetime import datetime
from typing import Any
from uuid import uuid4

from .base import SchedulerAdapter, UploadPayload


class MockScheduler(SchedulerAdapter):
    """Fallback scheduler adapter that prints the request payload."""

    def schedule_post(self, payload: UploadPayload) -> dict[str, Any]:
        response = {
            "id": f"mock_{uuid4().hex[:12]}",
            "provider": "mock",
            "scheduled_time": payload.scheduled_time.strftime("%Y-%m-%d %H:%M"),
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        request_data = {
            "channel_name": payload.channel_name,
            "account_id": payload.account_id,
            "video_path": str(payload.video_path),
            "caption": payload.caption,
            "scheduled_time": payload.scheduled_time.strftime("%Y-%m-%d %H:%M"),
        }
        print(json.dumps({"mock_request": request_data, "mock_response": response}, indent=2))
        return response

