from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


UPLOAD_INIT_URL = os.getenv(
    "TIKTOK_VIDEO_UPLOAD_INIT_URL",
    "https://open.tiktokapis.com/v2/post/publish/video/init/",
)


def _json_request(
    *,
    url: str,
    method: str,
    payload: dict[str, Any],
    access_token: str,
    timeout: int = 60,
) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url=url, data=body, method=method)
    req.add_header("Authorization", f"Bearer {access_token}")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(
            f"TikTok API request failed ({exc.code}) {url}\n{error_body}"
        ) from exc


def _upload_binary(upload_url: str, video_path: Path, timeout: int = 300) -> dict[str, Any]:
    raw_data = video_path.read_bytes()
    req = urllib.request.Request(url=upload_url, data=raw_data, method="PUT")
    req.add_header("Content-Type", "video/mp4")
    req.add_header("Content-Length", str(len(raw_data)))
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            response_body = response.read().decode("utf-8", errors="ignore")
            return {
                "status_code": response.status,
                "reason": response.reason,
                "body": response_body,
            }
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="ignore")
        return {
            "status_code": exc.code,
            "reason": str(exc.reason),
            "body": error_body,
        }


def upload_video_draft(access_token: str, video_path: Path, caption: str) -> dict[str, Any]:
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    payload = {
        "post_info": {
            "title": caption,
            "privacy_level": "SELF_ONLY",
            "disable_duet": False,
            "disable_comment": False,
            "disable_stitch": False,
        },
        "source_info": {
            "source": "FILE_UPLOAD",
            "video_size": video_path.stat().st_size,
            "chunk_size": video_path.stat().st_size,
            "total_chunk_count": 1,
        },
    }
    init_response = _json_request(
        url=UPLOAD_INIT_URL,
        method="POST",
        payload=payload,
        access_token=access_token,
    )

    data = init_response.get("data", {}) if isinstance(init_response, dict) else {}
    upload_url = (
        data.get("upload_url")
        or data.get("video_upload_url")
        or data.get("uploadUrl")
        or ""
    )

    upload_response: dict[str, Any] = {"status": "skipped", "reason": "upload_url_missing"}
    if upload_url:
        upload_response = _upload_binary(str(upload_url), video_path)

    return {
        "init_response": init_response,
        "upload_response": upload_response,
        "publish_id": data.get("publish_id") or data.get("publishId"),
    }

