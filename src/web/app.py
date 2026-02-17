from __future__ import annotations

import secrets
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel, Field

from ..pipeline import (
    DEFAULT_CHANNELS_CONFIG,
    create_job_id,
    load_channels_map,
    load_job_status,
    process_video_job,
    upload_job_drafts,
)
from ..tiktok.oauth import build_authorize_url, exchange_code_for_tokens, is_connected, load_tokens
from .jobs import JobStore


app = FastAPI(title="TikTok Scheduler Uploader", version="1.0.0")
jobs = JobStore()
oauth_states: set[str] = set()
UI_PATH = Path(__file__).with_name("index.html")


class ProcessRequest(BaseModel):
    url: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    channel: str = Field(..., min_length=1)
    crop_top_px: int = 120
    title_mask_px: int = 180
    video_y_scale: float = 2.08
    interval_min: int = 30
    part_seconds: int = 70
    output_width: int = 1080
    output_height: int = 1920
    render_preset: str = "legacy"
    part_label_position: str = "top-center"
    no_part_overlay: bool = False


class UploadRequest(BaseModel):
    job_id: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    channel: str = Field(..., min_length=1)
    interval_min: int = 30
    start_time: str | None = None


def _process_task(job_id: str, request: ProcessRequest) -> None:
    jobs.update(job_id, state="processing")

    def logger(message: str) -> None:
        jobs.append_log(job_id, message)

    try:
        payload = process_video_job(
            input_value=request.url,
            title=request.title,
            channel=request.channel,
            interval_min=request.interval_min,
            part_seconds=request.part_seconds,
            crop_top_px=request.crop_top_px,
            title_mask_px=request.title_mask_px,
            video_y_scale=request.video_y_scale,
            output_width=request.output_width,
            output_height=request.output_height,
            render_preset=request.render_preset,
            part_label_position=request.part_label_position,
            no_part_overlay=request.no_part_overlay,
            channels_config=DEFAULT_CHANNELS_CONFIG,
            job_id=job_id,
            log=logger,
        )
        jobs.update(
            job_id,
            state="processed",
            part_files=payload.get("part_files", []),
            output_dir=payload.get("output_dir"),
            persisted_status=payload,
        )
    except Exception as exc:  # noqa: BLE001
        jobs.append_log(job_id, f"Process failed: {exc}")
        jobs.update(job_id, state="failed", error=str(exc))


def _upload_task(job_id: str, request: UploadRequest) -> None:
    jobs.update(job_id, state="uploading")

    def logger(message: str) -> None:
        jobs.append_log(job_id, message)

    try:
        payload = upload_job_drafts(
            job_id=request.job_id,
            title=request.title,
            channel=request.channel,
            interval_min=request.interval_min,
            start_time=request.start_time,
            channels_config=DEFAULT_CHANNELS_CONFIG,
            log=logger,
        )
        jobs.update(
            job_id,
            state="uploaded",
            upload_results=payload.get("uploads", []),
            persisted_status=payload,
        )
    except Exception as exc:  # noqa: BLE001
        jobs.append_log(job_id, f"Upload failed: {exc}")
        jobs.update(job_id, state="upload_failed", error=str(exc))


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    if not UI_PATH.exists():
        raise HTTPException(status_code=500, detail="UI file is missing.")
    return UI_PATH.read_text(encoding="utf-8")


@app.post("/api/process")
def api_process(request: ProcessRequest, background_tasks: BackgroundTasks) -> dict[str, object]:
    job_id = create_job_id(request.channel)
    jobs.create(
        job_id,
        {
            "state": "queued",
            "channel": request.channel,
            "title": request.title,
            "part_files": [],
            "error": None,
        },
    )
    jobs.append_log(job_id, "Queued processing job.")
    background_tasks.add_task(_process_task, job_id, request)
    return {"job_id": job_id, "state": "queued", "part_files": []}


@app.get("/api/status/{job_id}")
def api_status(job_id: str) -> dict[str, object]:
    record = jobs.get(job_id)
    persisted_status: dict[str, object] | None = None
    try:
        persisted_status = load_job_status(job_id)
    except Exception:  # noqa: BLE001
        persisted_status = None

    if not record and not persisted_status:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    if not record:
        return {
            "job_id": job_id,
            "state": str(persisted_status.get("state", "unknown")) if persisted_status else "unknown",
            "logs": [],
            "part_files": persisted_status.get("part_files", []) if persisted_status else [],
            "status": persisted_status,
        }

    return {
        "job_id": job_id,
        "state": record.get("state", "unknown"),
        "logs": record.get("logs", []),
        "error": record.get("error"),
        "part_files": record.get("part_files", []),
        "output_dir": record.get("output_dir"),
        "status": persisted_status or record.get("persisted_status"),
        "updated_at": record.get("updated_at"),
    }


@app.get("/api/me")
def api_me() -> dict[str, object]:
    tokens = load_tokens()
    return {
        "connected": is_connected(),
        "has_tokens": bool(tokens),
        "open_id": tokens.get("open_id") if tokens else None,
        "expires_at_utc": tokens.get("expires_at_utc") if tokens else None,
    }


@app.get("/api/channels")
def api_channels() -> dict[str, object]:
    channels = load_channels_map(DEFAULT_CHANNELS_CONFIG)
    return {"channels": sorted(channels.keys())}


@app.get("/auth/start")
def auth_start() -> RedirectResponse:
    state = secrets.token_urlsafe(24)
    oauth_states.add(state)
    try:
        url = build_authorize_url(state=state)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return RedirectResponse(url=url)


@app.get("/auth/callback", response_class=HTMLResponse)
def auth_callback(
    code: str | None = Query(default=None),
    state: str | None = Query(default=None),
    error: str | None = Query(default=None),
) -> str:
    if error:
        return (
            "<html><body style='font-family:sans-serif;background:#111;color:#fff;'>"
            f"<h2>OAuth Error</h2><p>{error}</p></body></html>"
        )
    if not code:
        return (
            "<html><body style='font-family:sans-serif;background:#111;color:#fff;'>"
            "<h2>Missing code</h2><p>OAuth callback did not include code.</p></body></html>"
        )
    if state and state in oauth_states:
        oauth_states.remove(state)

    try:
        exchange_code_for_tokens(code)
        return (
            "<html><body style='font-family:sans-serif;background:#111;color:#fff;'>"
            "<h2>Connected ✅</h2><p>TikTok OAuth complete. You can return to the app.</p>"
            "<script>setTimeout(()=>window.close(),1500);</script>"
            "</body></html>"
        )
    except Exception as exc:  # noqa: BLE001
        return (
            "<html><body style='font-family:sans-serif;background:#111;color:#fff;'>"
            f"<h2>OAuth Failed</h2><p>{exc}</p></body></html>"
        )


@app.post("/api/upload")
def api_upload(request: UploadRequest, background_tasks: BackgroundTasks) -> dict[str, object]:
    if not jobs.exists(request.job_id):
        jobs.create(
            request.job_id,
            {
                "state": "queued",
                "channel": request.channel,
                "title": request.title,
                "part_files": [],
                "error": None,
            },
        )
    jobs.append_log(request.job_id, "Queued upload job.")
    background_tasks.add_task(_upload_task, request.job_id, request)
    return {"job_id": request.job_id, "state": "upload_queued"}
