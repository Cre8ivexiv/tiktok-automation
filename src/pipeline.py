from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable

from .captions import build_caption
from .download import is_http_url, resolve_input_video
from .render import RenderedPart, rendered_parts_to_dict, render_parts, resolve_segments
from .tiktok.oauth import get_valid_access_token, load_tokens
from .tiktok.posting import upload_video_draft


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHANNELS_CONFIG = PROJECT_ROOT / "config" / "channels.json"
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"
DOWNLOADS_ROOT = PROJECT_ROOT / "downloads"

LogFn = Callable[[str], None]


def _noop_log(_: str) -> None:
    return


def _slugify(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._-")
    return text[:80] if text else "job"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_channels_map(config_path: Path = DEFAULT_CHANNELS_CONFIG) -> dict[str, dict[str, Any]]:
    if not config_path.exists():
        raise FileNotFoundError(f"Channels config not found: {config_path}")

    data = read_json(config_path)
    if isinstance(data, dict) and "channels" in data and isinstance(data["channels"], dict):
        channels = data["channels"]
    elif isinstance(data, dict):
        channels = data
    else:
        raise ValueError("Invalid channels config format.")

    normalized: dict[str, dict[str, Any]] = {}
    for name, payload in channels.items():
        if isinstance(payload, dict):
            normalized[str(name)] = payload
    return normalized


def resolve_account_id(channel_payload: dict[str, Any], channel_name: str) -> str:
    return str(
        channel_payload.get("account_id")
        or channel_payload.get("provider_account_id")
        or channel_payload.get("id")
        or channel_name
    )


def parse_start_time(value: str | None) -> datetime:
    if value:
        return datetime.strptime(value, "%Y-%m-%d %H:%M")
    now = datetime.now()
    rounded = now.replace(minute=0, second=0, microsecond=0)
    if rounded <= now:
        rounded += timedelta(hours=1)
    return rounded


def extract_part_number(path: Path) -> int:
    match = re.search(r"part_(\d+)", path.stem, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Could not parse part number from file name: {path.name}")
    return int(match.group(1))


def discover_part_files(parts_dir: Path) -> list[Path]:
    files = [item for item in parts_dir.glob("part_*.mp4") if item.is_file()]
    return sorted(files, key=extract_part_number)


def create_job_id(channel_name: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{_slugify(channel_name)[:20]}_{uuid.uuid4().hex[:6]}"


def build_output_dir(channel_name: str, job_id: str) -> Path:
    return (OUTPUTS_ROOT / channel_name / job_id).resolve()


def find_job_output_dir(job_id: str, channel_name: str | None = None) -> Path:
    if channel_name:
        candidate = build_output_dir(channel_name, job_id)
        if candidate.exists():
            return candidate
    for channel_dir in OUTPUTS_ROOT.glob("*"):
        candidate = channel_dir / job_id
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not locate output directory for job_id={job_id}")


def _build_schedule_plan(
    title: str,
    part_count: int,
    interval_min: int,
    start_time: datetime,
) -> list[dict[str, str]]:
    plan: list[dict[str, str]] = []
    for index in range(1, part_count + 1):
        planned_time = start_time + timedelta(minutes=interval_min * (index - 1))
        plan.append(
            {
                "part_number": index,
                "caption": build_caption(title=title, part_number=index),
                "planned_time": planned_time.strftime("%Y-%m-%d %H:%M"),
            }
        )
    return plan


def process_video_job(
    *,
    input_value: str,
    title: str,
    channel: str,
    interval_min: int = 30,
    part_seconds: int = 70,
    crop_top_px: int = 120,
    title_mask_px: int = 180,
    video_y_scale: float = 2.08,
    output_width: int = 1080,
    output_height: int = 1920,
    render_preset: str = "legacy",
    part_label_position: str = "top-center",
    no_part_overlay: bool = False,
    cuts_path: Path | None = None,
    channels_config: Path = DEFAULT_CHANNELS_CONFIG,
    ffmpeg_bin: str = "ffmpeg",
    ffprobe_bin: str = "ffprobe",
    crf: int = 18,
    preset: str = "slow",
    start_time: str | None = None,
    job_id: str | None = None,
    log: LogFn | None = None,
) -> dict[str, Any]:
    log_fn = log or _noop_log
    channels = load_channels_map(channels_config)
    if channel not in channels:
        raise KeyError(f"Channel '{channel}' not found in {channels_config}")

    effective_job_id = job_id or create_job_id(channel)
    out_dir = build_output_dir(channel, effective_job_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_fn(f"Job created: {effective_job_id}")

    input_video, was_downloaded = resolve_input_video(
        input_value=input_value,
        downloads_root=DOWNLOADS_ROOT,
    )
    log_fn(f"Source ready: {input_video}")

    normalized_cuts_path = cuts_path.resolve() if cuts_path else None
    if normalized_cuts_path:
        log_fn(f"Using cuts override: {normalized_cuts_path}")
    else:
        log_fn(f"Auto-splitting into {part_seconds}s parts.")

    segments, cuts_overrides = resolve_segments(
        input_video=input_video,
        part_seconds=part_seconds,
        cuts_path=normalized_cuts_path,
        ffprobe_bin=ffprobe_bin,
    )
    log_fn(f"Segments ready: {len(segments)}")

    effective_crop_top = cuts_overrides.get("crop_top_px", crop_top_px)
    effective_out_w = cuts_overrides.get("output_width", output_width)
    effective_out_h = cuts_overrides.get("output_height", output_height)

    rendered: list[RenderedPart] = render_parts(
        input_video=input_video,
        out_dir=out_dir,
        segments=segments,
        crop_top_px=effective_crop_top,
        title_mask_px=title_mask_px,
        output_width=effective_out_w,
        output_height=effective_out_h,
        video_y_scale=video_y_scale,
        render_preset=render_preset,
        part_overlay_enabled=not no_part_overlay,
        part_label_position=part_label_position,
        ffmpeg_bin=ffmpeg_bin,
        ffprobe_bin=ffprobe_bin,
        crf=crf,
        preset=preset,
        log=log_fn,
    )
    log_fn(f"Rendered parts: {len(rendered)}")

    schedule_start = parse_start_time(start_time)
    schedule_plan = _build_schedule_plan(
        title=title,
        part_count=len(rendered),
        interval_min=interval_min,
        start_time=schedule_start,
    )

    status_payload: dict[str, Any] = {
        "mode": "process",
        "state": "processed",
        "job_id": effective_job_id,
        "input": input_value,
        "input_type": "url" if is_http_url(input_value) else "file",
        "downloaded_source": was_downloaded,
        "source_video": str(input_video.resolve()),
        "channel": channel,
        "title": title,
        "output_dir": str(out_dir),
        "part_seconds": part_seconds,
        "cuts_path": str(normalized_cuts_path) if normalized_cuts_path else None,
        "render_config": {
            "crop_top_px": effective_crop_top,
            "title_mask_px": title_mask_px,
            "video_y_scale": video_y_scale,
            "output_width": effective_out_w,
            "output_height": effective_out_h,
            "render_preset": render_preset,
            "part_overlay_enabled": not no_part_overlay,
            "part_label_position": part_label_position,
            "crf": crf,
            "preset": preset,
        },
        "rendered_parts": rendered_parts_to_dict(rendered),
        "render_manifest_path": str((out_dir / "render_manifest.json").resolve()),
        "part_files": [str(item.path.resolve()) for item in rendered],
        "upload_plan": {
            "start_time": schedule_start.strftime("%Y-%m-%d %H:%M"),
            "interval_min": interval_min,
            "items": schedule_plan,
        },
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_json(out_dir / "status.json", status_payload)
    log_fn(f"Status written: {out_dir / 'status.json'}")
    return status_payload


def upload_job_drafts(
    *,
    job_id: str,
    title: str,
    channel: str,
    interval_min: int = 30,
    start_time: str | None = None,
    channels_config: Path = DEFAULT_CHANNELS_CONFIG,
    log: LogFn | None = None,
) -> dict[str, Any]:
    log_fn = log or _noop_log
    channels = load_channels_map(channels_config)
    if channel not in channels:
        raise KeyError(f"Channel '{channel}' not found in {channels_config}")

    out_dir = find_job_output_dir(job_id=job_id, channel_name=channel)
    status_path = out_dir / "status.json"
    existing_status = read_json(status_path) if status_path.exists() else {}
    part_files = discover_part_files(out_dir)
    if not part_files:
        raise RuntimeError(f"No rendered parts found for job_id={job_id}")

    log_fn(f"Uploading {len(part_files)} part(s) as drafts...")
    tokens = load_tokens()
    if not tokens:
        raise RuntimeError("TikTok is not connected. Complete OAuth first.")
    access_token = get_valid_access_token()

    schedule_start = parse_start_time(start_time)
    account_id = resolve_account_id(channels[channel], channel)
    upload_results: list[dict[str, Any]] = []

    for index, video_path in enumerate(part_files, start=1):
        planned_time = schedule_start + timedelta(minutes=interval_min * (index - 1))
        caption = build_caption(title=title, part_number=index)
        log_fn(f"Uploading Part {index}: {video_path.name}")
        response = upload_video_draft(
            access_token=access_token,
            video_path=video_path,
            caption=caption,
        )
        upload_results.append(
            {
                "part_number": index,
                "video_path": str(video_path.resolve()),
                "caption": caption,
                "planned_time": planned_time.strftime("%Y-%m-%d %H:%M"),
                "response": response,
            }
        )

    existing_status.update(
        {
            "state": "uploaded",
            "job_id": job_id,
            "channel": channel,
            "account_id": account_id,
            "title": title,
            "upload_plan": {
                "start_time": schedule_start.strftime("%Y-%m-%d %H:%M"),
                "interval_min": interval_min,
                "items": [
                    {
                        "part_number": item["part_number"],
                        "caption": item["caption"],
                        "planned_time": item["planned_time"],
                    }
                    for item in upload_results
                ],
            },
            "uploads": upload_results,
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    )
    write_json(status_path, existing_status)
    log_fn(f"Upload status written: {status_path}")
    return existing_status


def load_job_status(job_id: str, channel: str | None = None) -> dict[str, Any]:
    out_dir = find_job_output_dir(job_id=job_id, channel_name=channel)
    status_path = out_dir / "status.json"
    if not status_path.exists():
        raise FileNotFoundError(f"No status.json found for job_id={job_id}")
    return read_json(status_path)
