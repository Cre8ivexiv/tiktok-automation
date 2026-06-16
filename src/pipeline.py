from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable

from .captions import build_caption
from .download import extract_youtube_chapters, is_http_url, resolve_input_video
from .render import RenderedPart, Segment, parse_time_to_seconds, rendered_parts_to_dict, render_parts, resolve_segments
from .tiktok.oauth import get_valid_access_token, load_tokens
from .tiktok.posting import upload_video_draft


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHANNELS_CONFIG = PROJECT_ROOT / "config" / "channels.json"
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"
DOWNLOADS_ROOT = PROJECT_ROOT / "downloads"

LogFn = Callable[[str], None]
RENDER_STYLE_KEYS: frozenset[str] = frozenset(
    {
        "crop_top_px",
        "title_mask_px",
        "edge_bar_px",
        "content_height_bump_px",
        "content_max_height_px",
        "video_y_scale",
        "y_scale_mode",
        "render_mode",
        "output_width",
        "output_height",
        "render_preset",
        "part_label_position",
        "label_x_pct",
        "label_y_pct",
        "show_part_label",
        "part_label_x_percent",
        "part_label_y_percent",
        "no_part_overlay",
        "hashtags",
        "chapter_title_position",
        "manual_caption_text",
        "overlay_x_percent",
        "overlay_y_percent",
        "show_youtube_credit",
        "youtube_credit_text",
        "youtube_credit_position",
        "playback_speed",
        "subtitles_enabled",
        "subtitle_style",
        "subtitle_language",
        "subtitle_offset_seconds",
        "video_style_scale",
        "crf",
        "preset",
        "ffmpeg_bin",
        "ffprobe_bin",
        "reaction_layout_enabled",
        "reaction_layout_mode",
        "reaction_layout_preset",
        "main_crop",
        "facecam_crop",
        "facecam_shape",
        "caption_text",
        "caption_position",
        "caption_duration_mode",
        "caption_duration_seconds",
        "reference_frame_url",
        "source_width",
        "source_height",
        "preview_frame_timestamp",
        "logo_enabled",
        "logo_path",
        "logo_x_percent",
        "logo_y_percent",
        "logo_width_percent",
        "logo_opacity",
        "reaction_layout_keyframes",
        "reaction_timeline",
        "imported_clip_plan",
    }
)


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


def normalize_video_style_scale(value: int | float | str | None) -> int:
    try:
        numeric = int(round(float(value))) if value is not None else 50
    except (TypeError, ValueError):
        numeric = 50
    return max(0, min(100, numeric))


def video_style_scale_to_bump_px(video_style_scale: int | float | str | None) -> int:
    """Map 0-100 user-facing style scale to existing height bump pixels."""
    return int((normalize_video_style_scale(video_style_scale) - 50) * 8)


def effective_content_height_bump_px(
    *,
    content_height_bump_px: int,
    video_style_scale: int | float | str | None,
) -> int:
    if int(content_height_bump_px) != 0:
        return int(content_height_bump_px)
    return video_style_scale_to_bump_px(video_style_scale)


def _format_hhmmss(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _normalize_imported_clip_plan(
    imported_clip_plan: dict[str, Any] | None,
) -> tuple[list[Segment], list[str], list[dict[str, Any]], dict[str, Any] | None]:
    if not isinstance(imported_clip_plan, dict):
        return [], [], [], None

    raw_clips = imported_clip_plan.get("clips")
    if not isinstance(raw_clips, list):
        return [], [], [], None

    segments: list[Segment] = []
    titles: list[str] = []
    metadata: list[dict[str, Any]] = []
    previous_end = -1.0
    for idx, raw_clip in enumerate(raw_clips, start=1):
        if not isinstance(raw_clip, dict) or raw_clip.get("enabled", True) is False:
            continue
        start = parse_time_to_seconds(str(raw_clip.get("start", "")))
        end = parse_time_to_seconds(str(raw_clip.get("end", "")))
        if end <= start:
            raise ValueError(f"Imported clip #{idx} end must be after start.")
        if previous_end >= 0 and start < previous_end:
            raise ValueError(f"Imported clip #{idx} overlaps the previous enabled clip.")
        previous_end = end

        title_text = str(raw_clip.get("title") or raw_clip.get("caption_text") or f"Clip {idx}").strip()
        caption_text = str(raw_clip.get("caption_text") or title_text).strip()
        segment = Segment(start=start, end=end)
        row = {
            "id": raw_clip.get("id", idx),
            "enabled": True,
            "start": _format_hhmmss(start),
            "end": _format_hhmmss(end),
            "title": title_text,
            "caption_text": caption_text,
            "hook": str(raw_clip.get("hook") or "").strip(),
            "summary": str(raw_clip.get("summary") or "").strip(),
            "day": str(raw_clip.get("day") or "").strip(),
            "clip_type": str(raw_clip.get("clip_type") or "").strip(),
            "mood": str(raw_clip.get("mood") or "").strip(),
            "characters_involved": raw_clip.get("characters_involved") if isinstance(raw_clip.get("characters_involved"), list) else [],
            "keywords": raw_clip.get("keywords") if isinstance(raw_clip.get("keywords"), list) else [],
            "subtitle_style": str(raw_clip.get("subtitle_style") or "").strip(),
            "suggested_layout": str(raw_clip.get("suggested_layout") or "").strip(),
            "suggested_zoom": str(raw_clip.get("suggested_zoom") or "").strip(),
            "render_preset": str(raw_clip.get("render_preset") or "").strip(),
            "importance": str(raw_clip.get("importance") or "").strip(),
            "start_seconds": start,
            "end_seconds": end,
            "duration_seconds": segment.duration,
        }
        segments.append(segment)
        titles.append(caption_text or title_text)
        metadata.append(row)

    if not segments:
        return [], [], [], None

    normalized_plan = {
        "video_title": str(imported_clip_plan.get("video_title") or "").strip(),
        "summary": str(imported_clip_plan.get("summary") or "").strip(),
        "characters": imported_clip_plan.get("characters") if isinstance(imported_clip_plan.get("characters"), list) else [],
        "clips": metadata,
        "import_ready_for_quickclips": True,
    }
    return segments, titles, metadata, normalized_plan


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
    part_descriptions: dict[int, str] | None = None,
) -> list[dict[str, Any]]:
    plan: list[dict[str, Any]] = []
    for index in range(1, part_count + 1):
        planned_time = start_time + timedelta(minutes=interval_min * (index - 1))
        plan.append(
            {
                "part_number": index,
                "caption": (part_descriptions or {}).get(index) or build_caption(title=title, part_number=index),
                "planned_time": planned_time.strftime("%Y-%m-%d %H:%M"),
            }
        )
    return plan


def _manifest_parts_by_number(out_dir: Path) -> dict[int, dict[str, Any]]:
    manifest_path = out_dir / "render_manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        manifest = read_json(manifest_path)
    except (OSError, json.JSONDecodeError):
        return {}
    parts = manifest.get("parts")
    if not isinstance(parts, list):
        return {}
    mapped: dict[int, dict[str, Any]] = {}
    for item in parts:
        if not isinstance(item, dict):
            continue
        try:
            part_number = int(item.get("part_number"))
        except (TypeError, ValueError):
            continue
        mapped[part_number] = item
    return mapped


def _description_files_from_manifest(out_dir: Path) -> dict[str, Any]:
    manifest_path = out_dir / "render_manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        manifest = read_json(manifest_path)
    except (OSError, json.JSONDecodeError):
        return {}
    files = manifest.get("description_files")
    descriptions = manifest.get("descriptions")
    return {
        "txt": str(files.get("txt") or "") if isinstance(files, dict) else "",
        "json": str(files.get("json") or "") if isinstance(files, dict) else "",
        "items": descriptions if isinstance(descriptions, list) else [],
    }


def process_video_job(
    *,
    input_value: str,
    title: str,
    channel: str,
    interval_min: int = 30,
    part_seconds: int = 70,
    split_mode: str = "duration",
    scene_threshold: float = 27.0,
    crop_top_px: int = 0,
    title_mask_px: int = 0,
    edge_bar_px: int = 45,
    content_height_bump_px: int = 0,
    content_max_height_px: int = 0,
    video_style_scale: int = 50,
    video_y_scale: float = 2.08,
    y_scale_mode: str = "letterbox",
    output_width: int = 1080,
    output_height: int = 1920,
    render_preset: str = "legacy",
    part_label_position: str = "top-center",
    label_x_pct: float = 0.5,
    label_y_pct: float = 0.05,
    show_part_label: bool = True,
    part_label_x_percent: float = 50.0,
    part_label_y_percent: float = 4.0,
    no_part_overlay: bool = False,
    hashtags: str = "",
    chapter_title_position: str = "top",
    manual_caption_text: str | None = None,
    overlay_x_percent: float = 50.0,
    overlay_y_percent: float = 12.0,
    manual_chapters: list[dict[str, Any]] | None = None,
    playback_speed: float = 1.0,
    subtitles_enabled: bool = False,
    subtitle_style: str = "hormozi",
    subtitle_language: str | None = None,
    subtitle_offset_seconds: float = 0.0,
    show_youtube_credit: bool = False,
    youtube_credit_text: str | None = None,
    youtube_credit_position: str = "below_frame",
    reaction_layout_enabled: bool = False,
    reaction_layout_mode: str = "stacked",
    reaction_layout_preset: str = "content_top_facecam_bottom",
    main_crop: dict[str, Any] | None = None,
    facecam_crop: dict[str, Any] | None = None,
    facecam_shape: str = "rectangle",
    caption_text: str | None = None,
    caption_position: str = "between",
    caption_duration_mode: str = "entire",
    caption_duration_seconds: float | None = None,
    reference_frame_url: str | None = None,
    source_width: int | None = None,
    source_height: int | None = None,
    preview_frame_timestamp: str = "00:00:05",
    logo_enabled: bool = False,
    logo_path: str | None = None,
    logo_x_percent: float = 82.0,
    logo_y_percent: float = 5.0,
    logo_width_percent: float = 15.0,
    logo_opacity: float = 100.0,
    reaction_layout_keyframes: list[dict[str, Any]] | None = None,
    reaction_timeline: list[dict[str, Any]] | None = None,
    imported_clip_plan: dict[str, Any] | None = None,
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

    input_is_url = is_http_url(input_value)
    input_video, was_downloaded, source_info = resolve_input_video(
        input_value=input_value,
        downloads_root=DOWNLOADS_ROOT,
        prefer_youtube_subtitles=subtitles_enabled,
        log=log_fn,
    )
    source_info_path = (input_video.parent / "source_info.json").resolve()
    source_info_exists = source_info_path.exists()
    source_cache_dir = input_video.parent.resolve() if source_info_exists else None

    if input_is_url:
        if was_downloaded:
            log_fn("Downloaded source video once and captured metadata during the same yt-dlp run.")
        else:
            log_fn("Using cached source video.")
    else:
        log_fn("Using local source video.")
    log_fn(f"Source ready: {input_video}")

    source_title = str(source_info.get("title") or "").strip()
    source_uploader = str(source_info.get("uploader") or "").strip()
    source_channel = str(source_info.get("channel") or "").strip()
    source_duration = source_info.get("duration")
    source_id = str(source_info.get("id") or "").strip()
    source_webpage_url = str(source_info.get("webpage_url") or "").strip()
    cached_chapters = extract_youtube_chapters(info_dict=source_info)
    chapter_count = len(cached_chapters)

    if source_info_exists:
        log_fn(f"Source metadata ready: {source_info_path}")
    if source_info:
        duration_text = f"{float(source_duration):.2f}s" if isinstance(source_duration, (int, float)) else "unknown"
        owner_text = source_uploader or source_channel or "unknown"
        log_fn(
            f"Source metadata: title={source_title or 'unknown'} | "
            f"chapters={chapter_count} | duration={duration_text} | owner={owner_text}"
        )

    normalized_cuts_path = cuts_path.resolve() if cuts_path else None
    if normalized_cuts_path:
        log_fn(f"Using cuts override: {normalized_cuts_path}")
    imported_segments, imported_titles, imported_metadata, normalized_imported_plan = _normalize_imported_clip_plan(imported_clip_plan)
    if imported_segments:
        log_fn(f"Imported AI clip plan: rendering {len(imported_segments)} enabled clip(s) from exact start/end timestamps.")
    elif split_mode == "manual":
        chapter_count_text = len(manual_chapters or [])
        log_fn(f"Manual chapters mode: building segments from {chapter_count_text} UI rows.")
    elif split_mode == "scene":
        log_fn(f"Scene detection mode (threshold={scene_threshold}) from the local source file.")
    elif split_mode == "chapters":
        log_fn("Chapters mode: building segments from cached source metadata.")
    else:
        log_fn(f"Auto-splitting local source into {part_seconds}s parts.")

    if subtitles_enabled:
        log_fn(
            "Subtitles enabled "
            f"(style={subtitle_style}, language={subtitle_language or 'auto'}, "
            f"offset={subtitle_offset_seconds:+.2f}s)"
        )
        log_fn("Subtitles: YouTube / cache / WhisperX")

    chapter_data: list[dict] | None = None
    if split_mode == "chapters":
        chapter_data = cached_chapters
        if not chapter_data:
            raise ValueError(
                "This video has no YouTube chapters. Use a different split mode."
            )
        metadata_source = "source_info.json" if source_info_exists else "resolved metadata"
        log_fn(f"Loaded {len(chapter_data)} chapters from {metadata_source}.")

    if imported_segments:
        segments = imported_segments
        cuts_overrides = {}
        chapter_titles = imported_titles
    else:
        segments, cuts_overrides, chapter_titles = resolve_segments(
            input_video=input_video,
            part_seconds=part_seconds,
            cuts_path=normalized_cuts_path,
            ffprobe_bin=ffprobe_bin,
            split_mode=split_mode,
            scene_threshold=scene_threshold,
            chapter_data=chapter_data,
            manual_chapters=manual_chapters,
            log=log_fn,
        )
    log_fn(f"Segments ready: {len(segments)}")

    effective_crop_top = cuts_overrides.get("crop_top_px", crop_top_px)
    effective_out_w = cuts_overrides.get("output_width", output_width)
    effective_out_h = cuts_overrides.get("output_height", output_height)
    safe_video_style_scale = normalize_video_style_scale(video_style_scale)
    effective_content_bump = effective_content_height_bump_px(
        content_height_bump_px=content_height_bump_px,
        video_style_scale=safe_video_style_scale,
    )
    log_fn("Rendering locally from the resolved source video only.")

    rendered: list[RenderedPart] = render_parts(
        input_video=input_video,
        out_dir=out_dir,
        segments=segments,
        crop_top_px=effective_crop_top,
        title_mask_px=title_mask_px,
        edge_bar_px=edge_bar_px,
        content_height_bump_px=effective_content_bump,
        content_max_height_px=content_max_height_px,
        output_width=effective_out_w,
        output_height=effective_out_h,
        video_y_scale=video_y_scale,
        y_scale_mode=y_scale_mode,
        render_preset=render_preset,
        part_overlay_enabled=show_part_label and not no_part_overlay,
        part_label_position=part_label_position,
        label_x_pct=label_x_pct,
        label_y_pct=label_y_pct,
        part_label_x_percent=part_label_x_percent,
        part_label_y_percent=part_label_y_percent,
        ffmpeg_bin=ffmpeg_bin,
        ffprobe_bin=ffprobe_bin,
        crf=crf,
        preset=preset,
        log=log_fn,
        chapter_titles=chapter_titles if chapter_titles else None,
        chapter_title_position=chapter_title_position,
        manual_caption_text=None if imported_segments else manual_caption_text,
        overlay_x_percent=overlay_x_percent,
        overlay_y_percent=overlay_y_percent,
        playback_speed=playback_speed,
        subtitles_enabled=subtitles_enabled,
        subtitle_style=subtitle_style,
        subtitle_language=subtitle_language,
        subtitle_offset_seconds=subtitle_offset_seconds,
        reaction_layout_enabled=reaction_layout_enabled,
        reaction_layout_mode=reaction_layout_mode,
        reaction_layout_preset=reaction_layout_preset,
        main_crop=main_crop,
        facecam_crop=facecam_crop,
        facecam_shape=facecam_shape,
        caption_text=caption_text,
        caption_position=caption_position,
        caption_duration_mode=caption_duration_mode,
        caption_duration_seconds=caption_duration_seconds,
        reaction_timeline=reaction_timeline,
        segment_metadata=imported_metadata if imported_segments else None,
        source_info=source_info,
        hashtags=hashtags,
        base_title=title,
        show_youtube_credit=show_youtube_credit,
        youtube_credit_text=youtube_credit_text or (source_channel or source_uploader),
        youtube_credit_position=youtube_credit_position,
        logo_enabled=logo_enabled,
        logo_path=logo_path,
        logo_x_percent=logo_x_percent,
        logo_y_percent=logo_y_percent,
        logo_width_percent=logo_width_percent,
        logo_opacity=logo_opacity,
    )
    log_fn(f"Rendered parts: {len(rendered)}")

    schedule_start = parse_start_time(start_time)
    manifest_parts = _manifest_parts_by_number(out_dir)
    description_files = _description_files_from_manifest(out_dir)
    part_descriptions = {
        part_number: str(item.get("upload_description") or "").strip()
        for part_number, item in manifest_parts.items()
        if str(item.get("upload_description") or "").strip()
    }
    schedule_plan = _build_schedule_plan(
        title=title,
        part_count=len(rendered),
        interval_min=interval_min,
        start_time=schedule_start,
        part_descriptions=part_descriptions,
    )

    status_payload: dict[str, Any] = {
        "mode": "process",
        "state": "processed",
        "job_id": effective_job_id,
        "input": input_value,
        "input_type": "url" if input_is_url else "file",
        "downloaded_source": was_downloaded,
        "source_cached": input_is_url and not was_downloaded,
        "source_video": str(input_video.resolve()),
        "source_cache_dir": str(source_cache_dir) if source_cache_dir else None,
        "source_info_path": str(source_info_path) if source_info_exists else None,
        "source_metadata": {
            "id": source_id,
            "title": source_title,
            "uploader": source_uploader,
            "channel": source_channel,
            "duration": source_duration,
            "webpage_url": source_webpage_url,
            "extractor": source_info.get("extractor"),
            "chapter_count": chapter_count,
        },
        "channel": channel,
        "title": title,
        "output_dir": str(out_dir),
        "part_seconds": part_seconds,
        "cuts_path": str(normalized_cuts_path) if normalized_cuts_path else None,
        "render_config": {
            "split_mode": split_mode,
            "imported_clip_plan_enabled": bool(imported_segments),
            "imported_clip_count": len(imported_segments),
            "scene_threshold": scene_threshold,
            "crop_top_px": effective_crop_top,
            "title_mask_px": title_mask_px,
            "edge_bar_px": edge_bar_px,
            "content_height_bump_px": effective_content_bump,
            "content_height_bump_px_requested": content_height_bump_px,
            "content_max_height_px": content_max_height_px,
            "video_style_scale": safe_video_style_scale,
            "video_y_scale": video_y_scale,
            "y_scale_mode": y_scale_mode,
            "output_width": effective_out_w,
            "output_height": effective_out_h,
            "render_preset": render_preset,
            "part_overlay_enabled": show_part_label and not no_part_overlay,
            "part_label_position": part_label_position,
            "label_x_pct": label_x_pct,
            "label_y_pct": label_y_pct,
            "show_part_label": show_part_label,
            "part_label_x_percent": part_label_x_percent,
            "part_label_y_percent": part_label_y_percent,
            "hashtags": hashtags,
            "crf": crf,
            "preset": preset,
            "chapter_title_position": chapter_title_position,
            "manual_caption_text": manual_caption_text,
            "overlay_x_percent": overlay_x_percent,
            "overlay_y_percent": overlay_y_percent,
            "chapter_titles": chapter_titles if chapter_titles else [],
            "manual_chapters": manual_chapters if manual_chapters else [],
            "playback_speed": playback_speed,
            "subtitles_enabled": subtitles_enabled,
            "subtitle_style": subtitle_style,
            "subtitle_language": subtitle_language,
            "subtitle_offset_seconds": subtitle_offset_seconds,
            "show_youtube_credit": show_youtube_credit,
            "youtube_credit_text": youtube_credit_text or (source_channel or source_uploader),
            "youtube_credit_position": youtube_credit_position,
            "reaction_layout_enabled": reaction_layout_enabled,
            "reaction_layout_mode": reaction_layout_mode,
            "reaction_layout_preset": reaction_layout_preset,
            "main_crop": main_crop or {},
            "facecam_crop": facecam_crop or {},
            "facecam_shape": facecam_shape,
            "caption_text": caption_text,
            "caption_position": caption_position,
            "caption_duration_mode": caption_duration_mode,
            "caption_duration_seconds": caption_duration_seconds,
            "reference_frame_url": reference_frame_url,
            "source_width": source_width,
            "source_height": source_height,
            "preview_frame_timestamp": preview_frame_timestamp,
            "logo_enabled": logo_enabled,
            "logo_path": logo_path,
            "logo_x_percent": logo_x_percent,
            "logo_y_percent": logo_y_percent,
            "logo_width_percent": logo_width_percent,
            "logo_opacity": logo_opacity,
            "reaction_layout_keyframes": reaction_layout_keyframes or [],
            "reaction_timeline": reaction_timeline or [],
            "imported_clip_plan": normalized_imported_plan or {},
        },
        "imported_clip_plan": normalized_imported_plan,
        "rendered_parts": rendered_parts_to_dict(rendered),
        "render_manifest_path": str((out_dir / "render_manifest.json").resolve()),
        "description_files": description_files,
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


def render_custom_segments(
    *,
    input_video: Path,
    segments: list[Segment],
    titles: list[str] | None,
    style_config: dict[str, Any],
    out_dir: Path,
    log_fn: LogFn | None = None,
) -> dict[str, Any]:
    """Render caller-provided source segments through the normal render pipeline."""
    log = log_fn or _noop_log
    filtered = {key: style_config.get(key) for key in RENDER_STYLE_KEYS if key in style_config}
    safe_video_style_scale = normalize_video_style_scale(filtered.get("video_style_scale", 50))
    requested_bump = int(filtered.get("content_height_bump_px") or 0)
    effective_bump = effective_content_height_bump_px(
        content_height_bump_px=requested_bump,
        video_style_scale=safe_video_style_scale,
    )
    overlay_x_percent = float(filtered["overlay_x_percent"]) if filtered.get("overlay_x_percent") is not None else 50.0
    overlay_y_percent = float(filtered["overlay_y_percent"]) if filtered.get("overlay_y_percent") is not None else 12.0

    parts = render_parts(
        input_video=input_video,
        out_dir=out_dir,
        segments=segments,
        crop_top_px=int(filtered.get("crop_top_px") or 0),
        title_mask_px=int(filtered.get("title_mask_px") or 0),
        edge_bar_px=int(filtered.get("edge_bar_px") or 45),
        content_height_bump_px=effective_bump,
        content_max_height_px=int(filtered.get("content_max_height_px") or 0),
        output_width=int(filtered.get("output_width") or 1080),
        output_height=int(filtered.get("output_height") or 1920),
        video_y_scale=float(filtered.get("video_y_scale") or 2.08),
        y_scale_mode=str(filtered.get("y_scale_mode") or filtered.get("render_mode") or "letterbox"),
        render_preset=str(filtered.get("render_preset") or "legacy"),
        part_overlay_enabled=bool(filtered.get("show_part_label", True)) and not bool(filtered.get("no_part_overlay", False)),
        part_label_position=str(filtered.get("part_label_position") or "top-center"),
        label_x_pct=float(filtered.get("label_x_pct") or 0.5),
        label_y_pct=float(filtered.get("label_y_pct") or 0.05),
        part_label_x_percent=float(filtered.get("part_label_x_percent") or 50.0),
        part_label_y_percent=float(filtered.get("part_label_y_percent") or 4.0),
        ffmpeg_bin=str(filtered.get("ffmpeg_bin") or "ffmpeg"),
        ffprobe_bin=str(filtered.get("ffprobe_bin") or "ffprobe"),
        crf=int(filtered.get("crf") or 18),
        preset=str(filtered.get("preset") or "slow"),
        log=log,
        chapter_titles=titles,
        chapter_title_position=str(filtered.get("chapter_title_position") or "top"),
        manual_caption_text=filtered.get("manual_caption_text"),
        overlay_x_percent=overlay_x_percent,
        overlay_y_percent=overlay_y_percent,
        playback_speed=float(filtered.get("playback_speed") or 1.0),
        subtitles_enabled=bool(filtered.get("subtitles_enabled", False)),
        subtitle_style=str(filtered.get("subtitle_style") or "hormozi"),
        subtitle_language=filtered.get("subtitle_language"),
        subtitle_offset_seconds=float(filtered.get("subtitle_offset_seconds") or 0.0),
        reaction_layout_enabled=bool(filtered.get("reaction_layout_enabled", False)),
        reaction_layout_mode=str(filtered.get("reaction_layout_mode") or "stacked"),
        reaction_layout_preset=str(filtered.get("reaction_layout_preset") or "content_top_facecam_bottom"),
        main_crop=filtered.get("main_crop") if isinstance(filtered.get("main_crop"), dict) else None,
        facecam_crop=filtered.get("facecam_crop") if isinstance(filtered.get("facecam_crop"), dict) else None,
        facecam_shape=str(filtered.get("facecam_shape") or "rectangle"),
        caption_text=filtered.get("caption_text"),
        caption_position=str(filtered.get("caption_position") or "between"),
        caption_duration_mode=str(filtered.get("caption_duration_mode") or "entire"),
        caption_duration_seconds=(
            float(filtered["caption_duration_seconds"])
            if filtered.get("caption_duration_seconds") is not None
            else None
        ),
        reaction_timeline=filtered.get("reaction_timeline") if isinstance(filtered.get("reaction_timeline"), list) else None,
        source_info=None,
        hashtags=str(filtered.get("hashtags") or ""),
        base_title=str(filtered.get("manual_caption_text") or "Livestream clip"),
        show_youtube_credit=bool(filtered.get("show_youtube_credit", False)),
        youtube_credit_text=filtered.get("youtube_credit_text"),
        youtube_credit_position=str(filtered.get("youtube_credit_position") or "below_frame"),
        logo_enabled=bool(filtered.get("logo_enabled", False)),
        logo_path=filtered.get("logo_path"),
        logo_x_percent=float(filtered.get("logo_x_percent") or 82.0),
        logo_y_percent=float(filtered.get("logo_y_percent") or 5.0),
        logo_width_percent=float(filtered.get("logo_width_percent") or 15.0),
        logo_opacity=float(filtered.get("logo_opacity") if filtered.get("logo_opacity") is not None else 100.0),
    )
    description_files = _description_files_from_manifest(out_dir)
    payload = {
        "state": "processed",
        "input": str(input_video.resolve()),
        "input_type": "file",
        "output_dir": str(out_dir.resolve()),
        "render_config": {
            **filtered,
            "content_height_bump_px": effective_bump,
            "content_height_bump_px_requested": requested_bump,
            "video_style_scale": safe_video_style_scale,
            "manual_caption_text": filtered.get("manual_caption_text"),
            "overlay_x_percent": overlay_x_percent,
            "overlay_y_percent": overlay_y_percent,
            "chapter_titles": titles or [],
        },
        "rendered_parts": rendered_parts_to_dict(parts),
        "render_manifest_path": str((out_dir / "render_manifest.json").resolve()),
        "description_files": description_files,
        "part_files": [str(item.path.resolve()) for item in parts],
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_json(out_dir / "status.json", payload)
    return payload


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
    manifest_parts = _manifest_parts_by_number(out_dir)

    for index, video_path in enumerate(part_files, start=1):
        planned_time = schedule_start + timedelta(minutes=interval_min * (index - 1))
        manifest_part = manifest_parts.get(index, {})
        caption = str(manifest_part.get("upload_description") or "").strip() or build_caption(title=title, part_number=index)
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
                "upload_description": caption,
                "title": str(manifest_part.get("title") or "").strip() or title,
                "hashtags": str(manifest_part.get("hashtags") or "").strip(),
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
                        "upload_description": item["upload_description"],
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
