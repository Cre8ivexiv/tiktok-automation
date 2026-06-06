from __future__ import annotations

import json
import re
import secrets
import hashlib
import subprocess
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator
from starlette.datastructures import UploadFile as StarletteUploadFile

from ..pipeline import (
    DEFAULT_CHANNELS_CONFIG,
    DOWNLOADS_ROOT,
    OUTPUTS_ROOT,
    RENDER_STYLE_KEYS,
    build_output_dir,
    create_job_id,
    discover_part_files,
    load_channels_map,
    load_job_status,
    process_video_job,
    render_custom_segments,
    upload_job_drafts,
)
from ..ai.gemini_client import GeminiConfigError, GeminiRequestError, generate_text
from ..download import resolve_cached_input_video
from ..live import LiveRecorderManager
from ..render import Segment, _probe_video_dimensions, _resolve_binary, parse_time_to_seconds
from ..tiktok.oauth import build_authorize_url, exchange_code_for_tokens, is_connected, load_tokens
from .jobs import JobStore


app = FastAPI(title="TikTok Scheduler Uploader", version="1.0.0")
jobs = JobStore()
live_recorders = LiveRecorderManager()
oauth_states: set[str] = set()
UI_PATH = Path(__file__).with_name("index.html")
RENDER_PRESETS_PATH = Path(__file__).resolve().parents[2] / "config" / "render_presets.json"
LAYOUT_PRESETS_PATH = Path(__file__).resolve().parents[2] / "config" / "layout_presets.json"
AI_PRESETS_PATH = Path(__file__).resolve().parents[2] / "config" / "ai_presets.json"
VIDEO_EXTENSIONS = {".mp4", ".mov", ".webm", ".mkv", ".avi"}
UPLOAD_EXTENSIONS = {".mp4", ".webm", ".mkv", ".mov", ".avi"}
UPLOADS_ROOT = Path(__file__).resolve().parents[2] / "downloads" / "uploads"
REFERENCE_FRAMES_ROOT = OUTPUTS_ROOT / "reference_frames"
MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB


_VALID_SPLIT_MODES: frozenset[str] = frozenset({"duration", "parts", "manual", "ai", "scene", "chapters"})


class ManualChapterInput(BaseModel):
    start: str = Field(..., min_length=1)
    end: str | None = None
    title: str = ""
    summary: str = ""
    clip_type: str = ""
    mood: str = ""
    keywords: str | list[str] = ""
    ai_notes: str = ""

    @field_validator("start")
    @classmethod
    def _validate_start(cls, value: str) -> str:
        normalized = str(value or "").strip()
        if not normalized:
            raise ValueError("start is required")
        return normalized

    @field_validator("title")
    @classmethod
    def _normalize_title(cls, value: str) -> str:
        return str(value or "").strip()

    @field_validator("end", "summary", "clip_type", "mood", "ai_notes", mode="before")
    @classmethod
    def _normalize_optional_strings(cls, value: Any) -> str | None:
        if value is None:
            return None
        return str(value or "").strip()


class ReactionCropInput(BaseModel):
    x_percent: float = Field(default=0.0, ge=0.0, le=100.0)
    y_percent: float = Field(default=0.0, ge=0.0, le=100.0)
    width_percent: float = Field(default=100.0, gt=1.0, le=100.0)
    height_percent: float = Field(default=100.0, gt=1.0, le=100.0)


class ReactionDividerInput(BaseModel):
    direction: str = "horizontal"
    ratio: float = Field(default=0.65, ge=0.1, le=0.9)

    @field_validator("direction")
    @classmethod
    def _validate_direction(cls, value: str) -> str:
        normalized = str(value or "horizontal").strip().lower()
        if normalized not in {"horizontal", "vertical"}:
            raise ValueError("direction must be horizontal or vertical")
        return normalized


class ReactionTimelineRowInput(BaseModel):
    id: str | None = None
    start: str = Field(..., min_length=1)
    end: str = Field(..., min_length=1)
    caption: str = ""
    layout_preset: str = "main_top_reaction_bottom"
    keep_aspect_ratio: bool = True
    caption_enabled: bool = True
    caption_duration: str = "entire"
    caption_duration_seconds: float | None = None
    main_crop: ReactionCropInput = Field(
        default_factory=lambda: ReactionCropInput(x_percent=0, y_percent=0, width_percent=100, height_percent=65)
    )
    facecam_crop: ReactionCropInput = Field(
        default_factory=lambda: ReactionCropInput(x_percent=60, y_percent=55, width_percent=35, height_percent=40)
    )
    facecam_shape: str = "rounded_rectangle"
    divider_split: ReactionDividerInput = Field(default_factory=ReactionDividerInput)
    main_region: ReactionCropInput | None = None
    facecam_region: ReactionCropInput | None = None
    caption_position: str = "between"
    caption_style: dict[str, Any] = Field(default_factory=dict)
    reference_frame_url: str | None = None
    reference_timestamp: str | None = None

    @field_validator("layout_preset")
    @classmethod
    def _validate_layout_preset(cls, value: str) -> str:
        normalized = str(value or "main_top_reaction_bottom").strip().lower()
        aliases = {
            "content_top_facecam_bottom": "main_top_reaction_bottom",
            "facecam_top_content_bottom": "reaction_top_main_bottom",
            "content_full_facecam_overlay": "facecam_right",
        }
        normalized = aliases.get(normalized, normalized)
        allowed = {
            "facecam_top",
            "facecam_bottom",
            "facecam_left",
            "facecam_right",
            "main_top_reaction_bottom",
            "reaction_top_main_bottom",
            "side_by_side",
            "custom",
            "top_right_facecam",
            "top_left_facecam",
            "bottom_right_facecam",
            "bottom_left_facecam",
            "reactor_fullscreen",
            "facecam_fullscreen",
            "no_facecam",
        }
        if normalized not in allowed:
            raise ValueError(f"layout_preset must be one of {sorted(allowed)}")
        return normalized

    @field_validator("facecam_shape")
    @classmethod
    def _validate_row_facecam_shape(cls, value: str) -> str:
        normalized = str(value or "rectangle").strip().lower()
        if normalized == "rounded":
            normalized = "rounded_rectangle"
        allowed = {"rectangle", "rounded_rectangle", "circle"}
        if normalized not in allowed:
            raise ValueError(f"facecam_shape must be one of {sorted(allowed)}")
        return normalized

    @field_validator("caption_duration")
    @classmethod
    def _validate_row_caption_duration(cls, value: str) -> str:
        normalized = str(value or "entire").strip().lower()
        aliases = {"3s": "first_3_seconds", "5s": "first_5_seconds", "10s": "first_10_seconds"}
        normalized = aliases.get(normalized, normalized)
        allowed = {"entire", "first_3_seconds", "first_5_seconds", "first_10_seconds", "custom"}
        if normalized not in allowed:
            raise ValueError(f"caption_duration must be one of {sorted(allowed)}")
        return normalized

    @field_validator("caption_position")
    @classmethod
    def _validate_row_caption_position(cls, value: str) -> str:
        normalized = str(value or "between").strip().lower()
        allowed = {"top", "center", "between", "bottom", "custom"}
        if normalized not in allowed:
            raise ValueError(f"caption_position must be one of {sorted(allowed)}")
        return normalized

    @model_validator(mode="after")
    def _validate_times_and_caption_duration(self) -> "ReactionTimelineRowInput":
        try:
            start = parse_time_to_seconds(self.start)
            end = parse_time_to_seconds(self.end)
        except ValueError as exc:
            raise ValueError("timeline row start/end must be seconds, MM:SS, or HH:MM:SS") from exc
        if end <= start:
            raise ValueError("timeline row end must be after start")
        if self.caption_duration == "custom" and (
            self.caption_duration_seconds is None or self.caption_duration_seconds <= 0
        ):
            raise ValueError("caption_duration_seconds must be > 0 when caption_duration is custom")
        self.caption = str(self.caption or "").strip()
        return self


class ProcessRequest(BaseModel):
    url: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    channel: str = Field(..., min_length=1)
    split_mode: str = "duration"
    scene_threshold: float = 27.0
    crop_top_px: int = 0
    title_mask_px: int = 0
    edge_bar_px: int = 45
    content_height_bump_px: int = 0
    content_max_height_px: int = 0
    video_style_scale: int = Field(default=50, ge=0, le=100)
    video_y_scale: float = 2.08
    y_scale_mode: str = "letterbox"
    interval_min: int = 30
    part_seconds: int = 70
    output_width: int = 1080
    output_height: int = 1920
    render_preset: str = "legacy"
    part_label_position: str = "top-center"
    label_x_pct: float = 0.5
    label_y_pct: float = 0.05
    show_part_label: bool = True
    part_label_x_percent: float = Field(default=50.0, ge=0.0, le=100.0)
    part_label_y_percent: float = Field(default=4.0, ge=0.0, le=100.0)
    no_part_overlay: bool = False
    hashtags: str = ""
    chapter_title_position: str = "top"
    manual_caption_text: str | None = None
    overlay_x_percent: float = Field(default=50.0, ge=0.0, le=100.0)
    overlay_y_percent: float = Field(default=12.0, ge=0.0, le=100.0)
    manual_chapters: list[ManualChapterInput] = Field(default_factory=list)
    playback_speed: float = 1.0
    subtitles_enabled: bool = False
    subtitle_style: str = "hormozi"
    subtitle_language: str | None = None
    subtitle_offset_seconds: float = Field(default=0.0, ge=-2.0, le=2.0)
    show_youtube_credit: bool = False
    youtube_credit_text: str | None = None
    youtube_credit_position: str = "below_frame"
    reaction_layout_enabled: bool = False
    reaction_layout_mode: str = "stacked"
    reaction_layout_preset: str = "content_top_facecam_bottom"
    main_crop: ReactionCropInput | None = None
    facecam_crop: ReactionCropInput | None = None
    facecam_shape: str = "rectangle"
    caption_text: str | None = None
    caption_position: str = "between"
    caption_duration_mode: str = "entire"
    caption_duration_seconds: float | None = None
    reference_frame_url: str | None = None
    source_width: int | None = Field(default=None, gt=0)
    source_height: int | None = Field(default=None, gt=0)
    reaction_layout_keyframes: list[dict[str, Any]] = Field(default_factory=list)
    reaction_timeline: list[ReactionTimelineRowInput] = Field(default_factory=list)
    imported_clip_plan: dict[str, Any] | None = None

    @field_validator("subtitle_style")
    @classmethod
    def _validate_subtitle_style(cls, value: str) -> str:
        allowed = {"hormozi", "standard", "minimal"}
        if value not in allowed:
            raise ValueError(f"subtitle_style must be one of {sorted(allowed)}")
        return value

    @field_validator("manual_caption_text")
    @classmethod
    def _normalize_manual_caption_text(cls, value: str | None) -> str | None:
        normalized = str(value or "").strip()
        return normalized or None

    @field_validator("caption_text")
    @classmethod
    def _normalize_caption_text(cls, value: str | None) -> str | None:
        normalized = str(value or "").strip()
        return normalized or None

    @field_validator("hashtags")
    @classmethod
    def _normalize_hashtags(cls, value: str) -> str:
        return re.sub(r"\s+", " ", str(value or "")).strip()

    @field_validator("youtube_credit_text")
    @classmethod
    def _normalize_youtube_credit_text(cls, value: str | None) -> str | None:
        normalized = str(value or "").strip()
        return normalized or None

    @field_validator("youtube_credit_position")
    @classmethod
    def _validate_youtube_credit_position(cls, value: str) -> str:
        normalized = str(value or "below_frame").strip().lower()
        allowed = {"below_frame", "bottom_left", "bottom_center", "bottom_right"}
        if normalized not in allowed:
            raise ValueError(f"youtube_credit_position must be one of {sorted(allowed)}")
        return normalized

    @field_validator("reaction_layout_mode")
    @classmethod
    def _validate_reaction_layout_mode(cls, value: str) -> str:
        allowed = {"stacked", "timeline"}
        if value not in allowed:
            raise ValueError(f"reaction_layout_mode must be one of {sorted(allowed)}")
        return value

    @field_validator("reaction_layout_preset")
    @classmethod
    def _validate_reaction_layout_preset(cls, value: str) -> str:
        allowed = {
            "content_top_facecam_bottom",
            "facecam_top_content_bottom",
            "content_full_facecam_overlay",
            "facecam_top",
            "facecam_bottom",
            "facecam_left",
            "facecam_right",
            "main_top_reaction_bottom",
            "reaction_top_main_bottom",
            "side_by_side",
            "custom",
        }
        if value not in allowed:
            raise ValueError(f"reaction_layout_preset must be one of {sorted(allowed)}")
        return value

    @field_validator("facecam_shape")
    @classmethod
    def _validate_facecam_shape(cls, value: str) -> str:
        if value == "rounded":
            value = "rounded_rectangle"
        allowed = {"rectangle", "rounded_rectangle", "circle"}
        if value not in allowed:
            raise ValueError(f"facecam_shape must be one of {sorted(allowed)}")
        return value

    @field_validator("caption_position")
    @classmethod
    def _validate_caption_position(cls, value: str) -> str:
        allowed = {"top", "center", "between", "bottom", "custom"}
        if value not in allowed:
            raise ValueError(f"caption_position must be one of {sorted(allowed)}")
        return value

    @field_validator("caption_duration_mode")
    @classmethod
    def _validate_caption_duration_mode(cls, value: str) -> str:
        aliases = {"3s": "first_3_seconds", "5s": "first_5_seconds", "10s": "first_10_seconds"}
        value = aliases.get(value, value)
        allowed = {"entire", "first_3_seconds", "first_5_seconds", "first_10_seconds", "custom"}
        if value not in allowed:
            raise ValueError(f"caption_duration_mode must be one of {sorted(allowed)}")
        return value

    @field_validator("playback_speed")
    @classmethod
    def _validate_playback_speed(cls, value: float) -> float:
        allowed = {1.0, 1.2, 1.5, 1.75, 2.0}
        if value not in allowed:
            raise ValueError(f"playback_speed must be one of {sorted(allowed)}")
        return value

    @field_validator("split_mode")
    @classmethod
    def _validate_split_mode(cls, value: str) -> str:
        if value not in _VALID_SPLIT_MODES:
            raise ValueError(f"split_mode must be one of {sorted(_VALID_SPLIT_MODES)}")
        return value

    @model_validator(mode="after")
    def _validate_manual_chapters(self) -> "ProcessRequest":
        imported_clips = []
        if isinstance(self.imported_clip_plan, dict):
            raw_clips = self.imported_clip_plan.get("clips")
            if raw_clips is not None and not isinstance(raw_clips, list):
                raise ValueError("imported_clip_plan.clips must be a list")
            imported_clips = [
                clip for clip in (raw_clips or [])
                if isinstance(clip, dict) and clip.get("enabled", True) is not False
            ]
            for idx, clip in enumerate(imported_clips, start=1):
                if not clip.get("start") or not clip.get("end"):
                    raise ValueError(f"imported_clip_plan clip #{idx} requires start and end")
                try:
                    start = parse_time_to_seconds(str(clip.get("start")))
                    end = parse_time_to_seconds(str(clip.get("end")))
                except ValueError as exc:
                    raise ValueError(f"imported_clip_plan clip #{idx} has invalid start/end") from exc
                if end <= start:
                    raise ValueError(f"imported_clip_plan clip #{idx} end must be after start")
        if self.split_mode == "manual" and not self.manual_chapters and not imported_clips:
            raise ValueError("manual_chapters is required when split_mode is 'manual'")
        if self.reaction_layout_enabled:
            if self.y_scale_mode != "reaction_layout":
                raise ValueError("y_scale_mode must be 'reaction_layout' when reaction_layout_enabled is true")
            if self.main_crop is None:
                self.main_crop = ReactionCropInput(x_percent=0, y_percent=0, width_percent=100, height_percent=65)
            if self.facecam_crop is None:
                self.facecam_crop = ReactionCropInput(x_percent=60, y_percent=55, width_percent=35, height_percent=40)
        if self.caption_duration_mode == "custom" and (
            self.caption_duration_seconds is None or self.caption_duration_seconds <= 0
        ):
            raise ValueError("caption_duration_seconds must be > 0 when caption_duration_mode is 'custom'")
        return self


class UploadRequest(BaseModel):
    job_id: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    channel: str = Field(..., min_length=1)
    interval_min: int = 30
    start_time: str | None = None


class LiveStartRequest(BaseModel):
    url: str = Field(..., min_length=1)


class LiveStopRequest(BaseModel):
    live_job_id: str = Field(..., min_length=1)


class ReferenceFrameRequest(BaseModel):
    source: str = Field(..., min_length=1)
    timestamp: str = "00:00:10"

    @field_validator("source")
    @classmethod
    def _normalize_source(cls, value: str) -> str:
        normalized = str(value or "").strip()
        if not normalized:
            raise ValueError("source is required")
        return normalized

    @field_validator("timestamp")
    @classmethod
    def _validate_timestamp(cls, value: str) -> str:
        normalized = str(value or "").strip() or "00:00:10"
        try:
            seconds = parse_time_to_seconds(normalized)
        except ValueError as exc:
            raise ValueError("timestamp must be seconds, MM:SS, or HH:MM:SS") from exc
        if seconds < 0:
            raise ValueError("timestamp must be zero or greater")
        return normalized


class LiveClipLastRequest(BaseModel):
    live_job_id: str = Field(..., min_length=1)
    seconds: int
    title: str = ""
    style_config: dict[str, Any] = Field(default_factory=dict)

    @field_validator("seconds")
    @classmethod
    def _validate_seconds(cls, value: int) -> int:
        allowed = {30, 60, 120}
        if value not in allowed:
            raise ValueError(f"seconds must be one of {sorted(allowed)}")
        return value


class LiveRenderRangeRequest(BaseModel):
    live_job_id: str = Field(..., min_length=1)
    start: str = Field(..., min_length=1)
    end: str = Field(..., min_length=1)
    title: str = ""
    style_config: dict[str, Any] = Field(default_factory=dict)


class AIChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    preset_id: str = "story_clips"
    source_url: str = ""
    video_title: str = ""
    extra_context: str = ""
    current_manual_chapters: list[dict[str, Any]] = Field(default_factory=list)


class AIGenerateTimelineRequest(BaseModel):
    preset_id: str = "day_by_day_recap"
    source_url: str = ""
    video_title: str = ""
    user_instructions: str = ""


def _extract_part_number(value: str) -> int:
    match = re.search(r"part[_\s-]*(\d+)", value, flags=re.IGNORECASE)
    if not match:
        return 0
    try:
        return int(match.group(1))
    except ValueError:
        return 0


def _sort_part_files(part_files: list[str]) -> list[str]:
    return sorted(
        part_files,
        key=lambda raw: (
            _extract_part_number(Path(str(raw)).name) or 9999,
            Path(str(raw)).name.lower(),
        ),
    )


def _merge_part_files(*groups: list[str]) -> list[str]:
    seen: set[str] = set()
    merged: list[str] = []
    for group in groups:
        for raw in group:
            text = str(raw).strip()
            if not text:
                continue
            name = Path(text).name.lower()
            if not name or name in seen:
                continue
            seen.add(name)
            merged.append(text)
    return _sort_part_files(merged)


def _read_render_presets() -> list[dict[str, Any]]:
    if not RENDER_PRESETS_PATH.exists():
        return []
    try:
        payload = json.loads(RENDER_PRESETS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail="Render presets file is not valid JSON.") from exc
    if not isinstance(payload, list):
        raise HTTPException(status_code=500, detail="Render presets file must contain a list.")
    return [item for item in payload if isinstance(item, dict)]


def _write_render_presets(presets: list[dict[str, Any]]) -> None:
    RENDER_PRESETS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RENDER_PRESETS_PATH.write_text(json.dumps(presets, indent=2), encoding="utf-8")


def _read_layout_presets() -> list[dict[str, Any]]:
    if not LAYOUT_PRESETS_PATH.exists():
        return []
    try:
        payload = json.loads(LAYOUT_PRESETS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail="Layout presets file is not valid JSON.") from exc
    if not isinstance(payload, list):
        raise HTTPException(status_code=500, detail="Layout presets file must contain a list.")
    return [item for item in payload if isinstance(item, dict)]


def _write_layout_presets(presets: list[dict[str, Any]]) -> None:
    LAYOUT_PRESETS_PATH.parent.mkdir(parents=True, exist_ok=True)
    LAYOUT_PRESETS_PATH.write_text(json.dumps(presets, indent=2), encoding="utf-8")


def _read_ai_presets() -> list[dict[str, Any]]:
    if not AI_PRESETS_PATH.exists():
        return []
    try:
        payload = json.loads(AI_PRESETS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail="AI presets file is not valid JSON.") from exc
    if not isinstance(payload, list):
        raise HTTPException(status_code=500, detail="AI presets file must contain a list.")
    return [item for item in payload if isinstance(item, dict)]


def _ai_preset_by_id(preset_id: str) -> dict[str, Any]:
    presets = _read_ai_presets()
    selected = str(preset_id or "").strip()
    for preset in presets:
        if str(preset.get("id") or "") == selected:
            return preset
    for preset in presets:
        if str(preset.get("id") or "") == "story_clips":
            return preset
    if presets:
        return presets[0]
    raise HTTPException(status_code=500, detail="No AI presets are configured.")


def _strip_json_fence(raw_text: str) -> str:
    text = str(raw_text or "").strip()
    match = re.search(r"```(?:json|javascript|js)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
    return (match.group(1) if match else text).strip()


def _extract_json_fragments(text: str) -> list[str]:
    values: list[str] = []
    start = -1
    stack: list[str] = []
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            if stack:
                in_string = True
            continue
        if char in "{[":
            if not stack:
                start = index
            stack.append(char)
            continue
        if char in "}]":
            if not stack:
                continue
            opener = stack[-1]
            if (opener, char) not in {("{", "}"), ("[", "]")}:
                raise ValueError("JSON brackets are not balanced.")
            stack.pop()
            if not stack and start >= 0:
                values.append(text[start:index + 1])
                start = -1
    if stack:
        raise ValueError("JSON brackets are not balanced.")
    return values


def _is_ai_metadata_object(value: Any) -> bool:
    if not isinstance(value, dict) or isinstance(value.get("clips"), list):
        return False
    allowed = {"import_ready_for_quickclips", "video_title", "summary", "characters", "source", "model", "notes"}
    keys = set(value.keys())
    return bool(keys) and keys.issubset(allowed)


def repair_ai_json(raw_text: str) -> Any:
    cleaned = _strip_json_fence(raw_text)
    if not cleaned:
        raise ValueError("Gemini returned an empty response.")
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        fragments = _extract_json_fragments(cleaned)
        if not fragments:
            raise ValueError("Gemini response did not contain valid JSON.")
        parsed = [json.loads(fragment) for fragment in fragments]
        if len(parsed) == 1:
            return parsed[0]
        main_index = next(
            (
                index for index, value in enumerate(parsed)
                if isinstance(value, list) or (isinstance(value, dict) and isinstance(value.get("clips"), list))
            ),
            -1,
        )
        if main_index < 0:
            raise ValueError("Gemini returned more than one top-level object and none contained clips.")
        main = parsed[main_index]
        metadata = [value for index, value in enumerate(parsed) if index != main_index and _is_ai_metadata_object(value)]
        if len(metadata) != len(parsed) - 1:
            raise ValueError("Gemini returned multiple top-level JSON objects that could not be merged safely.")
        if isinstance(main, list):
            return {"clips": main, **{key: val for item in metadata for key, val in item.items()}}
        return {**main, **{key: val for item in metadata for key, val in item.items()}}


def _normalize_ai_clip_plan(payload: Any) -> dict[str, Any]:
    plan = {"clips": payload} if isinstance(payload, list) else payload
    if not isinstance(plan, dict):
        raise ValueError("AI output must be a JSON object with clips, or a clips array.")
    raw_clips = plan.get("clips")
    if not isinstance(raw_clips, list) or not raw_clips:
        raise ValueError("AI output must include a non-empty clips array.")
    clips: list[dict[str, Any]] = []
    for idx, raw_clip in enumerate(raw_clips, start=1):
        if not isinstance(raw_clip, dict):
            raise ValueError(f"Clip #{idx} must be an object.")
        start = str(raw_clip.get("start") or "").strip()
        end = str(raw_clip.get("end") or "").strip()
        if not start or not end:
            raise ValueError(f"Clip #{idx} requires start and end.")
        start_seconds = parse_time_to_seconds(start)
        end_seconds = parse_time_to_seconds(end)
        if end_seconds <= start_seconds:
            raise ValueError(f"Clip #{idx} end must be after start.")
        title = str(raw_clip.get("title") or raw_clip.get("caption_text") or raw_clip.get("hook") or f"Clip {idx}").strip()
        caption = str(raw_clip.get("caption_text") or title).strip()
        clips.append(
            {
                "id": raw_clip.get("id", idx),
                "start": start,
                "end": end,
                "title": title,
                "caption_text": caption,
                "summary": str(raw_clip.get("summary") or "").strip(),
                "clip_type": str(raw_clip.get("clip_type") or "story_progression").strip() or "story_progression",
                "mood": str(raw_clip.get("mood") or "").strip(),
                "keywords": raw_clip.get("keywords") if isinstance(raw_clip.get("keywords"), list) else [],
                "hook": str(raw_clip.get("hook") or "").strip(),
            }
        )
    return {
        "video_title": str(plan.get("video_title") or "").strip(),
        "summary": str(plan.get("summary") or "").strip(),
        "characters": plan.get("characters") if isinstance(plan.get("characters"), list) else [],
        "clips": clips,
        "import_ready_for_quickclips": True,
    }


def _build_ai_prompt(*, preset: dict[str, Any], user_message: str, source_url: str, video_title: str, extra_context: str = "", current_manual_chapters: list[dict[str, Any]] | None = None, strict_json: bool = False) -> str:
    context = {
        "source_url": source_url,
        "video_title": video_title,
        "extra_context": extra_context,
        "current_manual_chapters": current_manual_chapters or [],
    }
    instruction = (
        (
            "Return strict JSON only. Do not wrap in markdown. "
            "Use this schema: "
            + str(preset.get("expected_output_schema_hint") or "")
        )
        if strict_json
        else "If you include an importable timeline, include a JSON object with a clips array."
    )
    prompt = "\n\n".join(
        [
            str(preset.get("system_prompt") or ""),
            instruction,
            "QuickClips context:",
            json.dumps(context, indent=2),
            "User request:",
            user_message,
            "Important: Gemini may need to estimate timestamps unless precise transcript/chapter context is provided. Prefer useful draft JSON over pretending certainty.",
        ]
    )
    max_prompt_chars = 24000
    if len(prompt) > max_prompt_chars:
        prompt = prompt[:max_prompt_chars] + "\n\n[Prompt truncated by QuickClips for cost control.]"
    return prompt


def _ai_source_warnings(source_url: str) -> list[str]:
    if str(source_url or "").strip():
        return [
            "Gemini only generates timeline JSON. QuickClips still downloads, cuts, captions, and renders locally.",
            "Gemini may estimate timestamps unless the prompt includes transcript, chapter, or scene context.",
        ]
    return [
        "No source URL/path was provided to Gemini. Add the source first for better timeline context.",
        "Gemini may estimate timestamps unless the prompt includes transcript, chapter, or scene context.",
    ]


def _call_gemini_or_http(prompt: str) -> tuple[str, str]:
    try:
        response = generate_text(prompt)
    except GeminiConfigError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except GeminiRequestError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return response.text, response.model


def _sanitize_layout_preset_payload(payload: dict[str, Any]) -> dict[str, Any]:
    name = re.sub(r"[^A-Za-z0-9_-]+", "_", str(payload.get("name") or "").strip()).strip("_").lower()
    if not name:
        raise HTTPException(status_code=400, detail="Layout preset name is required.")
    if len(name) > 80:
        raise HTTPException(status_code=400, detail="Layout preset name must be 80 characters or fewer.")

    label = str(payload.get("label") or name.replace("_", " ").title()).strip()[:120]
    layout_type = str(payload.get("layout_type") or "picture_in_picture").strip().lower()
    reaction_layout_preset = str(payload.get("reaction_layout_preset") or "content_full_facecam_overlay").strip()
    layout_preset = str(payload.get("layout_preset") or "custom").strip()

    def _crop(key: str, fallback: dict[str, float]) -> dict[str, float]:
        try:
            model = ReactionCropInput.model_validate(payload.get(key) or fallback)
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.errors()) from exc
        return model.model_dump()

    main_crop = _crop("main_crop", {"x_percent": 0, "y_percent": 0, "width_percent": 100, "height_percent": 100})
    facecam_crop = _crop("facecam_crop", {"x_percent": 60, "y_percent": 55, "width_percent": 35, "height_percent": 40})
    main_region = _crop("main_region", {"x_percent": 0, "y_percent": 0, "width_percent": 100, "height_percent": 100})
    facecam_region = _crop("facecam_region", {"x_percent": 61, "y_percent": 6, "width_percent": 34, "height_percent": 24})

    facecam_shape = str(payload.get("facecam_shape") or "rounded_rectangle").strip().lower()
    if facecam_shape == "rounded":
        facecam_shape = "rounded_rectangle"
    if facecam_shape not in {"rectangle", "rounded_rectangle", "circle"}:
        raise HTTPException(status_code=400, detail="facecam_shape must be rectangle, rounded_rectangle, or circle.")

    caption_position = str(payload.get("caption_position") or "between").strip().lower()
    if caption_position not in {"top", "center", "between", "bottom", "custom"}:
        caption_position = "between"

    return {
        "name": name,
        "label": label,
        "layout_type": layout_type,
        "reaction_layout_preset": reaction_layout_preset,
        "layout_preset": layout_preset,
        "main_crop": main_crop,
        "facecam_crop": facecam_crop,
        "main_region": main_region,
        "facecam_region": facecam_region,
        "facecam_shape": facecam_shape,
        "caption_position": caption_position,
        "keep_aspect_ratio": bool(payload.get("keep_aspect_ratio", True)),
    }


def _sanitize_preset_payload(payload: dict[str, Any]) -> dict[str, Any]:
    name = str(payload.get("name") or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="Preset name is required.")
    if len(name) > 80:
        raise HTTPException(status_code=400, detail="Preset name must be 80 characters or fewer.")

    raw_config = payload.get("config")
    source = raw_config if isinstance(raw_config, dict) else payload
    allowed_keys = set(RENDER_STYLE_KEYS) | {"render_mode"}
    sanitized = {
        key: value
        for key, value in source.items()
        if key in allowed_keys and key not in {"url", "live_job_id", "channel", "title"}
    }
    sanitized["name"] = name
    return sanitized


def _sanitize_style_config(style_config: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(style_config, dict):
        return {}
    allowed_keys = set(RENDER_STYLE_KEYS) | {"render_mode"}
    return {
        key: value
        for key, value in style_config.items()
        if key in allowed_keys
    }


def _live_output_dir(live_job_id: str, job_id: str) -> Path:
    return (OUTPUTS_ROOT / "livestreams" / live_job_id / job_id).resolve()


def _ensure_live_recording_ready(live_job_id: str) -> tuple[dict[str, Any], Path, float]:
    try:
        status = live_recorders.status(live_job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Livestream recording not found: {live_job_id}") from exc

    if status.get("status") != "recording":
        raise HTTPException(status_code=400, detail="Livestream recorder is not running.")

    try:
        buffer_path = live_recorders.buffer_path(live_job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail="Livestream buffer is not ready yet.") from exc

    duration = float(status.get("duration_seconds") or 0.0)
    if duration <= 0:
        duration = live_recorders.duration_seconds(live_job_id, allow_elapsed_fallback=True)
    if duration <= 0:
        raise HTTPException(status_code=400, detail="Livestream buffer duration is not available yet.")
    return status, buffer_path, duration


def _create_live_render_job(
    *,
    live_job_id: str,
    title: str,
    background_tasks: BackgroundTasks,
    input_video: Path,
    segment: Segment,
    style_config: dict[str, Any],
) -> dict[str, Any]:
    job_id = f"liveclip_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    out_dir = _live_output_dir(live_job_id, job_id)
    jobs.create(
        job_id,
        {
            "state": "queued",
            "channel": "livestreams",
            "title": title or "Livestream clip",
            "part_files": [],
            "output_dir": str(out_dir),
            "error": None,
            "live_job_id": live_job_id,
        },
    )
    jobs.append_log(job_id, "Queued livestream clip render.")
    background_tasks.add_task(
        _live_render_task,
        job_id,
        live_job_id,
        input_video,
        segment,
        title,
        style_config,
        out_dir,
    )
    return {
        "job_id": job_id,
        "live_job_id": live_job_id,
        "state": "queued",
        "part_files": [],
        "preview_parts": [],
    }


def _safe_upload_filename(filename: str) -> str:
    source_name = Path(filename or "upload").name.strip()
    suffix = Path(source_name).suffix.lower()
    if suffix not in UPLOAD_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix or 'none'}'. Allowed: {sorted(UPLOAD_EXTENSIONS)}",
        )

    stem = Path(source_name).stem.strip() or "upload"
    stem = re.sub(r"[^A-Za-z0-9._ -]+", "_", stem)
    stem = re.sub(r"\s+", " ", stem).strip(" ._-") or "upload"
    stem = stem[:80]
    unique = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    return f"{stem}_{unique}{suffix}"


async def _save_uploaded_video(file: UploadFile) -> dict[str, object]:
    filename = _safe_upload_filename(file.filename or "upload")
    UPLOADS_ROOT.mkdir(parents=True, exist_ok=True)
    dest = (UPLOADS_ROOT / filename).resolve()

    try:
        dest.relative_to(UPLOADS_ROOT.resolve())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid upload filename.") from exc

    written = 0
    chunk_size = 1024 * 1024  # 1 MB
    with dest.open("wb") as fh:
        while True:
            chunk = await file.read(chunk_size)
            if not chunk:
                break
            written += len(chunk)
            if written > MAX_UPLOAD_BYTES:
                dest.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail="File exceeds the 2 GB upload limit.")
            fh.write(chunk)

    if written <= 0:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    rel_path = f"downloads/uploads/{filename}"
    return {"path": rel_path, "filename": filename}


def _reference_timestamp_slug(timestamp: str) -> str:
    seconds = parse_time_to_seconds(timestamp)
    if seconds < 0:
        raise ValueError("timestamp must be zero or greater")
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", timestamp.strip()).strip("._-")
    if slug:
        return slug[:40]
    return f"{seconds:.3f}".replace(".", "_")


def _reference_frame_output_path(source_video: Path, timestamp: str) -> Path:
    stat = source_video.stat()
    digest_source = f"{source_video.resolve()}|{stat.st_mtime_ns}|{stat.st_size}|{timestamp}"
    digest = hashlib.sha1(digest_source.encode("utf-8")).hexdigest()[:16]
    return (REFERENCE_FRAMES_ROOT / f"{digest}_{_reference_timestamp_slug(timestamp)}.jpg").resolve()


def _run_reference_frame_extract(source_video: Path, timestamp: str, output_path: Path) -> None:
    ffmpeg_bin = _resolve_binary("ffmpeg")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg_bin,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        timestamp,
        "-i",
        str(source_video),
        "-frames:v",
        "1",
        "-q:v",
        "2",
        str(output_path),
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "ffmpeg failed to extract a reference frame.")
    if not output_path.exists() or output_path.stat().st_size <= 0:
        raise RuntimeError("ffmpeg did not produce a reference frame image.")


def _queue_upload_job(request: UploadRequest, background_tasks: BackgroundTasks) -> dict[str, object]:
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


def _discover_part_files(output_dir: str | Path | None) -> list[str]:
    if not output_dir:
        return []
    candidate = Path(str(output_dir)).resolve()
    if not candidate.exists() or not candidate.is_dir():
        return []
    return [str(item.resolve()) for item in discover_part_files(candidate)]


def _resolve_preview_title(
    status_payload: dict[str, object] | None,
    *,
    part_number: int,
    fallback_title: str = "",
) -> str:
    if isinstance(status_payload, dict):
        render_config = status_payload.get("render_config")
        if isinstance(render_config, dict):
            raw_titles = render_config.get("chapter_titles")
            if isinstance(raw_titles, list) and 0 < part_number <= len(raw_titles):
                title = str(raw_titles[part_number - 1] or "").strip()
                if title:
                    return title
    return fallback_title.strip()


def _build_preview_parts(
    job_id: str,
    part_files: list[str] | None,
    *,
    status_payload: dict[str, object] | None = None,
    fallback_title: str = "",
) -> list[dict[str, object]]:
    seen: set[str] = set()
    preview_parts: list[dict[str, object]] = []
    for raw in _sort_part_files([str(item) for item in (part_files or [])]):
        name = Path(str(raw)).name
        if not name or name in seen:
            continue
        seen.add(name)
        part_number = _extract_part_number(name) or (len(preview_parts) + 1)
        preview_parts.append(
            {
                "name": name,
                "path": str(raw),
                "url": f"/api/media/{job_id}/{name}",
                "part_number": part_number,
                "title": _resolve_preview_title(
                    status_payload,
                    part_number=part_number,
                    fallback_title=fallback_title,
                ),
                "ready": True,
            }
        )
    return preview_parts


def _resolve_output_dir(job_id: str) -> Path:
    record = jobs.get(job_id)
    if record and record.get("output_dir"):
        candidate = Path(str(record["output_dir"])).resolve()
        if candidate.exists():
            return candidate

    try:
        persisted_status = load_job_status(job_id)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}") from exc

    out_dir = persisted_status.get("output_dir")
    if not out_dir:
        raise HTTPException(status_code=404, detail=f"Output directory missing for job {job_id}")

    candidate = Path(str(out_dir)).resolve()
    if not candidate.exists():
        raise HTTPException(status_code=404, detail=f"Output directory does not exist: {candidate}")
    return candidate


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
            split_mode=request.split_mode,
            scene_threshold=request.scene_threshold,
            crop_top_px=request.crop_top_px,
            title_mask_px=request.title_mask_px,
            edge_bar_px=request.edge_bar_px,
            content_height_bump_px=request.content_height_bump_px,
            content_max_height_px=request.content_max_height_px,
            video_style_scale=request.video_style_scale,
            video_y_scale=request.video_y_scale,
            y_scale_mode=request.y_scale_mode,
            output_width=request.output_width,
            output_height=request.output_height,
            render_preset=request.render_preset,
            part_label_position=request.part_label_position,
            label_x_pct=request.label_x_pct,
            label_y_pct=request.label_y_pct,
            show_part_label=request.show_part_label,
            part_label_x_percent=request.part_label_x_percent,
            part_label_y_percent=request.part_label_y_percent,
            no_part_overlay=request.no_part_overlay,
            hashtags=request.hashtags,
            chapter_title_position=request.chapter_title_position,
            manual_caption_text=request.manual_caption_text,
            overlay_x_percent=request.overlay_x_percent,
            overlay_y_percent=request.overlay_y_percent,
            manual_chapters=[item.model_dump() for item in request.manual_chapters] or None,
            playback_speed=request.playback_speed,
            subtitles_enabled=request.subtitles_enabled,
            subtitle_style=request.subtitle_style,
            subtitle_language=request.subtitle_language,
            subtitle_offset_seconds=request.subtitle_offset_seconds,
            show_youtube_credit=request.show_youtube_credit,
            youtube_credit_text=request.youtube_credit_text,
            youtube_credit_position=request.youtube_credit_position,
            reaction_layout_enabled=request.reaction_layout_enabled,
            reaction_layout_mode=request.reaction_layout_mode,
            reaction_layout_preset=request.reaction_layout_preset,
            main_crop=request.main_crop.model_dump() if request.main_crop else None,
            facecam_crop=request.facecam_crop.model_dump() if request.facecam_crop else None,
            facecam_shape=request.facecam_shape,
            caption_text=request.caption_text,
            caption_position=request.caption_position,
            caption_duration_mode=request.caption_duration_mode,
            caption_duration_seconds=request.caption_duration_seconds,
            reference_frame_url=request.reference_frame_url,
            source_width=request.source_width,
            source_height=request.source_height,
            reaction_layout_keyframes=request.reaction_layout_keyframes,
            reaction_timeline=[item.model_dump() for item in request.reaction_timeline],
            imported_clip_plan=request.imported_clip_plan,
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
        exc_text = str(exc)
        if "WinError 2" in exc_text or "not found" in exc_text.lower():
            user_error = (
                "FFmpeg not found. Please install FFmpeg and add it to your system PATH, "
                "or place ffmpeg.exe in the project bin/ folder."
            )
        else:
            user_error = exc_text
        jobs.append_log(job_id, f"Process failed: {exc_text}")
        jobs.update(job_id, state="failed", error=user_error)


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


def _live_render_task(
    job_id: str,
    live_job_id: str,
    input_video: Path,
    segment: Segment,
    title: str,
    style_config: dict[str, Any],
    out_dir: Path,
) -> None:
    jobs.update(job_id, state="processing")

    def logger(message: str) -> None:
        jobs.append_log(job_id, message)

    try:
        payload = render_custom_segments(
            input_video=input_video,
            segments=[segment],
            titles=[title] if title else None,
            style_config=style_config,
            out_dir=out_dir,
            log_fn=logger,
        )
        payload.update(
            {
                "mode": "live_clip",
                "job_id": job_id,
                "live_job_id": live_job_id,
                "title": title,
                "state": "processed",
            }
        )
        jobs.update(
            job_id,
            state="processed",
            part_files=payload.get("part_files", []),
            output_dir=payload.get("output_dir"),
            persisted_status=payload,
        )
    except Exception as exc:  # noqa: BLE001
        jobs.append_log(job_id, f"Livestream clip render failed: {exc}")
        jobs.update(job_id, state="failed", error=str(exc))


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
            "output_dir": str(build_output_dir(request.channel, job_id)),
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
        _raw = persisted_status.get("part_files") if persisted_status else None
        persisted_parts: list[str] = [str(p) for p in _raw] if isinstance(_raw, list) else []
        output_dir = persisted_status.get("output_dir") if isinstance(persisted_status, dict) else None
        live_parts = _discover_part_files(output_dir)
        all_parts = _merge_part_files(persisted_parts, live_parts)
        return {
            "job_id": job_id,
            "state": str(persisted_status.get("state", "unknown")) if persisted_status else "unknown",
            "logs": [],
            "part_files": all_parts,
            "preview_parts": _build_preview_parts(
                job_id,
                all_parts,
                status_payload=persisted_status if isinstance(persisted_status, dict) else None,
                fallback_title=str(persisted_status.get("title", "")) if isinstance(persisted_status, dict) else "",
            ),
            "status": persisted_status,
        }

    _raw_files = record.get("part_files")
    part_files: list[str] = [str(p) for p in _raw_files] if isinstance(_raw_files, list) else []
    if not part_files and persisted_status:
        _raw_p = persisted_status.get("part_files")
        part_files = [str(p) for p in _raw_p] if isinstance(_raw_p, list) else []
    output_dir = record.get("output_dir") or (persisted_status.get("output_dir") if isinstance(persisted_status, dict) else None)
    live_parts = _discover_part_files(output_dir)
    part_files = _merge_part_files(part_files, live_parts)

    return {
        "job_id": job_id,
        "state": record.get("state", "unknown"),
        "logs": record.get("logs", []),
        "error": record.get("error"),
        "part_files": part_files,
        "preview_parts": _build_preview_parts(
            job_id,
            part_files,
            status_payload=persisted_status if isinstance(persisted_status, dict) else None,
            fallback_title=str(record.get("title", "")),
        ),
        "output_dir": output_dir,
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


@app.get("/api/presets")
def api_get_presets() -> dict[str, object]:
    return {"presets": _read_render_presets()}


@app.post("/api/presets")
async def api_save_preset(request: Request) -> dict[str, object]:
    try:
        raw_payload = await request.json()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="Expected JSON preset payload.") from exc
    if not isinstance(raw_payload, dict):
        raise HTTPException(status_code=400, detail="Preset payload must be an object.")

    preset = _sanitize_preset_payload(raw_payload)
    presets = _read_render_presets()
    preset_name = str(preset["name"]).casefold()
    if any(str(item.get("name", "")).casefold() == preset_name for item in presets):
        raise HTTPException(status_code=409, detail="A preset with that name already exists.")

    presets.append(preset)
    _write_render_presets(presets)
    return {"preset": preset, "presets": presets}


@app.delete("/api/presets/{preset_name}")
def api_delete_preset(preset_name: str) -> dict[str, object]:
    clean_name = str(preset_name or "").strip()
    if not clean_name:
        raise HTTPException(status_code=400, detail="Preset name is required.")

    presets = _read_render_presets()
    remaining = [item for item in presets if str(item.get("name", "")).casefold() != clean_name.casefold()]
    if len(remaining) == len(presets):
        raise HTTPException(status_code=404, detail=f"Preset not found: {clean_name}")

    _write_render_presets(remaining)
    return {"deleted": clean_name, "presets": remaining}


@app.get("/api/layout-presets")
def api_get_layout_presets() -> dict[str, object]:
    return {"presets": _read_layout_presets()}


@app.post("/api/layout-presets")
async def api_save_layout_preset(request: Request) -> dict[str, object]:
    try:
        raw_payload = await request.json()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="Expected JSON layout preset payload.") from exc
    if not isinstance(raw_payload, dict):
        raise HTTPException(status_code=400, detail="Layout preset payload must be an object.")

    preset = _sanitize_layout_preset_payload(raw_payload)
    presets = _read_layout_presets()
    preset_name = str(preset["name"]).casefold()
    replaced = False
    for index, item in enumerate(presets):
        if str(item.get("name", "")).casefold() == preset_name:
            presets[index] = preset
            replaced = True
            break
    if not replaced:
        presets.append(preset)
    _write_layout_presets(presets)
    return {"preset": preset, "presets": presets}


@app.delete("/api/layout-presets/{preset_name}")
def api_delete_layout_preset(preset_name: str) -> dict[str, object]:
    clean_name = str(preset_name or "").strip()
    if not clean_name:
        raise HTTPException(status_code=400, detail="Layout preset name is required.")
    presets = _read_layout_presets()
    remaining = [item for item in presets if str(item.get("name", "")).casefold() != clean_name.casefold()]
    if len(remaining) == len(presets):
        raise HTTPException(status_code=404, detail=f"Layout preset not found: {clean_name}")
    _write_layout_presets(remaining)
    return {"deleted": clean_name, "presets": remaining}


@app.get("/api/ai/presets")
def api_get_ai_presets() -> dict[str, object]:
    presets = _read_ai_presets()
    return {
        "presets": [
            {
                "id": str(preset.get("id") or ""),
                "name": str(preset.get("name") or preset.get("id") or ""),
                "description": str(preset.get("description") or ""),
                "expected_output_schema_hint": preset.get("expected_output_schema_hint") or {},
            }
            for preset in presets
        ]
    }


@app.post("/api/ai/chat")
def api_ai_chat(request: AIChatRequest) -> dict[str, object]:
    preset = _ai_preset_by_id(request.preset_id)
    print(
        "AI chat request "
        f"preset={preset.get('id', request.preset_id)} "
        f"source={'provided' if request.source_url.strip() else 'missing'}"
    )
    prompt = _build_ai_prompt(
        preset=preset,
        user_message=request.message,
        source_url=request.source_url,
        video_title=request.video_title,
        extra_context=request.extra_context,
        current_manual_chapters=request.current_manual_chapters,
        strict_json=False,
    )
    reply, model = _call_gemini_or_http(prompt)

    parsed_json: Any | None = None
    json_detected = False
    try:
        parsed_json = repair_ai_json(reply)
        json_detected = True
    except ValueError:
        parsed_json = None

    return {
        "reply": reply,
        "model": model,
        "json_detected": json_detected,
        "json": parsed_json,
        "warnings": _ai_source_warnings(request.source_url),
    }


@app.post("/api/ai/generate-timeline")
def api_ai_generate_timeline(request: AIGenerateTimelineRequest) -> dict[str, object]:
    preset = _ai_preset_by_id(request.preset_id)
    print(
        "AI timeline request "
        f"preset={preset.get('id', request.preset_id)} "
        f"source={'provided' if request.source_url.strip() else 'missing'}"
    )
    user_message = request.user_instructions.strip() or (
        "Generate a QuickClips short-form clip timeline for this source. "
        "Cover the important progression, story beats, challenges, reveals, and emotional turns."
    )
    prompt = _build_ai_prompt(
        preset=preset,
        user_message=user_message,
        source_url=request.source_url,
        video_title=request.video_title,
        strict_json=True,
    )
    reply, model = _call_gemini_or_http(prompt)

    try:
        parsed = repair_ai_json(reply)
        normalized = _normalize_ai_clip_plan(parsed)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"Gemini did not return importable QuickClips JSON: {exc}") from exc

    return {
        "reply": reply,
        "model": model,
        "json_detected": True,
        "json": normalized,
        "warnings": _ai_source_warnings(request.source_url),
    }


@app.post("/api/live/start")
def api_live_start(request: LiveStartRequest) -> dict[str, object]:
    try:
        return live_recorders.start(request.url)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/live/stop")
def api_live_stop(request: LiveStopRequest) -> dict[str, object]:
    try:
        return live_recorders.stop(request.live_job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Livestream recording not found: {request.live_job_id}") from exc


@app.get("/api/live/status/{live_job_id}")
def api_live_status(live_job_id: str) -> dict[str, object]:
    try:
        return live_recorders.status(live_job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Livestream recording not found: {live_job_id}") from exc


@app.post("/api/live/clip-last")
def api_live_clip_last(request: LiveClipLastRequest, background_tasks: BackgroundTasks) -> dict[str, object]:
    _status, buffer_path, duration = _ensure_live_recording_ready(request.live_job_id)
    start = max(0.0, duration - float(request.seconds))
    end = duration
    if end <= start:
        raise HTTPException(status_code=400, detail="Livestream buffer is too short for that clip.")

    return _create_live_render_job(
        live_job_id=request.live_job_id,
        title=request.title.strip() or f"Last {request.seconds}s",
        background_tasks=background_tasks,
        input_video=buffer_path,
        segment=Segment(start=start, end=end),
        style_config=_sanitize_style_config(request.style_config),
    )


@app.post("/api/live/render-range")
def api_live_render_range(request: LiveRenderRangeRequest, background_tasks: BackgroundTasks) -> dict[str, object]:
    _status, buffer_path, duration = _ensure_live_recording_ready(request.live_job_id)
    try:
        start = parse_time_to_seconds(request.start)
        end = parse_time_to_seconds(request.end)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if end <= start:
        raise HTTPException(status_code=400, detail="Manual live clip end time must be after start time.")
    if start < 0 or end > duration + 0.75:
        raise HTTPException(status_code=400, detail="Requested livestream range is outside the current buffer.")

    return _create_live_render_job(
        live_job_id=request.live_job_id,
        title=request.title.strip() or "Livestream clip",
        background_tasks=background_tasks,
        input_video=buffer_path,
        segment=Segment(start=start, end=min(end, duration)),
        style_config=_sanitize_style_config(request.style_config),
    )


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


@app.post("/api/upload-video")
async def api_upload_video(file: UploadFile = File(...)) -> dict[str, object]:
    return await _save_uploaded_video(file)


@app.post("/api/upload")
async def api_upload(request: Request, background_tasks: BackgroundTasks) -> dict[str, object]:
    content_type = request.headers.get("content-type", "").lower()
    if content_type.startswith("multipart/form-data"):
        form = await request.form()
        upload = form.get("file")
        if not isinstance(upload, StarletteUploadFile):
            raise HTTPException(status_code=400, detail="Upload must include a video file field named 'file'.")
        return await _save_uploaded_video(upload)

    try:
        payload = await request.json()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="Expected JSON upload job payload or multipart file upload.") from exc
    try:
        upload_request = UploadRequest.model_validate(payload)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc
    return _queue_upload_job(upload_request, background_tasks)


@app.post("/api/reference-frame")
def api_reference_frame(request: ReferenceFrameRequest) -> dict[str, object]:
    try:
        source_video, source_info = resolve_cached_input_video(
            input_value=request.source,
            downloads_root=DOWNLOADS_ROOT,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        ffprobe_bin = _resolve_binary("ffprobe")
        dimensions = _probe_video_dimensions(input_video=source_video, ffprobe_bin=ffprobe_bin)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    if dimensions is None:
        raise HTTPException(status_code=400, detail=f"Could not read source video dimensions: {source_video}")

    try:
        output_path = _reference_frame_output_path(source_video, request.timestamp)
        if not output_path.exists() or output_path.stat().st_size <= 0:
            _run_reference_frame_extract(source_video, request.timestamp, output_path)
    except (OSError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=500, detail=f"Reference frame extraction failed: {exc}") from exc

    width, height = dimensions
    return {
        "image_url": f"/api/reference-frame-file/{output_path.name}",
        "source_width": width,
        "source_height": height,
        "timestamp": request.timestamp,
        "source_metadata": {
            "title": str(source_info.get("title") or "").strip(),
            "uploader": str(source_info.get("uploader") or "").strip(),
            "channel": str(source_info.get("channel") or "").strip(),
        },
    }


@app.get("/api/reference-frame-file/{filename}")
def api_reference_frame_file(filename: str) -> FileResponse:
    safe_name = Path(filename).name
    if safe_name != filename:
        raise HTTPException(status_code=400, detail="Invalid file name.")
    if Path(safe_name).suffix.lower() not in {".jpg", ".jpeg"}:
        raise HTTPException(status_code=400, detail="Unsupported reference frame extension.")

    root = REFERENCE_FRAMES_ROOT.resolve()
    image_path = (root / safe_name).resolve()
    try:
        image_path.relative_to(root)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid reference frame path.") from exc
    if not image_path.exists():
        raise HTTPException(status_code=404, detail=f"Reference frame not found: {safe_name}")
    return FileResponse(path=image_path, media_type="image/jpeg", filename=safe_name)


@app.get("/api/media/{job_id}/{filename}")
def api_media(job_id: str, filename: str) -> FileResponse:
    safe_name = Path(filename).name
    if safe_name != filename:
        raise HTTPException(status_code=400, detail="Invalid file name.")
    if Path(safe_name).suffix.lower() not in VIDEO_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported media extension.")

    out_dir = _resolve_output_dir(job_id)
    media_path = (out_dir / safe_name).resolve()

    try:
        media_path.relative_to(out_dir)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid media path.") from exc

    if not media_path.exists():
        raise HTTPException(status_code=404, detail=f"Media file not found: {safe_name}")

    suffix = media_path.suffix.lower()
    media_type = "video/mp4"
    if suffix == ".webm":
        media_type = "video/webm"
    elif suffix == ".mov":
        media_type = "video/quicktime"
    elif suffix == ".mkv":
        media_type = "video/x-matroska"

    return FileResponse(path=media_path, media_type=media_type, filename=safe_name)
