from __future__ import annotations

import json
import math
import os
import re
import shlex
import shutil
import subprocess
import unicodedata
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

LOCKED_PRESET = True
KNOWN_GOOD_REFERENCE = Path("clips_2026_02_17_03_02_36") / "part_1.mp4"
FILL_TRIM_PX = 30
MANUAL_CHAPTER_END_TOLERANCE_SECONDS = 1.0


@dataclass(frozen=True)
class Segment:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class RenderedPart:
    part_number: int
    start: float
    end: float
    path: Path
    start_time: str = ""
    end_time: str = ""
    vf: str = ""
    ffmpeg_cmd: list[str] | None = None
    ffmpeg_cmd_path: Path | None = None
    ffmpeg_cmd_run_path: Path | None = None


def _resolve_binary(name: str) -> str:
    _bin_dir = Path(__file__).resolve().parents[1] / "bin"
    candidates = [name, f"{name}.exe", str(_bin_dir / name), str(_bin_dir / f"{name}.exe")]
    for candidate in candidates:
        if shutil.which(candidate):
            return candidate
    raise RuntimeError(
        f"{name} not found. Install it and add to PATH, or place {name}.exe in the project's bin/ folder."
    )


def _run_command(cmd: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(cwd) if cwd is not None else None,
    )
    if result.returncode != 0:
        command_text = " ".join(shlex.quote(item) for item in cmd)
        cwd_text = f"\ncwd:\n{cwd}" if cwd is not None else ""
        raise RuntimeError(
            f"Command failed ({result.returncode}): {command_text}{cwd_text}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return result


def _safe_probe_json(input_video: Path, ffprobe_bin: str) -> dict[str, Any]:
    cmd = [
        ffprobe_bin,
        "-v",
        "error",
        "-show_streams",
        "-show_format",
        "-of",
        "json",
        str(input_video),
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        return {
            "error": f"ffprobe failed ({result.returncode})",
            "stderr": result.stderr,
        }
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return {"raw": result.stdout}


def _safe_ffmpeg_version(ffmpeg_bin: str) -> str:
    result = subprocess.run(
        [ffmpeg_bin, "-version"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        return f"ffmpeg version probe failed ({result.returncode})"
    return result.stdout


def _command_to_shell_text(cmd: list[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(cmd)
    return shlex.join(cmd)


def _write_part_command_dump(path: Path, cmd: list[str], *, cwd: Path | None = None) -> None:
    shell_text = _command_to_shell_text(cmd)
    lines = [
        *(["Working directory:", str(cwd), ""] if cwd is not None else []),
        shell_text,
        "",
        "Args JSON:",
        json.dumps(cmd, indent=2),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_windows_cmd(path: Path, cmd: list[str], *, cwd: Path | None = None) -> None:
    shell_text = _command_to_shell_text(cmd)
    lines = ["@echo off"]
    if cwd is not None:
        lines.extend([f'pushd "{cwd}"', shell_text, "set CMD_EXIT=%ERRORLEVEL%", "popd", "exit /b %CMD_EXIT%", ""])
    else:
        lines.extend([shell_text, ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _safe_probe_short(input_video: Path, ffprobe_bin: str) -> dict[str, Any]:
    cmd = [
        ffprobe_bin,
        "-v",
        "error",
        "-show_entries",
        "stream=codec_name,width,height,sample_aspect_ratio,display_aspect_ratio,avg_frame_rate,bit_rate:format=bit_rate",
        "-of",
        "json",
        str(input_video),
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        return {"error": f"ffprobe failed ({result.returncode})", "stderr": result.stderr}
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return {"error": "ffprobe output parse failure", "raw": result.stdout}

    stream = {}
    for item in payload.get("streams", []):
        if item.get("codec_name"):
            stream = item
            break
    fmt = payload.get("format", {})
    return {
        "codec": stream.get("codec_name"),
        "resolution": f"{stream.get('width')}x{stream.get('height')}",
        "sar": stream.get("sample_aspect_ratio"),
        "dar": stream.get("display_aspect_ratio"),
        "fps": stream.get("avg_frame_rate"),
        "bitrate_stream": stream.get("bit_rate"),
        "bitrate_format": fmt.get("bit_rate"),
    }


def _safe_probe_tiktok_fields(input_video: Path, ffprobe_bin: str) -> dict[str, Any]:
    cmd = [
        ffprobe_bin,
        "-v",
        "error",
        "-show_entries",
        (
            "stream=codec_name,width,height,sample_aspect_ratio,display_aspect_ratio,"
            "pix_fmt,avg_frame_rate,color_range,color_space,color_transfer,color_primaries:"
            "format=format_name,bit_rate,duration"
        ),
        "-of",
        "json",
        str(input_video),
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        return {"error": f"ffprobe failed ({result.returncode})", "stderr": result.stderr}

    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return {"error": "ffprobe output parse failure", "raw": result.stdout}

    video_stream: dict[str, Any] = {}
    for stream in payload.get("streams", []):
        if stream.get("codec_name"):
            video_stream = stream
            break
    fmt = payload.get("format", {})
    return {
        "stream": {
            "width": video_stream.get("width"),
            "height": video_stream.get("height"),
            "sample_aspect_ratio": video_stream.get("sample_aspect_ratio"),
            "display_aspect_ratio": video_stream.get("display_aspect_ratio"),
            "pix_fmt": video_stream.get("pix_fmt"),
            "codec_name": video_stream.get("codec_name"),
            "avg_frame_rate": video_stream.get("avg_frame_rate"),
            "color_range": video_stream.get("color_range"),
            "color_space": video_stream.get("color_space"),
            "color_transfer": video_stream.get("color_transfer"),
            "color_primaries": video_stream.get("color_primaries"),
        },
        "format": {
            "format_name": fmt.get("format_name"),
            "bit_rate": fmt.get("bit_rate"),
            "duration": fmt.get("duration"),
        },
    }


def _expected_dar(width: int, height: int) -> str:
    if width <= 0 or height <= 0:
        return ""
    gcd = math.gcd(width, height)
    return f"{width // gcd}:{height // gcd}"


def _clamp_even(value: int, minimum: int, maximum: int) -> int:
    clamped = max(minimum, min(value, maximum))
    if clamped % 2 != 0:
        if clamped == maximum:
            clamped -= 1
        else:
            clamped += 1
    if clamped < minimum:
        clamped = minimum
    if clamped % 2 != 0:
        clamped += 1
    return max(2, clamped)


def _compute_zoom_target_height(
    *,
    source_width: int,
    source_height: int,
    crop_top_px: int,
    output_width: int,
    output_height: int,
    content_height_bump_px: int,
    content_max_height_px: int,
) -> dict[str, int | float]:
    safe_crop_top = max(0, min(crop_top_px, max(0, source_height - 2)))
    effective_source_height = max(2, source_height - safe_crop_top)
    raw_base_h = (effective_source_height * output_width) / source_width
    base_h = _clamp_even(int(round(raw_base_h)), minimum=2, maximum=output_height)
    max_height_cap = output_height
    if content_max_height_px > 0:
        max_height_cap = _clamp_even(content_max_height_px, minimum=2, maximum=output_height)
    target_h = _clamp_even(base_h + int(content_height_bump_px), minimum=2, maximum=max_height_cap)
    return {
        "source_width": source_width,
        "source_height": source_height,
        "crop_top_px": safe_crop_top,
        "effective_source_height": effective_source_height,
        "base_height_raw": raw_base_h,
        "base_height": base_h,
        "content_max_height_px": max_height_cap,
        "target_height": target_h,
        "content_height_bump_px": int(content_height_bump_px),
    }


def _build_vf_diagnostics(vf: str, output_width: int, output_height: int) -> dict[str, bool]:
    return {
        "setsar_1_in_filter_chain": "setsar=1" in vf,
        "vf_contains_scale_minus2_1920": "scale=-2:1920" in vf,
        "vf_contains_crop_1080_1920": "crop=1080:1920" in vf,
        "vf_contains_force_original_aspect_ratio": "force_original_aspect_ratio" in vf,
        "vf_contains_setsar": "setsar" in vf,
        "vf_contains_scale_minus2_output_height": f"scale=-2:{output_height}" in vf,
        "vf_contains_crop_output_dimensions": f"crop={output_width}:{output_height}" in vf,
    }


def _build_tiktok_risk_flags(
    *,
    probe: dict[str, Any],
    vf_diag: dict[str, bool],
    output_width: int,
    output_height: int,
) -> list[str]:
    flags: list[str] = []
    if "error" in probe:
        flags.append(f"ffprobe error: {probe.get('error')}")
        return flags

    stream = probe.get("stream", {})
    width = stream.get("width")
    height = stream.get("height")
    sar = stream.get("sample_aspect_ratio")
    dar = stream.get("display_aspect_ratio")
    pix_fmt = stream.get("pix_fmt")

    if width != output_width or height != output_height:
        flags.append(f"resolution is {width}x{height}, expected {output_width}x{output_height}")
    if sar and sar != "1:1":
        flags.append(f"SAR is {sar}, expected 1:1")

    expected_dar = _expected_dar(output_width, output_height)
    if dar and expected_dar and dar != expected_dar:
        flags.append(f"DAR is {dar}, expected {expected_dar}")
    if pix_fmt and pix_fmt != "yuv420p":
        flags.append(f"pix_fmt is {pix_fmt}, expected yuv420p")

    if not vf_diag.get("setsar_1_in_filter_chain", False):
        flags.append("filter chain missing setsar=1")
    if vf_diag.get("vf_contains_scale_minus2_1920", False) or vf_diag.get("vf_contains_scale_minus2_output_height", False):
        flags.append("filter chain includes height-fit scale (scale=-2:height)")
    if vf_diag.get("vf_contains_crop_1080_1920", False) or vf_diag.get("vf_contains_crop_output_dimensions", False):
        flags.append("filter chain includes full-frame crop (crop=width:height)")
    if vf_diag.get("vf_contains_force_original_aspect_ratio", False):
        flags.append("filter chain includes force_original_aspect_ratio")

    return flags


def _tiktok_probe_diff(good_probe: dict[str, Any], new_probe: dict[str, Any]) -> list[str]:
    keys = [
        ("stream", "width"),
        ("stream", "height"),
        ("stream", "sample_aspect_ratio"),
        ("stream", "display_aspect_ratio"),
        ("stream", "pix_fmt"),
        ("stream", "codec_name"),
        ("stream", "avg_frame_rate"),
        ("stream", "color_range"),
        ("stream", "color_space"),
        ("stream", "color_transfer"),
        ("stream", "color_primaries"),
        ("format", "format_name"),
        ("format", "bit_rate"),
        ("format", "duration"),
    ]
    rows: list[str] = []
    for scope, key in keys:
        good_value = (good_probe.get(scope) or {}).get(key)
        new_value = (new_probe.get(scope) or {}).get(key)
        marker = "==" if good_value == new_value else "!="
        rows.append(f"{scope}.{key}: {good_value} {marker} {new_value}")
    return rows


def _probe_video_dimensions(input_video: Path, ffprobe_bin: str) -> tuple[int, int] | None:
    cmd = [
        ffprobe_bin,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "csv=p=0:s=x",
        str(input_video),
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        return None

    raw = result.stdout.strip()
    if not raw or "x" not in raw:
        return None
    width_text, height_text = raw.split("x", 1)
    try:
        width = int(width_text.strip())
        height = int(height_text.strip())
    except ValueError:
        return None
    if width <= 0 or height <= 0:
        return None
    return width, height


def _compute_y_scale_debug(
    *,
    source_width: int,
    source_height: int,
    output_width: int,
    output_height: int,
    video_y_scale: float,
    y_scale_mode: str,
) -> dict[str, float | str]:
    # Match filter step: scale=output_width:-2 (width-fit with even height).
    fit_scale = output_width / source_width
    fit_width = float(output_width)
    raw_fit_height = source_height * fit_scale
    fit_height = max(2.0, float(int(math.floor(raw_fit_height / 2.0) * 2)))
    required_fill_scale = (output_height + (2 * FILL_TRIM_PX)) / fit_height if fit_height > 0 else 1.0
    if y_scale_mode == "fill":
        effective_y_scale = max(video_y_scale, required_fill_scale)
    elif y_scale_mode == "letterbox":
        effective_y_scale = 1.0
    else:
        effective_y_scale = video_y_scale

    return {
        "source_width": float(source_width),
        "source_height": float(source_height),
        "fit_width": float(fit_width),
        "fit_height": float(fit_height),
        "base_width": float(fit_width),
        "base_height": float(fit_height),
        "ih_after_fit": float(fit_height),
        "required_fill_scale": float(required_fill_scale),
        "computed_required_fill_scale": float(required_fill_scale),
        "video_y_scale_requested": float(video_y_scale),
        "effective_y_scale": float(effective_y_scale),
        "effective_y_scale_used": float(effective_y_scale),
        "y_scale_mode": y_scale_mode,
    }


def _short_probe_diff(good: dict[str, Any], new: dict[str, Any]) -> list[str]:
    keys = ["codec", "resolution", "sar", "dar", "fps", "bitrate_stream", "bitrate_format"]
    rows: list[str] = []
    for key in keys:
        gv = good.get(key)
        nv = new.get(key)
        marker = "==" if gv == nv else "!="
        rows.append(f"{key}: {gv} {marker} {nv}")
    return rows


def parse_time_to_seconds(value: str | int | float) -> float:
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()
    if not text:
        raise ValueError("Time value is empty.")

    try:
        return float(text)
    except ValueError:
        pass

    parts = text.split(":")
    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)

    raise ValueError(f"Unsupported time format: {value}")


def format_ffmpeg_time(seconds: float) -> str:
    total_ms = int(round(max(seconds, 0.0) * 1000))
    total_seconds, ms = divmod(total_ms, 1000)
    minutes, secs = divmod(total_seconds, 60)
    hours, mins = divmod(minutes, 60)
    return f"{hours:02d}:{mins:02d}:{secs:02d}.{ms:03d}"


def probe_duration_seconds(input_video: Path, ffprobe_bin: str = "ffprobe") -> float:
    if ffprobe_bin == "ffprobe":
        ffprobe_bin = _resolve_binary("ffprobe")
    cmd = [
        ffprobe_bin,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(input_video),
    ]
    result = _run_command(cmd)
    duration_text = result.stdout.strip()
    if not duration_text:
        raise RuntimeError(f"ffprobe returned empty duration output for {input_video}.")
    return float(duration_text)


def build_auto_segments(total_duration: float, part_seconds: int = 70) -> list[Segment]:
    if part_seconds <= 0:
        raise ValueError("part_seconds must be greater than 0.")
    if total_duration <= 0:
        raise ValueError("Input duration must be greater than 0.")

    segments: list[Segment] = []
    part_count = int(math.ceil(total_duration / part_seconds))
    for index in range(part_count):
        start = index * float(part_seconds)
        end = min((index + 1) * float(part_seconds), total_duration)
        if end - start <= 0:
            continue
        segments.append(Segment(start=start, end=end))

    if not segments:
        raise RuntimeError("No renderable segments were generated from input duration.")
    return segments


def load_cuts_override(cuts_path: Path) -> tuple[list[Segment], dict[str, int]]:
    data = json.loads(cuts_path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError("cuts.json must be a JSON object.")

    raw_parts = data.get("parts", [])
    segments: list[Segment] = []
    if raw_parts:
        if not isinstance(raw_parts, list):
            raise ValueError("cuts.json field 'parts' must be a list.")
        for idx, raw_part in enumerate(raw_parts, start=1):
            if not isinstance(raw_part, dict):
                raise ValueError(f"cuts.json part #{idx} must be an object.")
            start = parse_time_to_seconds(raw_part.get("start", ""))
            end = parse_time_to_seconds(raw_part.get("end", ""))
            if end <= start:
                raise ValueError(f"cuts.json part #{idx} has end <= start.")
            segments.append(Segment(start=start, end=end))

    overrides: dict[str, int] = {}
    for key in ("crop_top_px", "output_width", "output_height"):
        if key in data and data[key] is not None:
            overrides[key] = int(data[key])

    return segments, overrides


def build_manual_chapter_segments(
    manual_chapters: list[dict[str, Any]],
    *,
    total_duration: float,
    log: Callable[[str], None] | None = None,
) -> tuple[list[Segment], list[str]]:
    if not manual_chapters:
        raise ValueError("manual_chapters must contain at least one chapter.")

    normalized: list[tuple[float, float | None, str]] = []
    previous_start: float | None = None

    chapter_count = len(manual_chapters)
    for idx, raw_chapter in enumerate(manual_chapters, start=1):
        if not isinstance(raw_chapter, dict):
            raise ValueError(f"manual_chapters item #{idx} must be an object.")

        start = parse_time_to_seconds(raw_chapter.get("start", ""))
        raw_end = str(raw_chapter.get("end", "") or "").strip()
        end = parse_time_to_seconds(raw_end) if raw_end else None
        title = str(raw_chapter.get("title", "") or "").strip()

        if start < 0:
            raise ValueError(f"manual_chapters item #{idx} has a negative start.")
        if previous_start is not None and start <= previous_start:
            raise ValueError("manual_chapters start times must be strictly increasing.")
        if start >= total_duration:
            raise ValueError(f"manual_chapters item #{idx} starts at or after the video duration.")
        if end is not None:
            if end <= start:
                raise ValueError(f"manual_chapters item #{idx} has end <= start.")
            if end > total_duration + MANUAL_CHAPTER_END_TOLERANCE_SECONDS:
                raise ValueError(f"manual_chapters item #{idx} ends after the video duration.")
            if end > total_duration:
                end = total_duration
                if idx == chapter_count and log is not None:
                    log("Clamped final clip end to video duration.")

        normalized.append((start, end, title))
        previous_start = start

    segments: list[Segment] = []
    titles: list[str] = []
    for idx, (start, explicit_end, title) in enumerate(normalized):
        resolved_end = explicit_end if explicit_end is not None else (normalized[idx + 1][0] if idx + 1 < len(normalized) else total_duration)
        if resolved_end <= start:
            raise ValueError(f"manual_chapters item #{idx + 1} has end <= start.")
        if idx + 1 < len(normalized) and resolved_end > normalized[idx + 1][0]:
            raise ValueError(f"manual_chapters item #{idx + 1} overlaps the next chapter.")
        segments.append(Segment(start=start, end=resolved_end))
        titles.append(title)

    return segments, titles


def detect_scene_segments(
    input_video: Path,
    threshold: float = 27.0,
    ffprobe_bin: str = "ffprobe",  # noqa: ARG001
) -> list[Segment]:
    from scenedetect import ContentDetector, detect  # optional dep; imported locally

    scene_list = detect(str(input_video), ContentDetector(threshold=threshold))
    if not scene_list:
        raise RuntimeError("PySceneDetect found no scenes in the input video.")
    return [
        Segment(start=start_tc.get_seconds(), end=end_tc.get_seconds())
        for start_tc, end_tc in scene_list
    ]


_VALID_SPLIT_MODES: frozenset[str] = frozenset({"duration", "parts", "manual", "ai", "scene", "chapters"})


def resolve_segments(
    input_video: Path,
    part_seconds: int,
    cuts_path: Path | None,
    ffprobe_bin: str = "ffprobe",
    split_mode: str = "duration",
    scene_threshold: float = 27.0,
    chapter_data: list[dict] | None = None,
    manual_chapters: list[dict[str, Any]] | None = None,
    log: Callable[[str], None] | None = None,
) -> tuple[list[Segment], dict[str, int], list[str]]:
    if split_mode not in _VALID_SPLIT_MODES:
        raise ValueError(f"split_mode must be one of {sorted(_VALID_SPLIT_MODES)}; got {split_mode!r}")
    overrides: dict[str, int] = {}
    if cuts_path is not None:
        manual_segments, overrides = load_cuts_override(cuts_path)
        if manual_segments:
            return manual_segments, overrides, []

    if split_mode == "manual":
        if not manual_chapters:
            raise ValueError("manual split mode requires manual_chapters or cuts_path.")
        total_duration = probe_duration_seconds(input_video=input_video, ffprobe_bin=ffprobe_bin)
        manual_segments, manual_titles = build_manual_chapter_segments(
            manual_chapters,
            total_duration=total_duration,
            log=log,
        )
        return manual_segments, overrides, manual_titles

    if split_mode == "chapters":
        if not chapter_data:
            raise ValueError(
                "chapter_data is required for chapters split mode and must be non-empty."
            )
        chapter_segs: list[Segment] = []
        chapter_titles: list[str] = []
        for ch in chapter_data:
            start = float(ch["start_time"])
            end = float(ch["end_time"])
            if end > start:
                chapter_segs.append(Segment(start=start, end=end))
                chapter_titles.append(str(ch.get("title", "")))
        if not chapter_segs:
            raise ValueError("No valid chapter segments could be built from chapter_data.")
        return chapter_segs, overrides, chapter_titles

    if split_mode == "scene":
        return (
            detect_scene_segments(input_video=input_video, threshold=scene_threshold, ffprobe_bin=ffprobe_bin),
            overrides,
            [],
        )

    total_duration = probe_duration_seconds(input_video=input_video, ffprobe_bin=ffprobe_bin)
    return build_auto_segments(total_duration=total_duration, part_seconds=part_seconds), overrides, []


def _normalize_overlay_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(text or ""))
    normalized = (
        normalized
        .replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u02BC", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
    )

    cleaned_chars: list[str] = []
    for char in normalized:
        category = unicodedata.category(char)
        if category.startswith("C") and char not in {" ", "\t", "\n", "\r"}:
            continue
        cleaned_chars.append(char)

    return re.sub(r"\s+", " ", "".join(cleaned_chars)).strip()


def _escape_filter_path(path: Path) -> str:
    return (
        str(path)
        .replace("\\", "/")
        .replace(":", r"\:")
        .replace("'", r"\'")
        .replace(",", r"\,")
        .replace("[", r"\[")
        .replace("]", r"\]")
    )


def write_drawtext_textfile(
    text: str,
    out_dir: Path,
    name: str,
) -> Path:
    """Write normalized drawtext content to a UTF-8 file and return its path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    path.write_text(_normalize_overlay_text(text), encoding="utf-8")
    return path


def _escape_drawtext_textfile_path(textfile_path: Path) -> str:
    return _escape_filter_path(textfile_path)


def build_drawtext_filter_from_file(
    textfile_path: Path,
    fontsize: int,
    fontcolor: str,
    x_expr: str,
    y_expr: str,
    box: bool = True,
    boxcolor: str = "white@0.95",
    boxborderw: int = 24,
    borderw: int = 0,
    bordercolor: str = "black",
) -> str:
    parts = [
        f"drawtext=textfile='{_escape_drawtext_textfile_path(textfile_path)}'",
        f"fontsize={int(fontsize)}",
        f"fontcolor={fontcolor}",
    ]
    if box:
        parts.extend(
            [
                "box=1",
                f"boxcolor={boxcolor}",
                f"boxborderw={int(boxborderw)}",
            ]
        )
    parts.extend(
        [
            f"x={x_expr}",
            f"y={y_expr}",
            f"borderw={int(borderw)}",
            f"bordercolor={bordercolor}",
        ]
    )
    return ":".join(parts)


def _clamp_percent(value: float, fallback: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = fallback
    return max(0.0, min(100.0, numeric))


def _resolve_caption_xy(
    *,
    overlay_x_percent: float,
    overlay_y_percent: float,
) -> tuple[str, str]:
    if math.isclose(overlay_x_percent, 50.0, abs_tol=0.0001):
        x_expr = "(w-text_w)/2"
    else:
        x_expr = f"(w-text_w)*{overlay_x_percent:g}/100"
    y_expr = f"(h-text_h)*{overlay_y_percent:g}/100"
    return x_expr, y_expr


def _build_chapter_drawtext_filter(
    textfile_path: Path,
    font_file: Path | None,
    overlay_x_percent: float,
    overlay_y_percent: float,
) -> str:
    """Return a drawtext filter string that burns a caption/title onto the frame."""
    x_expr, y_expr = _resolve_caption_xy(
        overlay_x_percent=overlay_x_percent,
        overlay_y_percent=overlay_y_percent,
    )
    font_part = f":fontfile='{_escape_filter_path(font_file)}'" if font_file else ""
    return build_drawtext_filter_from_file(
        textfile_path=textfile_path,
        fontsize=64,
        fontcolor="black",
        x_expr=x_expr,
        y_expr=y_expr,
        box=True,
        boxcolor="white@0.95",
        boxborderw=24,
        borderw=0,
        bordercolor="black",
    ) + font_part


def _clamp_pct(value: float, fallback: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = fallback
    return max(0.0, min(1.0, numeric))


def _resolve_part_label_xy(
    *,
    part_label_position: str,
    label_x_pct: float,
    label_y_pct: float,
    part_label_x_percent: float | None = None,
    part_label_y_percent: float | None = None,
) -> tuple[str, str]:
    pos = (part_label_position or "top-center").lower()
    if pos in {"custom", "custom-drag"} and part_label_x_percent is not None and part_label_y_percent is not None:
        x_percent = _clamp_percent(part_label_x_percent, fallback=50.0)
        y_percent = _clamp_percent(part_label_y_percent, fallback=4.0)
        x_expr = "(w-text_w)/2" if math.isclose(x_percent, 50.0, abs_tol=0.0001) else f"(w-text_w)*{x_percent:g}/100"
        return x_expr, f"(h-text_h)*{y_percent:g}/100"
    if pos in {"custom", "custom-drag"}:
        return (
            f"(w-text_w)*{label_x_pct:g}",
            f"(h-text_h)*{label_y_pct:g}",
        )

    presets: dict[str, tuple[str, str]] = {
        "top-left": ("40", "40"),
        "top-center": ("(w-text_w)/2", "40"),
        "top-right": ("w-text_w-40", "40"),
        "middle-left": ("40", "(h-text_h)/2"),
        "middle-center": ("(w-text_w)/2", "(h-text_h)/2"),
        "middle-right": ("w-text_w-40", "(h-text_h)/2"),
        "bottom-left": ("40", "h-text_h-40"),
        "bottom-center": ("(w-text_w)/2", "h-text_h-40"),
        "bottom-right": ("w-text_w-40", "h-text_h-40"),
    }
    return presets.get(pos, presets["top-center"])


def _build_part_drawtext_filter(
    *,
    textfile_path: Path,
    part_label_position: str,
    label_x_pct: float,
    label_y_pct: float,
    part_label_x_percent: float | None,
    part_label_y_percent: float | None,
    font_file: Path | None,
) -> str:
    x_expr, y_expr = _resolve_part_label_xy(
        part_label_position=part_label_position,
        label_x_pct=label_x_pct,
        label_y_pct=label_y_pct,
        part_label_x_percent=part_label_x_percent,
        part_label_y_percent=part_label_y_percent,
    )
    font_part = f":fontfile='{_escape_filter_path(font_file)}'" if font_file else ""
    return build_drawtext_filter_from_file(
        textfile_path=textfile_path,
        fontsize=64,
        fontcolor="white",
        x_expr=x_expr,
        y_expr=y_expr,
        box=True,
        boxcolor="black@0.45",
        boxborderw=14,
        borderw=3,
        bordercolor="black",
    ) + font_part


def _build_youtube_credit_drawtext_filter(
    *,
    textfile_path: Path,
    youtube_credit_position: str,
    font_file: Path | None,
) -> str:
    position = (youtube_credit_position or "below_frame").strip().lower()
    x_presets = {
        "bottom_left": "42",
        "bottom_center": "(w-text_w)/2",
        "bottom_right": "w-text_w-42",
        "below_frame": "(w-text_w)/2",
    }
    y_presets = {
        "bottom_left": "h-text_h-120",
        "bottom_center": "h-text_h-120",
        "bottom_right": "h-text_h-120",
        "below_frame": "h-text_h-210",
    }
    x_expr = x_presets.get(position, x_presets["below_frame"])
    y_expr = y_presets.get(position, y_presets["below_frame"])
    font_part = f":fontfile='{_escape_filter_path(font_file)}'" if font_file else ""
    return build_drawtext_filter_from_file(
        textfile_path=textfile_path,
        fontsize=40,
        fontcolor="white",
        x_expr=x_expr,
        y_expr=y_expr,
        box=False,
        borderw=3,
        bordercolor="black",
    ) + font_part


def _sanitize_part_filename_title(title: str, *, max_length: int = 90) -> str:
    normalized = unicodedata.normalize("NFKD", str(title or "")).encode("ascii", "ignore").decode("ascii")
    normalized = normalized.lower()
    normalized = re.sub(r"[^a-z0-9\s_-]+", "", normalized)
    normalized = re.sub(r"[\s_-]+", "_", normalized).strip("_")
    return (normalized[:max_length].strip("_") or "clip")


def _build_upload_description(title: str, part_number: int, hashtags: str) -> str:
    safe_title = _normalize_overlay_text(title) or f"Part {part_number}"
    safe_hashtags = re.sub(r"\s+", " ", str(hashtags or "")).strip()
    return f"{safe_title} (Part {part_number}) {safe_hashtags}".strip()


def _normalize_reaction_crop(crop: dict[str, Any] | None, fallback: dict[str, float]) -> dict[str, float]:
    raw = crop if isinstance(crop, dict) else fallback

    def _value(key: str, default: float) -> float:
        try:
            numeric = float(raw.get(key, default))
        except (TypeError, ValueError, AttributeError):
            numeric = default
        return numeric

    x = max(0.0, min(100.0, _value("x_percent", fallback["x_percent"])))
    y = max(0.0, min(100.0, _value("y_percent", fallback["y_percent"])))
    width = max(2.0, min(100.0, _value("width_percent", fallback["width_percent"])))
    height = max(2.0, min(100.0, _value("height_percent", fallback["height_percent"])))
    width = min(width, 100.0 - x)
    height = min(height, 100.0 - y)
    if width < 2.0:
        width = 2.0
        x = max(0.0, min(x, 100.0 - width))
    if height < 2.0:
        height = 2.0
        y = max(0.0, min(y, 100.0 - height))
    return {
        "x_percent": x,
        "y_percent": y,
        "width_percent": width,
        "height_percent": height,
    }


def _percent_crop_to_pixels(
    crop: dict[str, Any],
    source_width: int,
    source_height: int,
) -> tuple[int, int, int, int]:
    if source_width <= 2 or source_height <= 2:
        raise ValueError("source dimensions must be greater than 2 pixels")

    normalized = _normalize_reaction_crop(
        crop,
        {"x_percent": 0.0, "y_percent": 0.0, "width_percent": 100.0, "height_percent": 100.0},
    )
    x = int(round(source_width * normalized["x_percent"] / 100.0))
    y = int(round(source_height * normalized["y_percent"] / 100.0))
    width = int(round(source_width * normalized["width_percent"] / 100.0))
    height = int(round(source_height * normalized["height_percent"] / 100.0))

    x = max(0, min(x, source_width - 2))
    y = max(0, min(y, source_height - 2))
    width = max(2, min(width, source_width - x))
    height = max(2, min(height, source_height - y))
    if width % 2:
        width -= 1 if width > 2 else 0
    if height % 2:
        height -= 1 if height > 2 else 0
    if x + width > source_width:
        x = max(0, source_width - width)
    if y + height > source_height:
        y = max(0, source_height - height)
    return x, y, max(2, width), max(2, height)


def _caption_duration_enable(
    caption_duration_mode: str,
    caption_duration_seconds: float | None,
) -> str:
    mode = (caption_duration_mode or "entire").lower()
    if mode in {"first_3_seconds", "3s", "first_3"}:
        return "between(t,0,3)"
    if mode == "first_5_seconds":
        return "between(t,0,5)"
    if mode in {"first_10_seconds", "10s", "first_10"}:
        return "between(t,0,10)"
    if mode == "custom" and caption_duration_seconds is not None:
        duration = max(0.1, float(caption_duration_seconds))
        return f"between(t,0,{duration:g})"
    return ""


def _caption_y_expr(caption_position: str, output_height: int, overlay_y_percent: float) -> str:
    position = (caption_position or "between").lower()
    if position == "top":
        return "120"
    if position == "center":
        return "(H-h)/2"
    if position == "bottom":
        return "H-h-180"
    if position == "custom":
        safe_percent = _clamp_percent(overlay_y_percent, fallback=50.0)
        return f"(H-h)*{safe_percent:g}/100"
    return f"{max(0, int(output_height / 2))}-h/2"


def _load_caption_font(font_size: int):
    from PIL import ImageFont

    candidates = [
        Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "arialbd.ttf",
        Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "segoeuib.ttf",
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        Path("/Library/Fonts/Arial Bold.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            try:
                return ImageFont.truetype(str(candidate), font_size)
            except OSError:
                continue
    return ImageFont.load_default()


def _text_size(draw: Any, text: str, font: Any) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])


def _wrap_caption_text(draw: Any, text: str, font: Any, max_width: int) -> list[str]:
    words = _normalize_overlay_text(text).split()
    if not words:
        return []
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        width, _height = _text_size(draw, candidate, font)
        if width <= max_width or not current:
            current = candidate
            continue
        lines.append(current)
        current = word
    if current:
        lines.append(current)
    return lines[:3]


def create_caption_overlay_png(
    text: str,
    output_path: Path,
    output_width: int = 1080,
    max_box_width: int = 940,
    font_size: int = 76,
    padding: int = 46,
    corner_radius: int = 34,
    background_opacity: float = 0.96,
    shadow_strength: float = 0.0,
    text_stroke: int = 0,
) -> Path:
    from PIL import Image, ImageDraw, ImageFilter

    cleaned = _normalize_overlay_text(text)
    if not cleaned:
        raise ValueError("caption text is empty")

    font = _load_caption_font(font_size)
    scratch = Image.new("RGBA", (output_width, 256), (0, 0, 0, 0))
    draw = ImageDraw.Draw(scratch)
    padding_x = max(18, int(padding))
    padding_y = max(14, int(round(padding * 0.65)))
    text_max_width = max(100, max_box_width - (padding_x * 2))
    lines = _wrap_caption_text(draw, cleaned, font, text_max_width)
    if not lines:
        lines = [cleaned]
    line_sizes = [_text_size(draw, line, font) for line in lines]
    line_gap = 12
    text_width = max(width for width, _height in line_sizes)
    text_height = sum(height for _width, height in line_sizes) + line_gap * max(0, len(lines) - 1)
    box_width = min(max_box_width, text_width + (padding_x * 2))
    box_height = text_height + (padding_y * 2)
    shadow_pad = int(max(0, min(40, shadow_strength * 16)))
    image = Image.new("RGBA", (box_width + shadow_pad * 2, box_height + shadow_pad * 2), (0, 0, 0, 0))
    if shadow_pad:
        shadow = Image.new("RGBA", image.size, (0, 0, 0, 0))
        shadow_draw = ImageDraw.Draw(shadow)
        shadow_draw.rounded_rectangle(
            (shadow_pad, shadow_pad + max(1, shadow_pad // 3), shadow_pad + box_width - 1, shadow_pad + box_height - 1),
            radius=min(max(0, int(corner_radius)), box_height // 2),
            fill=(0, 0, 0, int(70 * max(0.0, min(2.0, shadow_strength)))),
        )
        image.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(radius=max(1, shadow_pad // 2))))
    draw = ImageDraw.Draw(image)
    radius = min(max(0, int(corner_radius)), box_height // 2)
    alpha = int(max(0.1, min(1.0, float(background_opacity))) * 255)
    draw.rounded_rectangle(
        (shadow_pad, shadow_pad, shadow_pad + box_width - 1, shadow_pad + box_height - 1),
        radius=radius,
        fill=(255, 255, 255, alpha),
    )
    y = padding_y
    for line, (_width, height) in zip(lines, line_sizes):
        line_width, _line_height = _text_size(draw, line, font)
        draw.text(
            (shadow_pad + (box_width - line_width) / 2, shadow_pad + y),
            line,
            font=font,
            fill=(0, 0, 0, 255),
            stroke_width=max(0, int(text_stroke)),
            stroke_fill=(255, 255, 255, 180),
        )
        y += height + line_gap
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return output_path


def _build_region_filter(
    *,
    crop_pixels: tuple[int, int, int, int],
    target_width: int,
    target_height: int,
    label: str,
) -> str:
    x, y, width, height = crop_pixels
    return (
        f"[0:v]crop={width}:{height}:{x}:{y},"
        f"scale={target_width}:{target_height}:force_original_aspect_ratio=increase,"
        f"crop={target_width}:{target_height}:(iw-{target_width})/2:(ih-{target_height})/2,"
        f"setsar=1[{label}]"
    )


def _build_reaction_filter_complex(
    *,
    source_width: int,
    source_height: int,
    output_width: int,
    output_height: int,
    main_crop: dict[str, Any] | None,
    facecam_crop: dict[str, Any] | None,
    reaction_layout_preset: str,
    caption_text: str | None,
    caption_input_index: int | None,
    caption_position: str,
    caption_duration_mode: str,
    caption_duration_seconds: float | None,
    overlay_y_percent: float,
    playback_speed: float,
    subtitle_ass_path: Path | str | None,
    youtube_credit_textfile_path: Path | None = None,
    youtube_credit_position: str = "below_frame",
    font_file: Path | None = None,
) -> tuple[str, str, dict[str, Any]]:
    main_pixels = _percent_crop_to_pixels(
        main_crop or {"x_percent": 0, "y_percent": 0, "width_percent": 100, "height_percent": 65},
        source_width,
        source_height,
    )
    face_pixels = _percent_crop_to_pixels(
        facecam_crop or {"x_percent": 60, "y_percent": 55, "width_percent": 35, "height_percent": 40},
        source_width,
        source_height,
    )
    filters: list[str] = []
    preset = reaction_layout_preset or "content_top_facecam_bottom"
    if preset == "content_full_facecam_overlay":
        filters.append(
            _build_region_filter(
                crop_pixels=main_pixels,
                target_width=output_width,
                target_height=output_height,
                label="main_full",
            )
        )
        pip_size = _clamp_even(int(round(output_width * 0.34)), minimum=120, maximum=max(120, output_width - 80))
        filters.append(
            _build_region_filter(
                crop_pixels=face_pixels,
                target_width=pip_size,
                target_height=pip_size,
                label="face_pip",
            )
        )
        filters.append(f"[main_full][face_pip]overlay=x=W-w-56:y=120[layout0]")
    else:
        top_height = _clamp_even(output_height // 2, minimum=2, maximum=output_height - 2)
        bottom_height = output_height - top_height
        if bottom_height % 2:
            bottom_height -= 1
            top_height = output_height - bottom_height
        filters.append(
            _build_region_filter(
                crop_pixels=main_pixels,
                target_width=output_width,
                target_height=top_height,
                label="main_half",
            )
        )
        filters.append(
            _build_region_filter(
                crop_pixels=face_pixels,
                target_width=output_width,
                target_height=bottom_height,
                label="face_half",
            )
        )
        if preset == "facecam_top_content_bottom":
            filters.append("[face_half][main_half]vstack=inputs=2[layout0]")
        else:
            filters.append("[main_half][face_half]vstack=inputs=2[layout0]")

    current_label = "layout0"
    safe_speed = round(float(playback_speed), 2)
    if safe_speed != 1.0:
        filters.append(f"[{current_label}]setpts={1 / safe_speed:.4f}*PTS[layout_speed]")
        current_label = "layout_speed"

    if youtube_credit_textfile_path is not None:
        filters.append(
            f"[{current_label}]"
            f"{_build_youtube_credit_drawtext_filter(textfile_path=youtube_credit_textfile_path, youtube_credit_position=youtube_credit_position, font_file=font_file)}"
            "[layout_credit]"
        )
        current_label = "layout_credit"

    cleaned_caption = _normalize_overlay_text(caption_text or "")
    if cleaned_caption and caption_input_index is not None:
        enable = _caption_duration_enable(caption_duration_mode, caption_duration_seconds)
        enable_part = f":enable='{enable}'" if enable else ""
        y_expr = _caption_y_expr(caption_position, output_height, overlay_y_percent)
        filters.append(
            f"[{current_label}][{caption_input_index}:v]overlay=x=(W-w)/2:y={y_expr}{enable_part}[layout_caption]"
        )
        current_label = "layout_caption"

    if subtitle_ass_path is not None:
        from .subtitles import build_subtitle_filter

        filters.append(f"[{current_label}]{build_subtitle_filter(subtitle_ass_path)}[vout]")
        current_label = "vout"

    if current_label != "vout":
        filters.append(f"[{current_label}]format=yuv420p,setsar=1[vout]")
        current_label = "vout"
    else:
        filters.append("[vout]format=yuv420p,setsar=1[vout_fmt]")
        current_label = "vout_fmt"

    debug = {
        "main_crop_pixels": main_pixels,
        "facecam_crop_pixels": face_pixels,
        "reaction_layout_preset": preset,
    }
    return ";".join(filters), current_label, debug


def _normalize_facecam_shape(value: Any) -> str:
    shape = str(value or "rectangle").strip().lower()
    if shape == "rounded":
        return "rounded_rectangle"
    if shape in {"rectangle", "rounded_rectangle", "circle"}:
        return shape
    return "rectangle"


def _normalize_timeline_duration_mode(value: Any) -> str:
    mode = str(value or "entire").strip().lower()
    aliases = {
        "5s": "first_5_seconds",
        "first_5": "first_5_seconds",
        "3s": "first_3_seconds",
        "first_3": "first_3_seconds",
        "10s": "first_10_seconds",
        "first_10": "first_10_seconds",
    }
    mode = aliases.get(mode, mode)
    if mode in {"entire", "first_3_seconds", "first_5_seconds", "first_10_seconds", "custom"}:
        return mode
    return "entire"


def _normalize_layout_preset(value: Any) -> str:
    preset = str(value or "main_top_reaction_bottom").strip().lower()
    aliases = {
        "content_top_facecam_bottom": "main_top_reaction_bottom",
        "facecam_top_content_bottom": "reaction_top_main_bottom",
        "content_full_facecam_overlay": "facecam_right",
        "reaction_bottom": "main_top_reaction_bottom",
        "reaction_top": "reaction_top_main_bottom",
        "top_right_facecam": "facecam_right",
        "top_left_facecam": "facecam_left",
        "bottom_right_facecam": "facecam_right",
        "bottom_left_facecam": "facecam_left",
        "reactor_fullscreen": "custom",
        "facecam_fullscreen": "custom",
        "no_facecam": "custom",
    }
    preset = aliases.get(preset, preset)
    allowed = {
        "facecam_top",
        "facecam_bottom",
        "facecam_left",
        "facecam_right",
        "main_top_reaction_bottom",
        "reaction_top_main_bottom",
        "side_by_side",
        "custom",
    }
    return preset if preset in allowed else "main_top_reaction_bottom"


def _normalize_output_region(region: dict[str, Any] | None, fallback: dict[str, float]) -> dict[str, float]:
    return _normalize_reaction_crop(region, fallback)


def _timeline_ratio(row: dict[str, Any], fallback: float = 0.5) -> float:
    raw = row.get("divider_split") if isinstance(row, dict) else None
    if isinstance(raw, dict):
        value = raw.get("ratio", fallback)
    else:
        value = fallback
    try:
        ratio = float(value)
    except (TypeError, ValueError):
        ratio = fallback
    return max(0.1, min(0.9, ratio))


def _timeline_direction(row: dict[str, Any], fallback: str = "horizontal") -> str:
    raw = row.get("divider_split") if isinstance(row, dict) else None
    direction = str((raw or {}).get("direction", fallback) if isinstance(raw, dict) else fallback).lower()
    return "vertical" if direction == "vertical" else "horizontal"


def _timeline_caption_seconds(row: dict[str, Any]) -> float | None:
    value = row.get("caption_duration_seconds")
    if value is None:
        value = row.get("custom_caption_seconds")
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    return seconds if seconds > 0 else None


def _normalize_timeline_row(
    row: dict[str, Any],
    *,
    fallback: dict[str, Any],
) -> dict[str, Any]:
    start_raw = row.get("start", fallback.get("start", "00:00:00"))
    end_raw = row.get("end", fallback.get("end", "00:00:00"))
    start_seconds = parse_time_to_seconds(start_raw)
    end_seconds = parse_time_to_seconds(end_raw)
    if end_seconds <= start_seconds:
        raise ValueError(f"Reaction timeline row end must be after start: {start_raw} - {end_raw}")

    layout_preset = _normalize_layout_preset(row.get("layout_preset", fallback.get("layout_preset")))
    shape = _normalize_facecam_shape(row.get("facecam_shape", fallback.get("facecam_shape")))
    caption_duration_mode = _normalize_timeline_duration_mode(
        row.get("caption_duration", row.get("caption_duration_mode", fallback.get("caption_duration_mode", "entire")))
    )
    caption_enabled = bool(row.get("caption_enabled", fallback.get("caption_enabled", True)))
    caption_text = _normalize_overlay_text(row.get("caption", row.get("caption_text", fallback.get("caption", ""))))
    main_crop = _normalize_reaction_crop(row.get("main_crop"), fallback.get("main_crop", {}))
    facecam_crop = _normalize_reaction_crop(row.get("facecam_crop"), fallback.get("facecam_crop", {}))

    return {
        "id": str(row.get("id") or ""),
        "start": str(start_raw),
        "end": str(end_raw),
        "start_seconds": start_seconds,
        "end_seconds": end_seconds,
        "caption": caption_text,
        "layout_preset": layout_preset,
        "keep_aspect_ratio": bool(row.get("keep_aspect_ratio", fallback.get("keep_aspect_ratio", True))),
        "caption_enabled": caption_enabled,
        "caption_duration": caption_duration_mode,
        "caption_duration_seconds": _timeline_caption_seconds(row),
        "main_crop": main_crop,
        "facecam_crop": facecam_crop,
        "facecam_shape": shape,
        "divider_split": {
            "direction": _timeline_direction(row, fallback="vertical" if layout_preset == "side_by_side" else "horizontal"),
            "ratio": _timeline_ratio(row, fallback=0.5 if layout_preset == "side_by_side" else 0.65),
        },
        "main_region": row.get("main_region") if isinstance(row.get("main_region"), dict) else None,
        "facecam_region": row.get("facecam_region") if isinstance(row.get("facecam_region"), dict) else None,
        "caption_position": str(row.get("caption_position", fallback.get("caption_position", "between")) or "between"),
        "caption_style": row.get("caption_style", fallback.get("caption_style", {})),
        "reference_frame_url": row.get("reference_frame_url"),
        "reference_timestamp": row.get("reference_timestamp"),
    }


def _fallback_timeline_row(
    *,
    segment: Segment,
    main_crop: dict[str, Any] | None,
    facecam_crop: dict[str, Any] | None,
    reaction_layout_preset: str,
    facecam_shape: str,
    caption_text: str,
    caption_position: str,
    caption_duration_mode: str,
    caption_duration_seconds: float | None,
) -> dict[str, Any]:
    return {
        "id": "fallback",
        "start": format_ffmpeg_time(segment.start),
        "end": format_ffmpeg_time(segment.end),
        "start_seconds": segment.start,
        "end_seconds": segment.end,
        "caption": caption_text,
        "layout_preset": _normalize_layout_preset(reaction_layout_preset),
        "keep_aspect_ratio": True,
        "caption_enabled": bool(caption_text),
        "caption_duration": _normalize_timeline_duration_mode(caption_duration_mode),
        "caption_duration_seconds": caption_duration_seconds,
        "main_crop": _normalize_reaction_crop(
            main_crop,
            {"x_percent": 0, "y_percent": 0, "width_percent": 100, "height_percent": 65},
        ),
        "facecam_crop": _normalize_reaction_crop(
            facecam_crop,
            {"x_percent": 60, "y_percent": 55, "width_percent": 35, "height_percent": 40},
        ),
        "facecam_shape": _normalize_facecam_shape(facecam_shape),
        "divider_split": {"direction": "horizontal", "ratio": 0.5},
        "main_region": None,
        "facecam_region": None,
        "caption_position": caption_position,
        "caption_style": {},
    }


def _timeline_intervals_for_segment(
    *,
    segment: Segment,
    timeline_rows: list[dict[str, Any]],
    fallback_row: dict[str, Any],
) -> list[dict[str, Any]]:
    intersections: list[tuple[float, float, dict[str, Any]]] = []
    for row in sorted(timeline_rows, key=lambda item: (float(item["start_seconds"]), float(item["end_seconds"]))):
        start = max(0.0, float(row["start_seconds"]) - segment.start)
        end = min(segment.duration, float(row["end_seconds"]) - segment.start)
        if end - start <= 0.01:
            continue
        intersections.append((start, end, row))

    intervals: list[dict[str, Any]] = []
    cursor = 0.0
    for start, end, row in intersections:
        if start > cursor + 0.01:
            intervals.append({"start": cursor, "end": start, "row": fallback_row})
        if end > cursor + 0.01:
            intervals.append({"start": max(start, cursor), "end": end, "row": row})
            cursor = end
    if cursor < segment.duration - 0.01:
        intervals.append({"start": cursor, "end": segment.duration, "row": fallback_row})
    if not intervals:
        intervals.append({"start": 0.0, "end": segment.duration, "row": fallback_row})
    return intervals


def _region_percent_to_pixels(
    region: dict[str, Any],
    output_width: int,
    output_height: int,
) -> tuple[int, int, int, int]:
    normalized = _normalize_output_region(
        region,
        {"x_percent": 0, "y_percent": 0, "width_percent": 100, "height_percent": 100},
    )
    x = int(round(output_width * normalized["x_percent"] / 100.0))
    y = int(round(output_height * normalized["y_percent"] / 100.0))
    width = int(round(output_width * normalized["width_percent"] / 100.0))
    height = int(round(output_height * normalized["height_percent"] / 100.0))
    x = max(0, min(x, output_width - 2))
    y = max(0, min(y, output_height - 2))
    width = _clamp_even(width, minimum=2, maximum=output_width - x)
    height = _clamp_even(height, minimum=2, maximum=output_height - y)
    return x, y, width, height


def _timeline_region_layout(row: dict[str, Any]) -> tuple[dict[str, float], dict[str, float]]:
    preset = _normalize_layout_preset(row.get("layout_preset"))
    ratio = _timeline_ratio(row, fallback=0.5 if preset == "side_by_side" else 0.65)
    split_percent = ratio * 100
    remaining = 100 - split_percent
    default_face_overlay = {
        "facecam_top": {"x_percent": 29, "y_percent": 5, "width_percent": 42, "height_percent": 24},
        "facecam_bottom": {"x_percent": 29, "y_percent": 71, "width_percent": 42, "height_percent": 24},
        "facecam_left": {"x_percent": 5, "y_percent": 37, "width_percent": 34, "height_percent": 24},
        "facecam_right": {"x_percent": 61, "y_percent": 37, "width_percent": 34, "height_percent": 24},
    }
    if preset == "main_top_reaction_bottom":
        return (
            {"x_percent": 0, "y_percent": 0, "width_percent": 100, "height_percent": split_percent},
            {"x_percent": 0, "y_percent": split_percent, "width_percent": 100, "height_percent": remaining},
        )
    if preset == "reaction_top_main_bottom":
        return (
            {"x_percent": 0, "y_percent": remaining, "width_percent": 100, "height_percent": split_percent},
            {"x_percent": 0, "y_percent": 0, "width_percent": 100, "height_percent": remaining},
        )
    if preset == "side_by_side":
        return (
            {"x_percent": 0, "y_percent": 0, "width_percent": split_percent, "height_percent": 100},
            {"x_percent": split_percent, "y_percent": 0, "width_percent": remaining, "height_percent": 100},
        )
    if preset == "custom":
        return (
            _normalize_output_region(row.get("main_region"), {"x_percent": 0, "y_percent": 0, "width_percent": 100, "height_percent": 65}),
            _normalize_output_region(row.get("facecam_region"), {"x_percent": 0, "y_percent": 65, "width_percent": 100, "height_percent": 35}),
        )
    return (
        {"x_percent": 0, "y_percent": 0, "width_percent": 100, "height_percent": 100},
        _normalize_output_region(row.get("facecam_region"), default_face_overlay.get(preset, default_face_overlay["facecam_right"])),
    )


def _build_region_filter_from_label(
    *,
    input_label: str,
    crop_pixels: tuple[int, int, int, int],
    target_width: int,
    target_height: int,
    label: str,
    keep_aspect_ratio: bool,
) -> str:
    x, y, width, height = crop_pixels
    if keep_aspect_ratio:
        scaler = (
            f"scale={target_width}:{target_height}:force_original_aspect_ratio=increase,"
            f"crop={target_width}:{target_height}:(iw-{target_width})/2:(ih-{target_height})/2"
        )
    else:
        scaler = f"scale={target_width}:{target_height}"
    return f"[{input_label}]crop={width}:{height}:{x}:{y},{scaler},setsar=1[{label}]"


def _caption_style_params(style: Any) -> dict[str, Any]:
    raw = style if isinstance(style, dict) else {}
    preset = str(raw.get("preset") or raw.get("name") or "tiktok").lower()
    defaults: dict[str, dict[str, Any]] = {
        "tiktok": {"font_size": 76, "padding": 46, "corner_radius": 34, "background_opacity": 0.96, "shadow_strength": 0.8, "text_stroke": 0},
        "viral": {"font_size": 82, "padding": 50, "corner_radius": 38, "background_opacity": 0.98, "shadow_strength": 1.1, "text_stroke": 0},
        "gaming": {"font_size": 72, "padding": 42, "corner_radius": 20, "background_opacity": 0.94, "shadow_strength": 1.0, "text_stroke": 1},
        "anime_reaction": {"font_size": 76, "padding": 48, "corner_radius": 42, "background_opacity": 0.96, "shadow_strength": 0.9, "text_stroke": 0},
        "streamer": {"font_size": 70, "padding": 42, "corner_radius": 28, "background_opacity": 0.94, "shadow_strength": 1.2, "text_stroke": 0},
        "clean_minimal": {"font_size": 64, "padding": 34, "corner_radius": 24, "background_opacity": 0.9, "shadow_strength": 0.3, "text_stroke": 0},
    }
    params = dict(defaults.get(preset, defaults["tiktok"]))
    for key in ("font_size", "padding", "corner_radius", "text_stroke"):
        if raw.get(key) is not None:
            params[key] = int(max(0, float(raw[key])))
    for key in ("background_opacity", "shadow_strength"):
        if raw.get(key) is not None:
            params[key] = float(raw[key])
    return params


def _build_timeline_interval_layout(
    *,
    input_label: str,
    output_label: str,
    row: dict[str, Any],
    source_width: int,
    source_height: int,
    output_width: int,
    output_height: int,
    caption_input_index: int | None,
    overlay_y_percent: float,
    interval_duration: float,
    index: int,
) -> tuple[list[str], dict[str, Any]]:
    filters: list[str] = []
    main_pixels = _percent_crop_to_pixels(row.get("main_crop"), source_width, source_height)
    face_pixels = _percent_crop_to_pixels(row.get("facecam_crop"), source_width, source_height)
    main_region, face_region = _timeline_region_layout(row)
    main_x, main_y, main_w, main_h = _region_percent_to_pixels(main_region, output_width, output_height)
    face_x, face_y, face_w, face_h = _region_percent_to_pixels(face_region, output_width, output_height)
    keep_aspect = bool(row.get("keep_aspect_ratio", True))

    split_a = f"tl{index}_src_a"
    split_b = f"tl{index}_src_b"
    main_label = f"tl{index}_main"
    face_label = f"tl{index}_face"
    canvas_label = f"tl{index}_canvas"
    main_out = f"tl{index}_main_placed"
    filters.append(f"[{input_label}]split=2[{split_a}][{split_b}]")
    filters.append(
        _build_region_filter_from_label(
            input_label=split_a,
            crop_pixels=main_pixels,
            target_width=main_w,
            target_height=main_h,
            label=main_label,
            keep_aspect_ratio=keep_aspect,
        )
    )
    filters.append(
        _build_region_filter_from_label(
            input_label=split_b,
            crop_pixels=face_pixels,
            target_width=face_w,
            target_height=face_h,
            label=face_label,
            keep_aspect_ratio=keep_aspect,
        )
    )
    filters.append(f"color=c=0x050914:s={output_width}x{output_height}:d={max(0.05, interval_duration):.3f}[{canvas_label}]")
    filters.append(f"[{canvas_label}][{main_label}]overlay=x={main_x}:y={main_y}:shortest=1[{main_out}]")
    current_label = main_out
    face_out = f"tl{index}_face_placed"
    filters.append(f"[{current_label}][{face_label}]overlay=x={face_x}:y={face_y}:shortest=1[{face_out}]")
    current_label = face_out

    caption = _normalize_overlay_text(row.get("caption", ""))
    if caption and bool(row.get("caption_enabled", True)) and caption_input_index is not None:
        enable = _caption_duration_enable(str(row.get("caption_duration") or "entire"), row.get("caption_duration_seconds"))
        enable_part = f":enable='{enable}'" if enable else ""
        y_expr = _caption_y_expr(str(row.get("caption_position") or "between"), output_height, overlay_y_percent)
        caption_out = f"tl{index}_caption"
        filters.append(f"[{current_label}][{caption_input_index}:v]overlay=x=(W-w)/2:y={y_expr}{enable_part}[{caption_out}]")
        current_label = caption_out

    filters.append(f"[{current_label}]format=yuv420p,setsar=1[{output_label}]")
    debug = {
        "main_crop_pixels": main_pixels,
        "facecam_crop_pixels": face_pixels,
        "main_region_pixels": (main_x, main_y, main_w, main_h),
        "facecam_region_pixels": (face_x, face_y, face_w, face_h),
        "layout_preset": row.get("layout_preset"),
        "facecam_shape": row.get("facecam_shape"),
    }
    return filters, debug


def _build_reaction_timeline_filter_complex(
    *,
    source_width: int,
    source_height: int,
    output_width: int,
    output_height: int,
    intervals: list[dict[str, Any]],
    caption_input_indexes: dict[int, int],
    overlay_y_percent: float,
    playback_speed: float,
    subtitle_ass_path: Path | str | None,
    youtube_credit_textfile_path: Path | None = None,
    youtube_credit_position: str = "below_frame",
    font_file: Path | None = None,
) -> tuple[str, str, dict[str, Any]]:
    filters: list[str] = []
    output_labels: list[str] = []
    debug_intervals: list[dict[str, Any]] = []
    for index, interval in enumerate(intervals):
        start = float(interval["start"])
        end = float(interval["end"])
        duration = max(0.05, end - start)
        row = interval["row"]
        trim_label = f"tl{index}_trim"
        filters.append(f"[0:v]trim=start={start:.3f}:end={end:.3f},setpts=PTS-STARTPTS[{trim_label}]")
        output_label = f"tl{index}_out"
        interval_filters, interval_debug = _build_timeline_interval_layout(
            input_label=trim_label,
            output_label=output_label,
            row=row,
            source_width=source_width,
            source_height=source_height,
            output_width=output_width,
            output_height=output_height,
            caption_input_index=caption_input_indexes.get(index),
            overlay_y_percent=overlay_y_percent,
            interval_duration=duration,
            index=index,
        )
        filters.extend(interval_filters)
        output_labels.append(output_label)
        debug_intervals.append(
            {
                "start": start,
                "end": end,
                "row_id": row.get("id"),
                **interval_debug,
            }
        )

    current_label = "timeline_concat"
    if len(output_labels) == 1:
        filters.append(f"[{output_labels[0]}]null[{current_label}]")
    else:
        concat_inputs = "".join(f"[{label}]" for label in output_labels)
        filters.append(f"{concat_inputs}concat=n={len(output_labels)}:v=1:a=0[{current_label}]")

    safe_speed = round(float(playback_speed), 2)
    if safe_speed != 1.0:
        filters.append(f"[{current_label}]setpts={1 / safe_speed:.4f}*PTS[timeline_speed]")
        current_label = "timeline_speed"

    if youtube_credit_textfile_path is not None:
        filters.append(
            f"[{current_label}]"
            f"{_build_youtube_credit_drawtext_filter(textfile_path=youtube_credit_textfile_path, youtube_credit_position=youtube_credit_position, font_file=font_file)}"
            "[timeline_credit]"
        )
        current_label = "timeline_credit"

    if subtitle_ass_path is not None:
        from .subtitles import build_subtitle_filter

        filters.append(f"[{current_label}]{build_subtitle_filter(subtitle_ass_path)}[vout]")
        current_label = "vout"

    if current_label != "vout":
        filters.append(f"[{current_label}]format=yuv420p,setsar=1[vout]")
        current_label = "vout"
    else:
        filters.append("[vout]format=yuv420p,setsar=1[vout_fmt]")
        current_label = "vout_fmt"

    return ";".join(filters), current_label, {"timeline_intervals": debug_intervals}


def _cropdetect_cache_path(input_video: Path) -> Path:
    if input_video.stem == "source":
        return input_video.parent / "cropdetect.json"
    return input_video.parent / f"{input_video.stem}_cropdetect.json"


def _load_cropdetect_cache(input_video: Path) -> tuple[int, int, int, int] | None:
    cache_path = _cropdetect_cache_path(input_video)
    if not cache_path.exists():
        return None

    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None

    rect_payload = payload.get("rect")
    if not isinstance(rect_payload, dict):
        return None

    try:
        stat = input_video.stat()
        cached_name = payload.get("source_name")
        cached_size = payload.get("source_size")
        cached_mtime_ns = payload.get("source_mtime_ns")
        if cached_name and str(cached_name) != input_video.name:
            return None
        if cached_size is not None and int(cached_size) != stat.st_size:
            return None
        if cached_mtime_ns is not None and int(cached_mtime_ns) != stat.st_mtime_ns:
            return None

        width = int(rect_payload["width"])
        height = int(rect_payload["height"])
        x = int(rect_payload["x"])
        y = int(rect_payload["y"])
    except (KeyError, OSError, TypeError, ValueError):
        return None

    if width <= 0 or height <= 0:
        return None
    return (width, height, x, y)


def _save_cropdetect_cache(input_video: Path, rect: tuple[int, int, int, int]) -> None:
    cache_path = _cropdetect_cache_path(input_video)
    try:
        stat = input_video.stat()
        payload = {
            "source_name": input_video.name,
            "source_size": stat.st_size,
            "source_mtime_ns": stat.st_mtime_ns,
            "rect": {
                "width": rect[0],
                "height": rect[1],
                "x": rect[2],
                "y": rect[3],
            },
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError:
        return


def _detect_content_rect(
    input_video: Path,
    ffmpeg_bin: str,
    log: Callable[[str], None] | None = None,
) -> tuple[int, int, int, int] | None:
    """Run cropdetect on a short sample of the video and return the modal crop rect.

    Returns (width, height, x, y) matching FFmpeg's crop=W:H:X:Y convention,
    or None on any failure.
    """
    log_fn = log or (lambda _: None)
    cached_rect = _load_cropdetect_cache(input_video)
    if cached_rect is not None:
        log_fn(f"autozoom: using cached crop rect from {_cropdetect_cache_path(input_video).name}")
        return cached_rect

    try:
        log_fn("autozoom: cache miss, running cropdetect on first 20s sample (every 100th frame)...")
        cmd = [
            ffmpeg_bin,
            "-hide_banner",
            "-ss", "0",
            "-i", str(input_video),
            "-t", "20",
            "-vf", "select=not(mod(n\\,100)),cropdetect=limit=16:round=2:reset=0",
            "-vsync", "vfr",
            "-f", "null", "-",
        ]
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        _CROP_RE = re.compile(r"crop=(\d+):(\d+):(\d+):(\d+)")
        hits: list[tuple[int, int, int, int]] = []
        for match in _CROP_RE.finditer(result.stderr):
            w, h, x, y = (
                int(match.group(1)),
                int(match.group(2)),
                int(match.group(3)),
                int(match.group(4)),
            )
            if w > 0 and h > 0:
                hits.append((w, h, x, y))
        if not hits:
            return None
        mode_rect, _ = Counter(hits).most_common(1)[0]
        _save_cropdetect_cache(input_video, mode_rect)
        return mode_rect
    except Exception:  # noqa: BLE001
        return None


def _finalize_filter_chain(filters: list[str], subtitle_ass_path: Path | str | None) -> str:
    """Append the subtitle burn-in filter (if any) and return the joined chain."""
    if subtitle_ass_path is not None:
        from .subtitles import build_subtitle_filter  # lazy — whisperx not required
        filters.append(build_subtitle_filter(subtitle_ass_path))
    return ",".join(filters)


def build_video_filter(
    part_number: int,
    crop_top_px: int,
    output_width: int,
    output_height: int,
    video_y_scale: float = 2.08,
    y_scale_mode: str = "letterbox",
    edge_bar_px: int = 45,
    content_height_bump_px: int = 0,
    content_max_height_px: int = 0,
    zoom_target_height: int | None = None,
    effective_y_scale: float | None = None,
    autozoom_rect: tuple[int, int, int, int] | None = None,
    render_preset: str = "legacy",
    title_mask_px: int = 0,
    part_overlay_enabled: bool = True,
    part_label_position: str = "top-center",
    label_x_pct: float = 0.5,
    label_y_pct: float = 0.05,
    part_label_x_percent: float | None = None,
    part_label_y_percent: float | None = None,
    font_file: Path | None = None,
    chapter_title: str = "",
    chapter_title_position: str = "top",
    manual_caption_text: str | None = None,
    overlay_x_percent: float = 50.0,
    overlay_y_percent: float = 12.0,
    playback_speed: float = 1.0,
    subtitle_ass_path: Path | str | None = None,
    textfile_dir: Path | None = None,
    show_youtube_credit: bool = False,
    youtube_credit_text: str | None = None,
    youtube_credit_position: str = "below_frame",
) -> str:
    if LOCKED_PRESET and render_preset != "legacy":
        raise ValueError(
            "LOCKED_PRESET=true: render_preset is locked to 'legacy'. "
            "Set LOCKED_PRESET=False in src/render.py to unlock."
        )
    if render_preset != "legacy":
        raise ValueError(f"Unsupported render_preset: {render_preset}")

    safe_video_y_scale = float(video_y_scale)
    if safe_video_y_scale <= 0:
        raise ValueError("video_y_scale must be greater than 0.")
    if y_scale_mode not in {"manual", "fill", "letterbox", "zoom", "autozoom"}:
        raise ValueError("y_scale_mode must be one of: manual, fill, letterbox, zoom, autozoom")
    safe_edge_bar_px = max(0, min(int(edge_bar_px), 200))
    safe_content_height_bump_px = max(-1600, min(int(content_height_bump_px), 1600))
    safe_content_max_height_px = 0
    if int(content_max_height_px) > 0:
        safe_content_max_height_px = _clamp_even(int(content_max_height_px), minimum=2, maximum=output_height)
    safe_crop_top_px = max(0, int(crop_top_px))
    safe_label_x_pct = _clamp_pct(label_x_pct, fallback=0.5)
    safe_label_y_pct = _clamp_pct(label_y_pct, fallback=0.05)
    safe_part_label_x_percent = _clamp_percent(
        part_label_x_percent if part_label_x_percent is not None else safe_label_x_pct * 100.0,
        fallback=50.0,
    )
    safe_part_label_y_percent = _clamp_percent(
        part_label_y_percent if part_label_y_percent is not None else safe_label_y_pct * 100.0,
        fallback=4.0,
    )
    safe_overlay_x_percent = _clamp_percent(overlay_x_percent, fallback=50.0)
    safe_overlay_y_percent = _clamp_percent(overlay_y_percent, fallback=12.0)
    if (
        not _normalize_overlay_text(manual_caption_text or "")
        and (chapter_title_position or "").lower() == "bottom"
        and math.isclose(safe_overlay_y_percent, 12.0, abs_tol=0.0001)
    ):
        safe_overlay_y_percent = 78.0
    caption_text = _normalize_overlay_text(manual_caption_text or "") or _normalize_overlay_text(chapter_title or "")
    drawtext_dir = textfile_dir or Path.cwd()
    caption_textfile_path: Path | None = None
    part_label_textfile_path: Path | None = None
    youtube_credit_textfile_path: Path | None = None
    if caption_text:
        written = write_drawtext_textfile(caption_text, drawtext_dir, f"caption_part_{part_number}.txt")
        caption_textfile_path = Path(written.name) if textfile_dir is not None else written.resolve()
    if part_overlay_enabled:
        written = write_drawtext_textfile(f"Part {part_number}", drawtext_dir, f"part_label_{part_number}.txt")
        part_label_textfile_path = Path(written.name) if textfile_dir is not None else written.resolve()
    normalized_youtube_credit = _normalize_overlay_text(youtube_credit_text or "")
    if show_youtube_credit and normalized_youtube_credit:
        written = write_drawtext_textfile(f"YT: {normalized_youtube_credit}", drawtext_dir, f"yt_credit_part_{part_number}.txt")
        youtube_credit_textfile_path = Path(written.name) if textfile_dir is not None else written.resolve()
    safe_speed = round(float(playback_speed), 2)
    if not (1.0 <= safe_speed <= 2.0):
        raise ValueError("playback_speed must be between 1.0 and 2.0.")
    filters: list[str] = []
    if safe_speed != 1.0:
        filters.append(f"setpts={1 / safe_speed:.4f}*PTS")

    if y_scale_mode == "letterbox":
        filters.append(f"scale={output_width}:-2")
        if safe_crop_top_px > 0:
            filters.append(
                f"crop=iw:max(2\\,ih-{safe_crop_top_px}):0:min({safe_crop_top_px}\\,ih-2)"
            )
        if safe_content_height_bump_px != 0:
            target_height_expr = (
                f"trunc(max(2\\,min(ih{safe_content_height_bump_px:+d}\\,{output_height}))/2)*2"
            )
            target_width_expr = f"trunc(min(iw\\,{output_width})/2)*2"
            filters.append(f"scale=-2:{target_height_expr}")
            filters.append(f"crop={target_width_expr}:ih:(iw-{target_width_expr})/2:0")
        filters.append(f"pad={output_width}:{output_height}:(ow-iw)/2:(oh-ih)/2:color=black")
        if safe_edge_bar_px > 0:
            filters.append(f"drawbox=x=0:y=0:w=iw:h={safe_edge_bar_px}:color=black@1.0:t=fill")
            filters.append(
                f"drawbox=x=0:y=ih-{safe_edge_bar_px}:w=iw:h={safe_edge_bar_px}:color=black@1.0:t=fill"
            )
        if youtube_credit_textfile_path is not None:
            filters.append(
                _build_youtube_credit_drawtext_filter(
                    textfile_path=youtube_credit_textfile_path,
                    youtube_credit_position=youtube_credit_position,
                    font_file=font_file,
                )
            )
        if caption_textfile_path is not None:
            filters.append(
                _build_chapter_drawtext_filter(
                    caption_textfile_path,
                    font_file,
                    safe_overlay_x_percent,
                    safe_overlay_y_percent,
                )
            )
        if part_label_textfile_path is not None:
            filters.append(
                _build_part_drawtext_filter(
                    textfile_path=part_label_textfile_path,
                    part_label_position=part_label_position,
                    label_x_pct=safe_label_x_pct,
                    label_y_pct=safe_label_y_pct,
                    part_label_x_percent=safe_part_label_x_percent,
                    part_label_y_percent=safe_part_label_y_percent,
                    font_file=font_file,
                )
            )
        filters.append("setsar=1")
        return _finalize_filter_chain(filters, subtitle_ass_path)

    if y_scale_mode == "zoom":
        filters.append(f"scale={output_width}:-2")
        if safe_crop_top_px > 0:
            filters.append(
                f"crop=iw:max(2\\,ih-{safe_crop_top_px}):0:min({safe_crop_top_px}\\,ih-2)"
            )
        if zoom_target_height is not None:
            zoom_max_h = safe_content_max_height_px if safe_content_max_height_px > 0 else output_height
            safe_zoom_target_height = _clamp_even(int(zoom_target_height), minimum=2, maximum=zoom_max_h)
            target_width_expr = f"trunc(min(iw\\,{output_width})/2)*2"
            filters.append(f"scale=-2:{safe_zoom_target_height}")
            filters.append(f"crop={target_width_expr}:{safe_zoom_target_height}:(iw-{target_width_expr})/2:0")
        else:
            target_width_expr = f"trunc(min(iw\\,{output_width})/2)*2"
            filters.append(
                f"scale=-2:trunc(max(2\\,min(ih{safe_content_height_bump_px:+d}\\,{output_height}))/2)*2"
            )
            filters.append(f"crop={target_width_expr}:ih:(iw-{target_width_expr})/2:0")
        filters.append(f"pad={output_width}:{output_height}:(ow-iw)/2:(oh-ih)/2:color=black")
        if safe_edge_bar_px > 0:
            filters.append(f"drawbox=x=0:y=0:w=iw:h={safe_edge_bar_px}:color=black@1.0:t=fill")
            filters.append(
                f"drawbox=x=0:y=ih-{safe_edge_bar_px}:w=iw:h={safe_edge_bar_px}:color=black@1.0:t=fill"
            )
        if youtube_credit_textfile_path is not None:
            filters.append(
                _build_youtube_credit_drawtext_filter(
                    textfile_path=youtube_credit_textfile_path,
                    youtube_credit_position=youtube_credit_position,
                    font_file=font_file,
                )
            )
        if caption_textfile_path is not None:
            filters.append(
                _build_chapter_drawtext_filter(
                    caption_textfile_path,
                    font_file,
                    safe_overlay_x_percent,
                    safe_overlay_y_percent,
                )
            )
        if part_label_textfile_path is not None:
            filters.append(
                _build_part_drawtext_filter(
                    textfile_path=part_label_textfile_path,
                    part_label_position=part_label_position,
                    label_x_pct=safe_label_x_pct,
                    label_y_pct=safe_label_y_pct,
                    part_label_x_percent=safe_part_label_x_percent,
                    part_label_y_percent=safe_part_label_y_percent,
                    font_file=font_file,
                )
            )
        filters.append("setsar=1")
        return _finalize_filter_chain(filters, subtitle_ass_path)

    if y_scale_mode == "autozoom":
        # Crop out the content region (removes black pillarbars), then scale/crop
        # to fill the output frame vertically.
        if autozoom_rect is not None:
            az_w, az_h, az_x, az_y = autozoom_rect
            filters.append(f"crop={az_w}:{az_h}:{az_x}:{az_y}")
        filters.append(f"scale={output_width}:-2")
        if safe_crop_top_px > 0:
            filters.append(
                f"crop=iw:max(2\\,ih-{safe_crop_top_px}):0:min({safe_crop_top_px}\\,ih-2)"
            )
        # Center-crop if content exceeds output height; pad if shorter.
        filters.append(
            f"crop={output_width}:min(ih\\,{output_height}):0:(ih-min(ih\\,{output_height}))/2"
        )
        filters.append(f"pad={output_width}:{output_height}:(ow-iw)/2:(oh-ih)/2:color=black")
        if safe_edge_bar_px > 0:
            filters.append(f"drawbox=x=0:y=0:w=iw:h={safe_edge_bar_px}:color=black@1.0:t=fill")
            filters.append(
                f"drawbox=x=0:y=ih-{safe_edge_bar_px}:w=iw:h={safe_edge_bar_px}:color=black@1.0:t=fill"
            )
        if youtube_credit_textfile_path is not None:
            filters.append(
                _build_youtube_credit_drawtext_filter(
                    textfile_path=youtube_credit_textfile_path,
                    youtube_credit_position=youtube_credit_position,
                    font_file=font_file,
                )
            )
        if caption_textfile_path is not None:
            filters.append(
                _build_chapter_drawtext_filter(
                    caption_textfile_path,
                    font_file,
                    safe_overlay_x_percent,
                    safe_overlay_y_percent,
                )
            )
        if part_label_textfile_path is not None:
            filters.append(
                _build_part_drawtext_filter(
                    textfile_path=part_label_textfile_path,
                    part_label_position=part_label_position,
                    label_x_pct=safe_label_x_pct,
                    label_y_pct=safe_label_y_pct,
                    part_label_x_percent=safe_part_label_x_percent,
                    part_label_y_percent=safe_part_label_y_percent,
                    font_file=font_file,
                )
            )
        filters.append("setsar=1")
        return _finalize_filter_chain(filters, subtitle_ass_path)

    # Legacy preset filter chain:
    # width fit -> vertical-only scale -> vertical crop -> pad -> top/bottom edge bars -> drawtext -> setsar
    filters.append(f"scale={output_width}:-2")
    if effective_y_scale is not None:
        scale_factor_expr = f"{float(effective_y_scale):g}"
    elif y_scale_mode == "fill":
        scale_factor_expr = f"max({safe_video_y_scale:g}\\,{output_height + (2 * FILL_TRIM_PX)}/ih)"
    else:
        scale_factor_expr = f"{safe_video_y_scale:g}"
    filters.append(f"scale=iw:trunc(ih*{scale_factor_expr}/2)*2")
    if y_scale_mode == "fill":
        filters.append(
            f"crop={output_width}:min(ih\\,{output_height + (2 * FILL_TRIM_PX)}):0:(ih-min(ih\\,{output_height + (2 * FILL_TRIM_PX)}))/2"
        )
        filters.append(
            f"crop={output_width}:{output_height}:0:{FILL_TRIM_PX}"
        )
    else:
        filters.append(
            f"crop={output_width}:min(ih\\,{output_height}):0:(ih-min(ih\\,{output_height}))/2"
        )
    filters.append(f"pad={output_width}:{output_height}:(ow-iw)/2:(oh-ih)/2")
    if safe_edge_bar_px > 0:
        filters.append(f"drawbox=x=0:y=0:w=iw:h={safe_edge_bar_px}:color=black@1.0:t=fill")
        filters.append(
            f"drawbox=x=0:y=ih-{safe_edge_bar_px}:w=iw:h={safe_edge_bar_px}:color=black@1.0:t=fill"
        )
    if title_mask_px > 0:
        filters.append(f"drawbox=x=0:y=0:w=iw:h={int(title_mask_px)}:color=black@1.0:t=fill")

    if youtube_credit_textfile_path is not None:
        filters.append(
            _build_youtube_credit_drawtext_filter(
                textfile_path=youtube_credit_textfile_path,
                youtube_credit_position=youtube_credit_position,
                font_file=font_file,
            )
        )

    if caption_textfile_path is not None:
        filters.append(
            _build_chapter_drawtext_filter(
                caption_textfile_path,
                font_file,
                safe_overlay_x_percent,
                safe_overlay_y_percent,
            )
        )

    if part_label_textfile_path is not None:
        filters.append(
            _build_part_drawtext_filter(
                textfile_path=part_label_textfile_path,
                part_label_position=part_label_position,
                label_x_pct=safe_label_x_pct,
                label_y_pct=safe_label_y_pct,
                part_label_x_percent=safe_part_label_x_percent,
                part_label_y_percent=safe_part_label_y_percent,
                font_file=font_file,
            )
        )

    # Force square pixels for consistent platform detection.
    filters.append("setsar=1")

    return _finalize_filter_chain(filters, subtitle_ass_path)


def render_parts(
    input_video: Path,
    out_dir: Path,
    segments: list[Segment],
    crop_top_px: int = 0,
    output_width: int = 1080,
    output_height: int = 1920,
    video_y_scale: float = 2.08,
    y_scale_mode: str = "letterbox",
    edge_bar_px: int = 45,
    content_height_bump_px: int = 0,
    content_max_height_px: int = 0,
    render_preset: str = "legacy",
    title_mask_px: int = 0,
    raise_px: int | None = None,
    bottom_padding: int | None = None,
    part_overlay_enabled: bool = True,
    part_label_position: str = "top-center",
    label_x_pct: float = 0.5,
    label_y_pct: float = 0.05,
    part_label_x_percent: float | None = None,
    part_label_y_percent: float | None = None,
    font_file: Path | None = None,
    ffmpeg_bin: str = "ffmpeg",
    ffprobe_bin: str = "ffprobe",
    crf: int = 18,
    preset: str = "slow",
    log: Callable[[str], None] | None = None,
    chapter_titles: list[str] | None = None,
    chapter_title_position: str = "top",
    manual_caption_text: str | None = None,
    overlay_x_percent: float = 50.0,
    overlay_y_percent: float = 12.0,
    playback_speed: float = 1.0,
    subtitles_enabled: bool = False,
    subtitle_style: str = "hormozi",
    subtitle_language: str | None = None,
    subtitle_offset_seconds: float = 0.0,
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
    reaction_timeline: list[dict[str, Any]] | None = None,
    segment_metadata: list[dict[str, Any]] | None = None,
    source_info: dict[str, Any] | None = None,
    hashtags: str = "",
    base_title: str = "",
    show_youtube_credit: bool = False,
    youtube_credit_text: str | None = None,
    youtube_credit_position: str = "below_frame",
) -> list[RenderedPart]:
    out_dir = out_dir.resolve()
    input_video = input_video.resolve()
    if font_file is not None:
        font_file = font_file.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    log_fn = log or print

    if ffmpeg_bin == "ffmpeg":
        ffmpeg_bin = _resolve_binary("ffmpeg")
    if ffprobe_bin == "ffprobe":
        ffprobe_bin = _resolve_binary("ffprobe")

    if reaction_layout_enabled:
        y_scale_mode = "reaction_layout"
    if y_scale_mode not in {"manual", "fill", "letterbox", "zoom", "autozoom", "reaction_layout"}:
        raise ValueError("y_scale_mode must be one of: manual, fill, letterbox, zoom, autozoom, reaction_layout")

    safe_edge_bar_px = max(0, min(int(edge_bar_px), 200))
    safe_content_height_bump_px = max(-1600, min(int(content_height_bump_px), 1600))
    safe_content_max_height_px = 0
    if int(content_max_height_px) > 0:
        safe_content_max_height_px = _clamp_even(int(content_max_height_px), minimum=2, maximum=output_height)
    safe_label_x_pct = _clamp_pct(label_x_pct, fallback=0.5)
    safe_label_y_pct = _clamp_pct(label_y_pct, fallback=0.05)
    safe_part_label_x_percent = _clamp_percent(
        part_label_x_percent if part_label_x_percent is not None else safe_label_x_pct * 100.0,
        fallback=50.0,
    )
    safe_part_label_y_percent = _clamp_percent(
        part_label_y_percent if part_label_y_percent is not None else safe_label_y_pct * 100.0,
        fallback=4.0,
    )
    safe_overlay_x_percent = _clamp_percent(overlay_x_percent, fallback=50.0)
    safe_overlay_y_percent = _clamp_percent(overlay_y_percent, fallback=12.0)
    normalized_manual_caption = _normalize_overlay_text(manual_caption_text or "")
    normalized_reaction_caption = _normalize_overlay_text(caption_text or "")
    normalized_hashtags = re.sub(r"\s+", " ", str(hashtags or "")).strip()
    source_channel_credit = ""
    if isinstance(source_info, dict):
        source_channel_credit = _normalize_overlay_text(source_info.get("channel") or source_info.get("uploader") or "")
    normalized_youtube_credit = _normalize_overlay_text(youtube_credit_text or "") or source_channel_credit
    timeline_fallback = {
        "start": "00:00:00",
        "end": "00:00:01",
        "caption": normalized_reaction_caption,
        "layout_preset": reaction_layout_preset,
        "keep_aspect_ratio": True,
        "caption_enabled": bool(normalized_reaction_caption),
        "caption_duration_mode": caption_duration_mode,
        "main_crop": main_crop or {"x_percent": 0, "y_percent": 0, "width_percent": 100, "height_percent": 65},
        "facecam_crop": facecam_crop or {"x_percent": 60, "y_percent": 55, "width_percent": 35, "height_percent": 40},
        "facecam_shape": facecam_shape,
        "caption_position": caption_position,
    }
    normalized_reaction_timeline: list[dict[str, Any]] = []
    if isinstance(reaction_timeline, list):
        normalized_reaction_timeline = [
            _normalize_timeline_row(row, fallback=timeline_fallback)
            for row in reaction_timeline
            if isinstance(row, dict)
        ]
    normalized_segment_metadata = [
        item if isinstance(item, dict) else {}
        for item in (segment_metadata or [])
    ]
    safe_speed = round(float(playback_speed), 2)
    if not (1.0 <= safe_speed <= 2.0):
        raise ValueError("playback_speed must be between 1.0 and 2.0.")
    if safe_speed != 1.0:
        if safe_speed <= 2.0:
            atempo_filter = f"atempo={safe_speed:.2f}"
        else:
            atempo_filter = f"atempo=2.0,atempo={safe_speed / 2.0:.2f}"
    else:
        atempo_filter = ""

    rendered_parts: list[RenderedPart] = []
    segment_rows = [
        {
            "part_number": idx,
            "start_seconds": round(seg.start, 3),
            "end_seconds": round(seg.end, 3),
            "duration_seconds": round(seg.duration, 3),
            "start_time": format_ffmpeg_time(seg.start),
            "end_time": format_ffmpeg_time(seg.end),
        }
        for idx, seg in enumerate(segments, start=1)
    ]
    part_commands: list[dict[str, Any]] = []

    y_scale_debug: dict[str, float | str] | None = None
    effective_y_scale_for_filter: float | None = None
    zoom_target_height_for_filter: int | None = None
    zoom_debug: dict[str, int | float] | None = None
    autozoom_rect: tuple[int, int, int, int] | None = None
    source_dims = _probe_video_dimensions(input_video=input_video, ffprobe_bin=ffprobe_bin)
    if source_dims is not None:
        y_scale_debug = _compute_y_scale_debug(
            source_width=source_dims[0],
            source_height=source_dims[1],
            output_width=output_width,
            output_height=output_height,
            video_y_scale=video_y_scale,
            y_scale_mode=y_scale_mode,
        )
        if y_scale_mode == "fill":
            effective_y_scale_for_filter = float(y_scale_debug["effective_y_scale"])
        if y_scale_mode == "zoom":
            zoom_debug = _compute_zoom_target_height(
                source_width=source_dims[0],
                source_height=source_dims[1],
                crop_top_px=int(crop_top_px),
                output_width=output_width,
                output_height=output_height,
                content_height_bump_px=safe_content_height_bump_px,
                content_max_height_px=safe_content_max_height_px,
            )
            zoom_target_height_for_filter = int(zoom_debug["target_height"])
        log_fn(
            "y_scale_debug: "
            f"base_height={y_scale_debug['base_height']:.3f}, "
            f"required_fill_scale={y_scale_debug['required_fill_scale']:.6f}, "
            f"video_y_scale_requested={y_scale_debug['video_y_scale_requested']:.6f}, "
            f"effective_y_scale={y_scale_debug['effective_y_scale']:.6f}, "
            f"y_scale_mode={y_scale_debug['y_scale_mode']}"
        )
    else:
        log_fn("y_scale_debug: source dimensions unavailable from ffprobe; skipping computed fill metrics.")
    log_fn(
        f"render_config: y_scale_mode={y_scale_mode}, edge_bar_px={safe_edge_bar_px}, "
        f"content_height_bump_px={safe_content_height_bump_px}, content_max_height_px={safe_content_max_height_px}, "
        f"crop_top_px={crop_top_px}, title_mask_px={title_mask_px}, "
        f"part_label_position={part_label_position}, label_x_pct={safe_label_x_pct:.4f}, label_y_pct={safe_label_y_pct:.4f}, "
        f"part_label_x_percent={safe_part_label_x_percent:.2f}, part_label_y_percent={safe_part_label_y_percent:.2f}, "
        f"overlay_x_percent={safe_overlay_x_percent:.2f}, overlay_y_percent={safe_overlay_y_percent:.2f}, "
        f"manual_caption_text={'yes' if normalized_manual_caption else 'no'}, "
        f"hashtags={'yes' if normalized_hashtags else 'no'}, "
        f"youtube_credit={'yes' if show_youtube_credit and normalized_youtube_credit else 'no'}, "
        f"reaction_layout_enabled={y_scale_mode == 'reaction_layout'}, "
        f"reaction_timeline_rows={len(normalized_reaction_timeline)}"
    )
    if y_scale_mode == "reaction_layout" and facecam_shape in {"circle", "rounded_rectangle"}:
        log_fn(f"{facecam_shape.replace('_', ' ').title()} facecam mask not yet supported in FFmpeg output; using rectangle.")
    if y_scale_mode == "zoom" and zoom_debug is not None:
        log_fn(
            "zoom_debug: "
            f"base_height={zoom_debug['base_height']}, "
            f"target_height={zoom_debug['target_height']}, "
            f"content_max_height_px={zoom_debug['content_max_height_px']}, "
            f"content_height_bump_px={zoom_debug['content_height_bump_px']}"
        )
    if y_scale_mode == "autozoom":
        log_fn("autozoom: resolving crop rect from cache or local cropdetect analysis...")
        autozoom_rect = _detect_content_rect(
            input_video=input_video,
            ffmpeg_bin=ffmpeg_bin,
            log=log_fn,
        )
        if autozoom_rect is not None:
            az_w, az_h, az_x, az_y = autozoom_rect
            log_fn(f"autozoom: detected content rect crop={az_w}:{az_h}:{az_x}:{az_y}")
        else:
            log_fn("autozoom: cropdetect failed or found no crop; falling back to zoom mode.")
            y_scale_mode = "zoom"

    all_words: list[dict] | None = None
    if subtitles_enabled:
        try:
            from .subtitles import resolve_subtitles  # lazy import

            all_words = resolve_subtitles(
                input_video=input_video,
                info_dict=source_info,
                subtitle_language=subtitle_language,
                use_cache=True,
                ffmpeg_bin=ffmpeg_bin,
                subtitle_offset_seconds=subtitle_offset_seconds,
                log=log_fn,
            )
            if all_words:
                log_fn(f"Subtitles ready: {len(all_words)} word(s).")
            else:
                log_fn("Subtitles unavailable, continuing without burned captions.")
        except Exception as exc:  # noqa: BLE001
            log_fn(f"Subtitle resolution failed, continuing without subtitles: {exc}")
            all_words = None

    for idx, segment in enumerate(segments, start=1):
        if segment.duration <= 0:
            raise ValueError(f"Segment {idx} duration must be > 0.")

        output_path = out_dir / f"part_{idx}.mp4"
        part_metadata = normalized_segment_metadata[idx - 1] if idx <= len(normalized_segment_metadata) else {}
        segment_caption = _normalize_overlay_text(
            part_metadata.get("caption_text")
            or part_metadata.get("title")
            or ""
        )
        segment_layout = str(part_metadata.get("suggested_layout") or "").strip().lower()
        segment_render_preset = str(part_metadata.get("render_preset") or "").strip() or render_preset
        if LOCKED_PRESET and segment_render_preset != "legacy":
            segment_render_preset = "legacy"
        part_subtitle_style = str(part_metadata.get("subtitle_style") or subtitle_style).strip()
        if part_subtitle_style not in {"hormozi", "standard", "minimal"}:
            part_subtitle_style = subtitle_style
        segment_y_scale_mode = y_scale_mode
        if segment_layout in {"reaction_layout", "split_screen"} and (reaction_layout_enabled or main_crop or facecam_crop or normalized_reaction_timeline):
            segment_y_scale_mode = "reaction_layout"
        elif segment_layout == "full_screen" and y_scale_mode != "reaction_layout":
            segment_y_scale_mode = "fill"
        elif segment_layout == "gameplay" and y_scale_mode != "reaction_layout":
            segment_y_scale_mode = "zoom"
        part_reaction_caption = segment_caption or normalized_reaction_caption
        part_title = _normalize_overlay_text(
            part_metadata.get("title")
            or (chapter_titles[idx - 1] if chapter_titles and idx <= len(chapter_titles) else "")
            or normalized_manual_caption
            or segment_caption
            or normalized_reaction_caption
            or base_title
            or f"Part {idx}"
        )
        upload_description = _build_upload_description(part_title, idx, normalized_hashtags)
        suggested_filename = f"part_{idx}_{_sanitize_part_filename_title(part_title)}.mp4"

        subtitle_ass_path: Path | None = None
        if subtitles_enabled and all_words is not None:
            try:
                from .subtitles import build_ass_subtitles, slice_words_for_segment  # lazy import

                segment_words = slice_words_for_segment(
                    all_words,
                    segment_start=segment.start,
                    segment_end=segment.end,
                    subtitle_offset_seconds=subtitle_offset_seconds,
                )
                if segment_words:
                    ass_out = out_dir / f"part_{idx}.ass"
                    subtitle_ass_path = build_ass_subtitles(
                        segment_words,
                        ass_out,
                        style=part_subtitle_style,
                        clip_duration=segment.duration,
                    )
                    log_fn(f"ASS subtitles written (part {idx}): {ass_out.name}")
            except Exception as exc:  # noqa: BLE001
                log_fn(f"Subtitle generation failed for part {idx}, continuing without: {exc}")
                subtitle_ass_path = None

        subtitle_filter_target: str | None = None
        ffmpeg_cwd: Path | None = out_dir
        if subtitle_ass_path is not None:
            subtitle_filter_target = subtitle_ass_path.name

        youtube_credit_textfile_path: Path | None = None
        if show_youtube_credit and normalized_youtube_credit:
            written_credit = write_drawtext_textfile(f"YT: {normalized_youtube_credit}", out_dir, f"yt_credit_part_{idx}.txt")
            youtube_credit_textfile_path = Path(written_credit.name)

        reaction_debug: dict[str, Any] = {}
        if segment_y_scale_mode == "reaction_layout":
            if source_dims is None:
                raise RuntimeError("Reaction layout requires source dimensions from ffprobe.")
            if normalized_reaction_timeline:
                fallback_row = _fallback_timeline_row(
                    segment=segment,
                    main_crop=main_crop,
                    facecam_crop=facecam_crop,
                    reaction_layout_preset=reaction_layout_preset,
                    facecam_shape=facecam_shape,
                    caption_text=part_reaction_caption,
                    caption_position=caption_position,
                    caption_duration_mode=caption_duration_mode,
                    caption_duration_seconds=caption_duration_seconds,
                )
                intervals = _timeline_intervals_for_segment(
                    segment=segment,
                    timeline_rows=normalized_reaction_timeline,
                    fallback_row=fallback_row,
                )
                caption_pngs: list[Path] = []
                caption_input_indexes: dict[int, int] = {}
                for interval_index, interval in enumerate(intervals):
                    row = interval["row"]
                    row_caption = _normalize_overlay_text(row.get("caption", "")) or segment_caption
                    if not row_caption or not bool(row.get("caption_enabled", True)):
                        continue
                    style_params = _caption_style_params(row.get("caption_style"))
                    caption_path = out_dir / f"part_{idx:03d}_tl_{interval_index:02d}_caption.png"
                    create_caption_overlay_png(
                        row_caption,
                        caption_path,
                        output_width=output_width,
                        **style_params,
                    )
                    caption_pngs.append(caption_path)
                    caption_input_indexes[interval_index] = len(caption_pngs)
                video_filter, video_map, reaction_debug = _build_reaction_timeline_filter_complex(
                    source_width=source_dims[0],
                    source_height=source_dims[1],
                    output_width=output_width,
                    output_height=output_height,
                    intervals=intervals,
                    caption_input_indexes=caption_input_indexes,
                    overlay_y_percent=safe_overlay_y_percent,
                    playback_speed=safe_speed,
                    subtitle_ass_path=subtitle_filter_target,
                    youtube_credit_textfile_path=youtube_credit_textfile_path,
                    youtube_credit_position=youtube_credit_position,
                    font_file=font_file,
                )
                caption_inputs: list[str] = []
                for caption_png in caption_pngs:
                    caption_inputs.extend(["-loop", "1", "-i", str(caption_png)])
                cmd = [
                    ffmpeg_bin,
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-ss",
                    format_ffmpeg_time(segment.start),
                    "-t",
                    format_ffmpeg_time(segment.duration),
                    "-i",
                    str(input_video),
                    *caption_inputs,
                    "-filter_complex",
                    video_filter,
                    "-map",
                    f"[{video_map}]",
                    "-map",
                    "0:a?",
                    "-c:v",
                    "libx264",
                    "-preset",
                    preset,
                    "-crf",
                    str(crf),
                    "-pix_fmt",
                    "yuv420p",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "192k",
                    "-movflags",
                    "+faststart",
                    *((["-af", atempo_filter]) if atempo_filter else []),
                    str(output_path),
                ]
            else:
                caption_png: Path | None = None
                caption_input_index: int | None = None
                if part_reaction_caption:
                    caption_png = create_caption_overlay_png(
                        part_reaction_caption,
                        out_dir / f"part_{idx:03d}_caption.png",
                        output_width=output_width,
                    )
                    caption_input_index = 1
                video_filter, video_map, reaction_debug = _build_reaction_filter_complex(
                    source_width=source_dims[0],
                    source_height=source_dims[1],
                    output_width=output_width,
                    output_height=output_height,
                    main_crop=main_crop,
                    facecam_crop=facecam_crop,
                    reaction_layout_preset=reaction_layout_preset,
                    caption_text=part_reaction_caption,
                    caption_input_index=caption_input_index,
                    caption_position=caption_position,
                    caption_duration_mode=caption_duration_mode,
                    caption_duration_seconds=caption_duration_seconds,
                    overlay_y_percent=safe_overlay_y_percent,
                    playback_speed=safe_speed,
                    subtitle_ass_path=subtitle_filter_target,
                    youtube_credit_textfile_path=youtube_credit_textfile_path,
                    youtube_credit_position=youtube_credit_position,
                    font_file=font_file,
                )
                cmd = [
                    ffmpeg_bin,
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    str(input_video),
                    *((["-loop", "1", "-i", str(caption_png)]) if caption_png else []),
                    "-ss",
                    format_ffmpeg_time(segment.start),
                    "-t",
                    format_ffmpeg_time(segment.duration),
                    "-filter_complex",
                    video_filter,
                    "-map",
                    f"[{video_map}]",
                    "-map",
                    "0:a?",
                    "-c:v",
                    "libx264",
                    "-preset",
                    preset,
                    "-crf",
                    str(crf),
                    "-pix_fmt",
                    "yuv420p",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "192k",
                    "-movflags",
                    "+faststart",
                    *((["-af", atempo_filter]) if atempo_filter else []),
                    str(output_path),
                ]
        else:
            video_filter = build_video_filter(
                part_number=idx,
                crop_top_px=crop_top_px,
                output_width=output_width,
                output_height=output_height,
                video_y_scale=video_y_scale,
                y_scale_mode=segment_y_scale_mode,
                edge_bar_px=safe_edge_bar_px,
                content_height_bump_px=safe_content_height_bump_px,
                content_max_height_px=safe_content_max_height_px,
                zoom_target_height=zoom_target_height_for_filter,
                effective_y_scale=effective_y_scale_for_filter,
                autozoom_rect=autozoom_rect,
                render_preset=segment_render_preset,
                title_mask_px=title_mask_px,
                part_overlay_enabled=part_overlay_enabled,
                part_label_position=part_label_position,
                label_x_pct=safe_label_x_pct,
                label_y_pct=safe_label_y_pct,
                part_label_x_percent=safe_part_label_x_percent,
                part_label_y_percent=safe_part_label_y_percent,
                font_file=font_file,
                chapter_title=segment_caption or (chapter_titles[idx - 1] if chapter_titles and idx <= len(chapter_titles) else ""),
                chapter_title_position=chapter_title_position,
                manual_caption_text=normalized_manual_caption or None,
                overlay_x_percent=safe_overlay_x_percent,
                overlay_y_percent=safe_overlay_y_percent,
                playback_speed=safe_speed,
                subtitle_ass_path=subtitle_filter_target,
                textfile_dir=out_dir,
                show_youtube_credit=show_youtube_credit,
                youtube_credit_text=normalized_youtube_credit,
                youtube_credit_position=youtube_credit_position,
            )

            cmd = [
                ffmpeg_bin,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(input_video),
                "-ss",
                format_ffmpeg_time(segment.start),
                "-t",
                format_ffmpeg_time(segment.duration),
                "-vf",
                video_filter,
                "-c:v",
                "libx264",
                "-preset",
                preset,
                "-crf",
                str(crf),
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-movflags",
                "+faststart",
                *((["-af", atempo_filter]) if atempo_filter else []),
                str(output_path),
            ]
        cmd_path = out_dir / f"part_{idx:03d}.ffmpeg.txt"
        cmd_run_path = out_dir / f"part_{idx:03d}.render.cmd"
        _write_part_command_dump(cmd_path, cmd, cwd=ffmpeg_cwd)
        _write_windows_cmd(cmd_run_path, cmd, cwd=ffmpeg_cwd)

        command_shell_text = _command_to_shell_text(cmd)
        log_fn(f"FFmpeg {'filter_complex' if segment_y_scale_mode == 'reaction_layout' else '-vf'} (part {idx}): {video_filter}")
        if ffmpeg_cwd is not None:
            log_fn(f"FFmpeg cwd (part {idx}): {ffmpeg_cwd}")
        log_fn(f"FFmpeg cmd (part {idx}): {command_shell_text}")

        _run_command(cmd, cwd=ffmpeg_cwd)

        tiktok_probe = _safe_probe_tiktok_fields(output_path, ffprobe_bin)
        vf_diag = _build_vf_diagnostics(video_filter, output_width=output_width, output_height=output_height)
        tiktok_risk_flags = _build_tiktok_risk_flags(
            probe=tiktok_probe,
            vf_diag=vf_diag,
            output_width=output_width,
            output_height=output_height,
        )
        tiktok_probe_block = {
            **tiktok_probe,
            "filter_diagnostics": vf_diag,
            "risk_flags": tiktok_risk_flags,
        }
        if tiktok_risk_flags:
            log_fn(f"TikTok risk flags (part {idx}): {'; '.join(tiktok_risk_flags)}")
        else:
            log_fn(f"TikTok risk flags (part {idx}): none")

        part_commands.append(
            {
                "part_number": idx,
                "title": part_title,
                "upload_description": upload_description,
                "hashtags": normalized_hashtags,
                "suggested_filename": suggested_filename,
                "vf": video_filter,
                "ffmpeg_cmd": cmd,
                "ffmpeg_cmd_shell": command_shell_text,
                "ffmpeg_cmd_dump_path": str(cmd_path.resolve()),
                "ffmpeg_cmd_run_path": str(cmd_run_path.resolve()),
                "output_path": str(output_path.resolve()),
                "start_time": format_ffmpeg_time(segment.start),
                "end_time": format_ffmpeg_time(segment.end),
                "part_label_position": part_label_position,
                "label_x_pct": safe_label_x_pct,
                "label_y_pct": safe_label_y_pct,
                "part_label_x_percent": safe_part_label_x_percent,
                "part_label_y_percent": safe_part_label_y_percent,
                "manual_caption_text": normalized_manual_caption,
                "caption_text": part_reaction_caption if segment_y_scale_mode == "reaction_layout" else segment_caption,
                "segment_metadata": part_metadata,
                "suggested_layout": segment_layout,
                "subtitle_style": part_subtitle_style,
                "overlay_x_percent": safe_overlay_x_percent,
                "overlay_y_percent": safe_overlay_y_percent,
                "reaction_layout_enabled": segment_y_scale_mode == "reaction_layout",
                "reaction_layout_preset": reaction_layout_preset,
                "facecam_shape": facecam_shape,
                "reaction_timeline_rows": len(normalized_reaction_timeline),
                "reaction_layout_debug": reaction_debug,
                "show_youtube_credit": show_youtube_credit,
                "youtube_credit_text": normalized_youtube_credit,
                "youtube_credit_position": youtube_credit_position,
                "tiktok_probe": tiktok_probe_block,
            }
        )
        rendered_parts.append(
            RenderedPart(
                part_number=idx,
                start=segment.start,
                end=segment.end,
                path=output_path,
                start_time=format_ffmpeg_time(segment.start),
                end_time=format_ffmpeg_time(segment.end),
                vf=video_filter,
                ffmpeg_cmd=cmd,
                ffmpeg_cmd_path=cmd_path,
                ffmpeg_cmd_run_path=cmd_run_path,
            )
        )

    known_good_report: dict[str, Any] | None = None
    known_good_path = KNOWN_GOOD_REFERENCE.resolve()
    rendered_part1 = out_dir / "part_1.mp4"
    if known_good_path.exists() and rendered_part1.exists():
        known_good_stats = _safe_probe_short(known_good_path, ffprobe_bin)
        rendered_stats = _safe_probe_short(rendered_part1, ffprobe_bin)
        known_good_report = {
            "known_good_file": str(known_good_path),
            "rendered_file": str(rendered_part1.resolve()),
            "known_good_stats": known_good_stats,
            "rendered_stats": rendered_stats,
            "short_diff": _short_probe_diff(known_good_stats, rendered_stats),
            "known_good_tiktok_probe": _safe_probe_tiktok_fields(known_good_path, ffprobe_bin),
            "rendered_tiktok_probe": _safe_probe_tiktok_fields(rendered_part1, ffprobe_bin),
        }
        known_good_report["tiktok_probe_diff"] = _tiktok_probe_diff(
            known_good_report["known_good_tiktok_probe"],
            known_good_report["rendered_tiktok_probe"],
        )
        for line in known_good_report["short_diff"]:
            log_fn(f"known_good_diff: {line}")
        for line in known_good_report["tiktok_probe_diff"]:
            log_fn(f"known_good_tiktok_diff: {line}")

    manifest = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input_source": str(input_video.resolve()),
        "output_dir": str(out_dir.resolve()),
        "render_preset": render_preset,
        "render_params": {
            "crop_top_px": crop_top_px,
            "title_mask_px": title_mask_px,
            "output_width": output_width,
            "output_height": output_height,
            "video_y_scale": video_y_scale,
            "y_scale_mode": y_scale_mode,
            "edge_bar_px": safe_edge_bar_px,
            "content_height_bump_px": safe_content_height_bump_px,
            "content_max_height_px": safe_content_max_height_px,
            "zoom_target_height": (zoom_debug.get("target_height") if zoom_debug else None),
            "zoom_base_height": (zoom_debug.get("base_height") if zoom_debug else None),
            "base_height": (
                y_scale_debug.get("base_height")
                if y_scale_debug is not None
                else None
            ),
            "required_fill_scale": (
                y_scale_debug.get("required_fill_scale")
                if y_scale_debug is not None
                else None
            ),
            "effective_y_scale": (
                y_scale_debug.get("effective_y_scale")
                if y_scale_debug is not None
                else None
            ),
            "computed_required_fill_scale": (
                y_scale_debug.get("computed_required_fill_scale")
                if y_scale_debug is not None
                else None
            ),
            "effective_y_scale_used": (
                y_scale_debug.get("effective_y_scale_used")
                if y_scale_debug is not None
                else None
            ),
            "ih_after_fit": (
                y_scale_debug.get("ih_after_fit")
                if y_scale_debug is not None
                else None
            ),
            "raise_px": raise_px,
            "bottom_padding": bottom_padding,
            "locked_preset": LOCKED_PRESET,
            "part_overlay_enabled": part_overlay_enabled,
            "part_label_position": part_label_position,
            "label_x_pct": safe_label_x_pct,
            "label_y_pct": safe_label_y_pct,
            "part_label_x_percent": safe_part_label_x_percent,
            "part_label_y_percent": safe_part_label_y_percent,
            "manual_caption_text": normalized_manual_caption,
            "hashtags": normalized_hashtags,
            "overlay_x_percent": safe_overlay_x_percent,
            "overlay_y_percent": safe_overlay_y_percent,
            "show_youtube_credit": show_youtube_credit,
            "youtube_credit_text": normalized_youtube_credit,
            "youtube_credit_position": youtube_credit_position,
            "reaction_layout_enabled": y_scale_mode == "reaction_layout",
            "reaction_layout_mode": reaction_layout_mode,
            "reaction_layout_preset": reaction_layout_preset,
            "main_crop": main_crop or {},
            "facecam_crop": facecam_crop or {},
            "facecam_shape": facecam_shape,
            "reaction_timeline": normalized_reaction_timeline,
            "segment_metadata": normalized_segment_metadata,
            "caption_text": normalized_reaction_caption,
            "caption_position": caption_position,
            "caption_duration_mode": caption_duration_mode,
            "caption_duration_seconds": caption_duration_seconds,
            "crf": crf,
            "preset": preset,
            "playback_speed": safe_speed,
            "subtitles_enabled": subtitles_enabled,
            "subtitle_style": subtitle_style,
            "subtitle_offset_seconds": subtitle_offset_seconds,
        },
        "segments": segment_rows,
        "resolved_filter_chain": [item["vf"] for item in part_commands],
        "tiktok_probe": [item.get("tiktok_probe") for item in part_commands],
        "parts": part_commands,
        "ffmpeg_version": _safe_ffmpeg_version(ffmpeg_bin),
        "ffprobe_input": _safe_probe_json(input_video, ffprobe_bin),
        "y_scale_debug": y_scale_debug,
        "zoom_debug": zoom_debug,
        "known_good_comparison": known_good_report,
    }
    (out_dir / "render_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return rendered_parts


def rendered_parts_to_dict(parts: list[RenderedPart]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in parts:
        row: dict[str, Any] = {
            "part_number": item.part_number,
            "start_seconds": round(item.start, 3),
            "end_seconds": round(item.end, 3),
            "path": str(item.path),
        }
        if item.vf:
            row["vf"] = item.vf
        if item.ffmpeg_cmd is not None:
            row["ffmpeg_cmd"] = item.ffmpeg_cmd
        if item.ffmpeg_cmd_path is not None:
            row["ffmpeg_cmd_path"] = str(item.ffmpeg_cmd_path)
        if item.ffmpeg_cmd_run_path is not None:
            row["ffmpeg_cmd_run_path"] = str(item.ffmpeg_cmd_run_path)
        if item.start_time:
            row["start_time"] = item.start_time
        if item.end_time:
            row["end_time"] = item.end_time
        rows.append(row)
    return rows
