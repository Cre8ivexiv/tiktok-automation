from __future__ import annotations

import json
import math
import os
import shlex
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

LOCKED_PRESET = True
KNOWN_GOOD_REFERENCE = Path("clips_2026_02_17_03_02_36") / "part_1.mp4"
FILL_TRIM_PX = 30


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


def _run_command(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        command_text = " ".join(shlex.quote(item) for item in cmd)
        raise RuntimeError(
            f"Command failed ({result.returncode}): {command_text}\n"
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


def _write_part_command_dump(path: Path, cmd: list[str]) -> None:
    shell_text = _command_to_shell_text(cmd)
    lines = [
        shell_text,
        "",
        "Args JSON:",
        json.dumps(cmd, indent=2),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_windows_cmd(path: Path, cmd: list[str]) -> None:
    shell_text = _command_to_shell_text(cmd)
    lines = [
        "@echo off",
        shell_text,
        "",
    ]
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
    target_h = _clamp_even(base_h + max(0, content_height_bump_px), minimum=2, maximum=max_height_cap)
    return {
        "source_width": source_width,
        "source_height": source_height,
        "crop_top_px": safe_crop_top,
        "effective_source_height": effective_source_height,
        "base_height_raw": raw_base_h,
        "base_height": base_h,
        "content_max_height_px": max_height_cap,
        "target_height": target_h,
        "content_height_bump_px": max(0, content_height_bump_px),
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


def resolve_segments(
    input_video: Path,
    part_seconds: int,
    cuts_path: Path | None,
    ffprobe_bin: str = "ffprobe",
) -> tuple[list[Segment], dict[str, int]]:
    overrides: dict[str, int] = {}
    if cuts_path is not None:
        manual_segments, overrides = load_cuts_override(cuts_path)
        if manual_segments:
            return manual_segments, overrides

    total_duration = probe_duration_seconds(input_video=input_video, ffprobe_bin=ffprobe_bin)
    return build_auto_segments(total_duration=total_duration, part_seconds=part_seconds), overrides


def _escape_drawtext_text(text: str) -> str:
    return (
        text.replace("\\", r"\\")
        .replace(":", r"\:")
        .replace("'", r"\'")
        .replace(",", r"\,")
        .replace("%", r"\%")
        .replace("[", r"\[")
        .replace("]", r"\]")
    )


def _escape_filter_path(path: Path) -> str:
    return str(path).replace("\\", "/").replace(":", r"\:").replace("'", r"\'")


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
    render_preset: str = "legacy",
    title_mask_px: int = 0,
    part_overlay_enabled: bool = True,
    part_label_position: str = "top-center",
    font_file: Path | None = None,
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
    if y_scale_mode not in {"manual", "fill", "letterbox", "zoom"}:
        raise ValueError("y_scale_mode must be one of: manual, fill, letterbox, zoom")
    safe_edge_bar_px = max(0, min(int(edge_bar_px), 200))
    safe_content_height_bump_px = max(0, int(content_height_bump_px))
    safe_content_max_height_px = 0
    if int(content_max_height_px) > 0:
        safe_content_max_height_px = _clamp_even(int(content_max_height_px), minimum=2, maximum=output_height)
    safe_crop_top_px = max(0, int(crop_top_px))
    filters: list[str] = []

    if y_scale_mode == "letterbox":
        filters.append(f"scale={output_width}:-2")
        if safe_crop_top_px > 0:
            filters.append(
                f"crop=iw:max(2\\,ih-{safe_crop_top_px}):0:min({safe_crop_top_px}\\,ih-2)"
            )
        filters.append(f"pad={output_width}:{output_height}:(ow-iw)/2:(oh-ih)/2:color=black")
        if safe_edge_bar_px > 0:
            filters.append(f"drawbox=x=0:y=0:w=iw:h={safe_edge_bar_px}:color=black@1.0:t=fill")
            filters.append(
                f"drawbox=x=0:y=ih-{safe_edge_bar_px}:w=iw:h={safe_edge_bar_px}:color=black@1.0:t=fill"
            )
        if part_overlay_enabled:
            label_text = _escape_drawtext_text(f"Part {part_number}")
            if part_label_position == "top-left":
                x_expr = "40"
            else:
                x_expr = "(w-text_w)/2"

            font_arg = ""
            if font_file:
                font_arg = f"fontfile='{_escape_filter_path(font_file)}':"

            filters.append(
                "drawtext="
                f"{font_arg}"
                f"text='{label_text}':"
                "fontsize=64:"
                "fontcolor=white:"
                "borderw=3:"
                "bordercolor=black:"
                "box=1:"
                "boxcolor=black@0.45:"
                "boxborderw=14:"
                f"x={x_expr}:"
                "y=40"
            )
        filters.append("setsar=1")
        return ",".join(filters)

    if y_scale_mode == "zoom":
        filters.append(f"scale={output_width}:-2")
        if safe_crop_top_px > 0:
            filters.append(
                f"crop=iw:max(2\\,ih-{safe_crop_top_px}):0:min({safe_crop_top_px}\\,ih-2)"
            )
        if zoom_target_height is not None:
            zoom_max_h = safe_content_max_height_px if safe_content_max_height_px > 0 else output_height
            safe_zoom_target_height = _clamp_even(int(zoom_target_height), minimum=2, maximum=zoom_max_h)
            filters.append(f"scale=-2:{safe_zoom_target_height}")
            filters.append(f"crop={output_width}:{safe_zoom_target_height}:(iw-{output_width})/2:0")
        else:
            filters.append(
                f"scale=-2:trunc(min(ih+{safe_content_height_bump_px}\\,{output_height})/2)*2"
            )
            filters.append(f"crop={output_width}:ih:(iw-{output_width})/2:0")
        filters.append(f"pad={output_width}:{output_height}:(ow-iw)/2:(oh-ih)/2:color=black")
        if safe_edge_bar_px > 0:
            filters.append(f"drawbox=x=0:y=0:w=iw:h={safe_edge_bar_px}:color=black@1.0:t=fill")
            filters.append(
                f"drawbox=x=0:y=ih-{safe_edge_bar_px}:w=iw:h={safe_edge_bar_px}:color=black@1.0:t=fill"
            )
        if part_overlay_enabled:
            label_text = _escape_drawtext_text(f"Part {part_number}")
            if part_label_position == "top-left":
                x_expr = "40"
            else:
                x_expr = "(w-text_w)/2"

            font_arg = ""
            if font_file:
                font_arg = f"fontfile='{_escape_filter_path(font_file)}':"

            filters.append(
                "drawtext="
                f"{font_arg}"
                f"text='{label_text}':"
                "fontsize=64:"
                "fontcolor=white:"
                "borderw=3:"
                "bordercolor=black:"
                "box=1:"
                "boxcolor=black@0.45:"
                "boxborderw=14:"
                f"x={x_expr}:"
                "y=40"
            )
        filters.append("setsar=1")
        return ",".join(filters)

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

    if part_overlay_enabled:
        label_text = _escape_drawtext_text(f"Part {part_number}")
        if part_label_position == "top-left":
            x_expr = "40"
        else:
            x_expr = "(w-text_w)/2"

        font_arg = ""
        if font_file:
            font_arg = f"fontfile='{_escape_filter_path(font_file)}':"

        filters.append(
            "drawtext="
            f"{font_arg}"
            f"text='{label_text}':"
            "fontsize=64:"
            "fontcolor=white:"
            "borderw=3:"
            "bordercolor=black:"
            "box=1:"
            "boxcolor=black@0.45:"
            "boxborderw=14:"
            f"x={x_expr}:"
            "y=40"
        )

    # Force square pixels for consistent platform detection.
    filters.append("setsar=1")

    return ",".join(filters)


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
    font_file: Path | None = None,
    ffmpeg_bin: str = "ffmpeg",
    ffprobe_bin: str = "ffprobe",
    crf: int = 18,
    preset: str = "slow",
    log: Callable[[str], None] | None = None,
) -> list[RenderedPart]:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_fn = log or print

    if y_scale_mode not in {"manual", "fill", "letterbox", "zoom"}:
        raise ValueError("y_scale_mode must be one of: manual, fill, letterbox, zoom")

    safe_edge_bar_px = max(0, min(int(edge_bar_px), 200))
    safe_content_height_bump_px = max(0, int(content_height_bump_px))
    safe_content_max_height_px = 0
    if int(content_max_height_px) > 0:
        safe_content_max_height_px = _clamp_even(int(content_max_height_px), minimum=2, maximum=output_height)

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
        f"crop_top_px={crop_top_px}, title_mask_px={title_mask_px}"
    )
    if y_scale_mode == "letterbox" and safe_content_height_bump_px > 0:
        log_fn("render_config: content_height_bump_px is ignored in letterbox mode.")
    if y_scale_mode == "zoom" and zoom_debug is not None:
        log_fn(
            "zoom_debug: "
            f"base_height={zoom_debug['base_height']}, "
            f"target_height={zoom_debug['target_height']}, "
            f"content_max_height_px={zoom_debug['content_max_height_px']}, "
            f"content_height_bump_px={zoom_debug['content_height_bump_px']}"
        )

    for idx, segment in enumerate(segments, start=1):
        if segment.duration <= 0:
            raise ValueError(f"Segment {idx} duration must be > 0.")

        output_path = out_dir / f"part_{idx}.mp4"
        video_filter = build_video_filter(
            part_number=idx,
            crop_top_px=crop_top_px,
            output_width=output_width,
            output_height=output_height,
            video_y_scale=video_y_scale,
            y_scale_mode=y_scale_mode,
            edge_bar_px=safe_edge_bar_px,
            content_height_bump_px=safe_content_height_bump_px,
            content_max_height_px=safe_content_max_height_px,
            zoom_target_height=zoom_target_height_for_filter,
            effective_y_scale=effective_y_scale_for_filter,
            render_preset=render_preset,
            title_mask_px=title_mask_px,
            part_overlay_enabled=part_overlay_enabled,
            part_label_position=part_label_position,
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
            str(output_path),
        ]
        cmd_path = out_dir / f"part_{idx:03d}.ffmpeg.txt"
        cmd_run_path = out_dir / f"part_{idx:03d}.render.cmd"
        _write_part_command_dump(cmd_path, cmd)
        _write_windows_cmd(cmd_run_path, cmd)

        command_shell_text = _command_to_shell_text(cmd)
        log_fn(f"FFmpeg -vf (part {idx}): {video_filter}")
        log_fn(f"FFmpeg cmd (part {idx}): {command_shell_text}")

        _run_command(cmd)

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
                "vf": video_filter,
                "ffmpeg_cmd": cmd,
                "ffmpeg_cmd_shell": command_shell_text,
                "ffmpeg_cmd_dump_path": str(cmd_path.resolve()),
                "ffmpeg_cmd_run_path": str(cmd_run_path.resolve()),
                "output_path": str(output_path.resolve()),
                "start_time": format_ffmpeg_time(segment.start),
                "end_time": format_ffmpeg_time(segment.end),
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
            "crf": crf,
            "preset": preset,
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
