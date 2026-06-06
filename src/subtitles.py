from __future__ import annotations

import json
import os
import re
import subprocess
from html import unescape
from pathlib import Path
from typing import Any, Callable


_VALID_SUBTITLE_STYLES: frozenset[str] = frozenset({"hormozi", "standard", "minimal"})
_SUPPORTED_YOUTUBE_SUBTITLE_EXTS: frozenset[str] = frozenset({"json3", "srv3", "vtt", "srt"})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seconds_to_ass_time(seconds: float) -> str:
    """Convert seconds to ASS timestamp format H:MM:SS.cc (centiseconds)."""
    total_cs = int(round(max(seconds, 0.0) * 100))
    cs = total_cs % 100
    total_s = total_cs // 100
    s = total_s % 60
    total_m = total_s // 60
    m = total_m % 60
    h = total_m // 60
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _escape_ass_path(path: Path | str) -> str:
    """Escape an ASS path for FFmpeg's subtitles filter."""
    path_obj = Path(path)
    if not path_obj.is_absolute() and path_obj.parent == Path("."):
        s = path_obj.name
    else:
        cwd = Path.cwd().resolve()
        resolved = path_obj.resolve()
        s = resolved.name if resolved.parent == cwd else str(resolved)
    if os.name == "nt":
        s = s.replace("\\", "/")
        # FFmpeg filter syntax needs the drive colon escaped, but not double-escaped,
        # when the filter string is passed directly via subprocess argv.
        if len(s) >= 2 and s[1] == ":":
            s = s[0] + "\\:" + s[2:]
    s = s.replace("'", "\\'")
    s = s.replace("[", "\\[").replace("]", "\\]")
    return s


def _escape_ass_text(text: str) -> str:
    """Escape text for safe inclusion in an ASS Dialogue line."""
    return text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")


def _transcript_cache_path(input_video: Path) -> Path:
    return input_video.parent / f"{input_video.stem}_transcript.json"


def _transcription_audio_path(input_video: Path, sample_rate: int = 16000) -> Path:
    if sample_rate == 16000:
        suffix = "16k"
    else:
        suffix = f"{sample_rate}hz"
    return input_video.parent / f"{input_video.stem}_audio_{suffix}.wav"


def _normalize_word_record(raw_word: Any) -> dict[str, Any] | None:
    if not isinstance(raw_word, dict):
        return None

    word_text = str(raw_word.get("word", "")).strip()
    start = raw_word.get("start")
    end = raw_word.get("end")
    if not word_text or start is None or end is None:
        return None

    try:
        start_f = float(start)
        end_f = float(end)
    except (TypeError, ValueError):
        return None

    if end_f <= start_f:
        return None

    normalized = {
        "word": word_text,
        "start": start_f,
        "end": end_f,
    }
    speaker = raw_word.get("speaker")
    if speaker is not None:
        normalized["speaker"] = speaker
    return normalized


def _clean_caption_text(text: str) -> str:
    cleaned = unescape(text)
    cleaned = (
        cleaned.replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u02bc", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
    )
    cleaned = cleaned.replace("\r", " ").replace("\n", " ")
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = re.sub(r"\{[^}]+\}", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _split_text_evenly(text: str, start: float, end: float) -> list[dict[str, float | str]]:
    cleaned = _clean_caption_text(text)
    if not cleaned:
        return []

    words = cleaned.split()
    if not words:
        return []

    start_f = max(0.0, float(start))
    end_f = max(start_f, float(end))
    if end_f <= start_f:
        end_f = start_f + 0.1

    step = (end_f - start_f) / len(words)
    return [
        {
            "word": word,
            "start": start_f + index * step,
            "end": start_f + (index + 1) * step,
        }
        for index, word in enumerate(words)
    ]


def _load_transcript_cache(input_video: Path) -> list[dict] | None:
    cache_path = _transcript_cache_path(input_video)
    if not cache_path.exists():
        return None

    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return None

    cached_words: list[dict] = []
    for item in payload:
        normalized = _normalize_word_record(item)
        if normalized is not None:
            cached_words.append(normalized)
    return cached_words


def _resolve_binary_if_available(binary: str, default_name: str) -> str:
    raw_value = str(binary or default_name).strip() or default_name
    if any(sep in raw_value for sep in ("\\", "/")):
        return raw_value

    try:
        from .render import _resolve_binary  # lazy import to avoid a hard dependency at module import time
    except Exception:  # noqa: BLE001
        return raw_value

    candidate = default_name if raw_value.lower() in {default_name.lower(), f"{default_name.lower()}.exe"} else raw_value
    try:
        return _resolve_binary(candidate)
    except Exception:  # noqa: BLE001
        return raw_value


def _normalize_language(value: str | None) -> str:
    if not value:
        return ""
    return value.strip().lower().replace("_", "-")


def _language_priority(language: str | None) -> list[str]:
    preferred: list[str] = []
    normalized = _normalize_language(language)
    if normalized:
        preferred.append(normalized)
        base = normalized.split("-", 1)[0]
        if base and base not in preferred:
            preferred.append(base)
    else:
        preferred.extend(["en", "en-us", "en-gb"])
    return preferred


def _choose_language(available_languages: list[str], language: str | None) -> str | None:
    if not available_languages:
        return None

    normalized_map = {_normalize_language(item): item for item in available_languages}
    for candidate in _language_priority(language):
        if candidate in normalized_map:
            return normalized_map[candidate]

    for candidate in _language_priority(language):
        for normalized, original in normalized_map.items():
            if normalized.startswith(f"{candidate}-"):
                return original

    return sorted(available_languages)[0]


def _parse_timestamp(value: str) -> float:
    text = value.strip().replace(",", ".")
    parts = text.split(":")
    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    raise ValueError(f"Unsupported subtitle timestamp: {value}")


def _parse_webvtt_or_srt(text: str) -> list[dict[str, float | str]]:
    words: list[dict[str, float | str]] = []
    lines = text.replace("\ufeff", "").splitlines()

    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line or line.upper().startswith("WEBVTT") or line.upper().startswith("NOTE"):
            index += 1
            continue

        if re.fullmatch(r"\d+", line):
            index += 1
            if index >= len(lines):
                break
            line = lines[index].strip()

        if "-->" not in line:
            index += 1
            continue

        start_text, end_text = [part.strip() for part in line.split("-->", 1)]
        end_text = end_text.split(" ", 1)[0].strip()
        try:
            start = _parse_timestamp(start_text)
            end = _parse_timestamp(end_text)
        except ValueError:
            index += 1
            continue

        index += 1
        caption_lines: list[str] = []
        while index < len(lines) and lines[index].strip():
            caption_lines.append(lines[index].strip())
            index += 1

        words.extend(_split_text_evenly(" ".join(caption_lines), start, end))
        index += 1

    return words


def _parse_json3_payload(payload: dict[str, Any]) -> list[dict[str, float | str]]:
    words: list[dict[str, float | str]] = []
    raw_events = payload.get("events")
    if not isinstance(raw_events, list):
        return words

    for event in raw_events:
        if not isinstance(event, dict):
            continue
        start_ms = event.get("tStartMs")
        duration_ms = event.get("dDurationMs")
        segs = event.get("segs")
        if start_ms is None or not isinstance(segs, list):
            continue

        timed_parts: list[tuple[str, float | None]] = []
        for seg in segs:
            if not isinstance(seg, dict):
                continue
            utf8 = str(seg.get("utf8", ""))
            if utf8:
                offset_ms = seg.get("tOffsetMs")
                offset_seconds = None
                if isinstance(offset_ms, (int, float)):
                    offset_seconds = float(offset_ms) / 1000.0
                timed_parts.append((utf8, offset_seconds))

        text_value = "".join(part for part, _offset in timed_parts).strip()
        if not text_value:
            continue

        start = float(start_ms) / 1000.0
        end = start + (float(duration_ms) / 1000.0 if duration_ms is not None else 0.0)
        if any(offset is not None for _part, offset in timed_parts):
            for index, (part_text, offset) in enumerate(timed_parts):
                part_clean = _clean_caption_text(part_text)
                if not part_clean:
                    continue
                part_start = start + (offset or 0.0)
                next_offset = None
                for _next_text, candidate_offset in timed_parts[index + 1:]:
                    if candidate_offset is not None:
                        next_offset = candidate_offset
                        break
                part_end = start + next_offset if next_offset is not None else end
                words.extend(_split_text_evenly(part_clean, part_start, part_end))
        else:
            words.extend(_split_text_evenly(text_value, start, end))

    return words


def _parse_subtitle_file(path: Path) -> list[dict[str, float | str]] | None:
    if not path.exists() or not path.is_file():
        return None

    ext = path.suffix.lower().lstrip(".")
    if ext not in _SUPPORTED_YOUTUBE_SUBTITLE_EXTS:
        return None

    raw_text = path.read_text(encoding="utf-8", errors="ignore")
    if ext in {"vtt", "srt"}:
        return _parse_webvtt_or_srt(raw_text)
    if ext in {"json3", "srv3"}:
        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError:
            return None
        return _parse_json3_payload(payload)
    return None


def _coerce_track_entries(raw_entries: Any) -> list[dict[str, Any]]:
    if isinstance(raw_entries, list):
        return [entry for entry in raw_entries if isinstance(entry, dict)]
    if isinstance(raw_entries, dict):
        return [raw_entries]
    return []


def _available_track_groups(info_dict: dict[str, Any]) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    subtitles = info_dict.get("subtitles")
    automatic = info_dict.get("automatic_captions")
    if isinstance(subtitles, dict):
        groups.append(subtitles)
    if isinstance(automatic, dict):
        groups.append(automatic)
    return groups


def _info_has_subtitle_tracks(info_dict: dict[str, Any] | None) -> bool:
    if not isinstance(info_dict, dict):
        return False
    return any(isinstance(info_dict.get(key), dict) and info_dict.get(key) for key in ("subtitles", "automatic_captions"))


def _format_subtitle_error(exc: Exception) -> str:
    text = str(exc).strip()
    lowered = text.lower()
    if "429" in lowered or "too many requests" in lowered:
        return "HTTP 429"
    if not text:
        return "unknown error"
    return text.splitlines()[0][:140]


def _has_local_subtitle_file(info_dict: dict[str, Any] | None) -> bool:
    if not isinstance(info_dict, dict):
        return False
    for key in ("subtitles", "automatic_captions"):
        group = info_dict.get(key)
        if not isinstance(group, dict):
            continue
        for entries in group.values():
            for entry in _coerce_track_entries(entries):
                filepath = str(entry.get("filepath", "")).strip()
                if filepath and Path(filepath).exists():
                    return True
    return False


def _iter_track_candidates(info_dict: dict[str, Any], language: str | None) -> list[tuple[str, list[dict[str, Any]]]]:
    candidates: list[tuple[str, list[dict[str, Any]]]] = []
    for group in _available_track_groups(info_dict):
        available_languages = [str(key) for key in group.keys()]
        selected_language = _choose_language(available_languages, language)
        if not selected_language:
            continue
        entries = _coerce_track_entries(group.get(selected_language))
        if entries:
            candidates.append((selected_language, entries))
    return candidates


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_subtitle_filter(ass_path: Path | str) -> str:
    """Return the FFmpeg -vf subtitles filter string for an ASS file."""
    escaped = _escape_ass_path(ass_path)
    # Use the explicit filename= form so FFmpeg does not have to infer which
    # positional value belongs to the file path once extra options are appended.
    return f"subtitles=filename='{escaped}':force_style='Encoding=UTF-8'"


def get_youtube_subtitles(
    info_dict: dict[str, Any],
    language: str | None = None,
    log: Callable[[str], None] | None = None,
) -> list[dict] | None:
    """Parse already-downloaded YouTube subtitle files referenced by *info_dict*."""
    log_fn = log or (lambda _: None)
    try:
        for _selected_language, entries in _iter_track_candidates(info_dict, language):
            sorted_entries = sorted(
                entries,
                key=lambda item: (
                    0 if str(item.get("ext", "")).lower() in {"json3", "srv3"} else 1,
                    0 if str(item.get("ext", "")).lower() == "vtt" else 1,
                    str(item.get("filepath", "")),
                ),
            )
            for entry in sorted_entries:
                filepath_raw = str(entry.get("filepath", "")).strip()
                if not filepath_raw:
                    continue
                words = _parse_subtitle_file(Path(filepath_raw))
                if words:
                    return [
                        {
                            "word": str(item["word"]).strip(),
                            "start": float(item["start"]),
                            "end": float(item["end"]),
                        }
                        for item in words
                        if str(item.get("word", "")).strip()
                    ]
    except Exception as exc:  # noqa: BLE001
        log_fn(
            f"YouTube subtitles unavailable ({_format_subtitle_error(exc)}), falling back to Whisper cache/transcription"
        )
        return None
    return None


def transcribe_video(
    input_video: Path,
    device: str | None = None,
    language: str | None = None,
    model_size: str | None = None,
    use_cache: bool = True,
    ffmpeg_bin: str = "ffmpeg",
    subtitle_offset_seconds: float = 0.0,
    log: Callable[[str], None] | None = None,
) -> list[dict]:
    """
    Transcribe *input_video* with WhisperX and return per-word timestamps.

    Returns a flat list of dicts::

        [{"word": "hello", "start": 0.12, "end": 0.45}, ...]

    Falls back to sentence-level segments (words split evenly) when word-level
    alignment is unavailable.
    """
    _log = log or (lambda _: None)

    if use_cache:
        cached_words = _load_transcript_cache(input_video)
        if cached_words is not None:
            _log("Transcript cache found, skipping WhisperX transcription")
            return cached_words

    # Keep the cached transcript anchored to the full source timeline. The
    # manual subtitle offset is applied later when each clip is sliced.
    _ = subtitle_offset_seconds

    try:
        import whisperx  # type: ignore[import]
    except ImportError as exc:
        raise RuntimeError(
            "WhisperX is not installed. Run: pip install whisperx"
        ) from exc

    model_name = model_size or os.getenv("WHISPER_MODEL") or "medium"

    try:
        import torch  # type: ignore[import]
        auto_device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001
        auto_device = "cpu"
    resolved_device = os.getenv("WHISPER_DEVICE") or device or auto_device
    compute_type = "float16" if str(resolved_device).lower().startswith("cuda") else "int8"

    audio_path = extract_audio_for_transcription(
        input_video=input_video,
        ffmpeg_bin=ffmpeg_bin,
        sample_rate=16000,
        log=_log,
    )

    _log(f"WhisperX transcribing with model={model_name}, device={resolved_device}")

    model = whisperx.load_model(model_name, device=resolved_device, compute_type=compute_type, language=language)
    audio = whisperx.load_audio(str(audio_path))
    result = model.transcribe(audio, language=language)

    try:
        align_model, metadata = whisperx.load_align_model(
            language_code=result.get("language") or language or "en",
            device=resolved_device,
        )
        aligned = whisperx.align(
            result["segments"],
            align_model,
            metadata,
            audio,
            device=resolved_device,
            return_char_alignments=False,
        )
        segments = aligned.get("segments", [])
        _log("WhisperX alignment complete")
    except Exception as exc:  # noqa: BLE001
        _log(f"WhisperX alignment unavailable, using transcript segment timing: {exc}")
        segments = result.get("segments", [])

    words: list[dict] = []
    for seg in segments:
        seg_words = seg.get("words")
        if seg_words:
            for w in seg_words:
                normalized = _normalize_word_record(w)
                if normalized is not None:
                    words.append(normalized)
        else:
            seg_text = str(seg.get("text", "")).strip()
            seg_start = float(seg.get("start", 0))
            seg_end = float(seg.get("end", seg_start))
            words.extend(_split_text_evenly(seg_text, seg_start, seg_end))

    if use_cache:
        cache_path = _transcript_cache_path(input_video)
        cache_path.write_text(json.dumps(words, ensure_ascii=False), encoding="utf-8")
        _log(f"Transcript cached to {cache_path}")

    return words


def extract_audio_for_transcription(
    input_video: Path,
    ffmpeg_bin: str = "ffmpeg",
    sample_rate: int = 16000,
    log: Callable[[str], None] | None = None,
) -> Path:
    """Extract mono WAV audio for transcription and reuse it while it is fresh."""
    log_fn = log or (lambda _: None)
    audio_path = _transcription_audio_path(input_video, sample_rate=sample_rate)
    if (
        audio_path.exists()
        and audio_path.stat().st_size > 0
        and audio_path.stat().st_mtime >= input_video.stat().st_mtime
    ):
        log_fn("Using cached extracted audio...")
        return audio_path

    resolved_ffmpeg = _resolve_binary_if_available(ffmpeg_bin, "ffmpeg")
    log_fn("Extracting audio for transcription...")
    cmd = [
        resolved_ffmpeg,
        "-y",
        "-i",
        str(input_video),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-c:a",
        "pcm_s16le",
        str(audio_path),
    ]
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        audio_path.unlink(missing_ok=True)
        stderr_text = result.stderr.strip() or result.stdout.strip() or "unknown ffmpeg error"
        raise RuntimeError(
            f"Failed to extract transcription audio from {input_video}: {stderr_text.splitlines()[-1]}"
        )
    return audio_path


def resolve_subtitles(
    input_video: Path,
    info_dict: dict | None,
    subtitle_language: str | None,
    use_cache: bool,
    ffmpeg_bin: str = "ffmpeg",
    subtitle_offset_seconds: float = 0.0,
    log: Callable[[str], None] | None = None,
) -> list[dict] | None:
    """Resolve subtitles from YouTube tracks, transcript cache, or WhisperX."""
    log_fn = log or (lambda _: None)

    if info_dict:
        try:
            youtube_words = get_youtube_subtitles(info_dict, language=subtitle_language, log=log_fn)
            if youtube_words:
                log_fn("Subtitles: YouTube")
                return youtube_words
        except Exception:  # noqa: BLE001
            pass
        if _info_has_subtitle_tracks(info_dict) and not _has_local_subtitle_file(info_dict):
            log_fn("YouTube subtitles unavailable, falling back to Whisper cache/transcription")

    if use_cache:
        try:
            cached_words = _load_transcript_cache(input_video)
            if cached_words:
                log_fn("Subtitles: cache")
                log_fn("Using cached Whisper transcript")
                return cached_words
        except Exception:  # noqa: BLE001
            pass

    try:
        log_fn("Subtitles: WhisperX")
        log_fn("Using WhisperX transcription")
        words = transcribe_video(
            input_video=input_video,
            language=subtitle_language,
            use_cache=use_cache,
            ffmpeg_bin=ffmpeg_bin,
            subtitle_offset_seconds=subtitle_offset_seconds,
            log=log_fn,
        )
        return words or None
    except Exception as exc:  # noqa: BLE001
        log_fn(f"Subtitles unavailable, continuing without subtitles: {exc}")
        return None


def slice_words_for_segment(
    words: list[dict],
    segment_start: float,
    segment_end: float,
    subtitle_offset_seconds: float = 0.0,
) -> list[dict]:
    """Return clip-local words sliced from a full-source word list."""
    source_start = float(segment_start)
    source_end = max(source_start, float(segment_end))
    clip_duration = max(0.0, source_end - source_start)
    offset = float(subtitle_offset_seconds)

    sliced: list[dict] = []
    for raw_word in words:
        normalized = _normalize_word_record(raw_word)
        if normalized is None:
            continue

        word_start = float(normalized["start"])
        word_end = float(normalized["end"])
        if word_end <= source_start or word_start >= source_end:
            continue

        local_start = (word_start - source_start) + offset
        local_end = (word_end - source_start) + offset
        clamped_start = max(0.0, local_start)
        clamped_end = min(clip_duration, local_end)
        if clamped_end <= clamped_start:
            continue

        local_word = {
            "word": normalized["word"],
            "start": clamped_start,
            "end": clamped_end,
        }
        if "speaker" in normalized:
            local_word["speaker"] = normalized["speaker"]
        sliced.append(local_word)

    return sliced


def build_ass_subtitles(
    words: list[dict],
    output_path: Path,
    style: str = "hormozi",
    segment_start: float = 0.0,
    clip_duration: float | None = None,
) -> Path:
    """
    Generate an ASS subtitle file from per-word timestamp dicts.

    Words are expected to already be clip-local. *segment_start* is kept only
    for backward compatibility with older absolute-timestamp callers.

    Returns the path to the generated ``.ass`` file.
    """
    if style not in _VALID_SUBTITLE_STYLES:
        raise ValueError(f"subtitle style must be one of {sorted(_VALID_SUBTITLE_STYLES)}")

    safe_segment_start = float(segment_start)
    safe_clip_duration = max(0.0, float(clip_duration)) if clip_duration is not None else None
    adjusted: list[dict] = []
    for raw_word in words:
        normalized = _normalize_word_record(raw_word)
        if normalized is None:
            continue

        start = float(normalized["start"]) - safe_segment_start
        end = float(normalized["end"]) - safe_segment_start
        if safe_clip_duration is not None:
            if start >= safe_clip_duration or end <= 0.0:
                continue
            start = max(0.0, min(start, safe_clip_duration))
            end = max(0.0, min(end, safe_clip_duration))
        else:
            start = max(0.0, start)
            end = max(0.0, end)

        if end <= start:
            continue

        adjusted_word = {
            "word": normalized["word"],
            "start": start,
            "end": end,
        }
        if "speaker" in normalized:
            adjusted_word["speaker"] = normalized["speaker"]
        adjusted.append(adjusted_word)

    if style == "hormozi":
        style_line = (
            "Style: Default,Arial,72,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,"
            "1,0,0,0,100,100,0,0,1,3,0,2,10,10,10,1"
        )
    elif style == "standard":
        style_line = (
            "Style: Default,Arial,56,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,"
            "1,0,0,0,100,100,0,0,1,3,0,2,10,10,10,1"
        )
    else:
        style_line = (
            "Style: Default,Arial,44,&H99FFFFFF,&H000000FF,&H00000000,&H00000000,"
            "0,0,0,0,100,100,0,0,1,0,0,2,10,10,10,1"
        )

    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        "PlayResX: 1080\n"
        "PlayResY: 1920\n"
        "Collisions: Normal\n"
        "PlayDepth: 0\n"
        "\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"{style_line}\n"
        "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )

    events: list[str] = []

    if style == "hormozi":
        _build_hormozi_events(adjusted, events)
    elif style == "standard":
        _build_standard_events(adjusted, events)
    else:
        _build_minimal_events(adjusted, events)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(header + "\n".join(events) + "\n", encoding="utf-8")
    return output_path


# ---------------------------------------------------------------------------
# Per-style event builders (each appends ASS Dialogue lines to *events*)
# ---------------------------------------------------------------------------

def _build_hormozi_events(words: list[dict], events: list[str]) -> None:
    """
    Hormozi / TikTok viral style:
    - 3-word chunks
    - All caps
    - Currently spoken word highlighted in yellow; others white
    - Positioned in the bottom third of a 1080x1920 frame
    """
    chunk_size = 3
    for i in range(0, len(words), chunk_size):
        chunk = words[i : i + chunk_size]
        if not chunk:
            continue
        for active_idx, active_word in enumerate(chunk):
            w_start = active_word["start"]
            w_end = active_word["end"]
            if w_end <= w_start:
                w_end = w_start + 0.1

            parts: list[str] = []
            for j, w in enumerate(chunk):
                text = _escape_ass_text(w["word"].upper())
                if j == active_idx:
                    parts.append(f"{{\\c&H0000FFFF&}}{text}{{\\c&H00FFFFFF&}}")
                else:
                    parts.append(text)

            line_text = "{\\an2\\pos(540,1650)}" + " ".join(parts)
            t_start = _seconds_to_ass_time(w_start)
            t_end = _seconds_to_ass_time(w_end)
            events.append(f"Dialogue: 0,{t_start},{t_end},Default,,0,0,0,,{line_text}")


def _build_standard_events(words: list[dict], events: list[str]) -> None:
    """Standard style: 8-word chunks, white text, black outline, no highlight."""
    chunk_size = 8
    for i in range(0, len(words), chunk_size):
        chunk = words[i : i + chunk_size]
        if not chunk:
            continue
        chunk_start = chunk[0]["start"]
        chunk_end = chunk[-1]["end"]
        if chunk_end <= chunk_start:
            chunk_end = chunk_start + 0.1
        text = _escape_ass_text(" ".join(w["word"] for w in chunk))
        line_text = "{\\an2\\pos(540,1800)}" + text
        t_start = _seconds_to_ass_time(chunk_start)
        t_end = _seconds_to_ass_time(chunk_end)
        events.append(f"Dialogue: 0,{t_start},{t_end},Default,,0,0,0,,{line_text}")


def _build_minimal_events(words: list[dict], events: list[str]) -> None:
    """Minimal style: full sentences grouped by silence gaps > 1 s."""
    if not words:
        return

    def _flush(chunk: list[dict]) -> None:
        if not chunk:
            return
        chunk_start = chunk[0]["start"]
        chunk_end = chunk[-1]["end"]
        if chunk_end <= chunk_start:
            chunk_end = chunk_start + 0.1
        text = _escape_ass_text(" ".join(w["word"] for w in chunk))
        line_text = "{\\an2\\pos(540,1750)}" + text
        t_start = _seconds_to_ass_time(chunk_start)
        t_end = _seconds_to_ass_time(chunk_end)
        events.append(f"Dialogue: 0,{t_start},{t_end},Default,,0,0,0,,{line_text}")

    current: list[dict] = [words[0]]
    for w in words[1:]:
        if w["start"] - current[-1]["end"] > 1.0:
            _flush(current)
            current = [w]
        else:
            current.append(w)
    _flush(current)
