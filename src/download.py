from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse


VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".avi"}
REJECTED_SOURCE_SUFFIXES = {".json", ".json3", ".vtt", ".srt", ".ass", ".txt"}
SUBTITLE_SUFFIXES = {".json3", ".srv3", ".vtt", ".srt"}
SUPPORTED_JS_RUNTIMES = ("deno", "node", "bun", "quickjs")
JS_RUNTIME_ENV = "YT_DLP_JS_RUNTIMES"
IMPERSONATE_ENV = "YT_DLP_IMPERSONATE"
YOUTUBE_HOST_SUFFIXES = (
    "youtube.com",
    "youtu.be",
)
_JS_RUNTIME_EXECUTABLES: dict[str, tuple[str, ...]] = {
    "deno": ("deno",),
    "node": ("node",),
    "bun": ("bun",),
    "quickjs": ("qjs", "quickjs"),
}
_WARNED_MESSAGES: set[str] = set()
YOUTUBE_CHALLENGE_HELP = "YouTube blocked extraction. Update yt-dlp and install Node.js LTS, then retry."

LogFn = Callable[[str], None]


def _noop_log(_: str) -> None:
    return


def is_http_url(value: str) -> bool:
    parsed = urlparse(value.strip())
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _slugify(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._-")
    return text[:80] if text else "source"


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _normalize_duration(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _warn_once(key: str, message: str, log: LogFn | None) -> None:
    if key in _WARNED_MESSAGES:
        return
    _WARNED_MESSAGES.add(key)
    (log or _noop_log)(message)


def _normalized_host(url: str) -> str:
    return urlparse(url.strip()).netloc.lower().split(":", 1)[0]


def _is_youtube_url(value: str) -> bool:
    if not is_http_url(value):
        return False
    host = _normalized_host(value)
    return any(host == suffix or host.endswith(f".{suffix}") for suffix in YOUTUBE_HOST_SUFFIXES)


def _extract_youtube_video_id(url: str) -> str | None:
    if not _is_youtube_url(url):
        return None

    parsed = urlparse(url.strip())
    host = _normalized_host(url)
    query = parse_qs(parsed.query)
    path_parts = [part for part in parsed.path.split("/") if part]

    candidate: str | None = None
    if host == "youtu.be":
        candidate = path_parts[0] if path_parts else None
    elif query.get("v"):
        candidate = query["v"][0]
    elif len(path_parts) >= 2 and path_parts[0] in {"shorts", "embed", "live", "v"}:
        candidate = path_parts[1]

    if not candidate:
        return None

    normalized = candidate.strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]{6,}", normalized):
        return None
    return normalized


def _safe_name_from_url(url: str) -> str:
    if _is_youtube_url(url):
        video_id = _extract_youtube_video_id(url)
        if not video_id:
            raise ValueError(f"Invalid YouTube URL: could not determine video ID from {url}")
        return video_id

    parsed = urlparse(url)
    path_slug = Path(parsed.path).stem
    if path_slug:
        return _slugify(path_slug)

    query = parse_qs(parsed.query)
    for values in query.values():
        if values:
            return _slugify(values[0])
    return _slugify(parsed.netloc or "source")


def _source_cache_dir(url: str, downloads_root: Path) -> Path:
    return (downloads_root / _safe_name_from_url(url)).resolve()


def _source_info_path(cache_dir: Path) -> Path:
    return cache_dir / "source_info.json"


def _chapters_cache_path(cache_dir: Path) -> Path:
    return cache_dir / "chapters.json"


def _subtitle_filename_candidates(cache_dir: Path, language: str, ext: str | None = None) -> list[Path]:
    lang = language.strip().lower()
    expected_ext = f".{ext.strip().lower()}" if ext else ""
    candidates: list[Path] = []
    for item in sorted(cache_dir.glob("source.*")):
        if not item.is_file():
            continue
        suffix = item.suffix.lower()
        if suffix not in SUBTITLE_SUFFIXES:
            continue
        if expected_ext and suffix != expected_ext:
            continue
        name_lower = item.name.lower()
        if name_lower.startswith(f"source.{lang}.") or f".{lang}." in name_lower:
            candidates.append(item.resolve())
    return candidates


def _parse_chapters_from_info(info: dict[str, Any]) -> list[dict[str, float | str]]:
    raw_chapters = info.get("chapters") or []
    chapters: list[dict[str, float | str]] = []
    if not isinstance(raw_chapters, list):
        return chapters

    for ch in raw_chapters:
        if not isinstance(ch, dict):
            continue
        start = ch.get("start_time")
        end = ch.get("end_time")
        if start is None or end is None:
            continue
        try:
            start_f = float(start)
            end_f = float(end)
        except (TypeError, ValueError):
            continue
        if end_f <= start_f:
            continue
        chapters.append(
            {
                "title": _normalize_text(ch.get("title")),
                "start_time": start_f,
                "end_time": end_f,
            }
        )
    return chapters


def _has_subtitle_tracks(info: dict[str, Any] | None) -> bool:
    if not isinstance(info, dict):
        return False
    for key in ("subtitles", "automatic_captions"):
        group = info.get(key)
        if isinstance(group, dict) and group:
            return True
    return False


def _cached_subtitle_sidecars(cache_dir: Path) -> list[Path]:
    return [
        item.resolve()
        for item in sorted(cache_dir.glob("source.*"))
        if item.is_file() and item.suffix.lower() in SUBTITLE_SUFFIXES
    ]


def _normalize_subtitle_group(raw_group: Any, cache_dir: Path) -> dict[str, list[dict[str, str]]]:
    normalized: dict[str, list[dict[str, str]]] = {}
    if not isinstance(raw_group, dict):
        return normalized

    for language, raw_entries in raw_group.items():
        entries: list[dict[str, str]] = []
        if isinstance(raw_entries, list):
            iterable = raw_entries
        elif isinstance(raw_entries, dict):
            iterable = [raw_entries]
        else:
            iterable = []

        for entry in iterable:
            if not isinstance(entry, dict):
                continue
            ext = _normalize_text(entry.get("ext")).lower()
            name = _normalize_text(entry.get("name"))
            filepath = _normalize_text(entry.get("filepath"))
            url = _normalize_text(entry.get("url"))

            if filepath:
                path_obj = Path(filepath)
                if path_obj.exists():
                    filepath = str(path_obj.resolve())
                else:
                    filepath = ""

            if not filepath:
                guessed = _subtitle_filename_candidates(cache_dir, str(language), ext)
                if guessed:
                    filepath = str(guessed[0])

            record = {
                "ext": ext,
                "name": name,
                "filepath": filepath,
                "url": url,
            }
            if filepath or ext or url:
                entries.append(record)

        if entries:
            normalized[str(language)] = entries

    return normalized


def _normalize_source_info(info: dict[str, Any], cache_dir: Path) -> dict[str, Any]:
    normalized: dict[str, Any] = {
        "id": _normalize_text(info.get("id")),
        "title": _normalize_text(info.get("title")),
        "uploader": _normalize_text(info.get("uploader")),
        "channel": _normalize_text(info.get("channel")),
        "duration": _normalize_duration(info.get("duration")),
        "webpage_url": _normalize_text(
            info.get("webpage_url")
            or info.get("original_url")
            or info.get("url")
        ),
        "extractor": _normalize_text(info.get("extractor") or info.get("extractor_key")),
        "chapters": _parse_chapters_from_info(info),
        "subtitles": _normalize_subtitle_group(info.get("subtitles"), cache_dir),
        "automatic_captions": _normalize_subtitle_group(info.get("automatic_captions"), cache_dir),
    }
    return normalized


def _load_cached_info(cache_dir: Path) -> dict[str, Any] | None:
    info_path = _source_info_path(cache_dir)
    if not info_path.exists():
        return None

    payload = json.loads(info_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Cached metadata file is invalid: {info_path}")
    return payload


def _save_cached_info(cache_dir: Path, info: dict[str, Any]) -> dict[str, Any]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    normalized = _normalize_source_info(info, cache_dir)
    info_path = _source_info_path(cache_dir)
    info_path.write_text(json.dumps(normalized, indent=2), encoding="utf-8")

    chapters_path = _chapters_cache_path(cache_dir)
    chapters_path.write_text(json.dumps(normalized.get("chapters", []), indent=2), encoding="utf-8")
    return normalized


def _validate_source_info(info: dict[str, Any], *, source_url: str) -> None:
    if _is_youtube_url(source_url) and not _normalize_text(info.get("id")):
        raise RuntimeError(f"yt-dlp metadata is missing the video id for URL: {source_url}")


def _has_local_subtitle_files(info: dict[str, Any] | None, cache_dir: Path) -> bool:
    if _cached_subtitle_sidecars(cache_dir):
        return True
    if not isinstance(info, dict):
        return False
    for key in ("subtitles", "automatic_captions"):
        group = info.get(key)
        if not isinstance(group, dict):
            continue
        for entries in group.values():
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                filepath = _normalize_text(entry.get("filepath"))
                if filepath and Path(filepath).exists():
                    return True
    return False


def _format_subtitle_failure_reason(exc: Exception) -> str:
    text = str(exc).strip()
    lowered = text.lower()
    if "429" in lowered or "too many requests" in lowered:
        return "HTTP 429"
    first_line = text.splitlines()[0].strip() if text else "unknown error"
    first_line = re.sub(r"^yt-dlp [^:]+ failed for .*?:\s*", "", first_line)
    return first_line[:140]


def _parse_js_runtimes_env(value: str, log: LogFn | None) -> dict[str, dict[str, str]]:
    normalized = value.strip()
    if not normalized or normalized.lower() in {"none", "off", "false", "0"}:
        return {}

    runtimes: dict[str, dict[str, str]] = {}
    raw_parts = [part.strip() for part in re.split(r"[;,]", normalized) if part.strip()]
    for item in raw_parts:
        name, raw_path = [piece.strip() for piece in item.split(":", 1)] if ":" in item else (item, "")
        runtime_name = name.lower()
        if runtime_name not in SUPPORTED_JS_RUNTIMES:
            _warn_once(
                f"js-runtime-unsupported:{runtime_name}",
                (
                    f"yt-dlp: ignoring unsupported JS runtime '{runtime_name}'. "
                    f"Supported runtimes: {', '.join(SUPPORTED_JS_RUNTIMES)}."
                ),
                log,
            )
            continue

        resolved_path = raw_path
        if not resolved_path:
            for candidate in _JS_RUNTIME_EXECUTABLES.get(runtime_name, (runtime_name,)):
                found = shutil.which(candidate)
                if found:
                    resolved_path = found
                    break

        if raw_path and not Path(raw_path).exists():
            _warn_once(
                f"js-runtime-missing:{runtime_name}:{raw_path}",
                f"yt-dlp: configured JS runtime path not found for '{runtime_name}': {raw_path}",
                log,
            )
            continue
        if not resolved_path:
            _warn_once(
                f"js-runtime-missing:{runtime_name}",
                f"yt-dlp: configured JS runtime '{runtime_name}' is not available on PATH.",
                log,
            )
            continue

        runtimes[runtime_name] = {"path": str(Path(resolved_path).resolve())}

    return runtimes


def _resolve_js_runtimes(log: LogFn | None) -> dict[str, dict[str, str]]:
    configured = os.getenv(JS_RUNTIME_ENV, "").strip()
    if configured:
        return _parse_js_runtimes_env(configured, log)

    node_path = shutil.which("node")
    deno_path = shutil.which("deno")
    if not node_path and not deno_path:
        _warn_once(
            "js-runtime-node-deno:none",
            "YouTube may fail: install Node.js LTS or Deno for yt-dlp JS challenge solving.",
            log,
        )
    else:
        if node_path:
            (log or _noop_log)(f"yt-dlp JS challenge runtime detected: node={node_path}")
        if deno_path:
            (log or _noop_log)(f"yt-dlp JS challenge runtime detected: deno={deno_path}")

    runtimes: dict[str, dict[str, str]] = {}
    for runtime_name in SUPPORTED_JS_RUNTIMES:
        for candidate in _JS_RUNTIME_EXECUTABLES.get(runtime_name, (runtime_name,)):
            found = shutil.which(candidate)
            if found:
                runtimes[runtime_name] = {"path": str(Path(found).resolve())}
                break

    if not runtimes:
        _warn_once(
            "js-runtime:none",
            (
                "yt-dlp: no supported JS runtime found for YouTube extraction. "
                "Continuing without configured JS runtimes. Install Deno, Node, Bun, or QuickJS, "
                f"or set {JS_RUNTIME_ENV}=runtime[:path]."
            ),
            log,
        )

    return runtimes


def _is_youtube_challenge_error(text: str) -> bool:
    lowered = text.lower()
    return (
        "n challenge" in lowered
        or "challenge solving failed" in lowered
        or "js challenge" in lowered
        or "javascript runtime" in lowered
        or "remote components challenge solver" in lowered
    )


def _resolve_impersonate_target(js_runtimes: dict[str, dict[str, str]], log: LogFn | None) -> Any | None:
    configured = os.getenv(IMPERSONATE_ENV, "").strip()
    if not configured:
        return None

    _ensure_ytdlp_available()
    import yt_dlp
    from yt_dlp.networking.impersonate import ImpersonateTarget

    try:
        requested_target = (
            ImpersonateTarget()
            if configured.lower() in {"any", "true", "1"}
            else ImpersonateTarget.from_str(configured.lower())
        )
    except Exception as exc:  # noqa: BLE001
        _warn_once(
            f"impersonate-invalid:{configured}",
            f"yt-dlp: invalid impersonation target '{configured}', disabling impersonation ({exc}).",
            log,
        )
        return None

    try:
        with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "js_runtimes": js_runtimes}) as ydl:
            if ydl._impersonate_target_available(requested_target):
                return requested_target
    except Exception as exc:  # noqa: BLE001
        _warn_once(
            f"impersonate-check-error:{configured}",
            (
                f"yt-dlp: could not validate impersonation target '{configured}', disabling impersonation ({exc}). "
                "Install dependencies that provide impersonation support, or run "
                "yt-dlp --list-impersonate-targets to verify availability."
            ),
            log,
        )
        return None

    _warn_once(
        f"impersonate-unavailable:{configured}",
        (
            f"yt-dlp: impersonation target '{configured}' is not available, disabling impersonation. "
            "Install dependencies that provide impersonation support, or run "
            "yt-dlp --list-impersonate-targets to verify availability."
        ),
        log,
    )
    return None


class _YtDlpLogger:
    def __init__(self, log: LogFn | None) -> None:
        self._log = log or _noop_log

    def debug(self, _message: str) -> None:
        return

    def info(self, _message: str) -> None:
        return

    def warning(self, message: str) -> None:
        text = str(message).strip()
        if not text:
            return
        lowered = text.lower()
        if "impersonate target" in lowered or "no impersonate target is available" in lowered:
            _warn_once(f"ytdlp-warning:{lowered}", f"yt-dlp: {text}", self._log)
            return
        if "javascript runtime" in lowered or "js runtime" in lowered:
            _warn_once(f"ytdlp-warning:{lowered}", f"yt-dlp: {text}", self._log)
            return
        _warn_once(f"ytdlp-warning:{lowered}", f"yt-dlp warning: {text}", self._log)

    def error(self, message: str) -> None:
        text = str(message).strip()
        if text:
            _warn_once(f"ytdlp-error:{text.lower()}", f"yt-dlp: {text}", self._log)


def _ensure_ytdlp_available() -> None:
    try:
        import yt_dlp  # noqa: F401
    except ImportError as exc:
        raise SystemExit("yt-dlp is required for URL inputs. Install with: pip install yt-dlp") from exc


def _is_rejected_source_file(path: Path) -> bool:
    name_lower = path.name.lower()
    if name_lower.endswith(".info.json"):
        return True
    return path.suffix.lower() in REJECTED_SOURCE_SUFFIXES


def _is_valid_source_media_file(path: Path) -> bool:
    return path.is_file() and not _is_rejected_source_file(path) and path.suffix.lower() in VIDEO_SUFFIXES


def _resolve_local_input_path(raw_input: str, downloads_root: Path) -> Path:
    raw_path = Path(raw_input).expanduser()
    if raw_path.is_absolute():
        return raw_path.resolve()

    repo_root = Path(__file__).resolve().parents[1]
    downloads_base = downloads_root.expanduser()
    if not downloads_base.is_absolute():
        downloads_base = repo_root / downloads_base

    candidates = [raw_path.resolve(), (repo_root / raw_path).resolve()]
    if not raw_path.parts or raw_path.parts[0].lower() != downloads_base.name.lower():
        candidates.append((downloads_base / raw_path).resolve())

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return candidate
    return candidates[0]


def _source_candidate_sort_key(path: Path) -> tuple[int, int, int, str]:
    extension_priority = {
        ".mp4": 0,
        ".webm": 1,
        ".mkv": 2,
        ".mov": 3,
        ".avi": 4,
    }
    try:
        size_key = -int(path.stat().st_size)
    except OSError:
        size_key = 0
    stem_priority = 0 if path.stem.lower() == "source" else 1
    suffix_priority = extension_priority.get(path.suffix.lower(), 99)
    return (stem_priority, suffix_priority, size_key, path.name.lower())


def _select_source_media_file(video_dir: Path) -> Path:
    candidates = [
        item.resolve()
        for item in video_dir.glob("source*")
        if _is_valid_source_media_file(item)
    ]
    if not candidates:
        raise RuntimeError(
            "No valid source media file found. Expected one of: .mp4, .webm, .mkv, .mov, .avi"
        )
    return sorted(candidates, key=_source_candidate_sort_key)[0]


def _find_downloaded_source(download_dir: Path, info_dict: dict[str, Any] | None = None) -> Path:
    preferred_name = _normalize_text((info_dict or {}).get("source_filename"))
    if preferred_name:
        preferred_path = (download_dir / preferred_name).resolve()
        if _is_valid_source_media_file(preferred_path):
            return preferred_path

    try:
        return _select_source_media_file(download_dir)
    except RuntimeError as exc:
        raise RuntimeError(f"Source video not found in cache folder: {download_dir}. {exc}") from exc


def _build_ytdlp_options(
    *,
    output_template: str | None,
    cookies_file: Path | None,
    skip_media_download: bool,
    include_youtube_subtitles: bool,
    js_runtimes: dict[str, dict[str, str]] | None,
    impersonate: Any | None,
    log: LogFn | None,
) -> dict[str, Any]:
    ydl_opts: dict[str, Any] = {
        "sleep_interval": 3,
        "max_sleep_interval": 6,
        "retries": 5,
        "fragment_retries": 5,
        "ratelimit": 2_000_000,
        "quiet": True,
        "no_warnings": True,
        "logger": _YtDlpLogger(log),
        "js_runtimes": js_runtimes or {},
        "extractor_args": {
            "youtube": {
                "player_client": ["web", "android"],
            },
        },
        "writesubtitles": False,
        "writeautomaticsub": False,
    }
    if output_template:
        ydl_opts["outtmpl"] = output_template
    if skip_media_download:
        ydl_opts["skip_download"] = True
    if cookies_file and cookies_file.exists():
        ydl_opts["cookiefile"] = str(cookies_file)
    if impersonate is not None:
        ydl_opts["impersonate"] = impersonate
    if include_youtube_subtitles:
        log_fn = log or _noop_log
        log_fn("Fetching YouTube subtitles separately from the main video download.")
        ydl_opts["writesubtitles"] = True
        ydl_opts["writeautomaticsub"] = True
        ydl_opts["subtitleslangs"] = ["all"]
        ydl_opts["subtitlesformat"] = "json3/vtt/best"
    return ydl_opts


def _fetch_ytdlp_info(
    url: str,
    *,
    download: bool,
    output_template: str | None,
    cookies_file: Path | None,
    include_youtube_subtitles: bool = False,
    skip_media_download: bool | None = None,
    js_runtimes: dict[str, dict[str, str]] | None = None,
    impersonate: Any | None = None,
    log: LogFn | None = None,
) -> dict[str, Any]:
    _ensure_ytdlp_available()
    import yt_dlp

    ydl_opts = _build_ytdlp_options(
        output_template=output_template,
        cookies_file=cookies_file,
        skip_media_download=(not download) if skip_media_download is None else skip_media_download,
        include_youtube_subtitles=include_youtube_subtitles and _is_youtube_url(url),
        js_runtimes=js_runtimes,
        impersonate=impersonate,
        log=log,
    )

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=download) or {}
    except yt_dlp.utils.DownloadError as exc:
        action = "download" if download else "metadata fetch"
        if _is_youtube_url(url) and _is_youtube_challenge_error(str(exc)):
            raise RuntimeError(YOUTUBE_CHALLENGE_HELP) from exc
        raise RuntimeError(f"yt-dlp {action} failed for {url}:\n{exc}") from exc

    if not isinstance(info, dict):
        raise RuntimeError(f"yt-dlp returned unexpected metadata for {url}")
    return info


def _try_fetch_optional_youtube_subtitles(
    *,
    url: str,
    cache_dir: Path,
    info_dict: dict[str, Any],
    cookies_file: Path | None,
    js_runtimes: dict[str, dict[str, str]] | None,
    impersonate: Any | None,
    log: LogFn | None,
) -> dict[str, Any]:
    log_fn = log or _noop_log
    if not _is_youtube_url(url):
        return info_dict
    if not _has_subtitle_tracks(info_dict):
        return info_dict
    if _has_local_subtitle_files(info_dict, cache_dir):
        return _save_cached_info(cache_dir, info_dict)

    try:
        subtitle_info = _fetch_ytdlp_info(
            url,
            download=True,
            output_template=str(cache_dir / "source.%(ext)s"),
            cookies_file=cookies_file,
            include_youtube_subtitles=True,
            skip_media_download=True,
            js_runtimes=js_runtimes,
            impersonate=impersonate,
            log=log_fn,
        )
    except Exception as exc:  # noqa: BLE001
        reason = _format_subtitle_failure_reason(exc if isinstance(exc, Exception) else RuntimeError(str(exc)))
        log_fn(f"YouTube subtitles unavailable ({reason}), falling back to Whisper cache/transcription")
        return _save_cached_info(cache_dir, info_dict)

    normalized = _save_cached_info(cache_dir, subtitle_info)
    if _has_local_subtitle_files(normalized, cache_dir):
        log_fn("YouTube subtitles cached for local reuse.")
    return normalized


def _load_local_source_info(local_path: Path) -> dict[str, Any]:
    info_path = local_path.parent / "source_info.json"
    if not info_path.exists():
        return {}

    payload = json.loads(info_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Cached metadata file is invalid: {info_path}")
    return payload


def extract_youtube_chapters(url: str | None = None, info_dict: dict[str, Any] | None = None) -> list[dict[str, float | str]]:
    """Return normalized chapter metadata for a YouTube video.

    If *info_dict* contains chapters, they are used directly. Only when no usable
    info dict is available and *url* is provided does this fall back to a
    metadata-only yt-dlp request.
    """
    if info_dict is not None:
        return _parse_chapters_from_info(info_dict)

    if not url:
        return []

    try:
        cookies_file = Path(__file__).resolve().parents[1] / "cookies.txt"
        js_runtimes = _resolve_js_runtimes(None if not _is_youtube_url(url) else _noop_log)
        impersonate = _resolve_impersonate_target(js_runtimes, None)
        info = _fetch_ytdlp_info(
            url,
            download=False,
            output_template=None,
            cookies_file=cookies_file,
            js_runtimes=js_runtimes,
            impersonate=impersonate,
        )
    except Exception:  # noqa: BLE001
        return []
    return _parse_chapters_from_info(info)


def resolve_input_video(
    input_value: str,
    downloads_root: Path,
    *,
    prefer_youtube_subtitles: bool = False,
    log: LogFn | None = None,
) -> tuple[Path, bool, dict[str, Any]]:
    """Resolve a local file path or download URL.

    Returns ``(video_path, was_downloaded, info_dict)`` where *info_dict* is a
    compact metadata payload loaded from or saved to ``source_info.json``.
    """
    log_fn = log or _noop_log
    raw_input = input_value.strip()
    if not is_http_url(raw_input):
        local_path = _resolve_local_input_path(raw_input, downloads_root)
        if not local_path.exists():
            raise FileNotFoundError(f"Input video not found: {local_path}")
        if not _is_valid_source_media_file(local_path):
            raise ValueError(
                f"Input path is not a supported video file: {local_path}. "
                "Expected one of: .mp4, .webm, .mkv, .mov, .avi"
            )
        return local_path, False, _load_local_source_info(local_path)

    if _is_youtube_url(raw_input):
        video_id = _extract_youtube_video_id(raw_input)
        if not video_id:
            raise ValueError(f"Invalid YouTube URL: could not determine video ID from {raw_input}")

    cache_dir = _source_cache_dir(raw_input, downloads_root)
    cache_dir.mkdir(parents=True, exist_ok=True)
    info_path = _source_info_path(cache_dir)
    cookies_file = Path(__file__).resolve().parents[1] / "cookies.txt"
    youtube_source = _is_youtube_url(raw_input)
    js_runtimes = _resolve_js_runtimes(log_fn if youtube_source else None)
    impersonate = _resolve_impersonate_target(js_runtimes, log_fn if youtube_source else None)

    cached_source_exists = any(_is_valid_source_media_file(path) for path in cache_dir.glob("source*"))
    cached_info = _load_cached_info(cache_dir) if info_path.exists() else None

    if cached_info and not cached_source_exists:
        raise RuntimeError(
            f"Cached metadata exists but the source video is missing in {cache_dir}. "
            "Delete the stale cache folder and retry."
        )

    if cached_source_exists and cached_info:
        _validate_source_info(cached_info, source_url=raw_input)
        source_path = _find_downloaded_source(cache_dir, cached_info)
        if prefer_youtube_subtitles and youtube_source:
            cached_info = _try_fetch_optional_youtube_subtitles(
                url=raw_input,
                cache_dir=cache_dir,
                info_dict=cached_info,
                cookies_file=cookies_file,
                js_runtimes=js_runtimes,
                impersonate=impersonate,
                log=log_fn,
            )
            source_path = _find_downloaded_source(cache_dir, cached_info)
        if _normalize_text(cached_info.get("source_filename")) != source_path.name:
            cached_info = dict(cached_info)
            cached_info["source_filename"] = source_path.name
            info_path.write_text(json.dumps(cached_info, indent=2), encoding="utf-8")
        return source_path, False, cached_info

    output_template = str(cache_dir / "source.%(ext)s")

    if cached_source_exists and not cached_info:
        metadata_info = _fetch_ytdlp_info(
            raw_input,
            download=False,
            output_template=output_template,
            cookies_file=cookies_file,
            js_runtimes=js_runtimes,
            impersonate=impersonate,
            log=log_fn,
        )
        normalized_info = _save_cached_info(cache_dir, metadata_info)
        if prefer_youtube_subtitles and youtube_source:
            normalized_info = _try_fetch_optional_youtube_subtitles(
                url=raw_input,
                cache_dir=cache_dir,
                info_dict=normalized_info,
                cookies_file=cookies_file,
                js_runtimes=js_runtimes,
                impersonate=impersonate,
                log=log_fn,
            )
        _validate_source_info(normalized_info, source_url=raw_input)
        source_path = _find_downloaded_source(cache_dir, normalized_info)
        normalized_info["source_filename"] = source_path.name
        info_path.write_text(json.dumps(normalized_info, indent=2), encoding="utf-8")
        return source_path, False, normalized_info

    captured_info = _fetch_ytdlp_info(
        raw_input,
        download=True,
        output_template=output_template,
        cookies_file=cookies_file,
        include_youtube_subtitles=False,
        js_runtimes=js_runtimes,
        impersonate=impersonate,
        log=log_fn,
    )
    normalized_info = _save_cached_info(cache_dir, captured_info)
    if prefer_youtube_subtitles and youtube_source:
        normalized_info = _try_fetch_optional_youtube_subtitles(
            url=raw_input,
            cache_dir=cache_dir,
            info_dict=normalized_info,
            cookies_file=cookies_file,
            js_runtimes=js_runtimes,
            impersonate=impersonate,
            log=log_fn,
        )
    _validate_source_info(normalized_info, source_url=raw_input)
    source_path = _find_downloaded_source(cache_dir, normalized_info)

    if not source_path.exists():
        raise RuntimeError(f"Source video not found after download: {source_path}")

    normalized_info["source_filename"] = source_path.name
    info_path.write_text(json.dumps(normalized_info, indent=2), encoding="utf-8")

    return source_path, True, normalized_info


def resolve_cached_input_video(input_value: str, downloads_root: Path) -> tuple[Path, dict[str, Any]]:
    """Resolve a local source or an already-cached URL without downloading.

    This is used by preview/reference-frame flows where a fast local frame
    extraction is useful, but triggering a fresh yt-dlp download would be
    surprising.
    """
    raw_input = input_value.strip()
    if not raw_input:
        raise ValueError("Source is required.")

    if not is_http_url(raw_input):
        local_path = _resolve_local_input_path(raw_input, downloads_root)
        if not local_path.exists():
            raise FileNotFoundError(f"Input video not found: {local_path}")
        if not _is_valid_source_media_file(local_path):
            raise ValueError(
                f"Input path is not a supported video file: {local_path}. "
                "Expected one of: .mp4, .webm, .mkv, .mov, .avi"
            )
        return local_path, _load_local_source_info(local_path)

    cache_dir = _source_cache_dir(raw_input, downloads_root)
    if not cache_dir.exists():
        raise FileNotFoundError(
            "Source URL is not cached locally yet. Process or upload the video first, "
            "then load a reference frame."
        )

    info_path = _source_info_path(cache_dir)
    cached_info = _load_cached_info(cache_dir) if info_path.exists() else None
    cached_source_exists = any(_is_valid_source_media_file(path) for path in cache_dir.glob("source*"))
    if not cached_source_exists:
        raise FileNotFoundError(
            f"Cached source video is missing in {cache_dir}. Process or upload the video first."
        )

    if cached_info:
        _validate_source_info(cached_info, source_url=raw_input)
    source_path = _find_downloaded_source(cache_dir, cached_info)
    return source_path, cached_info or {}
