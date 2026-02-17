from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from urllib.parse import parse_qs, urlparse


VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi", ".m2ts"}


def is_http_url(value: str) -> bool:
    parsed = urlparse(value.strip())
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _slugify(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._-")
    return text[:80] if text else "source"


def _safe_name_from_url(url: str) -> str:
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    video_id = query.get("v", [None])[0]
    if video_id:
        return _slugify(video_id)
    path_slug = Path(parsed.path).stem
    if path_slug:
        return _slugify(path_slug)
    return _slugify(parsed.netloc or "source")


def _ensure_ytdlp_available() -> str:
    ytdlp_bin = shutil.which("yt-dlp")
    if not ytdlp_bin:
        raise SystemExit("yt-dlp is required for URL inputs. Install with: pip install yt-dlp")
    return ytdlp_bin


def _find_downloaded_source(download_dir: Path) -> Path:
    candidates = sorted(download_dir.glob("source.*"), key=lambda p: p.stat().st_mtime, reverse=True)
    for candidate in candidates:
        if candidate.is_file() and candidate.suffix.lower() in VIDEO_SUFFIXES:
            return candidate.resolve()
    if candidates:
        return candidates[0].resolve()
    raise RuntimeError(f"yt-dlp finished but no downloaded source file found in {download_dir}")


def resolve_input_video(input_value: str, downloads_root: Path) -> tuple[Path, bool]:
    """Resolve a local file path or download URL, returning (video_path, was_downloaded)."""
    raw_input = input_value.strip()
    if not is_http_url(raw_input):
        local_path = Path(raw_input).expanduser().resolve()
        if not local_path.exists():
            raise FileNotFoundError(f"Input video not found: {local_path}")
        return local_path, False

    ytdlp_bin = _ensure_ytdlp_available()
    safe_name = _safe_name_from_url(raw_input)
    download_dir = downloads_root / safe_name
    download_dir.mkdir(parents=True, exist_ok=True)
    output_template = download_dir / "source.%(ext)s"

    cmd = [ytdlp_bin, "-o", str(output_template), raw_input]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"yt-dlp download failed ({result.returncode}).\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    return _find_downloaded_source(download_dir), True

