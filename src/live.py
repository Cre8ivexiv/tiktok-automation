from __future__ import annotations

import json
import shutil
import subprocess
import sys
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LIVESTREAMS_ROOT = PROJECT_ROOT / "livestreams"

LogFn = Callable[[str], None]


def _now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _new_live_job_id() -> str:
    return f"live_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"


def _is_supported_live_url(url: str) -> bool:
    parsed = urlparse(url.strip())
    return parsed.scheme.lower() in {"http", "https", "rtmp", "rtmps"} and bool(parsed.netloc)


def _resolve_binary(name: str) -> str:
    bin_dir = PROJECT_ROOT / "bin"
    candidates = [name, f"{name}.exe", str(bin_dir / name), str(bin_dir / f"{name}.exe")]
    for candidate in candidates:
        if shutil.which(candidate):
            return candidate
    raise RuntimeError(
        f"{name} not found. Install it and add to PATH, or place {name}.exe in the project's bin/ folder."
    )


def _tail_text(path: Path, limit: int = 1600) -> str:
    if not path.exists():
        return ""
    data = path.read_bytes()[-limit:]
    return data.decode("utf-8", errors="ignore").strip()


@dataclass
class LiveRecording:
    live_job_id: str
    url: str
    output_dir: Path
    output_path: Path
    start_time: str
    status: str = "starting"
    process_id: int | None = None
    error: str | None = None
    command: list[str] = field(default_factory=list)
    process: subprocess.Popen[str] | None = field(default=None, repr=False)

    @property
    def metadata_path(self) -> Path:
        return self.output_dir / "metadata.json"

    @property
    def stderr_path(self) -> Path:
        return self.output_dir / "ffmpeg.stderr.log"

    @property
    def stdout_path(self) -> Path:
        return self.output_dir / "ffmpeg.stdout.log"

    def to_dict(self, *, include_process: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "live_job_id": self.live_job_id,
            "url": self.url,
            "start_time": self.start_time,
            "output_dir": str(self.output_dir.resolve()),
            "output_path": str(self.output_path.resolve()),
            "process_id": self.process_id,
            "status": self.status,
            "error": self.error,
            "command": self.command,
        }
        if include_process:
            payload["process"] = self.process
        return payload

    def write_metadata(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


class LiveRecorderManager:
    def __init__(self, root: Path = LIVESTREAMS_ROOT) -> None:
        self.root = root
        self._recordings: dict[str, LiveRecording] = {}
        self._lock = threading.Lock()

    def start(self, url: str, *, ffmpeg_bin: str = "ffmpeg", log: LogFn | None = None) -> dict[str, Any]:
        clean_url = str(url or "").strip()
        if not _is_supported_live_url(clean_url):
            raise ValueError("Enter a valid livestream URL.")

        live_job_id = _new_live_job_id()
        output_dir = (self.root / live_job_id).resolve()
        recording = LiveRecording(
            live_job_id=live_job_id,
            url=clean_url,
            output_dir=output_dir,
            output_path=output_dir / "buffer.ts",
            start_time=_now_text(),
        )
        recording.write_metadata()
        with self._lock:
            self._recordings[live_job_id] = recording

        thread = threading.Thread(
            target=self._start_process,
            args=(live_job_id, ffmpeg_bin, log),
            daemon=True,
        )
        thread.start()
        return self.status(live_job_id)

    def _get(self, live_job_id: str) -> LiveRecording:
        with self._lock:
            recording = self._recordings.get(live_job_id)
        if recording is None:
            raise KeyError(f"Livestream recording not found: {live_job_id}")
        return recording

    def _resolve_stream_url(self, url: str) -> str:
        direct_exts = (".m3u8", ".mpd", ".ts")
        if urlparse(url).path.lower().endswith(direct_exts):
            return url

        cmd = [
            sys.executable,
            "-m",
            "yt_dlp",
            "-f",
            "best",
            "-g",
            url,
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=45)
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip() or "yt-dlp could not resolve this livestream URL."
            raise RuntimeError(detail.splitlines()[-1])

        candidates = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if not candidates:
            raise RuntimeError("yt-dlp did not return a playable livestream URL.")
        return candidates[0]

    def _start_process(self, live_job_id: str, ffmpeg_bin: str, log: LogFn | None) -> None:
        log_fn = log or (lambda _message: None)
        recording = self._get(live_job_id)
        try:
            resolved_ffmpeg = _resolve_binary(ffmpeg_bin)
            stream_url = self._resolve_stream_url(recording.url)
            cmd = [
                resolved_ffmpeg,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-reconnect",
                "1",
                "-reconnect_streamed",
                "1",
                "-reconnect_delay_max",
                "5",
                "-i",
                stream_url,
                "-c",
                "copy",
                "-f",
                "mpegts",
                str(recording.output_path),
            ]
            recording.command = cmd
            recording.output_dir.mkdir(parents=True, exist_ok=True)
            stdout_fh = recording.stdout_path.open("w", encoding="utf-8")
            stderr_fh = recording.stderr_path.open("w", encoding="utf-8")
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=stdout_fh,
                stderr=stderr_fh,
                text=True,
            )
            recording.process = process
            recording.process_id = process.pid
            recording.status = "recording"
            recording.error = None
            recording.write_metadata()
            log_fn(f"Livestream recording started: {live_job_id}")
        except Exception as exc:  # noqa: BLE001
            recording.status = "failed"
            recording.error = str(exc)
            recording.write_metadata()
            log_fn(f"Livestream recording failed: {exc}")

    def stop(self, live_job_id: str) -> dict[str, Any]:
        recording = self._get(live_job_id)
        process = recording.process
        if process is None:
            if recording.status in {"failed", "stopped"}:
                return self.status(live_job_id)
            recording.status = "stopped"
            recording.write_metadata()
            return self.status(live_job_id)

        if process.poll() is None:
            recording.status = "stopping"
            recording.write_metadata()
            try:
                if process.stdin:
                    process.stdin.write("q\n")
                    process.stdin.flush()
                process.wait(timeout=10)
            except Exception:  # noqa: BLE001
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)

        recording.status = "stopped" if process.returncode in {0, None} else "failed"
        if recording.status == "failed":
            recording.error = _tail_text(recording.stderr_path) or f"ffmpeg exited with {process.returncode}"
        recording.write_metadata()
        return self.status(live_job_id)

    def status(self, live_job_id: str) -> dict[str, Any]:
        recording = self._get(live_job_id)
        process = recording.process
        if process is not None:
            returncode = process.poll()
            if returncode is None and recording.status in {"starting", "recording"}:
                recording.status = "recording"
            elif returncode is not None and recording.status not in {"stopped", "failed"}:
                recording.status = "stopped" if returncode == 0 else "failed"
                if recording.status == "failed":
                    recording.error = _tail_text(recording.stderr_path) or f"ffmpeg exited with {returncode}"
            recording.write_metadata()

        payload = recording.to_dict()
        payload["duration_seconds"] = self.duration_seconds(live_job_id, allow_elapsed_fallback=True)
        payload["buffer_exists"] = recording.output_path.exists()
        payload["buffer_size_bytes"] = recording.output_path.stat().st_size if recording.output_path.exists() else 0
        return payload

    def duration_seconds(self, live_job_id: str, *, allow_elapsed_fallback: bool = False) -> float:
        recording = self._get(live_job_id)
        if recording.output_path.exists() and recording.output_path.stat().st_size > 0:
            try:
                ffprobe_bin = _resolve_binary("ffprobe")
                result = subprocess.run(
                    [
                        ffprobe_bin,
                        "-v",
                        "error",
                        "-show_entries",
                        "format=duration",
                        "-of",
                        "default=noprint_wrappers=1:nokey=1",
                        str(recording.output_path),
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=10,
                )
                if result.returncode == 0:
                    duration_text = result.stdout.strip()
                    if duration_text and duration_text.upper() != "N/A":
                        return max(0.0, float(duration_text))
            except Exception:  # noqa: BLE001
                pass

        if allow_elapsed_fallback and recording.status == "recording":
            try:
                started_at = datetime.strptime(recording.start_time, "%Y-%m-%d %H:%M:%S")
                return max(0.0, (datetime.now() - started_at).total_seconds())
            except Exception:  # noqa: BLE001
                return 0.0
        return 0.0

    def buffer_path(self, live_job_id: str) -> Path:
        recording = self._get(live_job_id)
        if not recording.output_path.exists() or recording.output_path.stat().st_size <= 0:
            raise FileNotFoundError("Livestream buffer is not ready yet.")
        return recording.output_path
