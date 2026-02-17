
#!/usr/bin/env python3
"""
Dependencies:
    pip install -U yt-dlp
    pip install srt
    FFmpeg must be installed and available on PATH.
    FFmpeg build must include:
      - NVIDIA encoder: hevc_nvenc
      - filters: subtitles, drawtext, pad

Usage examples:
    Windows batch style (dynamic script directory):
        python "%~dp0<your_script_name>.py" --url "https://www.youtube.com/watch?v=VIDEO_ID" --srt "C:/path/subtitles.srt" --splits "C:/path/splits.csv" --bottom-padding 220

    PowerShell equivalent:
        python "$PSScriptRoot/<your_script_name>.py" --url "https://www.youtube.com/watch?v=VIDEO_ID" --srt "/path/to/subtitles.srt" --splits "/path/to/splits.csv" --bottom-padding 220
"""

import argparse
import csv
import shlex
import shutil
import subprocess
import tempfile
import textwrap
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import List, Optional

import srt


@dataclass
class Segment:
    start: float
    end: float


def run_command(cmd: List[str]) -> None:
    print("Running:", " ".join(shlex.quote(x) for x in cmd))
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}):\n{result.stdout}")


def ensure_nvenc_available(ffmpeg_bin: str) -> None:
    result = subprocess.run(
        [ffmpeg_bin, "-hide_banner", "-encoders"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if result.returncode != 0 or "hevc_nvenc" not in result.stdout:
        raise SystemExit(
            "FFmpeg is missing 'hevc_nvenc'. Install an FFmpeg build with NVIDIA NVENC support."
        )


def parse_time_to_seconds(value: str) -> float:
    text = value.strip().replace(",", ".")
    if not text:
        raise ValueError("Empty time value")

    try:
        return float(text)  # raw seconds
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
    total_ms = int(round(max(seconds, 0) * 1000))
    total_sec, ms = divmod(total_ms, 1000)
    minutes, sec = divmod(total_sec, 60)
    hours, minute = divmod(minutes, 60)
    return f"{hours:02d}:{minute:02d}:{sec:02d}.{ms:03d}"


def read_splits_csv(csv_path: Path) -> List[Segment]:
    segments: List[Segment] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("CSV has no header. Expected columns: start,end")

        normalized = {name.strip().lower(): name for name in reader.fieldnames}
        start_col = normalized.get("start")
        end_col = normalized.get("end")
        if not start_col or not end_col:
            raise ValueError(f"CSV must contain columns 'start' and 'end'. Found: {reader.fieldnames}")

        for row_num, row in enumerate(reader, start=2):
            start_raw = (row.get(start_col) or "").strip()
            end_raw = (row.get(end_col) or "").strip()
            if not start_raw and not end_raw:
                continue

            start = parse_time_to_seconds(start_raw)
            end = parse_time_to_seconds(end_raw)
            if end <= start:
                raise ValueError(f"Invalid segment at line {row_num}: end must be > start")

            segments.append(Segment(start=start, end=end))

    if not segments:
        raise ValueError("No valid segments found in CSV.")
    return segments


def format_hhmmss_from_seconds(seconds: int) -> str:
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def download_with_ytdlp(url: str, work_dir: Path, test_first_seconds: int) -> tuple[Path, str]:
    title_cmd = ["yt-dlp", "--no-warnings", "--get-title", url]
    try:
        title_result = subprocess.run(
            title_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("yt-dlp is not installed or not found in PATH.") from exc

    if title_result.returncode != 0:
        title_output = "\n".join(
            part for part in [title_result.stdout, title_result.stderr] if part
        ).strip()
        raise RuntimeError(
            f"yt-dlp title fetch failed ({title_result.returncode}):\n{title_output}"
        )

    title_string = title_result.stdout.strip()
    title_string = " ".join(title_string.splitlines()).strip()
    if not title_string:
        title_string = "Untitled"

    output_template = work_dir / "source_merged.%(ext)s"
    download_cmd = [
        "yt-dlp",
        "--no-warnings",
        "-f",
        "bv*+ba/b",
        "--merge-output-format",
        "mp4",
        "-o",
        str(output_template),
    ]
    if test_first_seconds > 0:
        end_time = format_hhmmss_from_seconds(test_first_seconds)
        download_cmd.extend(["--download-sections", f"*00:00:00-{end_time}"])
    download_cmd.append(url)

    try:
        download_result = subprocess.run(
            download_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("yt-dlp is not installed or not found in PATH.") from exc

    if download_result.returncode != 0:
        raise RuntimeError(
            f"yt-dlp download failed ({download_result.returncode}):\n{download_result.stdout}"
        )

    merged_path = work_dir / "source_merged.mp4"
    if not merged_path.exists():
        raise RuntimeError(
            "yt-dlp download did not produce source_merged.mp4:\n"
            f"{download_result.stdout}"
        )

    return merged_path, title_string


def escape_filter_value(value: str) -> str:
    return (
        value.replace("\\", "\\\\")
        .replace("\n", r"\n")
        .replace(":", r"\:")
        .replace("'", r"\'")
        .replace(",", r"\,")
        .replace("[", r"\[")
        .replace("]", r"\]")
        .replace("%", r"\%")
    )


def wrap_title(text: str, max_chars: int, max_lines: int) -> str:
    clean = " ".join(text.split())
    if not clean:
        return "Untitled"

    wrapped = textwrap.wrap(
        clean,
        width=max(1, max_chars),
        break_long_words=True,
        break_on_hyphens=False,
    )
    if not wrapped:
        return "Untitled"

    if len(wrapped) > max_lines:
        wrapped = wrapped[:max_lines]
        last = wrapped[-1].rstrip()
        if len(last) >= max_chars:
            last = last[: max(1, max_chars - 1)].rstrip()
        wrapped[-1] = f"{last}\u2026"

    return "\n".join(wrapped)


def write_title_file(path: Path, title_text: str) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(title_text)


def filter_path(path: Path) -> str:
    return escape_filter_value(path.resolve().as_posix())


def pick_fontfile() -> Optional[Path]:
    candidates = [
        Path("C:/Windows/Fonts/arial.ttf"),  # Windows
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),  # Linux common
        Path("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"),  # Linux alt
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def build_video_filter(
    shifted_srt: Optional[Path],
    title_text_file: Path,
    part_number: int,
    bottom_padding: int,
    out_width: int,
    out_height: int,
    raise_px: int,
    video_y_scale: float,
    top_bar_height: int,
    title_font_size: int,
    part_font_size: int,
    fontfile: Optional[Path],
    content_scale: float = 1.0,
) -> str:
    title_file = filter_path(title_text_file)
    part_text = escape_filter_value(f"Part {part_number}")

    if fontfile:
        font_expr = f"fontfile='{filter_path(fontfile)}'"
    else:
        font_expr = "font='Sans'"

    safe_video_y_scale = max(video_y_scale, 0.01)

    # Vertical output pipeline:
    # 1) Scale source to fixed width (aspect ratio preserved)
    # 2) Apply vertical-only scaling (width stays fixed)
    # 3) Pad to out_width x out_height on black canvas (centered)
    # 4) Shift content upward via overlay on black base
    # 4) Burn subtitles and draw title/part labels
    filters = [
        "[0:v]setpts=PTS-STARTPTS,"
        f"scale={out_width}:-2,"
        f"scale={out_width}:ih*{safe_video_y_scale}:eval=frame,"
        f"pad={out_width}:{out_height}:({out_width}-iw)/2:({out_height}-ih)/2:color=black[padded]",
        f"color=c=black:size={out_width}x{out_height}[base]",
        f"[base][padded]overlay=x=0:y=-{raise_px}:shortest=1[composed]",
    ]

    if bottom_padding > 0:
        part_y = f"h-{bottom_padding}+({bottom_padding}-text_h)/2"
    else:
        part_y = "h-text_h-40"

    title_input = "composed"
    if shifted_srt is not None:
        subs_path = filter_path(shifted_srt)
        filters.append(f"[composed]subtitles=filename='{subs_path}'[subbed]")
        title_input = "subbed"

    filters.append(
        f"[{title_input}]drawtext={font_expr}:textfile='{title_file}':"
        "x=(w-text_w)/2:y=30:"
        "line_spacing=10:"
        f"fontsize={title_font_size}:fontcolor=black:"
        "box=1:boxcolor=white@0.95:boxborderw=26[title]"
    )
    filters.append(
        f"[title]drawtext={font_expr}:text='{part_text}':"
        f"x=(w-text_w)/2:y={part_y}:"
        f"fontsize={part_font_size}:fontcolor=black:"
        "box=1:boxcolor=white@0.95:boxborderw=26[vout]"
    )

    return ";".join(filters)


def shift_subtitles_for_segment(
    subtitles: List[srt.Subtitle], segment_start: float, segment_end: float
) -> List[srt.Subtitle]:
    start_td = timedelta(seconds=segment_start)
    end_td = timedelta(seconds=segment_end)

    shifted: List[srt.Subtitle] = []
    for sub in subtitles:
        overlap_start = max(sub.start, start_td)
        overlap_end = min(sub.end, end_td)
        if overlap_end <= overlap_start:
            continue

        shifted.append(
            srt.Subtitle(
                index=len(shifted) + 1,
                start=overlap_start - start_td,
                end=overlap_end - start_td,
                content=sub.content,
                proprietary=sub.proprietary,
            )
        )
    return shifted


def render_segment(
    ffmpeg_bin: str,
    source_video: Path,
    shifted_srt: Optional[Path],
    title_text_file: Path,
    output_path: Path,
    segment: Segment,
    part_number: int,
    bottom_padding: int,
    out_width: int,
    out_height: int,
    raise_px: int,
    video_y_scale: float,
    content_scale: float,
    top_bar_height: int,
    title_font_size: int,
    part_font_size: int,
    fontfile: Optional[Path],
) -> None:
    duration = segment.end - segment.start
    if duration <= 0:
        raise ValueError("Segment duration must be positive.")

    vf = build_video_filter(
        shifted_srt=shifted_srt,
        title_text_file=title_text_file,
        part_number=part_number,
        bottom_padding=bottom_padding,
        out_width=out_width,
        out_height=out_height,
        raise_px=raise_px,
        video_y_scale=video_y_scale,
        content_scale=content_scale,
        top_bar_height=top_bar_height,
        title_font_size=title_font_size,
        part_font_size=part_font_size,
        fontfile=fontfile,
    )

    cmd = [
        ffmpeg_bin,
        "-y",
        "-ss",
        format_ffmpeg_time(segment.start),
        "-t",
        format_ffmpeg_time(duration),
        "-i",
        str(source_video),
        "-filter_complex",
        vf,
        "-af",
        "asetpts=PTS-STARTPTS",
        "-map",
        "[vout]",
        "-map",
        "0:a:0",
        "-c:v",
        "hevc_nvenc",
        "-preset",
        "p7",
        "-rc",
        "vbr_hq",
        "-cq",
        "14",
        "-b:v",
        "0",
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
    run_command(cmd)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download YouTube video, split into parts, shift/burn subtitles, and export with NVENC."
    )
    parser.add_argument("--url", help="YouTube URL")
    parser.add_argument("--input-video", type=Path, help="Local input video file path")
    parser.add_argument("--title", help="Optional main title override")
    parser.add_argument("--srt", type=Path, help="Path to original SRT subtitle file (optional)")
    parser.add_argument("--splits", type=Path, help="CSV file with columns: start,end")
    parser.add_argument(
        "--bottom-padding",
        type=int,
        default=260,
        help="Bottom padding height in pixels (controls empty area + 'Part X' label space).",
    )
    parser.add_argument("--out-width", type=int, default=1080, help="Final output width in pixels")
    parser.add_argument("--out-height", type=int, default=1920, help="Final output height in pixels")
    parser.add_argument("--video-y-scale", type=float, default=1.0, help="Vertical stretch factor")
    parser.add_argument("--content-scale", type=float, default=1.0, help="Scale factor for composed content")
    parser.add_argument("--raise-px", type=int, default=140, help="Move video up by this many pixels")
    parser.add_argument(
        "--test-first-seconds",
        type=int,
        default=0,
        help="Optional: only download the first N seconds from YouTube for test runs.",
    )
    parser.add_argument("--top-bar-height", type=int, default=120, help="Top title bar height in pixels")
    parser.add_argument("--title-font-size", type=int, default=52, help="Top title text size")
    parser.add_argument("--title-max-chars", type=int, default=28, help="Approximate max chars per title line")
    parser.add_argument("--title-max-lines", type=int, default=2, help="Maximum number of title lines")
    parser.add_argument("--part-font-size", type=int, default=42, help="'Part X' text size")
    parser.add_argument("--ffmpeg-bin", default="ffmpeg", help="FFmpeg executable path/name")
    args = parser.parse_args()

    from datetime import datetime

    script_dir = Path(__file__).resolve().parent
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    auto_output_dir = script_dir / f"clips_{timestamp}"
    auto_output_dir.mkdir(parents=True, exist_ok=True)

    args.output_dir = auto_output_dir

    print(f"Output directory created: {args.output_dir}")

    if shutil.which(args.ffmpeg_bin) is None:
        raise SystemExit(f"FFmpeg not found: {args.ffmpeg_bin}")
    ensure_nvenc_available(args.ffmpeg_bin)

    if args.bottom_padding < 0:
        raise SystemExit("--bottom-padding must be >= 0")
    if args.out_width <= 0:
        raise SystemExit("--out-width must be > 0")
    if args.out_height <= 0:
        raise SystemExit("--out-height must be > 0")
    if args.content_scale <= 0:
        raise SystemExit("--content-scale must be > 0")
    if args.video_y_scale <= 0:
        raise SystemExit("--video-y-scale must be > 0")
    if args.title_max_chars <= 0:
        raise SystemExit("--title-max-chars must be > 0")
    if args.title_max_lines <= 0:
        raise SystemExit("--title-max-lines must be > 0")
    if args.raise_px < 0:
        raise SystemExit("--raise-px must be >= 0")
    if args.test_first_seconds < 0:
        raise SystemExit("--test-first-seconds must be >= 0")
    if args.input_video is None and not args.url:
        raise SystemExit("Provide --input-video or --url.")
    if args.input_video is not None and not args.input_video.exists():
        raise SystemExit(f"Input video file not found: {args.input_video}")
    if args.srt is not None and not args.srt.exists():
        raise SystemExit(f"SRT file not found: {args.srt}")
    if args.test_first_seconds == 0:
        if args.splits is None:
            raise SystemExit("--splits is required when --test-first-seconds is 0")
        if not args.splits.exists():
            raise SystemExit(f"CSV file not found: {args.splits}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    original_subs: List[srt.Subtitle] = []
    if args.srt is not None:
        original_subs = list(srt.parse(args.srt.read_text(encoding="utf-8-sig")))
    if args.test_first_seconds > 0:
        segments = [Segment(start=0.0, end=float(args.test_first_seconds))]
    else:
        segments = read_splits_csv(args.splits)
    fontfile = pick_fontfile()

    if fontfile:
        print(f"Using font file: {fontfile}")
    else:
        print("No known font file found. drawtext will use font='Sans'.")

    with tempfile.TemporaryDirectory(prefix="yt_split_work_") as temp_dir_str:
        temp_dir = Path(temp_dir_str)

        if args.input_video is not None:
            source_video = args.input_video.resolve()
            main_title = args.input_video.stem
            print(f"Using local source video: {source_video}")
        else:
            print("Downloading source video...")
            source_video, main_title = download_with_ytdlp(args.url, temp_dir, args.test_first_seconds)
            print(f"Prepared source video: {source_video}")

        if args.title and args.title.strip():
            main_title = args.title
        main_title = " ".join(main_title.splitlines()).strip()
        if not main_title:
            main_title = "Untitled"

        for i, seg in enumerate(segments, start=1):
            shifted_srt_path: Optional[Path] = None
            if original_subs:
                shifted_subs = shift_subtitles_for_segment(original_subs, seg.start, seg.end)
                shifted_srt_path = temp_dir / f"shifted_part_{i}.srt"
                shifted_srt_path.write_text(srt.compose(shifted_subs), encoding="utf-8")

            wrapped_title = wrap_title(main_title, args.title_max_chars, args.title_max_lines)
            title_text_path = temp_dir / f"title_part_{i}.txt"
            write_title_file(title_text_path, wrapped_title)

            out_path = args.output_dir / f"part_{i}.mp4"
            print(
                f"Rendering part {i}: {format_ffmpeg_time(seg.start)} -> "
                f"{format_ffmpeg_time(seg.end)}"
            )
            render_segment(
                ffmpeg_bin=args.ffmpeg_bin,
                source_video=source_video,
                shifted_srt=shifted_srt_path,
                title_text_file=title_text_path,
                output_path=out_path,
                segment=seg,
                part_number=i,
                bottom_padding=args.bottom_padding,
                out_width=args.out_width,
                out_height=args.out_height,
                raise_px=args.raise_px,
                video_y_scale=args.video_y_scale,
                content_scale=args.content_scale,
                top_bar_height=args.top_bar_height,
                title_font_size=args.title_font_size,
                part_font_size=args.part_font_size,
                fontfile=fontfile,
            )

    print(f"Done. Files saved in: {args.output_dir}")


if __name__ == "__main__":
    main()

