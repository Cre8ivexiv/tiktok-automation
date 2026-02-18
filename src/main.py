from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .captions import build_caption
from .download import is_http_url, resolve_input_video
from .render import (
    RenderedPart,
    rendered_parts_to_dict,
    render_parts,
    resolve_segments,
)
from .scheduler.base import UploadPayload
from .scheduler.mock import MockScheduler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHANNELS_CONFIG = PROJECT_ROOT / "config" / "channels.json"
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".m4v"}


def load_channels_map(config_path: Path) -> dict[str, dict[str, Any]]:
    if not config_path.exists():
        raise FileNotFoundError(f"Channels config not found: {config_path}")

    data = json.loads(config_path.read_text(encoding="utf-8"))
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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


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


def detect_source_video(channel_dir: Path) -> Path | None:
    preferred = [
        channel_dir / "source.mp4",
        channel_dir / "source.mov",
        channel_dir / "source.mkv",
        channel_dir / "source.webm",
    ]
    for candidate in preferred:
        if candidate.exists():
            return candidate

    discovered = sorted(
        [
            path
            for path in channel_dir.iterdir()
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
        ]
    )
    return discovered[0] if discovered else None


def infer_title(run_title: str | None, source_video: Path, channel_dir: Path) -> str:
    if run_title and run_title.strip():
        return run_title.strip()

    title_file = channel_dir / "title.txt"
    if title_file.exists():
        title_value = title_file.read_text(encoding="utf-8").strip()
        if title_value:
            return title_value

    return source_video.stem


def build_process_output_dir(channel_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (PROJECT_ROOT / "outputs" / channel_name / timestamp).resolve()


def schedule_parts(
    parts_dir: Path,
    title: str,
    channel_name: str,
    account_id: str,
    start_time: datetime,
    interval_min: int,
) -> list[dict[str, Any]]:
    scheduler = MockScheduler()
    part_files = discover_part_files(parts_dir)
    if not part_files:
        raise RuntimeError(f"No rendered files found in {parts_dir}")

    scheduled_items: list[dict[str, Any]] = []
    for index, video_path in enumerate(part_files, start=1):
        scheduled_time = start_time + timedelta(minutes=interval_min * (index - 1))
        caption = build_caption(title=title, part_number=index)
        payload = UploadPayload(
            channel_name=channel_name,
            account_id=account_id,
            video_path=video_path.resolve(),
            caption=caption,
            scheduled_time=scheduled_time,
        )
        response = scheduler.schedule_post(payload)
        scheduled_items.append(
            {
                "part_number": index,
                "video_path": str(video_path.resolve()),
                "caption": caption,
                "scheduled_time": scheduled_time.strftime("%Y-%m-%d %H:%M"),
                "scheduler_response_id": response.get("id"),
                "scheduler_response": response,
            }
        )
    return scheduled_items


def render_pipeline(
    input_video: Path,
    out_dir: Path,
    part_seconds: int,
    cuts_path: Path | None,
    crop_top_px: int,
    title_mask_px: int,
    edge_bar_px: int,
    letterbox_bump_px: int,
    video_y_scale: float,
    y_scale_mode: str,
    render_preset: str,
    output_width: int,
    output_height: int,
    part_overlay_enabled: bool,
    part_label_position: str,
    ffmpeg_bin: str,
    ffprobe_bin: str,
    crf: int,
    preset: str,
) -> tuple[list[RenderedPart], dict[str, Any]]:
    segments, cuts_overrides = resolve_segments(
        input_video=input_video,
        part_seconds=part_seconds,
        cuts_path=cuts_path,
        ffprobe_bin=ffprobe_bin,
    )
    effective_crop_top = cuts_overrides.get("crop_top_px", crop_top_px)
    effective_out_w = cuts_overrides.get("output_width", output_width)
    effective_out_h = cuts_overrides.get("output_height", output_height)

    rendered = render_parts(
        input_video=input_video,
        out_dir=out_dir,
        segments=segments,
        crop_top_px=effective_crop_top,
        title_mask_px=title_mask_px,
        edge_bar_px=edge_bar_px,
        letterbox_bump_px=letterbox_bump_px,
        video_y_scale=video_y_scale,
        y_scale_mode=y_scale_mode,
        render_preset=render_preset,
        output_width=effective_out_w,
        output_height=effective_out_h,
        part_overlay_enabled=part_overlay_enabled,
        part_label_position=part_label_position,
        ffmpeg_bin=ffmpeg_bin,
        ffprobe_bin=ffprobe_bin,
        crf=crf,
        preset=preset,
    )
    return rendered, {
        "crop_top_px": effective_crop_top,
        "title_mask_px": title_mask_px,
        "edge_bar_px": edge_bar_px,
        "letterbox_bump_px": letterbox_bump_px,
        "video_y_scale": video_y_scale,
        "y_scale_mode": y_scale_mode,
        "render_preset": render_preset,
        "output_width": effective_out_w,
        "output_height": effective_out_h,
    }


def cmd_render(args: argparse.Namespace) -> int:
    input_video, was_downloaded = resolve_input_video(
        input_value=args.input,
        downloads_root=(PROJECT_ROOT / "downloads"),
    )
    out_dir = args.out.resolve()
    cuts_path = args.cuts.resolve() if args.cuts else None

    rendered, effective = render_pipeline(
        input_video=input_video,
        out_dir=out_dir,
        part_seconds=args.part_seconds,
        cuts_path=cuts_path,
        crop_top_px=args.crop_top_px,
        title_mask_px=args.title_mask_px,
        edge_bar_px=args.edge_bar_px,
        letterbox_bump_px=args.letterbox_bump_px,
        video_y_scale=args.video_y_scale,
        y_scale_mode=args.y_scale_mode,
        render_preset=args.render_preset,
        output_width=args.output_width,
        output_height=args.output_height,
        part_overlay_enabled=not args.no_part_overlay,
        part_label_position=args.part_label_position,
        ffmpeg_bin=args.ffmpeg_bin,
        ffprobe_bin=args.ffprobe_bin,
        crf=args.crf,
        preset=args.preset,
    )

    manifest = {
        "mode": "render",
        "input": args.input,
        "input_type": "url" if is_http_url(args.input) else "file",
        "downloaded_source": was_downloaded,
        "input_video": str(input_video),
        "title": args.title,
        "part_seconds": args.part_seconds,
        "cuts_path": str(cuts_path) if cuts_path else None,
        "render_config": effective,
        "render_manifest_path": str((out_dir / "render_manifest.json").resolve()),
        "part_overlay_enabled": not args.no_part_overlay,
        "rendered_parts": rendered_parts_to_dict(rendered),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_json(out_dir / "parts_manifest.json", manifest)
    print(f"Rendered {len(rendered)} part(s) to {out_dir}")
    return 0


def cmd_schedule(args: argparse.Namespace) -> int:
    parts_dir = args.parts.resolve()
    channels = load_channels_map(args.channels_config.resolve())
    if args.channel not in channels:
        raise KeyError(f"Channel '{args.channel}' not found in {args.channels_config}")

    channel_payload = channels[args.channel]
    account_id = resolve_account_id(channel_payload, args.channel)
    start_time = parse_start_time(args.start_time)

    scheduled = schedule_parts(
        parts_dir=parts_dir,
        title=args.title,
        channel_name=args.channel,
        account_id=account_id,
        start_time=start_time,
        interval_min=args.interval_min,
    )

    status_path = args.status_file.resolve() if args.status_file else (parts_dir / "status.json")
    status = {
        "mode": "schedule",
        "channel": args.channel,
        "account_id": account_id,
        "title": args.title,
        "parts_dir": str(parts_dir),
        "start_time": start_time.strftime("%Y-%m-%d %H:%M"),
        "interval_min": args.interval_min,
        "rendered_parts": [str(path.resolve()) for path in discover_part_files(parts_dir)],
        "scheduled": scheduled,
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_json(status_path, status)
    print(f"Scheduled {len(scheduled)} part(s). Status saved to {status_path}")
    return 0


def cmd_process(args: argparse.Namespace) -> int:
    channels = load_channels_map(args.channels_config.resolve())
    if args.channel not in channels:
        raise KeyError(f"Channel '{args.channel}' not found in {args.channels_config}")

    channel_payload = channels[args.channel]
    account_id = resolve_account_id(channel_payload, args.channel)
    start_time = parse_start_time(args.start_time)

    input_video, was_downloaded = resolve_input_video(
        input_value=args.input,
        downloads_root=(PROJECT_ROOT / "downloads"),
    )
    cuts_path = args.cuts.resolve() if args.cuts else None
    out_dir = build_process_output_dir(args.channel)

    rendered, effective = render_pipeline(
        input_video=input_video,
        out_dir=out_dir,
        part_seconds=args.part_seconds,
        cuts_path=cuts_path,
        crop_top_px=args.crop_top_px,
        title_mask_px=args.title_mask_px,
        edge_bar_px=args.edge_bar_px,
        letterbox_bump_px=args.letterbox_bump_px,
        video_y_scale=args.video_y_scale,
        y_scale_mode=args.y_scale_mode,
        render_preset=args.render_preset,
        output_width=args.output_width,
        output_height=args.output_height,
        part_overlay_enabled=not args.no_part_overlay,
        part_label_position=args.part_label_position,
        ffmpeg_bin=args.ffmpeg_bin,
        ffprobe_bin=args.ffprobe_bin,
        crf=args.crf,
        preset=args.preset,
    )

    scheduled = schedule_parts(
        parts_dir=out_dir,
        title=args.title,
        channel_name=args.channel,
        account_id=account_id,
        start_time=start_time,
        interval_min=args.interval_min,
    )

    status = {
        "mode": "process",
        "completed": True,
        "input": args.input,
        "input_type": "url" if is_http_url(args.input) else "file",
        "downloaded_source": was_downloaded,
        "source_video": str(input_video.resolve()),
        "channel": args.channel,
        "account_id": account_id,
        "title": args.title,
        "output_dir": str(out_dir),
        "part_seconds": args.part_seconds,
        "cuts_path": str(cuts_path) if cuts_path else None,
        "render_config": effective,
        "render_manifest_path": str((out_dir / "render_manifest.json").resolve()),
        "part_overlay_enabled": not args.no_part_overlay,
        "rendered_parts": rendered_parts_to_dict(rendered),
        "schedule_start_time": start_time.strftime("%Y-%m-%d %H:%M"),
        "interval_min": args.interval_min,
        "scheduled": scheduled,
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    status_path = out_dir / "status.json"
    write_json(status_path, status)
    print(f"Processed {len(rendered)} part(s). Status saved to {status_path}")
    return 0


def cmd_run_folder(args: argparse.Namespace) -> int:
    root = args.root.resolve()
    if not root.exists():
        raise FileNotFoundError(f"Root folder not found: {root}")

    channels = load_channels_map(args.channels_config.resolve())
    folder_start_time = parse_start_time(args.start_time)

    processed = 0
    skipped = 0
    for channel_dir in sorted([item for item in root.iterdir() if item.is_dir()]):
        channel_name = channel_dir.name
        status_path = channel_dir / "status.json"

        if not args.force and status_path.exists():
            existing = json.loads(status_path.read_text(encoding="utf-8"))
            if existing.get("completed") is True:
                skipped += 1
                print(f"Skipping {channel_name}: already completed (use --force to rerun).")
                continue

        if channel_name not in channels:
            skipped += 1
            print(f"Skipping {channel_name}: missing channel mapping in {args.channels_config}.")
            continue

        source_video = detect_source_video(channel_dir)
        if source_video is None:
            skipped += 1
            print(f"Skipping {channel_name}: no source video file found.")
            continue

        cuts_path = channel_dir / "cuts.json"
        cuts_override = cuts_path if cuts_path.exists() else None
        out_dir = channel_dir / "rendered_parts"
        title = infer_title(args.title, source_video=source_video, channel_dir=channel_dir)

        rendered, effective = render_pipeline(
            input_video=source_video,
            out_dir=out_dir,
            part_seconds=args.part_seconds,
            cuts_path=cuts_override,
            crop_top_px=args.crop_top_px,
            title_mask_px=args.title_mask_px,
            edge_bar_px=args.edge_bar_px,
            letterbox_bump_px=args.letterbox_bump_px,
            video_y_scale=args.video_y_scale,
            y_scale_mode=args.y_scale_mode,
            render_preset=args.render_preset,
            output_width=args.output_width,
            output_height=args.output_height,
            part_overlay_enabled=not args.no_part_overlay,
            part_label_position=args.part_label_position,
            ffmpeg_bin=args.ffmpeg_bin,
            ffprobe_bin=args.ffprobe_bin,
            crf=args.crf,
            preset=args.preset,
        )

        channel_payload = channels[channel_name]
        account_id = resolve_account_id(channel_payload, channel_name)
        scheduled = schedule_parts(
            parts_dir=out_dir,
            title=title,
            channel_name=channel_name,
            account_id=account_id,
            start_time=folder_start_time,
            interval_min=args.interval_min,
        )

        status = {
            "completed": True,
            "channel": channel_name,
            "account_id": account_id,
            "source_video": str(source_video.resolve()),
            "title": title,
            "cuts_path": str(cuts_override.resolve()) if cuts_override else None,
            "render_config": effective,
            "render_manifest_path": str((out_dir / "render_manifest.json").resolve()),
            "part_seconds": args.part_seconds,
            "part_overlay_enabled": not args.no_part_overlay,
            "rendered_parts": rendered_parts_to_dict(rendered),
            "schedule_start_time": folder_start_time.strftime("%Y-%m-%d %H:%M"),
            "interval_min": args.interval_min,
            "scheduled": scheduled,
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        write_json(status_path, status)
        processed += 1
        print(f"Processed channel folder: {channel_name}")

    print(f"run-folder complete. processed={processed}, skipped={skipped}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TikTok Scheduler Uploader pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    render_parser = subparsers.add_parser("render", help="Render Part 1..N videos")
    render_parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Source video file path or URL (http/https)",
    )
    render_parser.add_argument("--out", type=Path, required=True, help="Output folder for rendered parts")
    render_parser.add_argument("--title", type=str, required=True, help="Caption title used later for schedule")
    render_parser.add_argument(
        "--part-seconds",
        type=int,
        default=70,
        help="Auto split duration per part in seconds (default: 70)",
    )
    render_parser.add_argument("--cuts", type=Path, default=None, help="Optional cuts.json override")
    render_parser.add_argument("--crop-top-px", type=int, default=0, help="Top pixels to crop")
    render_parser.add_argument("--title-mask-px", type=int, default=0, help="Top title mask height in pixels")
    render_parser.add_argument("--edge-bar-px", type=int, default=45, help="Top/bottom dark band size in pixels")
    render_parser.add_argument(
        "--letterbox-bump-px",
        type=int,
        default=20,
        help="Increase letterbox-mode content height by this many pixels before padding",
    )
    render_parser.add_argument("--video-y-scale", type=float, default=2.08, help="Vertical scale multiplier")
    render_parser.add_argument(
        "--y-scale-mode",
        choices=["manual", "fill", "letterbox"],
        default="letterbox",
        help="manual=use video-y-scale directly, fill=auto-bump to fill frame height, letterbox=no zoom",
    )
    render_parser.add_argument("--render-preset", type=str, default="legacy", help="Rendering preset")
    render_parser.add_argument("--output-width", type=int, default=1080, help="Output width")
    render_parser.add_argument("--output-height", type=int, default=1920, help="Output height")
    render_parser.add_argument(
        "--part-label-position",
        choices=["top-left", "top-center"],
        default="top-center",
        help="Part label position",
    )
    render_parser.add_argument(
        "--no-part-overlay",
        action="store_true",
        help="Disable the in-video Part X overlay",
    )
    render_parser.add_argument("--ffmpeg-bin", type=str, default="ffmpeg", help="FFmpeg executable")
    render_parser.add_argument("--ffprobe-bin", type=str, default="ffprobe", help="FFprobe executable")
    render_parser.add_argument("--crf", type=int, default=18, help="libx264 CRF quality")
    render_parser.add_argument("--preset", type=str, default="slow", help="libx264 preset")
    render_parser.set_defaults(func=cmd_render)

    process_parser = subparsers.add_parser(
        "process",
        help="Download (if URL), render parts, and schedule sequential posts",
    )
    process_parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Source video file path or URL (http/https)",
    )
    process_parser.add_argument("--channel", type=str, required=True, help="Channel name key")
    process_parser.add_argument("--title", type=str, required=True, help="Base caption title")
    process_parser.add_argument(
        "--channels-config",
        type=Path,
        default=DEFAULT_CHANNELS_CONFIG,
        help="Path to config/channels.json",
    )
    process_parser.add_argument(
        "--start-time",
        type=str,
        default=None,
        help='First post local time: "YYYY-MM-DD HH:MM". If missing, rounds to next hour.',
    )
    process_parser.add_argument(
        "--interval-min",
        type=int,
        default=30,
        help="Minutes between each post",
    )
    process_parser.add_argument(
        "--part-seconds",
        type=int,
        default=70,
        help="Auto split duration per part in seconds (default: 70)",
    )
    process_parser.add_argument("--cuts", type=Path, default=None, help="Optional cuts.json override")
    process_parser.add_argument("--crop-top-px", type=int, default=0, help="Top pixels to crop")
    process_parser.add_argument("--title-mask-px", type=int, default=0, help="Top title mask height in pixels")
    process_parser.add_argument("--edge-bar-px", type=int, default=45, help="Top/bottom dark band size in pixels")
    process_parser.add_argument(
        "--letterbox-bump-px",
        type=int,
        default=20,
        help="Increase letterbox-mode content height by this many pixels before padding",
    )
    process_parser.add_argument("--video-y-scale", type=float, default=2.08, help="Vertical scale multiplier")
    process_parser.add_argument(
        "--y-scale-mode",
        choices=["manual", "fill", "letterbox"],
        default="letterbox",
        help="manual=use video-y-scale directly, fill=auto-bump to fill frame height, letterbox=no zoom",
    )
    process_parser.add_argument("--render-preset", type=str, default="legacy", help="Rendering preset")
    process_parser.add_argument("--output-width", type=int, default=1080, help="Output width")
    process_parser.add_argument("--output-height", type=int, default=1920, help="Output height")
    process_parser.add_argument(
        "--part-label-position",
        choices=["top-left", "top-center"],
        default="top-center",
        help="Part label position",
    )
    process_parser.add_argument(
        "--no-part-overlay",
        action="store_true",
        help="Disable the in-video Part X overlay",
    )
    process_parser.add_argument("--ffmpeg-bin", type=str, default="ffmpeg", help="FFmpeg executable")
    process_parser.add_argument("--ffprobe-bin", type=str, default="ffprobe", help="FFprobe executable")
    process_parser.add_argument("--crf", type=int, default=18, help="libx264 CRF quality")
    process_parser.add_argument("--preset", type=str, default="slow", help="libx264 preset")
    process_parser.set_defaults(func=cmd_process)

    schedule_parser = subparsers.add_parser("schedule", help="Schedule rendered parts")
    schedule_parser.add_argument("--parts", type=Path, required=True, help="Folder with part_*.mp4")
    schedule_parser.add_argument("--title", type=str, required=True, help="Base caption title")
    schedule_parser.add_argument("--channel", type=str, required=True, help="Channel name key")
    schedule_parser.add_argument(
        "--channels-config",
        type=Path,
        default=DEFAULT_CHANNELS_CONFIG,
        help="Path to config/channels.json",
    )
    schedule_parser.add_argument(
        "--start-time",
        type=str,
        default=None,
        help='First post local time: "YYYY-MM-DD HH:MM". If missing, rounds to next hour.',
    )
    schedule_parser.add_argument(
        "--interval-min",
        type=int,
        default=30,
        help="Minutes between each post",
    )
    schedule_parser.add_argument(
        "--status-file",
        type=Path,
        default=None,
        help="Optional explicit status.json output path",
    )
    schedule_parser.set_defaults(func=cmd_schedule)

    run_folder_parser = subparsers.add_parser(
        "run-folder",
        help="Find channel folders, render source video, then schedule parts",
    )
    run_folder_parser.add_argument("--root", type=Path, required=True, help="Root folder (uploads/tiktok)")
    run_folder_parser.add_argument(
        "--channels-config",
        type=Path,
        default=DEFAULT_CHANNELS_CONFIG,
        help="Path to config/channels.json",
    )
    run_folder_parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional title override for all channels in this run",
    )
    run_folder_parser.add_argument(
        "--part-seconds",
        type=int,
        default=70,
        help="Auto split duration per part in seconds (default: 70)",
    )
    run_folder_parser.add_argument("--crop-top-px", type=int, default=0, help="Top pixels to crop")
    run_folder_parser.add_argument("--title-mask-px", type=int, default=0, help="Top title mask height in pixels")
    run_folder_parser.add_argument("--edge-bar-px", type=int, default=45, help="Top/bottom dark band size in pixels")
    run_folder_parser.add_argument(
        "--letterbox-bump-px",
        type=int,
        default=20,
        help="Increase letterbox-mode content height by this many pixels before padding",
    )
    run_folder_parser.add_argument("--video-y-scale", type=float, default=2.08, help="Vertical scale multiplier")
    run_folder_parser.add_argument(
        "--y-scale-mode",
        choices=["manual", "fill", "letterbox"],
        default="letterbox",
        help="manual=use video-y-scale directly, fill=auto-bump to fill frame height, letterbox=no zoom",
    )
    run_folder_parser.add_argument("--render-preset", type=str, default="legacy", help="Rendering preset")
    run_folder_parser.add_argument("--output-width", type=int, default=1080, help="Output width")
    run_folder_parser.add_argument("--output-height", type=int, default=1920, help="Output height")
    run_folder_parser.add_argument(
        "--part-label-position",
        choices=["top-left", "top-center"],
        default="top-center",
        help="Part label position",
    )
    run_folder_parser.add_argument(
        "--no-part-overlay",
        action="store_true",
        help="Disable the in-video Part X overlay",
    )
    run_folder_parser.add_argument(
        "--start-time",
        type=str,
        default=None,
        help='First post local time: "YYYY-MM-DD HH:MM". If missing, rounds to next hour.',
    )
    run_folder_parser.add_argument(
        "--interval-min",
        type=int,
        default=30,
        help="Minutes between each post",
    )
    run_folder_parser.add_argument("--ffmpeg-bin", type=str, default="ffmpeg", help="FFmpeg executable")
    run_folder_parser.add_argument("--ffprobe-bin", type=str, default="ffprobe", help="FFprobe executable")
    run_folder_parser.add_argument("--crf", type=int, default=18, help="libx264 CRF quality")
    run_folder_parser.add_argument("--preset", type=str, default="slow", help="libx264 preset")
    run_folder_parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess folders even if status.json says completed=true",
    )
    run_folder_parser.set_defaults(func=cmd_run_folder)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
