from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import (
    DEFAULT_CHANNELS_CONFIG,
    process_video_job,
    upload_job_drafts,
)
from .tiktok.oauth import build_authorize_url


def cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("uvicorn is required. Install with: pip install uvicorn fastapi") from exc

    uvicorn.run("src.web.app:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def cmd_process(args: argparse.Namespace) -> int:
    payload = process_video_job(
        input_value=args.url,
        title=args.title,
        channel=args.channel,
        interval_min=args.interval_min,
        part_seconds=args.part_seconds,
        crop_top_px=args.crop_top_px,
        title_mask_px=args.title_mask_px,
        edge_bar_px=args.edge_bar_px,
        content_height_bump_px=args.content_height_bump_px,
        video_y_scale=args.video_y_scale,
        y_scale_mode=args.y_scale_mode,
        output_width=args.output_width,
        output_height=args.output_height,
        render_preset=args.render_preset,
        part_label_position=args.part_label_position,
        no_part_overlay=args.no_part_overlay,
        cuts_path=args.cuts,
        channels_config=args.channels_config.resolve(),
        log=lambda message: print(f"[process] {message}"),
    )
    print(json.dumps(payload, indent=2))
    return 0


def cmd_upload(args: argparse.Namespace) -> int:
    payload = upload_job_drafts(
        job_id=args.job_id,
        title=args.title,
        channel=args.channel,
        interval_min=args.interval_min,
        start_time=args.start_time,
        channels_config=args.channels_config.resolve(),
        log=lambda message: print(f"[upload] {message}"),
    )
    print(json.dumps(payload, indent=2))
    return 0


def cmd_auth_start(_: argparse.Namespace) -> int:
    try:
        print(build_authorize_url())
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(str(exc)) from exc
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CLI wrapper for TikTok Scheduler Uploader")
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve_parser = subparsers.add_parser("serve", help="Run FastAPI web UI server")
    serve_parser.add_argument("--host", type=str, default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8080)
    serve_parser.add_argument("--reload", action="store_true")
    serve_parser.set_defaults(func=cmd_serve)

    process_parser = subparsers.add_parser("process", help="Download/process a URL or local file")
    process_parser.add_argument("--url", type=str, required=True, help="URL or local input path")
    process_parser.add_argument("--title", type=str, required=True)
    process_parser.add_argument("--channel", type=str, required=True)
    process_parser.add_argument("--interval-min", type=int, default=30)
    process_parser.add_argument("--part-seconds", type=int, default=70)
    process_parser.add_argument("--crop-top-px", type=int, default=0)
    process_parser.add_argument("--title-mask-px", type=int, default=0)
    process_parser.add_argument("--edge-bar-px", type=int, default=45)
    process_parser.add_argument(
        "--content-height-bump-px",
        type=int,
        default=0,
        help="Zoom-mode only: increase content height in pixels before center pad",
    )
    process_parser.add_argument(
        "--video-y-scale",
        type=float,
        default=2.08,
        help="Vertical scale multiplier for legacy manual/fill modes",
    )
    process_parser.add_argument(
        "--y-scale-mode",
        choices=["letterbox", "zoom", "manual", "fill"],
        default="letterbox",
    )
    process_parser.add_argument("--render-preset", type=str, default="legacy")
    process_parser.add_argument("--output-width", type=int, default=1080)
    process_parser.add_argument("--output-height", type=int, default=1920)
    process_parser.add_argument(
        "--part-label-position",
        choices=["top-left", "top-center"],
        default="top-center",
    )
    process_parser.add_argument("--no-part-overlay", action="store_true")
    process_parser.add_argument("--cuts", type=Path, default=None)
    process_parser.add_argument(
        "--channels-config",
        type=Path,
        default=DEFAULT_CHANNELS_CONFIG,
    )
    process_parser.set_defaults(func=cmd_process)

    upload_parser = subparsers.add_parser("upload", help="Upload rendered parts as TikTok drafts")
    upload_parser.add_argument("--job-id", type=str, required=True)
    upload_parser.add_argument("--title", type=str, required=True)
    upload_parser.add_argument("--channel", type=str, required=True)
    upload_parser.add_argument("--interval-min", type=int, default=30)
    upload_parser.add_argument("--start-time", type=str, default=None)
    upload_parser.add_argument(
        "--channels-config",
        type=Path,
        default=DEFAULT_CHANNELS_CONFIG,
    )
    upload_parser.set_defaults(func=cmd_upload)

    auth_parser = subparsers.add_parser("auth-start", help="Print TikTok OAuth authorize URL")
    auth_parser.set_defaults(func=cmd_auth_start)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
