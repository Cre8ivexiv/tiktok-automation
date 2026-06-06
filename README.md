# Quick Clips Pipeline

A TikTok video clipping and scheduling bot. Download or supply a source video, split it into vertical short-form parts, apply overlays, and schedule/upload drafts to TikTok.

---

## Requirements

| Dependency | Notes |
|---|---|
| Python 3.10+ | Tested on 3.11/3.12 |
| [FFmpeg](https://ffmpeg.org/download.html) | Must be on `PATH`. Needs `drawtext`, `pad`, `crop` filters. |
| [ffprobe](https://ffmpeg.org/download.html) | Ships with FFmpeg. Must be on `PATH`. |
| [yt-dlp](https://github.com/yt-dlp/yt-dlp) | Only needed when passing a YouTube URL as input. |

Install Python packages:

```bash
pip install fastapi uvicorn yt-dlp scenedetect[opencv]
```

Or if a full `requirements.txt` is present:

```bash
pip install -r requirements.txt
```

---

## Development Setup

Downloads are stored locally in `downloads/`.
Rendered clips are stored locally in `outputs/`.

These folders are intentionally excluded from Git because they contain generated media, cached source files, and job output. Create your own `.env` file locally for secrets and machine-specific settings; do not commit it.

---

## Project layout

```
quick clips/
├── config/
│   ├── channels.json          # Channel definitions (name → account_id, etc.)
│   └── example_cuts.json      # Example manual cuts file
├── src/
│   ├── cli.py                 # CLI entry point (serve / process / upload / auth-start)
│   ├── main.py                # Alternate CLI (render / process / schedule / run-folder)
│   ├── pipeline.py            # process_video_job() — main orchestration
│   ├── render.py              # Segment detection, FFmpeg filter building, render_parts()
│   ├── download.py            # yt-dlp wrapper + local file resolver
│   ├── captions.py            # Caption/title builder
│   └── web/
│       ├── app.py             # FastAPI web backend
│       ├── index.html         # Browser UI
│       └── jobs.py            # In-memory job store
├── outputs/                   # Rendered parts land here (auto-created)
├── downloads/                 # Downloaded source videos (auto-created)
├── smoke_test.py              # Quick sanity tests (no pytest needed)
├── python.py                  # Legacy standalone script (NVENC / yt-dlp)
└── requirements.txt
```

---

## Quickstart — Web UI (recommended)

**1. Set up your channel config**

Edit `config/channels.json`. Minimum shape:

```json
{
  "channels": {
    "my channel": {
      "account_id": "123456789"
    }
  }
}
```

**2. Start the server**

```bash
# From the project root (the "quick clips" folder)
python -m src.cli serve
```

Server starts at **http://127.0.0.1:8080** by default.

Optional flags:
```bash
python -m src.cli serve --host 0.0.0.0 --port 9000 --reload
```

**3. Open the UI**

Go to [http://127.0.0.1:8080](http://127.0.0.1:8080) in your browser.

Fill in the form and click **Process**. The job runs in the background and the UI polls for progress.

---

## Quickstart — Command Line

### Render only (no scheduling)

```bash
python -m src.main render \
  --input "C:/path/to/video.mp4" \
  --out "C:/path/to/output_dir" \
  --title "My Video" \
  --part-seconds 70
```

### Process + schedule (full pipeline)

```bash
python -m src.cli process \
  --url "https://youtube.com/watch?v=..." \
  --title "Solo Leveling Recap" \
  --channel "my channel"
```

Or with a local file:

```bash
python -m src.cli process \
  --url "C:/path/to/video.mp4" \
  --title "Solo Leveling Recap" \
  --channel "my channel"
```

### Upload already-rendered parts to TikTok

```bash
python -m src.cli upload \
  --job-id "20260330_120000_my_channel_abc123" \
  --title "Solo Leveling Recap" \
  --channel "my channel"
```

---

## Split Modes

The **Split Mode** field (web UI) or `--split-mode` flag controls how the source video is segmented.

| Mode | Behaviour |
|---|---|
| `duration` | Auto-split by fixed seconds (`part_seconds`, default 70s) |
| `parts` | Same as duration (fixed-length parts) |
| `manual` | Load cut points from a `cuts.json` file (pass via `--cuts`) |
| `ai` | AI-assisted cut detection (future / placeholder) |
| `scene` | **PySceneDetect** — detects scene changes automatically |

### Scene Detection (`scene` mode)

Select **Scene Detection (Auto)** in the UI. Adjust the **Scene Sensitivity** slider:

- **Lower value** = more sensitive = more cuts (e.g. `15`)
- **Higher value** = less sensitive = fewer cuts (e.g. `40`)
- Default: `27`

Via CLI (not yet wired to `src.cli` — use the web UI or call `process_video_job()` directly):

```python
from src.pipeline import process_video_job

process_video_job(
    input_value="video.mp4",
    title="My Video",
    channel="my channel",
    split_mode="scene",
    scene_threshold=27.0,
)
```

---

## Manual Cuts File (`cuts.json`)

Pass a `cuts.json` to override auto-splitting entirely. Example:

```json
{
  "parts": [
    { "start": "0:00", "end": "1:10" },
    { "start": "1:10", "end": "2:30" },
    { "start": "2:30", "end": "3:45" }
  ],
  "crop_top_px": 50,
  "output_width": 1080,
  "output_height": 1920
}
```

Times can be `"MM:SS"`, `"HH:MM:SS"`, or plain seconds (`90.5`).

---

## Render Modes (`y_scale_mode`)

Controls how the source video is scaled to fill the 9:16 frame.

| Mode | Behaviour |
|---|---|
| `letterbox` | Fit width, black bars top/bottom |
| `zoom` | Zoom in vertically to fill height, centre-crop |
| `fill` | Scale up to cover full frame (may crop) |
| `manual` | Apply `video_y_scale` multiplier directly |

---

## TikTok OAuth

```bash
python -m src.cli auth-start
```

Prints the TikTok authorization URL. Open it in a browser, authorize, and the callback will save tokens locally.

---

## Smoke Test

Quick sanity check — no test framework needed:

```bash
python smoke_test.py
```

- `test_filter_string()` — verifies the FFmpeg filter chain builds correctly.
- `run_render_test()` — renders a 3-second clip from `temp_test.webm` if present.
- `test_scene_detection()` — runs PySceneDetect on `temp_test.webm` if present and prints detected scenes.

---

## Outputs

Each job writes to `outputs/<channel>/<job_id>/`:

```
outputs/
└── my channel/
    └── 20260330_120000_my_channel_abc123/
        ├── part_1.mp4
        ├── part_2.mp4
        ├── ...
        ├── status.json          # Full job record
        └── render_manifest.json # FFmpeg commands used
```

---

## Common issues

**`ffmpeg not found`** — Install FFmpeg and make sure it is on your system `PATH`.

**`No module named 'scenedetect'`** — Run `pip install scenedetect[opencv]`.

**`Channel 'x' not found`** — Add the channel to `config/channels.json`.

**`yt-dlp not found`** — Install with `pip install yt-dlp` (only needed for YouTube URLs).

**Port already in use** — Pass a different port: `python -m src.cli serve --port 9000`.
