# TikTok Scheduler Uploader

Demo-ready pipeline with:
- URL/local input processing (`download -> split -> render`)
- Web UI for app-review screen recording
- TikTok Login Kit OAuth connect flow
- Draft upload flow using TikTok Content Posting API helpers

## What The Pipeline Does
- Splits source into sequential parts of `70` seconds by default (last part kept even if shorter).
- Renders each part with FFmpeg:
  - crops top using `crop_top_px` (to remove burned-in title text)
  - scales/pads to `1080x1920`
  - overlays only `Part X` (or disables with `--no-part-overlay`)
- Caption format per part:
  - `{TITLE} (Part X) #partx #fyp #animevideos #anime #movie #manhwa #animerecap #animeedits #isekaianime #animeedit #recap #animerecommendations`
  - `#partx` is lowercase and matches part number (`#part1`, `#part2`, ...)
- Title is in caption only, not burned into the video.

## Requirements
- Python 3.10+
- FFmpeg + ffprobe on PATH
- `yt-dlp` (URL input)
- FastAPI + Uvicorn (web server)

Install:
```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -U fastapi uvicorn yt-dlp
```

## Project Structure
```text
src/
  pipeline.py
  download.py
  render.py
  captions.py
  cli.py
  web/
    app.py
    jobs.py
    index.html
  tiktok/
    oauth.py
    posting.py
config/
  channels.json
uploads/
  tiktok/
    anime recaps/
```

## Web UI (Demo Flow)
Start server:
```powershell
python -m src.cli serve --host 127.0.0.1 --port 8080
```

Open:
- `http://127.0.0.1:8080`

UI flow:
1. Paste URL + Title + Channel.
2. Click **Process Video**.
3. Click **Connect TikTok**.
4. Click **Upload Drafts**.

UI routes:
- `GET /`
- `POST /api/process`
- `GET /api/status/{job_id}`
- `GET /api/me`
- `GET /auth/start`
- `GET /auth/callback`
- `POST /api/upload`

## CLI Wrapper
Serve web app:
```powershell
python -m src.cli serve --host 127.0.0.1 --port 8080
```

Process:
```powershell
python -m src.cli process --url "https://www.youtube.com/watch?v=..." --title "Solo Leveling Recap" --channel "anime recaps"
```

Upload drafts:
```powershell
python -m src.cli upload --job-id "YOUR_JOB_ID" --title "Solo Leveling Recap" --channel "anime recaps"
```

Print OAuth authorize URL:
```powershell
python -m src.cli auth-start
```

## OAuth / Token Storage
- OAuth helper: `src/tiktok/oauth.py`
- Tokens file: `.secrets/tiktok_tokens.json`
- `.secrets/` is ignored by `.gitignore`

Configure TikTok credentials using env vars or `.secrets/tiktok_oauth.json`:
- `TIKTOK_CLIENT_KEY`
- `TIKTOK_CLIENT_SECRET`
- `TIKTOK_REDIRECT_URI` (default `http://127.0.0.1:8080/auth/callback`)
- optional `TIKTOK_SCOPES` (default `user.info.basic,video.upload`)

Example `.secrets/tiktok_oauth.json`:
```json
{
  "client_key": "YOUR_CLIENT_KEY",
  "client_secret": "YOUR_CLIENT_SECRET",
  "redirect_uri": "http://127.0.0.1:8080/auth/callback",
  "scopes": "user.info.basic,video.upload"
}
```

## Draft Upload Notes
- Upload module: `src/tiktok/posting.py`
- Uploads are attempted in part order.
- `interval_min` is used to compute intended schedule timestamps and saved in `status.json`.
- If API behavior differs by app setup, adjust endpoint env var:
  - `TIKTOK_VIDEO_UPLOAD_INIT_URL`
- Existing scheduler module (`src/scheduler/mock.py`) is still mock-only and does not post to TikTok.

## Output + Status
Each process job writes to:
- `outputs/<channel>/<job_id>/part_1.mp4`, `part_2.mp4`, ...
- `outputs/<channel>/<job_id>/status.json`

`status.json` includes:
- source path
- rendered parts
- upload plan timestamps
- upload responses

## Demo Checklist (for App Review Recording)
1. Run server: `python -m src.cli serve --host 127.0.0.1 --port 8080`
2. Open UI and paste URL + title.
3. Click **Process Video** and show live logs + generated parts.
4. Click **Connect TikTok** and complete OAuth consent.
5. Return to UI and click **Upload Drafts**.
6. Show upload log updates and resulting `status.json`.

## Repo Hygiene
- Videos are local-only and should not be committed.
- Put source videos under `uploads/...` or `downloads/...` as needed by your workflow.
- Rendered outputs go under `outputs/...`.
- Secrets belong in `.secrets/` (and `.env*`) and are never committed.
