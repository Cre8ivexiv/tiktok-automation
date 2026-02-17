from __future__ import annotations

import json
import os
import secrets
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SECRETS_DIR = PROJECT_ROOT / ".secrets"
TOKENS_PATH = SECRETS_DIR / "tiktok_tokens.json"
OAUTH_CONFIG_PATH = SECRETS_DIR / "tiktok_oauth.json"

AUTHORIZE_URL = "https://www.tiktok.com/v2/auth/authorize/"
TOKEN_URL = "https://open.tiktokapis.com/v2/oauth/token/"


def _ensure_secrets_dir() -> None:
    SECRETS_DIR.mkdir(parents=True, exist_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    _ensure_secrets_dir()
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_oauth_config() -> dict[str, Any]:
    file_cfg: dict[str, Any] = {}
    if OAUTH_CONFIG_PATH.exists():
        file_cfg = _read_json(OAUTH_CONFIG_PATH)

    env_scopes = os.getenv("TIKTOK_SCOPES")
    default_scopes = "user.info.basic,video.upload"

    cfg: dict[str, Any] = {
        "client_key": os.getenv("TIKTOK_CLIENT_KEY", file_cfg.get("client_key", "")),
        "client_secret": os.getenv("TIKTOK_CLIENT_SECRET", file_cfg.get("client_secret", "")),
        "redirect_uri": os.getenv(
            "TIKTOK_REDIRECT_URI",
            file_cfg.get("redirect_uri", "http://127.0.0.1:8080/auth/callback"),
        ),
        "scopes": env_scopes or file_cfg.get("scopes", default_scopes),
    }
    if isinstance(cfg["scopes"], str):
        cfg["scopes"] = [item.strip() for item in cfg["scopes"].split(",") if item.strip()]
    return cfg


def build_authorize_url(state: str | None = None) -> str:
    cfg = load_oauth_config()
    if not cfg.get("client_key"):
        raise RuntimeError(
            "Missing TikTok client_key. Set TIKTOK_CLIENT_KEY env var or .secrets/tiktok_oauth.json."
        )

    oauth_state = state or secrets.token_urlsafe(24)
    params = {
        "client_key": cfg["client_key"],
        "response_type": "code",
        "redirect_uri": cfg["redirect_uri"],
        "scope": ",".join(cfg["scopes"]),
        "state": oauth_state,
    }
    return f"{AUTHORIZE_URL}?{urllib.parse.urlencode(params)}"


def _token_request(data: dict[str, Any]) -> dict[str, Any]:
    encoded = urllib.parse.urlencode(data).encode("utf-8")
    req = urllib.request.Request(TOKEN_URL, data=encoded, method="POST")
    req.add_header("Content-Type", "application/x-www-form-urlencoded")
    with urllib.request.urlopen(req, timeout=30) as response:
        body = response.read().decode("utf-8")
    return json.loads(body)


def _normalize_tokens(response_payload: dict[str, Any]) -> dict[str, Any]:
    token_data = response_payload.get("data") if isinstance(response_payload, dict) else None
    if not isinstance(token_data, dict):
        token_data = response_payload
    if not isinstance(token_data, dict) or "access_token" not in token_data:
        raise RuntimeError(f"Unexpected token response: {response_payload}")

    expires_in = int(token_data.get("expires_in", 0) or 0)
    expires_at = datetime.utcnow() + timedelta(seconds=expires_in) if expires_in else None
    normalized = dict(token_data)
    normalized["fetched_at_utc"] = datetime.utcnow().isoformat()
    normalized["expires_at_utc"] = expires_at.isoformat() if expires_at else None
    return normalized


def exchange_code_for_tokens(code: str) -> dict[str, Any]:
    cfg = load_oauth_config()
    if not cfg.get("client_key") or not cfg.get("client_secret"):
        raise RuntimeError(
            "Missing TikTok client credentials. Configure client_key and client_secret."
        )

    payload = {
        "client_key": cfg["client_key"],
        "client_secret": cfg["client_secret"],
        "code": code,
        "grant_type": "authorization_code",
        "redirect_uri": cfg["redirect_uri"],
    }
    response = _token_request(payload)
    tokens = _normalize_tokens(response)
    save_tokens(tokens)
    return tokens


def refresh_access_token(refresh_token: str) -> dict[str, Any]:
    cfg = load_oauth_config()
    if not cfg.get("client_key") or not cfg.get("client_secret"):
        raise RuntimeError(
            "Missing TikTok client credentials. Configure client_key and client_secret."
        )

    payload = {
        "client_key": cfg["client_key"],
        "client_secret": cfg["client_secret"],
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
    }
    response = _token_request(payload)
    refreshed = _normalize_tokens(response)
    if "refresh_token" not in refreshed:
        refreshed["refresh_token"] = refresh_token
    save_tokens(refreshed)
    return refreshed


def save_tokens(tokens: dict[str, Any]) -> None:
    _write_json(TOKENS_PATH, tokens)


def load_tokens() -> dict[str, Any] | None:
    if not TOKENS_PATH.exists():
        return None
    payload = _read_json(TOKENS_PATH)
    return payload if isinstance(payload, dict) else None


def is_connected() -> bool:
    tokens = load_tokens()
    return bool(tokens and tokens.get("access_token"))


def get_valid_access_token() -> str:
    tokens = load_tokens()
    if not tokens:
        raise RuntimeError("TikTok tokens not found. Connect TikTok first.")

    access_token = str(tokens.get("access_token", "")).strip()
    refresh_token = str(tokens.get("refresh_token", "")).strip()
    expires_at_raw = str(tokens.get("expires_at_utc", "")).strip()

    if access_token and expires_at_raw:
        try:
            expires_at = datetime.fromisoformat(expires_at_raw)
            if datetime.utcnow() >= expires_at and refresh_token:
                refreshed = refresh_access_token(refresh_token)
                return str(refreshed.get("access_token", "")).strip()
        except ValueError:
            pass

    if not access_token and refresh_token:
        refreshed = refresh_access_token(refresh_token)
        access_token = str(refreshed.get("access_token", "")).strip()

    if not access_token:
        raise RuntimeError("Access token unavailable after refresh.")
    return access_token

