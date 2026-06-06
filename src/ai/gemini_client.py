from __future__ import annotations

import os
from dataclasses import dataclass


DEFAULT_GEMINI_MODEL = "gemini-2.5-pro"
FALLBACK_GEMINI_MODEL = "gemini-2.5-flash"


class GeminiConfigError(RuntimeError):
    """Raised when Gemini is not configured for this server."""


class GeminiRequestError(RuntimeError):
    """Raised when Gemini request execution fails."""


@dataclass(frozen=True)
class GeminiResponse:
    text: str
    model: str


def _api_key() -> str:
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key:
        raise GeminiConfigError("GEMINI_API_KEY is not set. Add it to your environment and restart QuickClips.")
    return key


def _generation_config() -> object | None:
    try:
        from google.genai import types
    except Exception:  # noqa: BLE001
        return None

    kwargs: dict[str, object] = {}
    max_tokens = os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "").strip()
    temperature = os.getenv("GEMINI_TEMPERATURE", "").strip()
    if max_tokens:
        try:
            kwargs["max_output_tokens"] = int(max_tokens)
        except ValueError:
            pass
    if temperature:
        try:
            kwargs["temperature"] = float(temperature)
        except ValueError:
            pass
    return types.GenerateContentConfig(**kwargs) if kwargs else None


def _response_text(response: object) -> str:
    text = getattr(response, "text", None)
    if text:
        return str(text)
    return str(response or "")


def generate_text(prompt: str, *, model: str | None = None) -> GeminiResponse:
    key = _api_key()
    try:
        from google import genai
    except ImportError as exc:
        raise GeminiConfigError("google-genai is not installed. Run: pip install -U google-genai") from exc

    requested_model = (model or os.getenv("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL).strip()
    config = _generation_config()
    client = genai.Client(api_key=key)
    tried: list[str] = []
    last_error: Exception | None = None

    for candidate in [requested_model, FALLBACK_GEMINI_MODEL]:
        if not candidate or candidate in tried:
            continue
        tried.append(candidate)
        try:
            kwargs: dict[str, object] = {"model": candidate, "contents": prompt}
            if config is not None:
                kwargs["config"] = config
            response = client.models.generate_content(**kwargs)
            return GeminiResponse(text=_response_text(response), model=candidate)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if candidate == FALLBACK_GEMINI_MODEL:
                break

    raise GeminiRequestError(f"Gemini request failed: {last_error}") from last_error
