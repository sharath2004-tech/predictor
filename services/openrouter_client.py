import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import httpx
from dotenv import dotenv_values

OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"
DEFAULT_OPENROUTER_MODEL = "google/gemma-3-4b-it:free"
_OPENROUTER_BASE = "https://openrouter.ai/api/v1"


def _resolve_openrouter_key(explicit: Optional[str] = None) -> Optional[str]:
    """Find the OpenRouter API key from explicit arg, Streamlit secrets, or env."""
    if explicit:
        return explicit
    try:
        import streamlit as st  # type: ignore

        if hasattr(st, "secrets") and OPENROUTER_API_KEY_ENV in st.secrets:
            val = st.secrets[OPENROUTER_API_KEY_ENV]
            if val:
                return val
    except Exception:
        pass
    env_val = os.getenv(OPENROUTER_API_KEY_ENV)
    if env_val:
        return env_val

    try:
        env_path = Path(__file__).resolve().parent.parent / ".env"
        if env_path.exists():
            values = dotenv_values(env_path)
            candidate = values.get(OPENROUTER_API_KEY_ENV)
            if candidate:
                os.environ.setdefault(OPENROUTER_API_KEY_ENV, candidate)
                return candidate
    except Exception:
        pass
    return None


def _default_headers(api_key: str) -> Dict[str, str]:
    referer = os.getenv("OPENROUTER_SITE_URL", "http://localhost")
    title = os.getenv("OPENROUTER_APP_NAME", "Advanced Stock Predictor AI")
    return {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": referer,
        "X-Title": title,
        "Content-Type": "application/json",
    }


def openrouter_chat(
    messages: List[Dict[str, str]],
    *,
    api_key: Optional[str] = None,
    model: str = DEFAULT_OPENROUTER_MODEL,
    temperature: float = 0.2,
    max_tokens: Optional[int] = None,
    timeout: float = 60.0,
) -> str:
    """Send a chat completion request to OpenRouter."""
    key = _resolve_openrouter_key(api_key)
    if not key:
        return "AI error: OpenRouter API key missing. Set OPENROUTER_API_KEY in .env or Streamlit secrets."

    payload: Dict[str, object] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if max_tokens:
        payload["max_tokens"] = max_tokens

    try:
        with httpx.Client(timeout=timeout) as client:
            res = client.post(
                f"{_OPENROUTER_BASE}/chat/completions",
                json=payload,
                headers=_default_headers(key),
            )
            if res.status_code != 200:
                body = res.text[:400]
                if res.status_code == 401:
                    return (
                        "AI error: OpenRouter authentication failed. Double-check your key and account status. "
                        f"Raw: {body}"
                    )
                if res.status_code == 404:
                    return (
                        "AI error: Requested OpenRouter model not found. Try refreshing available models. "
                        f"Raw: {body}"
                    )
                return f"AI error: OpenRouter HTTP {res.status_code}. Detail: {body}"

            data = res.json() or {}
            choices = data.get("choices") or []
            if not choices:
                return "(No response)"
            message = choices[0].get("message") or {}
            content = (message.get("content") or "").strip()
            return content or "(No response)"
    except httpx.ConnectError:
        return "AI error: Cannot reach OpenRouter. Check your internet connection."
    except Exception as exc:
        return f"AI error: {exc}"


def list_openrouter_models(
    *, api_key: Optional[str] = None, limit: int = 100, timeout: float = 15.0
) -> List[str]:
    key = _resolve_openrouter_key(api_key)
    if not key:
        return []
    try:
        with httpx.Client(timeout=timeout) as client:
            res = client.get(
                f"{_OPENROUTER_BASE}/models",
                headers=_default_headers(key),
            )
            if res.status_code != 200:
                return []
            data = res.json() or {}
            names: List[str] = []
            for model in data.get("data", []):
                name = model.get("id") if isinstance(model, dict) else None
                if name:
                    names.append(name)
                if len(names) >= limit:
                    break
            return names
    except Exception:
        return []


def openrouter_health(*, api_key: Optional[str] = None, timeout: float = 5.0) -> Tuple[bool, str]:
    key = _resolve_openrouter_key(api_key)
    if not key:
        return False, "missing-key"
    try:
        with httpx.Client(timeout=timeout) as client:
            res = client.get(
                f"{_OPENROUTER_BASE}/models",
                headers=_default_headers(key),
            )
            if res.status_code == 200:
                return True, "ok"
            return False, f"HTTP {res.status_code}"
    except Exception as exc:
        return False, str(exc)
