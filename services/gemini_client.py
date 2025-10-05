import os
from typing import List, Dict, Optional
import httpx


GEMINI_API_KEY_ENV = "GEMINI_API_KEY"
DEFAULT_GEMINI_MODEL = "gemini-1.5-flash-latest"
_GEMINI_API_VERSIONS = ["v1beta", "v1"]  # try beta first, then stable


def _resolve_gemini_key(explicit: Optional[str] = None) -> Optional[str]:
    """Return the Gemini API key from (in order): explicit param, Streamlit secrets, env var.

    This allows flexible configuration without hard‑coding the key in code.
    """
    if explicit:
        return explicit
    # Try Streamlit secrets if available
    try:
        import streamlit as st  # type: ignore
        if hasattr(st, "secrets") and GEMINI_API_KEY_ENV in st.secrets:
            return st.secrets[GEMINI_API_KEY_ENV]
    except Exception:
        pass
    return os.getenv(GEMINI_API_KEY_ENV)


def _messages_to_prompt(messages: List[Dict[str, str]]) -> str:
    parts: List[str] = []
    for m in messages:
        role = m.get("role", "user")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if role == "system":
            parts.append(f"System: {content}")
        elif role == "assistant":
            parts.append(f"Assistant: {content}")
        else:
            parts.append(f"User: {content}")
    parts.append("Assistant:")
    return "\n\n".join(parts)


def list_gemini_models(api_key: Optional[str] = None, timeout: float = 10.0) -> List[str]:
    key = _resolve_gemini_key(api_key)
    if not key:
        return []
    found: List[str] = []
    for ver in _GEMINI_API_VERSIONS:
        url = f"https://generativelanguage.googleapis.com/{ver}/models?key={key}"
        try:
            with httpx.Client(timeout=timeout) as client:
                r = client.get(url)
                if r.status_code != 200:
                    continue
                data = r.json() or {}
                for m in data.get("models", []):
                    name = m.get("name")
                    if name and name not in found:
                        found.append(name)
        except Exception:
            continue
    return found


def gemini_chat(messages: List[Dict[str, str]], *, api_key: Optional[str] = None, model: str = DEFAULT_GEMINI_MODEL, timeout: float = 60.0) -> str:
    """Call Google Gemini API (text / multi-turn simplified).

    We convert the conversation into a single prompt (Gemini can accept structured content,
    but for simplicity we flatten here) and invoke the generateContent endpoint.
    """
    key = _resolve_gemini_key(api_key)
    if not key:
        return "AI error: Gemini API key missing. Set GEMINI_API_KEY env var or provide it in the UI."

    prompt = _messages_to_prompt(messages)
    payload = {
        "contents": [
            {"parts": [{"text": prompt}]}
        ]
    }
    last_error = None
    for ver in _GEMINI_API_VERSIONS:
        url = f"https://generativelanguage.googleapis.com/{ver}/models/{model}:generateContent?key={key}"
        try:
            with httpx.Client(timeout=timeout) as client:
                resp = client.post(url, json=payload)
                if resp.status_code != 200:
                    body_preview = resp.text[:400]
                    # Specific guidance for key issues
                    if resp.status_code == 400 and "API key not valid" in body_preview:
                        return (
                            "AI error: Gemini API key invalid. Steps: rotate key, update GEMINI_API_KEY, restart app, ensure API enabled. Raw: " + body_preview
                        )
                    if resp.status_code == 404 and ver != _GEMINI_API_VERSIONS[-1]:
                        # Try next version automatically
                            last_error = body_preview
                            continue
                    if resp.status_code == 404:
                        return (
                            "AI error: Model not found. Use the 'Refresh Gemini Models' button to list available models. Raw: " + body_preview
                        )
                    last_error = body_preview
                    continue
                data = resp.json() or {}
                candidates = data.get("candidates") or []
                if not candidates:
                    return "(No response)"
                parts = (candidates[0].get("content") or {}).get("parts") or []
                texts = [p.get("text", "") for p in parts if p.get("text")]
                combined = "\n".join(texts).strip()
                return combined or "(No response)"
        except Exception as e:
            last_error = str(e)
            continue
    return f"AI error: Gemini request failed. Last error: {last_error}"


def gemini_health(api_key: Optional[str] = None, timeout: float = 5.0) -> bool:
    """Very lightweight health check – validates key with a trivial empty prompt."""
    key = _resolve_gemini_key(api_key)
    if not key:
        return False
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{DEFAULT_GEMINI_MODEL}:generateContent?key={key}"
    payload = {"contents": [{"parts": [{"text": "Ping"}]}]}
    try:
        with httpx.Client(timeout=timeout) as client:
            r = client.post(url, json=payload)
            return r.status_code == 200
    except Exception:
        return False
