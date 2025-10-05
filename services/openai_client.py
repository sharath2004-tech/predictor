import os
from typing import List, Dict, Optional

try:
    from openai import OpenAI  # New style client (>=1.0)
except Exception:
    OpenAI = None

try:
    import openai as _openai_any
except Exception:
    _openai_any = None

_OPENAI_MAJOR = None
if _openai_any is not None:
    try:
        import importlib.metadata as _md
        _OPENAI_MAJOR = int(_md.version("openai").split(".")[0])
    except Exception:
        _OPENAI_MAJOR = None

OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"

def _resolve_openai_key(explicit: Optional[str] = None) -> Optional[str]:
    if explicit:
        return explicit
    # Streamlit secrets support
    try:
        import streamlit as st  # type: ignore
        if hasattr(st, "secrets") and OPENAI_API_KEY_ENV in st.secrets:
            # Only use secrets value if non-empty; otherwise fall back to environment variable
            _val = st.secrets[OPENAI_API_KEY_ENV]
            if _val:
                return _val
    except Exception:
        pass
    return os.getenv(OPENAI_API_KEY_ENV)

def openai_chat(
    messages: List[Dict[str, str]],
    *,
    api_key: Optional[str] = None,
    model: str = DEFAULT_OPENAI_MODEL,
    timeout: float = 60.0,
    temperature: float = 0.3,
    max_tokens: Optional[int] = None,
) -> str:
    """Send a chat completion request using the modern OpenAI client.

    We removed the automatic legacy fallback because openai>=1.x emits
    a clear migration error when ChatCompletion is used. This avoids
    confusing 'legacy fallback failed' messages.
    """
    key = _resolve_openai_key(api_key)
    if not key:
        return "AI error: OpenAI API key missing. Set OPENAI_API_KEY in secrets or environment."

    if OpenAI is None:
        return "AI error: openai package not importable. Run: pip install --upgrade openai"

    # If version is >=1 we must use new style API only.
    if _OPENAI_MAJOR is not None and _OPENAI_MAJOR >= 1:
        try:
            client = OpenAI(api_key=key, timeout=timeout)
            create_kwargs = dict(model=model, messages=messages, temperature=temperature)
            # Support both max_tokens and max_completion_tokens depending on SDK version
            if max_tokens and max_tokens > 0:
                # Try new param first
                try:
                    create_kwargs["max_completion_tokens"] = max_tokens
                except Exception:
                    create_kwargs["max_tokens"] = max_tokens
            resp = client.chat.completions.create(**create_kwargs)
            choice = resp.choices[0]
            return (choice.message.content or "").strip() if hasattr(choice, 'message') else "(No response)"
        except Exception as e:
            msg = str(e)
            low = msg.lower()
            if 'invalid_api_key' in low or 'no auth' in low or '401' in low:
                return (
                    "AI error: 401 authentication failure. Checklist: 1) Ensure OPENAI_API_KEY is set (no quotes/spaces). "
                    "2) If you edited .streamlit/secrets.toml, restart Streamlit. 3) Rotate key if it was ever committed. "
                    f"Raw: {msg[:160]}"
                )
            if 'model' in msg.lower() and 'not found' in msg.lower():
                return (
                    f"AI error: Model '{model}' not found. Use 'List OpenAI Models' to select an available model. Raw: {msg[:180]}"
                )
            return f"AI error: {msg}"

    # If somehow an older version (<1) is installed (rare in this setup), attempt legacy call
    if _openai_any is not None and hasattr(_openai_any, 'ChatCompletion'):
        try:
            _openai_any.api_key = key
            legacy_kwargs = dict(model=model, messages=messages, temperature=temperature, timeout=timeout)
            if max_tokens and max_tokens > 0:
                legacy_kwargs["max_tokens"] = max_tokens
            legacy_resp = _openai_any.ChatCompletion.create(**legacy_kwargs)
            return legacy_resp['choices'][0]['message']['content'].strip()
        except Exception as le:
            return f"AI error: Legacy OpenAI client failed: {le}"

    return "AI error: No compatible OpenAI client available. Install latest with: pip install --upgrade openai"

def openai_health(api_key: Optional[str] = None) -> bool:
    key = _resolve_openai_key(api_key)
    if not key:
        return False
    # Try modern first
    if OpenAI is not None:
        try:
            client = OpenAI(api_key=key)
            _ = client.models.list()
            return True
        except Exception:
            pass
    # Legacy fallback
    return False

def list_openai_models(api_key: Optional[str] = None, limit: int = 50) -> List[str]:
    key = _resolve_openai_key(api_key)
    if not key:
        return []
    # Modern client
    if OpenAI is not None:
        try:
            client = OpenAI(api_key=key)
            models = client.models.list()
            names: List[str] = []
            for m in getattr(models, 'data', []):
                name = getattr(m, 'id', None) or (m.get('id') if isinstance(m, dict) else None)
                if name:
                    names.append(name)
                if len(names) >= limit:
                    break
            if names:
                return names
        except Exception:
            pass
    # Legacy fallback
    return []
