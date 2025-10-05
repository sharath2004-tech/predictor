import os
from typing import List, Dict, Optional

import httpx


def _ollama_base_url() -> str:
    return os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")


def _messages_to_prompt(messages: List[Dict[str, str]]) -> str:
    """Convert chat-style messages into a single prompt suitable for /api/generate."""
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
    # Nudge the model to produce the assistant response
    parts.append("Assistant:")
    return "\n\n".join(parts)


def _pick_available_model(requested: str, client: httpx.Client, base: str) -> str:
    """Return the best available model name. If requested isn't found, try the base name before ':'."""
    try:
        tags_url = f"{base}/api/tags"
        r = client.get(tags_url)
        if r.status_code != 200:
            return requested
        data = r.json() or {}
        models = [m.get("name") for m in data.get("models", []) if m.get("name")]
        if requested in models:
            return requested
        # Try base (e.g., 'mistral' from 'mistral:instruct')
        base_name = requested.split(":", 1)[0]
        if base_name in models:
            return base_name
        # As a last resort, pick the first available model if any exist
        if models:
            return models[0]
        return requested
    except Exception:
        return requested


def ollama_chat(messages: List[Dict[str, str]], model: str = "mistral", host: Optional[str] = None, timeout: float = 60.0, *, temperature: Optional[float] = None) -> str:
    """Call the local Ollama API. Try /api/chat; on 404, fall back to /api/generate.

    Parameters
    - messages: list of {role: 'system'|'user'|'assistant', content: str}
    - model: ollama model name (e.g., 'mistral', 'mistral:instruct')
    - host: override base url (default: env OLLAMA_HOST or http://127.0.0.1:11434)
    - timeout: request timeout in seconds
    """
    base = host or _ollama_base_url()

    try:
        with httpx.Client(timeout=timeout) as client:
            # Normalize model to one that actually exists if we can detect it
            model_to_use = _pick_available_model(model, client, base)

            # First try /api/chat (newer Ollama)
            chat_url = f"{base}/api/chat"
            payload_chat = {
                "model": model_to_use,
                "messages": messages,
                "stream": False,
            }
            if temperature is not None:
                # Ollama expects options for parameters
                payload_chat["options"] = {"temperature": float(temperature)}
            res = client.post(chat_url, json=payload_chat)
            if res.status_code == 404:
                # Fall back to /api/generate (older Ollama)
                prompt = _messages_to_prompt(messages)
                gen_url = f"{base}/api/generate"
                payload_gen = {
                    "model": model_to_use,
                    "prompt": prompt,
                    "stream": False,
                }
                if temperature is not None:
                    payload_gen["options"] = {"temperature": float(temperature)}
                res = client.post(gen_url, json=payload_gen)
                res.raise_for_status()
                data = res.json()
                text = (data.get("response") or "").strip()
                return text or "(No response)"

            res.raise_for_status()
            data = res.json()
            msg = (data.get("message") or {}).get("content", "").strip()
            return msg or "(No response)"
    except httpx.ConnectError:
        return "AI error: Cannot connect to Ollama at 127.0.0.1:11434. Is 'ollama serve' running?"
    except httpx.HTTPStatusError as e:
        return f"AI error: HTTP {e.response.status_code} from Ollama. Detail: {e.response.text[:200]}"
    except Exception as e:
        return f"AI error: {e}"


def ollama_health(host: Optional[str] = None, timeout: float = 5.0):
    """Quick health check for Ollama server. Returns (ok, info).

    ok: bool -> True if reachable and responded 200 to /api/tags
    info: list[str] of available model names if ok, otherwise error detail string
    """
    base = host or _ollama_base_url()
    try:
        with httpx.Client(timeout=timeout) as client:
            r = client.get(f"{base}/api/tags")
            if r.status_code == 200:
                data = r.json() or {}
                models = [m.get("name") for m in data.get("models", []) if m.get("name")]
                return True, models
            return False, f"HTTP {r.status_code}: {r.text[:200]}"
    except Exception as e:
        return False, str(e)


essential_system_prompt = (
    "You are a concise stock analysis assistant embedded in a Streamlit app. "
    "Be helpful, factual, and clear. Use short bullets. Avoid financial advice; include a disclaimer."
)


def ask_ai(question: str, *, ticker: Optional[str] = None, context: Optional[str] = None, model: str = "mistral", host: Optional[str] = None, timeout: float = 60.0, temperature: Optional[float] = None) -> str:
    """High-level helper that builds the prompt and calls Ollama."""
    preface = []
    if ticker:
        preface.append(f"Ticker: {ticker}")
    if context:
        preface.append(f"Context:\n{context}")

    user_content = "\n\n".join(preface + [f"Question:\n{question}"]).strip()

    messages = [
        {"role": "system", "content": essential_system_prompt},
        {"role": "user", "content": user_content},
    ]

    return ollama_chat(messages, model=model, host=host, timeout=timeout, temperature=temperature)
