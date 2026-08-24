import os

import requests

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
REQUIRED_MODELS = ("llama3.2", "nomic-embed-text")


def ollama_error():
    """Return a human-readable error message, or None if Ollama is ready."""
    try:
        resp = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=3)
        resp.raise_for_status()
    except requests.RequestException as exc:
        return (
            f"Could not reach Ollama at {OLLAMA_HOST} "
            f"({exc.__class__.__name__}). Start it with `ollama serve` and try again."
        )

    installed = {m["name"].split(":")[0] for m in resp.json().get("models", [])}
    missing = [m for m in REQUIRED_MODELS if m not in installed]
    if missing:
        pulls = " && ".join(f"ollama pull {m}" for m in missing)
        return f"Missing Ollama model(s): {', '.join(missing)}. Run: {pulls}"

    return None
