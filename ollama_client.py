"""Small client for Ollama's local chat endpoint."""

import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class OllamaClient:
    def __init__(self, base_url=None, model=None):
        self.base_url = (
            base_url or os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
        ).rstrip("/")
        if "://" not in self.base_url:
            self.base_url = "http://" + self.base_url
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3:8b")

    def chat(self, messages):
        # JSON decisions work with Llama 3 without native function calling.
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0, "num_ctx": 8192},
        }
        request = Request(
            self.base_url + "/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=180) as response:
                body = json.load(response)
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(
                f"Ollama returned HTTP {error.code}: {detail}"
            ) from error
        except (URLError, TimeoutError) as error:
            raise RuntimeError(
                "Cannot reach Ollama. Start it with `ollama serve` and install "
                f"the model with `ollama pull {self.model}`. Details: {error}"
            ) from error
        try:
            content = body["message"]["content"]
        except (KeyError, TypeError) as error:
            raise RuntimeError("Ollama returned an invalid chat response.") from error
        if not isinstance(content, str):
            raise RuntimeError("Ollama returned non-text message content.")
        return content
