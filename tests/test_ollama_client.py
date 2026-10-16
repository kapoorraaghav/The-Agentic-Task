import io
import json
from urllib.error import URLError

import pytest

from ollama_client import OllamaClient


def test_chat_request(monkeypatch):
    def respond(request, timeout):
        body = json.loads(request.data)
        assert request.full_url == "http://127.0.0.1:11434/api/chat"
        assert body["model"] == "llama3:8b"
        assert body["format"] == "json"
        assert body["stream"] is False
        return io.BytesIO(
            json.dumps({"message": {"content": '{"answer":"hello"}'}}).encode()
        )

    monkeypatch.setattr("ollama_client.urlopen", respond)
    client = OllamaClient(base_url="http://127.0.0.1:11434", model="llama3:8b")
    assert json.loads(client.chat([])) == {"answer": "hello"}


def test_connection_error(monkeypatch):
    def unavailable(*args, **kwargs):
        raise URLError("Connection refused")

    monkeypatch.setattr("ollama_client.urlopen", unavailable)
    with pytest.raises(RuntimeError, match="ollama serve"):
        OllamaClient().chat([])
