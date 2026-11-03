import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from pinpoint.llm import LLMError, make_llm, openai_compatible_llm


@pytest.fixture
def server():
    """A local OpenAI-compatible server. `state["reply"]` = (status, bytes) to send; requests recorded."""
    state = {"queue": [], "reply": (200, json.dumps({"choices": [{"message": {"content": "hi"}}], "usage": {}}).encode()),
             "requests": []}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state["requests"].append({"path": self.path, "auth": self.headers.get("Authorization"), "body": body})
            status, data, *hdrs = state["queue"].pop(0) if state["queue"] else state["reply"]
            self.send_response(status)
            for k, v in hdrs:
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    httpd = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    state["url"] = f"http://127.0.0.1:{httpd.server_port}/v1/"
    yield state
    httpd.shutdown()


def test_sends_model_and_key_to_chat_completions(server):
    call = openai_compatible_llm(server["url"], "m-1", "sk-secret")
    out = call({"messages": [{"role": "user", "content": "x"}], "temperature": 0})
    req = server["requests"][0]
    assert out["choices"][0]["message"]["content"] == "hi"
    assert req["path"] == "/v1/chat/completions" and req["auth"] == "Bearer sk-secret"
    assert req["body"] == {"messages": [{"role": "user", "content": "x"}], "temperature": 0, "model": "m-1"}


def test_no_key_sends_no_auth_header(server):  # e.g. a local Ollama
    openai_compatible_llm(server["url"], "m")({"messages": []})
    assert server["requests"][0]["auth"] is None


@pytest.mark.parametrize("reply, msg", [
    ((429, b'{"error": "rate limited"}'), "HTTP 429"),
    ((200, b"OK"), "non-JSON"),                       # a parked host (what models.github.ai does now)
    ((200, b'{"error": "bad model"}'), "no choices"),
])
def test_errors_raise_llmerror_without_the_key(server, reply, msg):
    server["reply"] = reply
    with pytest.raises(LLMError, match=msg) as e:
        openai_compatible_llm(server["url"], "m", "sk-secret", sleep=lambda s: None)({"messages": []})
    assert "sk-secret" not in str(e.value)


def test_make_llm_modes(monkeypatch):
    for v in ("PINPOINT_LLM", "PINPOINT_LLM_URL", "PINPOINT_LLM_MODEL", "PINPOINT_LLM_KEY"):
        monkeypatch.delenv(v, raising=False)
    call, label = make_llm()
    assert callable(call) and label == "Llama 3.3 70B"          # default: Databricks, not connected yet
    monkeypatch.setenv("PINPOINT_LLM", "none")
    assert make_llm() == (None, None)
    monkeypatch.setenv("PINPOINT_LLM", "openai")
    with pytest.raises(SystemExit, match="PINPOINT_LLM_URL"):
        make_llm()
    monkeypatch.setenv("PINPOINT_LLM_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("PINPOINT_LLM_MODEL", "qwen3:32b")
    call, label = make_llm()
    assert callable(call) and label == "qwen3:32b"
    monkeypatch.setenv("PINPOINT_LLM", "gpt")
    with pytest.raises(SystemExit, match="one of"):
        make_llm()


def test_rate_limits_are_retried_with_backoff(server):
    server["queue"] = [(429, b"slow down", ("Retry-After", "3")), (503, b"busy")]
    waits = []
    out = openai_compatible_llm(server["url"], "m", sleep=waits.append)({"messages": []})
    assert out["choices"] and len(server["requests"]) == 3
    assert waits == [3.0, 4]                       # server's Retry-After, then 2 ** 2


def test_client_errors_are_not_retried(server):
    server["reply"] = (401, b"bad key")
    with pytest.raises(LLMError, match="HTTP 401"):
        openai_compatible_llm(server["url"], "m", sleep=lambda s: None)({"messages": []})
    assert len(server["requests"]) == 1
