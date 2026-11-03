"""Which LLM the front-ends (bot, live CLI, API, page) talk to, picked from the environment.

    PINPOINT_LLM=databricks  (default) Llama 3.3 70B on a Databricks Foundation Model endpoint
    PINPOINT_LLM=openai      any OpenAI-compatible /chat/completions server with tool calling:
                             PINPOINT_LLM_URL   base URL, e.g. https://api.example.com/v1 or http://localhost:11434/v1
                             PINPOINT_LLM_MODEL the model name that server expects
                             PINPOINT_LLM_KEY   API key, if the server needs one (sent as a Bearer token)
    PINPOINT_LLM=none        no LLM: dense retrieval only, no agent and no explanation (needs no secrets)

Every call_llm takes an OpenAI-format chat body (messages, tools, tool_choice, ...) and returns the
OpenAI-format response dict; the agent and explain.py never know which one they got.
The agent was tuned on Llama 3.3 70B; other models are unmeasured until they get an eval row.
"""
import json
import os
import time
import urllib.error
import urllib.request

MODES = ("databricks", "openai", "none")
TIMEOUT = 120  # seconds per request: a 70B model with ~20k prompt tokens can take a while
RETRY_STATUS = {429, 502, 503}  # rate limited / briefly down: free tiers return 429 a lot
RETRIES = 4                      # waits 2, 4, 8, 16 s (or the server's Retry-After, capped at 60)


class LLMError(RuntimeError):
    pass


def openai_compatible_llm(base_url: str, model: str, api_key: str | None = None, timeout: int = TIMEOUT,
                          retries: int = RETRIES, sleep=time.sleep):
    """call_llm for an OpenAI-compatible server. The body gets the `model` field the server needs
    (Databricks takes the model from the URL instead). Errors raise LLMError with the server's
    message, never the key; live.triage turns them into the dense fallback. 429/502/503 are retried
    with exponential backoff, as the Databricks SDK does for the default provider."""
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    def call(body: dict) -> dict:
        data = json.dumps({**body, "model": model}).encode()
        for attempt in range(retries + 1):
            try:
                with urllib.request.urlopen(urllib.request.Request(url, data=data, headers=headers), timeout=timeout) as r:
                    raw = r.read()
                break
            except urllib.error.HTTPError as e:
                if e.code not in RETRY_STATUS or attempt == retries:
                    raise LLMError(f"HTTP {e.code} from {url}: {e.read()[:300].decode(errors='replace')}") from None
                wait = e.headers.get("Retry-After", "")
                sleep(min(float(wait), 60) if wait.replace(".", "", 1).isdigit() else 2 ** (attempt + 1))
        try:
            out = json.loads(raw)
        except json.JSONDecodeError:  # e.g. a parked host answering 200 "OK" to everything
            raise LLMError(f"non-JSON response from {url}: {raw[:100]!r}") from None
        if "choices" not in out:
            raise LLMError(f"no choices in response from {url}: {str(out)[:300]}")
        return out
    return call


def lazy_databricks_llm():
    """Connects on the first call, not here: an expired login or a quota block then fails inside
    the agent's try (the report degrades to dense) instead of failing the whole request."""
    client = {}

    def call(body):
        if "fn" not in client:
            from pinpoint.agent import databricks_llm

            client["fn"] = databricks_llm()
        return client["fn"](body)
    return call


def make_llm():
    """(call_llm or None, label for the comment footer) from the environment; see the module docstring."""
    mode = os.environ.get("PINPOINT_LLM", "databricks").lower()
    if mode == "none":
        return None, None
    if mode == "databricks":
        return lazy_databricks_llm(), "Llama 3.3 70B"
    if mode == "openai":
        url, model = os.environ.get("PINPOINT_LLM_URL"), os.environ.get("PINPOINT_LLM_MODEL")
        if not url or not model:
            raise SystemExit("PINPOINT_LLM=openai needs PINPOINT_LLM_URL and PINPOINT_LLM_MODEL")
        return openai_compatible_llm(url, model, os.environ.get("PINPOINT_LLM_KEY") or None), model
    raise SystemExit(f"PINPOINT_LLM={mode!r}: use one of {', '.join(MODES)}")
