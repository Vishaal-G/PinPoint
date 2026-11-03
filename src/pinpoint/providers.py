"""Shared, process-wide resources for the long-running front-ends (api.py, app.py).

One embedder (its in-memory vector cache is the expensive part), one LLM client, and one lock per
repo: requests share one checkout per repo, and `git checkout` would move it under a running request.
Tests swap these out (FastAPI: app.dependency_overrides; Streamlit: monkeypatch this module).
"""
import os
import threading
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

CACHE_DIR = Path("data/silver")  # the embedding cache shared with the SWE-bench runs


@lru_cache(maxsize=1)
def get_embedder():
    """Databricks bge-large by default. PINPOINT_EMBEDDER=local runs bge-small on this machine instead
    (offline demo while the endpoint is down; needs the `local` uv group). Rankings differ between the two."""
    if os.environ.get("PINPOINT_EMBEDDER") == "local":
        from pinpoint.embed import MODEL, Embedder

        return Embedder(MODEL, cache_dir=CACHE_DIR)
    from pinpoint.embed import ENDPOINT_MODEL, EndpointEmbedder

    return EndpointEmbedder(ENDPOINT_MODEL, cache_dir=CACHE_DIR)


@lru_cache(maxsize=1)
def get_llm():
    """A call_llm that connects on first use, not here: an expired login or a quota block then fails
    inside the agent's try (the report degrades to dense) instead of failing the whole request."""
    client = {}

    def call(body):
        if "fn" not in client:
            from pinpoint.agent import databricks_llm

            client["fn"] = databricks_llm()
        return client["fn"](body)
    return call


_locks: defaultdict[str, threading.Lock] = defaultdict(threading.Lock)


def repo_lock(repo: str) -> threading.Lock:
    return _locks[repo]
