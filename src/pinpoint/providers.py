"""Shared, process-wide resources for the long-running front-ends (api.py, app.py).

One embedder (its in-memory vector cache is the expensive part), one LLM client, and one lock per
repo: requests share one checkout per repo, and `git checkout` would move it under a running request.
Tests swap these out (FastAPI: app.dependency_overrides; Streamlit: monkeypatch this module).
"""
import threading
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

CACHE_DIR = Path("data/silver")  # the embedding cache shared with the SWE-bench runs


@lru_cache(maxsize=1)
def get_embedder():
    """embed.default_embedder: Databricks bge-large, or bge-small here with PINPOINT_EMBEDDER=local."""
    from pinpoint.embed import default_embedder

    return default_embedder(CACHE_DIR)


@lru_cache(maxsize=1)
def _llm():
    from pinpoint.llm import make_llm

    return make_llm()


def get_llm():
    """llm.make_llm's call_llm (Databricks by default, connects lazily; None with PINPOINT_LLM=none)."""
    return _llm()[0]


def llm_label() -> str | None:
    return _llm()[1]


_locks: defaultdict[str, threading.Lock] = defaultdict(threading.Lock)


def repo_lock(repo: str) -> threading.Lock:
    return _locks[repo]
