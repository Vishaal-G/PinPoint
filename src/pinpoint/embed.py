"""Embed text, caching each passage vector by the sha256 of its text.

    Embedder          bge-small run locally with fastembed (ONNX Runtime, no PyTorch)
    EndpointEmbedder  a Databricks Foundation Model embedding endpoint (bge-large-en), for scale

The cache is one parquet file per model, so a text is embedded once across every
snapshot and every run. `make_embedder(name)` picks the class from the model name.
"""
import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd

MODEL = "BAAI/bge-small-en-v1.5"
# bge is trained for short query -> long passage; queries get this prefix, passages don't.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
ENDPOINT_MODEL = "databricks-bge-large-en"


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class Embedder:
    """Local model through fastembed: the same bge weights exported to ONNX, run by onnxruntime.
    Vectors match sentence-transformers (cosine 1.0000 on 120 pytest bodies, incl. ones truncated at
    512 tokens), so caches written by either are interchangeable; the install is ~25 MB instead of torch."""

    def __init__(self, model_name: str = MODEL, cache_dir: Path = Path("data/silver")):
        from fastembed import TextEmbedding  # onnxruntime import; only when used

        model_dir = os.path.expanduser(os.environ.get("FASTEMBED_CACHE_PATH", "~/.cache/fastembed"))
        self.model_name = model_name
        self.model = TextEmbedding(model_name, cache_dir=model_dir)  # downloads ~130 MB once
        self._tokenizer = None
        self.cache_path = Path(cache_dir) / f"embeddings_{model_name.split('/')[-1]}.parquet"
        self.cache: dict[str, np.ndarray] = {}
        if self.cache_path.exists():
            df = pd.read_parquet(self.cache_path)
            self.cache = dict(zip(df.hash, (np.asarray(v, dtype=np.float32) for v in df.embedding)))
        self.hits = self.misses = 0

    def _run(self, texts: list[str]) -> np.ndarray:
        v = np.asarray(list(self.model.embed(texts, batch_size=32)), dtype=np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    def count_tokens(self, lines: list[str]) -> list[int]:
        """Token count per line, without the [CLS]/[SEP] the model adds per input."""
        if not lines:
            return []
        if self._tokenizer is None:
            from tokenizers import Tokenizer  # comes with fastembed

            self._tokenizer = Tokenizer.from_pretrained(self.model_name)
        return [len(e.ids) for e in self._tokenizer.encode_batch(lines, add_special_tokens=False)]

    def encode(self, texts: list[str]) -> np.ndarray:
        """Unit-length vectors, shape (len(texts), dim). Only texts not in the cache are run."""
        hashes = [text_hash(t) for t in texts]
        todo = {h: t for h, t in zip(hashes, texts) if h not in self.cache}
        self.misses += len(todo)
        self.hits += len(texts) - len(todo)
        if todo:
            self.cache.update(zip(todo, self._run(list(todo.values()))))
        if not hashes:
            return np.empty((0, self.model.embedding_size), np.float32)
        return np.stack([self.cache[h] for h in hashes])

    def encode_query(self, text: str) -> np.ndarray:
        """Queries are not cached: each issue is embedded once per run anyway."""
        return self._run([QUERY_PREFIX + text])[0]

    def save(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"hash": list(self.cache), "embedding": [v.tolist() for v in self.cache.values()]}
                     ).to_parquet(self.cache_path, index=False)


class EndpointEmbedder(Embedder):
    """Same interface and cache as Embedder, but vectors come from a Databricks serving endpoint.
    The endpoint truncates each input to the model's 512 tokens, like the local model does."""

    batch_size = 16   # Free Edition: 16 inputs per request sustain ~24 texts/s; 32+ always get HTTP 429

    def __init__(self, endpoint: str = ENDPOINT_MODEL, cache_dir: Path = Path("data/silver"), client=None):
        if client is None:
            from pinpoint.cloud import workspace

            client = workspace()
        self.client, self.endpoint = client, endpoint
        self.cache_path = Path(cache_dir) / f"embeddings_{endpoint}.parquet"
        self.cache = {}
        if self.cache_path.exists():
            df = pd.read_parquet(self.cache_path)
            self.cache = dict(zip(df.hash, (np.asarray(v, dtype=np.float32) for v in df.embedding)))
        self.hits = self.misses = 0
        self.dim = None

    def _call(self, texts: list[str]) -> np.ndarray:
        """One request -> unit-length float32 vectors, in input order."""
        r = self.client.api_client.do("POST", f"/serving-endpoints/{self.endpoint}/invocations", body={"input": texts})
        rows = sorted(r["data"], key=lambda d: d["index"])
        v = np.asarray([d["embedding"] for d in rows], dtype=np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    def encode(self, texts: list[str]) -> np.ndarray:
        hashes = [text_hash(t) for t in texts]
        todo = {h: t for h, t in zip(hashes, texts) if h not in self.cache}
        self.misses += len(todo)
        self.hits += len(texts) - len(todo)
        items = list(todo.items())
        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]
            self.cache.update(zip((h for h, _ in batch), self._call([t for _, t in batch])))
        if not hashes:
            return np.empty((0, self.dim or 0), np.float32)
        return np.stack([self.cache[h] for h in hashes])

    def encode_query(self, text: str) -> np.ndarray:
        return self._call([QUERY_PREFIX + text])[0]

    def count_tokens(self, lines: list[str]) -> list[int]:
        raise NotImplementedError("token counts need the local tokenizer; use Embedder")


def make_embedder(model: str = MODEL, **kw) -> Embedder:
    """Embedder for a local (fastembed) model, EndpointEmbedder for a databricks-* endpoint."""
    return EndpointEmbedder(model, **kw) if model.startswith("databricks-") else Embedder(model, **kw)


def default_embedder(cache_dir: Path) -> Embedder:
    """What the front-ends (bot, API, page, live CLI) use. Databricks bge-large by default;
    PINPOINT_EMBEDDER=local runs bge-small on this machine (no Databricks needed for retrieval;
    `pip install "pinpoint[local-embed]"`). Rankings differ between the two models."""
    if os.environ.get("PINPOINT_EMBEDDER") == "local":
        return Embedder(MODEL, cache_dir=cache_dir)
    return EndpointEmbedder(ENDPOINT_MODEL, cache_dir=cache_dir)
