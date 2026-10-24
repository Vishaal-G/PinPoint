"""Embed text with bge-small, caching each passage vector by the sha256 of its text.

The cache is one parquet file per model, so a text is embedded once across every
snapshot and every run.
"""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

MODEL = "BAAI/bge-small-en-v1.5"
# bge is trained for short query -> long passage; queries get this prefix, passages don't.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class Embedder:
    def __init__(self, model_name: str = MODEL, cache_dir: Path = Path("data/silver")):
        from sentence_transformers import SentenceTransformer  # slow import; only when used

        self.model = SentenceTransformer(model_name)
        self.cache_path = cache_dir / f"embeddings_{model_name.split('/')[-1]}.parquet"
        self.cache: dict[str, np.ndarray] = {}
        if self.cache_path.exists():
            df = pd.read_parquet(self.cache_path)
            self.cache = dict(zip(df.hash, (np.asarray(v, dtype=np.float32) for v in df.embedding)))
        self.hits = self.misses = 0

    def count_tokens(self, lines: list[str]) -> list[int]:
        """Token count per line, without the [CLS]/[SEP] the model adds per input."""
        if not lines:
            return []
        return [len(ids) for ids in self.model.tokenizer(lines, add_special_tokens=False)["input_ids"]]

    def encode(self, texts: list[str]) -> np.ndarray:
        """Unit-length vectors, shape (len(texts), dim). Only texts not in the cache are run."""
        hashes = [text_hash(t) for t in texts]
        todo = {h: t for h, t in zip(hashes, texts) if h not in self.cache}
        self.misses += len(todo)
        self.hits += len(texts) - len(todo)
        if todo:
            vecs = self.model.encode(list(todo.values()), batch_size=32, normalize_embeddings=True)
            self.cache.update(zip(todo, vecs))
        dim = self.model.get_embedding_dimension()
        return np.stack([self.cache[h] for h in hashes]) if hashes else np.empty((0, dim), np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        """Queries are not cached: each issue is embedded once per run anyway."""
        return self.model.encode(QUERY_PREFIX + text, normalize_embeddings=True)

    def save(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"hash": list(self.cache), "embedding": [v.tolist() for v in self.cache.values()]}
                     ).to_parquet(self.cache_path, index=False)
