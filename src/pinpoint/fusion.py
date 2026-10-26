"""Strategy 4L: our own hybrid. Fuse BM25 and dense rankings with reciprocal rank fusion (RRF).

Databricks' HYBRID fused dense with its own keyword engine, which was weak on this data
(file acc@1 .091). This runs the same idea with our BM25 (whole-identifier tokens), to tell
"fusion doesn't help" apart from "that keyword side doesn't help".

    uv run python -m pinpoint.fusion             # pytest-dev/pytest tune issues
"""
import sys
from collections.abc import Sequence

import pandas as pd
from rank_bm25 import BM25Okapi

from pinpoint import bm25, dense, dense_functions
from pinpoint.data import load_issues
from pinpoint.embed import MODEL, Embedder
from pinpoint.metrics import dedupe, evaluate
from pinpoint.repos import get_snapshot
from pinpoint.split import tune_ids
from pinpoint.text import tokenize
from pinpoint.tracking import log_run


def rrf(rankings: Sequence[Sequence[str]], k: int = 60, weights: Sequence[float] | None = None) -> list[str]:
    """Reciprocal rank fusion: score(d) = sum over lists of weight / (k + rank of d), rank from 1.

    Only positions are used, so lists with incomparable scores (BM25 vs cosine) can be fused.
    An item missing from a list gets nothing from it. Ties keep first-seen order.
    """
    weights = weights or [1.0] * len(rankings)
    scores: dict[str, float] = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        for rank, item in enumerate(dedupe(ranking), start=1):
            scores[item] = scores.get(item, 0.0) + weight / (k + rank)
    return sorted(scores, key=lambda item: -scores[item])


def rank_functions_bm25(fns: pd.DataFrame, query: str, mode: str = "whole") -> list[str]:
    """Function ids by BM25 of the query against each function's source (IDF from this snapshot)."""
    index = BM25Okapi([tokenize(s, mode) for s in fns.source], k1=1.5, b=0.75)
    scores = index.get_scores(tokenize(query, mode))
    order = sorted(range(len(fns)), key=lambda i: -scores[i])
    return fns.function_id.values[order].tolist()


def side_rankings(issues: pd.DataFrame, candidates: pd.DataFrame) -> dict[str, dict[str, list[str]]]:
    """Per issue, the four single-retriever rankings that get fused."""
    embedder = Embedder()
    out = {"bm25_file": {}, "dense_file": {}, "bm25_fn": {}, "dense_fn": {}}
    for _, issue in issues.iterrows():
        iid, text = issue["instance_id"], issue["problem_statement"]
        fns = candidates[candidates.instance_id == iid]
        scores = dense.search(embedder.encode(fns.source.tolist()), embedder.encode_query(text))
        out["bm25_file"][iid] = bm25.rank_files(get_snapshot(issue), text, "whole")
        out["dense_file"][iid] = dense.rank_files(fns, scores)
        out["bm25_fn"][iid] = rank_functions_bm25(fns, text)
        out["dense_fn"][iid] = dense_functions.rank_functions(fns, scores)
        print(f"  {iid}: {len(fns)} functions", flush=True)
    return out


if __name__ == "__main__":
    repo = sys.argv[1] if len(sys.argv) > 1 else "pytest-dev/pytest"
    issues = load_issues(repo)
    issues = issues[issues.instance_id.isin(tune_ids())]  # never score held-out here
    gt = pd.read_parquet("data/gold/ground_truth.parquet")
    gold = {"file": dict(zip(gt.instance_id, gt.gold_files)), "fn": dict(zip(gt.instance_id, gt.gold_functions))}
    gold = {level: {i: g[i] for i in issues.instance_id} for level, g in gold.items()}
    sides = side_rankings(issues, dense_functions.load_candidates())

    def fused(level: str, k: int = 60, w_dense: float = 1.0) -> dict[str, list[str]]:
        return {i: rrf([sides[f"bm25_{level}"][i], sides[f"dense_{level}"][i]], k, [1.0, w_dense])
                for i in sides[f"bm25_{level}"]}

    # Each side alone first (the Databricks lesson: check both lists before fusing), then fused.
    rows = []
    for level in ("file", "fn"):
        variants = {"bm25 alone": sides[f"bm25_{level}"], "dense alone": sides[f"dense_{level}"],
                    "rrf k=60": fused(level), "rrf k=10": fused(level, k=10),
                    "rrf k=60, dense x2": fused(level, w_dense=2.0)}
        for name, rankings in variants.items():
            r = evaluate(rankings, gold[level])
            rows.append({"level": level, "variant": name, **{m: round(r[m], 3) for m in ("acc@1", "acc@5", "acc@10", "mrr")},
                         "n": r["n"]})
    print(pd.DataFrame(rows).to_string(index=False))

    # Log only the pre-registered setting (k=60, equal weights); the others are a sensitivity check
    # on 11 issues, too few to pick a winner from without overfitting.
    params = {"strategy": "rrf-bm25-dense", "model": MODEL, "bm25_tokens": "whole", "k": 60,
              "weights": "1,1", "slice": f"{repo} tune"}
    log_run("rrf-file", {**params, "level": "file"}, fused("file"), gold["file"])
    log_run("rrf-function", {**params, "level": "function"}, fused("fn"), gold["fn"])
    log_run("bm25-function", {"strategy": "bm25", "level": "function", "tokens": "whole",
                              "slice": f"{repo} tune"}, sides["bm25_fn"], gold["fn"])
