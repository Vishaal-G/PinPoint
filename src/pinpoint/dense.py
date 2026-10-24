"""Strategy 2: fixed ~500-token chunks -> bge-small -> brute-force cosine search.

Chunk scores become file scores (max over the file's chunks) and function scores
(max over the chunks that overlap the function).

    uv run python -m pinpoint.dense                 # pytest-dev/pytest tune issues
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from pinpoint.chunking import chunk_text
from pinpoint.data import load_issues
from pinpoint.embed import MODEL, Embedder
from pinpoint.repos import get_snapshot, source_files
from pinpoint.split import tune_ids
from pinpoint.tracking import log_run


def snapshot_chunks(snapshot: Path, embedder: Embedder, size: int = 500, overlap: int = 50) -> pd.DataFrame:
    """One row per chunk of every source file: path, start_line, end_line, text."""
    rows = []
    for path in source_files(snapshot):
        text = (snapshot / path).read_text(errors="replace")
        for start, end, chunk in chunk_text(text, embedder.count_tokens, size, overlap):
            rows.append({"path": path, "start_line": start, "end_line": end, "text": chunk})
    return pd.DataFrame(rows, columns=["path", "start_line", "end_line", "text"])


def search(chunk_vecs: np.ndarray, query_vec: np.ndarray) -> np.ndarray:
    """Cosine of the query with every chunk. Vectors are unit length, so cosine = dot product,
    and one matrix-vector product scores them all: (n, 384) @ (384,) -> (n,)."""
    return chunk_vecs @ query_vec


def rank_files(chunks: pd.DataFrame, scores: np.ndarray) -> list[str]:
    """Files by their best chunk, best first."""
    best = pd.Series(scores, index=chunks.path).groupby(level=0, sort=False).max()
    return best.sort_values(ascending=False, kind="stable").index.tolist()


def rank_functions(chunks: pd.DataFrame, scores: np.ndarray, functions: pd.DataFrame) -> list[str]:
    """Function ids by the best chunk overlapping each function, best first.

    Every function inside one chunk gets that chunk's score, so ties are common; they are
    broken by file order (path, start_line), which is arbitrary but deterministic.
    """
    chunks = chunks.assign(score=scores)
    by_path = {p: g for p, g in chunks.groupby("path", sort=False)}
    ids, best = [], []
    for path, fns in functions.sort_values(["path", "start_line"]).groupby("path", sort=False):
        if path not in by_path:  # e.g. test files: not candidates
            continue
        c = by_path[path]
        # overlap[i, j]: function i and chunk j share at least one line.
        overlap = ((fns.start_line.values[:, None] <= c.end_line.values[None, :])
                   & (fns.end_line.values[:, None] >= c.start_line.values[None, :]))
        best.extend(np.where(overlap, c.score.values[None, :], -np.inf).max(axis=1))
        ids.extend(fns.function_id)
    order = np.argsort(-np.asarray(best), kind="stable")
    return [ids[i] for i in order]


def run(issues: pd.DataFrame, gold_files: dict, gold_functions: dict, membership: pd.DataFrame,
        size: int = 500, overlap: int = 50, slice_name: str = "") -> dict[str, dict]:
    """Rank files and functions for every issue; log one MLflow run per level."""
    embedder = Embedder()
    file_rankings, fn_rankings = {}, {}
    total_chunks = 0
    for _, issue in issues.iterrows():
        iid = issue["instance_id"]
        chunks = snapshot_chunks(get_snapshot(issue), embedder, size, overlap)
        total_chunks += len(chunks)
        scores = search(embedder.encode(chunks.text.tolist()), embedder.encode_query(issue["problem_statement"]))
        file_rankings[iid] = rank_files(chunks, scores)
        fn_rankings[iid] = rank_functions(chunks, scores, membership[membership.instance_id == iid])
        embedder.save()  # every issue: CPU embedding is slow, don't lose it to a crash
        print(f"  {iid}: {len(chunks)} chunks, {embedder.misses} embedded so far", flush=True)

    params = {"strategy": "dense-fixed", "model": MODEL, "chunk_tokens": size, "overlap_tokens": overlap,
              "slice": slice_name, "chunks_total": total_chunks, "chunks_embedded": embedder.misses}
    # Score only the issues ranked here (see bm25.run).
    return {
        "file": log_run("dense-fixed-file", {**params, "level": "file"},
                        file_rankings, {i: gold_files[i] for i in file_rankings}),
        "function": log_run("dense-fixed-function", {**params, "level": "function"},
                            fn_rankings, {i: gold_functions[i] for i in fn_rankings}),
    }


if __name__ == "__main__":
    repo = sys.argv[1] if len(sys.argv) > 1 else "pytest-dev/pytest"
    issues = load_issues(repo)
    issues = issues[issues.instance_id.isin(tune_ids())]  # never score held-out here
    gt = pd.read_parquet("data/gold/ground_truth.parquet")
    membership = pd.read_parquet("data/silver/snapshot_membership.parquet")
    results = run(issues, dict(zip(gt.instance_id, gt.gold_files)),
                  dict(zip(gt.instance_id, gt.gold_functions)), membership, slice_name=f"{repo} tune")
    for level, r in results.items():
        print(f"dense fixed chunks, {level:8s}  " + "  ".join(
            f"{k}={v:.3f}" for k, v in r.items() if k != "n") + f"  n={r['n']}")
