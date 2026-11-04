"""Strategy 3: one chunk per function (from silver) -> bge-small -> brute-force cosine search.

Functions come from silver/snapshot_membership + silver/functions, so the same source is
embedded once across every snapshot (the cache key is the same sha256 as the silver hash).

    uv run python -m pinpoint.dense_functions             # pytest-dev/pytest tune issues
    uv run python -m pinpoint.dense_functions --header    # prepend "path\\nqualname" to each chunk
    uv run python -m pinpoint.dense_functions --model=databricks-bge-large-en   # Databricks endpoint
    uv run python -m pinpoint.dense_functions --slice=heldout   # a cross-repo slice (needs pipeline.py --slice)
"""
import sys

import numpy as np
import pandas as pd

from pinpoint.data import load_issues
from pinpoint.dense import rank_files, search
from pinpoint.embed import MODEL, make_embedder
from pinpoint.repos import is_source_file
from pinpoint.split import tune_ids
from pinpoint.tracking import log_run


def chunk_texts(fns: pd.DataFrame, header: bool = False) -> list[str]:
    """The text embedded for each function: its source, optionally under a path + qualname header."""
    if not header:
        return fns.source.tolist()
    return [f"{p}\n{q}\n{s}" for p, q, s in zip(fns.path, fns.qualname, fns.source)]


def rank_functions(fns: pd.DataFrame, scores: np.ndarray) -> list[str]:
    """Function ids by their own cosine, best first."""
    order = np.argsort(-scores, kind="stable")
    return fns.function_id.values[order].tolist()


def run(issues: pd.DataFrame, gold_files: dict, gold_functions: dict, candidates: pd.DataFrame,
        header: bool = False, slice_name: str = "", model: str = MODEL) -> dict[str, dict]:
    """`candidates`: membership rows (source files only) joined with each function's source."""
    embedder = make_embedder(model)
    file_rankings, fn_rankings = {}, {}
    for _, issue in issues.iterrows():
        iid = issue["instance_id"]
        fns = candidates[candidates.instance_id == iid]
        scores = search(embedder.encode(chunk_texts(fns, header)), embedder.encode_query(issue["problem_statement"]))
        fn_rankings[iid] = rank_functions(fns, scores)
        file_rankings[iid] = rank_files(fns, scores)  # file = its best function
        embedder.save()
        print(f"  {iid}: {len(fns)} functions, {embedder.misses} embedded so far", flush=True)

    ranked = candidates[candidates.instance_id.isin(fn_rankings)]
    params = {"strategy": "dense-functions", "model": model, "header": header, "slice": slice_name,
              "chunks_total": len(ranked), "chunks_unique": ranked.hash.nunique(),
              "chunks_embedded": embedder.misses}
    return {
        "file": log_run("dense-functions-file", {**params, "level": "file"},
                        file_rankings, {i: gold_files[i] for i in file_rankings}),
        "function": log_run("dense-functions-function", {**params, "level": "function"},
                            fn_rankings, {i: gold_functions[i] for i in fn_rankings}),
    }


def load_candidates(ids: set[str] | None = None) -> pd.DataFrame:
    """Membership (source files only) joined with source. Filter to `ids` before the join: joining
    all 2M membership rows to their source text takes several GB."""
    membership = pd.read_parquet("data/silver/snapshot_membership.parquet")
    if ids is not None:
        membership = membership[membership.instance_id.isin(ids)]
    membership = membership[membership.path.map(is_source_file)]
    functions = pd.read_parquet("data/silver/functions.parquet", filters=[("hash", "in", list(set(membership.hash)))])
    return membership.merge(functions, on="hash", how="left").sort_values(["instance_id", "path", "start_line"])


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    slice_ = next((a.removeprefix("--slice=") for a in sys.argv if a.startswith("--slice=")), None)
    if slice_:  # recheck / tune / heldout across repos
        from pinpoint.split import slice_ids

        ids, slice_name = slice_ids(slice_), slice_
        issues = load_issues()
        issues = issues[issues.instance_id.isin(ids)]
    else:
        repo = args[0] if args else "pytest-dev/pytest"
        issues = load_issues(repo)
        issues = issues[issues.instance_id.isin(tune_ids())]  # never score held-out here
        ids, slice_name = set(issues.instance_id), f"{repo} tune"
    gt = pd.read_parquet("data/gold/ground_truth.parquet")
    results = run(issues, dict(zip(gt.instance_id, gt.gold_files)), dict(zip(gt.instance_id, gt.gold_functions)),
                  load_candidates(ids), header="--header" in sys.argv, slice_name=slice_name,
                  model=next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--model=")), MODEL))
    for level, r in results.items():
        print(f"dense functions, {level:8s}  " + "  ".join(
            f"{k}={v:.3f}" for k, v in r.items() if k != "n") + f"  n={r['n']}")
