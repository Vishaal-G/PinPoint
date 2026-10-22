"""Strategy 1: BM25 over whole files at each issue's base commit.

    uv run python -m pinpoint.bm25                  # pytest-dev/pytest, all tokenizer modes
"""
import sys
from pathlib import Path

import pandas as pd
from rank_bm25 import BM25Okapi

from pinpoint.data import load_issues
from pinpoint.repos import get_snapshot, source_files
from pinpoint.split import tune_ids
from pinpoint.text import tokenize
from pinpoint.tracking import log_run


def rank_files(snapshot: Path, query: str, mode: str = "both",
               k1: float = 1.5, b: float = 0.75) -> list[str]:
    """Every source file in the snapshot, best BM25 match for `query` first."""
    paths = source_files(snapshot)
    docs = [tokenize((snapshot / p).read_text(errors="replace"), mode) for p in paths]
    # IDF and average length come from this snapshot's files only.
    index = BM25Okapi(docs, k1=k1, b=b)
    scores = index.get_scores(tokenize(query, mode))
    order = sorted(range(len(paths)), key=lambda i: -scores[i])
    return [paths[i] for i in order]


def run(issues: pd.DataFrame, gold: dict, mode: str = "both",
        k1: float = 1.5, b: float = 0.75, slice_name: str = "") -> dict[str, float]:
    """Rank files for every issue, then score and log the run to MLflow."""
    rankings = {
        issue["instance_id"]: rank_files(get_snapshot(issue), issue["problem_statement"], mode, k1, b)
        for _, issue in issues.iterrows()
    }
    # Score only the issues ranked here: evaluate() counts a gold issue with no ranking as a miss.
    gold = {iid: gold[iid] for iid in rankings}
    params = {"strategy": "bm25", "level": "file", "tokens": mode, "k1": k1, "b": b,
              "slice": slice_name}
    return log_run(f"bm25-file-{mode}", params, rankings, gold)


if __name__ == "__main__":
    repo = sys.argv[1] if len(sys.argv) > 1 else "pytest-dev/pytest"
    issues = load_issues(repo)
    issues = issues[issues.instance_id.isin(tune_ids())]  # never score held-out here
    gt = pd.read_parquet("data/gold/ground_truth.parquet")
    gold = dict(zip(gt.instance_id, gt.gold_files))
    for mode in ("whole", "split", "both"):
        r = run(issues, gold, mode, slice_name=f"{repo} tune")
        print(f"bm25 files, tokens={mode:5s}  " + "  ".join(
            f"{k}={v:.3f}" for k, v in r.items() if k != "n") + f"  n={r['n']}")
