"""Log every evaluation as an MLflow run, so strategies can be compared side by side.

View the runs:  uv run mlflow ui --backend-store-uri sqlite:///data/mlflow.db
"""
from collections.abc import Iterable, Mapping

import pandas as pd

from pinpoint.metrics import evaluate, reciprocal_rank

TRACKING_URI = "sqlite:///data/mlflow.db"
EXPERIMENT = "pinpoint"


def per_issue(rankings: Mapping[str, list[str]], gold: Mapping[str, Iterable[str]]) -> pd.DataFrame:
    """One row per scored issue: where the first gold id landed (None = not ranked) and the top hit.
    This is the table to open when asking *why* a strategy missed."""
    rows = []
    for iid, g in gold.items():
        g = set(g)
        if not g:
            continue
        ranking = rankings.get(iid, [])
        rr = reciprocal_rank(ranking, g)
        rows.append({
            "instance_id": iid,
            "gold_rank": round(1 / rr) if rr else None,
            "top1": ranking[0] if ranking else None,
            "gold": sorted(g),
        })
    return pd.DataFrame(rows)


def log_run(
    name: str,
    params: Mapping[str, object],
    rankings: Mapping[str, list[str]],
    gold: Mapping[str, Iterable[str]],
    ks: Iterable[int] = (1, 5, 10),
) -> dict[str, float]:
    """Score `rankings` against `gold` and record params, metrics and the per-issue table."""
    results = evaluate(rankings, gold, ks)
    import mlflow  # imported here: it takes ~40s to import from /mnt/c, and most callers never log

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=name):
        mlflow.log_params({**params, "n": results["n"]})
        # MLflow metric names can't contain "@".
        mlflow.log_metrics({k.replace("@", "_at_"): v for k, v in results.items() if k != "n"})
        mlflow.log_text(per_issue(rankings, gold).to_csv(index=False), "per_issue.csv")
    return results
