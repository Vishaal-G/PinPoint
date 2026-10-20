"""Build the silver and gold tables for one repo's issues.

    uv run python -m pinpoint.pipeline                      # pytest-dev/pytest
    uv run python -m pinpoint.pipeline django/django        # another repo
"""
import sys
from pathlib import Path

import pandas as pd

from pinpoint.data import load_issues
from pinpoint.ground_truth import function_id, gold_functions
from pinpoint.parse import parse_snapshot
from pinpoint.patches import changed_source_lines
from pinpoint.repos import get_snapshot

SILVER = Path("data/silver")
GOLD = Path("data/gold")


def build_tables(issues: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Parse each issue's snapshot and return the tables.

    functions:            one row per unique function body (keyed by hash), source stored once
    edges:                (hash, callee) for every name a function body calls; keyed by hash
                          because calls depend only on the source, so each body's edges are stored once
    snapshot_membership:  which functions exist in which issue's snapshot, and where
    ground_truth:         per issue, the files and functions the gold fix touched
    """
    membership, sources, calls, truth = [], {}, {}, []
    for _, issue in issues.iterrows():
        funcs = parse_snapshot(get_snapshot(issue))
        sources.update(zip(funcs["hash"], funcs["source"]))
        calls.update(zip(funcs["hash"], funcs["calls"]))
        membership.append(
            funcs.drop(columns=["source", "calls"]).assign(
                instance_id=issue["instance_id"],
                function_id=[function_id(p, q) for p, q in zip(funcs["path"], funcs["qualname"])],
            )
        )

        changed = changed_source_lines(issue["patch"])
        gold = gold_functions(changed, funcs)
        truth.append({
            "instance_id": issue["instance_id"],
            "repo": issue["repo"],
            "gold_files": sorted(changed),
            "gold_functions": gold,
            "has_function": bool(gold),
        })
        print(f"{issue['instance_id']:30s} {len(funcs):5d} functions  gold={gold}")

    return {
        "functions": pd.DataFrame({"hash": list(sources), "source": list(sources.values())}),
        "edges": pd.DataFrame(
            [(h, callee) for h, names in calls.items() for callee in names],
            columns=["hash", "callee"],
        ),
        "snapshot_membership": pd.concat(membership, ignore_index=True)[
            ["instance_id", "function_id", "path", "qualname", "start_line", "end_line", "hash"]
        ],
        "ground_truth": pd.DataFrame(truth),
    }


def save_tables(tables: dict[str, pd.DataFrame]) -> None:
    SILVER.mkdir(parents=True, exist_ok=True)
    GOLD.mkdir(parents=True, exist_ok=True)
    tables["functions"].to_parquet(SILVER / "functions.parquet", index=False)
    tables["edges"].to_parquet(SILVER / "edges.parquet", index=False)
    tables["snapshot_membership"].to_parquet(SILVER / "snapshot_membership.parquet", index=False)
    tables["ground_truth"].to_parquet(GOLD / "ground_truth.parquet", index=False)


if __name__ == "__main__":
    repo = sys.argv[1] if len(sys.argv) > 1 else "pytest-dev/pytest"
    tables = build_tables(load_issues(repo))
    save_tables(tables)
    gt = tables["ground_truth"]
    print(f"\n{gt['has_function'].sum()}/{len(gt)} issues have >=1 gold function")
    print(f"{len(tables['snapshot_membership'])} membership rows, "
          f"{len(tables['functions'])} unique functions, {len(tables['edges'])} call edges")
