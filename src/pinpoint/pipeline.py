"""Build the silver and gold tables for one repo's issues, or for a slice across repos.

    uv run python -m pinpoint.pipeline                      # pytest-dev/pytest
    uv run python -m pinpoint.pipeline django/django        # another repo
    uv run python -m pinpoint.pipeline --slice=recheck      # the 52-issue stratified slice (split.recheck_ids)

New rows are merged into the existing parquet files, with the same rules as the Databricks job:
hash-keyed tables keep one row per hash, per-issue tables replace exactly the issues just built.
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


PATHS = {"functions": SILVER / "functions.parquet", "edges": SILVER / "edges.parquet",
         "snapshot_membership": SILVER / "snapshot_membership.parquet", "ground_truth": GOLD / "ground_truth.parquet"}


def merge_tables(old: dict[str, pd.DataFrame], new: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Add `new` to `old`: hash-keyed tables gain only unseen hashes (a known hash = identical
    content), per-issue tables drop the rebuilt issues' old rows first. Rebuilding is idempotent."""
    out = {}
    for name in ("functions", "edges"):
        known = set(old[name].hash) if name in old else set()
        fresh = new[name][~new[name].hash.isin(known)]
        out[name] = pd.concat([old[name], fresh], ignore_index=True) if name in old else fresh
    for name in ("snapshot_membership", "ground_truth"):
        if name not in old:
            out[name] = new[name]
            continue
        kept = old[name][~old[name].instance_id.isin(set(new[name].instance_id))]
        out[name] = pd.concat([kept, new[name]], ignore_index=True)
    return out


def save_tables(tables: dict[str, pd.DataFrame]) -> None:
    """Merge into the parquet files already on disk (see merge_tables)."""
    SILVER.mkdir(parents=True, exist_ok=True)
    GOLD.mkdir(parents=True, exist_ok=True)
    old = {name: pd.read_parquet(path) for name, path in PATHS.items() if path.exists()}
    for name, df in merge_tables(old, tables).items():
        df.to_parquet(PATHS[name], index=False)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    slice_ = next((a.removeprefix("--slice=") for a in sys.argv if a.startswith("--slice=")), None)
    if slice_:  # recheck (52) or tune (200)
        from pinpoint.split import recheck_ids, tune_ids

        issues = load_issues()
        issues = issues[issues.instance_id.isin(recheck_ids() if slice_ == "recheck" else tune_ids())]
    else:
        issues = load_issues(args[0] if args else "pytest-dev/pytest")
    tables = build_tables(issues)
    save_tables(tables)
    gt = tables["ground_truth"]
    print(f"\n{gt['has_function'].sum()}/{len(gt)} issues have >=1 gold function")
    print(f"{len(tables['snapshot_membership'])} membership rows, "
          f"{len(tables['functions'])} unique functions, {len(tables['edges'])} call edges")
