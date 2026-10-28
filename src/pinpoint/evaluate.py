"""Score the non-LLM strategies on any slice, from tables only (no worktrees, no local model).

The Databricks job's `evaluate` task and a local run call the same functions, so cloud and
laptop numbers come from one code path. Inputs are plain DataFrames / dicts:

    candidates   membership rows for source files, joined with each function's source
    vectors      body hash -> unit vector (any embedding model, e.g. databricks-bge-large-en)
    query_vecs   instance_id -> unit vector of the issue text (same model)
    edges        silver/edges (hash, callee)

    uv run python -m pinpoint.evaluate             # recheck slice, local parquet tables
"""
from collections.abc import Mapping

import numpy as np
import pandas as pd

from pinpoint import dense, dense_functions
from pinpoint.fusion import rank_functions_bm25, rrf
from pinpoint.graph import expand, snapshot_graph
from pinpoint.metrics import evaluate as score
from pinpoint.metrics import files_from_functions
from pinpoint.repos import is_source_file

# strategy -> level. Names match the NOTES.md rows: 3 = dense function chunks, 6 = + call graph.
STRATEGIES = {"dense_file": "file", "dense_fn": "fn", "graph_fn": "fn", "bm25_fn": "fn", "rrf_fn": "fn",
              "graph_file": "file", "bm25_file": "file"}
METRICS = ("acc@1", "acc@5", "acc@10", "mrr")
SEEDS_K = 10
RANKINGS_K = 50


def candidates_from(membership: pd.DataFrame, functions: pd.DataFrame) -> pd.DataFrame:
    """Same rows as dense_functions.load_candidates, from any two DataFrames."""
    membership = membership[membership.path.map(is_source_file)]
    return membership.merge(functions, on="hash", how="left").sort_values(["instance_id", "path", "start_line"])


def rankings(issues: pd.DataFrame, candidates: pd.DataFrame, vectors: Mapping[str, np.ndarray],
             query_vecs: Mapping[str, np.ndarray], edges: pd.DataFrame) -> dict[str, dict[str, list[str]]]:
    """strategy -> instance_id -> ranked ids (function ids, or paths for file strategies)."""
    out = {name: {} for name in STRATEGIES}
    by_issue = dict(tuple(candidates.groupby("instance_id")))
    for _, issue in issues.iterrows():
        iid = issue["instance_id"]
        fns = by_issue[iid]
        scores = dense.search(np.stack([vectors[h] for h in fns.hash]), query_vecs[iid])
        dense_fn = dense_functions.rank_functions(fns, scores)
        graph_fn = expand(dense_fn, snapshot_graph(fns, edges), k=5, alpha=1.0, direction="both")
        bm25_fn = rank_functions_bm25(fns, issue["problem_statement"])
        out["dense_file"][iid] = dense.rank_files(fns, scores)
        out["dense_fn"][iid] = dense_fn
        out["graph_fn"][iid] = graph_fn
        out["bm25_fn"][iid] = bm25_fn
        out["rrf_fn"][iid] = rrf([bm25_fn, dense_fn])
        out["graph_file"][iid] = files_from_functions(graph_fn)
        out["bm25_file"][iid] = files_from_functions(bm25_fn)  # file = its best function (not Strategy 1)
        print(f"  {iid}: {len(fns)} functions", flush=True)
    return out


def results_table(ranked: dict[str, dict[str, list[str]]], ground_truth: pd.DataFrame) -> pd.DataFrame:
    """One row per (strategy, repo): the metrics over that repo's issues, plus repo = "all"."""
    gt = ground_truth.set_index("instance_id")
    rows = []
    for name, level in STRATEGIES.items():
        col = "gold_files" if level == "file" else "gold_functions"
        ids = list(ranked[name])
        groups = {"all": ids, **{repo: [i for i in ids if gt.loc[i, "repo"] == repo]
                                 for repo in sorted(gt.loc[ids, "repo"].unique())}}
        for repo, members in groups.items():
            r = score({i: ranked[name][i] for i in members}, {i: gt.loc[i, col] for i in members})
            rows.append({"strategy": name, "level": level, "repo": repo, "n": r["n"],
                         **{m: round(r[m], 3) for m in METRICS}})
    return pd.DataFrame(rows)


def rankings_table(ranked: dict[str, dict[str, list[str]]], k: int = RANKINGS_K) -> pd.DataFrame:
    """Every strategy's top-k per issue, long format: the table to open when asking *why* a strategy
    missed on a slice too big to rerun locally."""
    rows = [{"strategy": name, "instance_id": iid, "rank": r, "item": item}
            for name, per_issue in ranked.items() for iid, ranking in per_issue.items()
            for r, item in enumerate(ranking[:k], 1)]
    return pd.DataFrame(rows)


def seeds_table(ranked: dict[str, dict[str, list[str]]], candidates: pd.DataFrame, k: int = SEEDS_K) -> pd.DataFrame:
    """The top-k of Strategy 6 (graph_fn) with line ranges: what the agent (Strategy H) is seeded with."""
    lines = candidates.set_index(["instance_id", "function_id"])[["start_line", "end_line"]]
    rows = [{"instance_id": iid, "rank": r, "function_id": f,
             "start_line": int(lines.loc[(iid, f), "start_line"]), "end_line": int(lines.loc[(iid, f), "end_line"])}
            for iid, ranking in ranked["graph_fn"].items() for r, f in enumerate(ranking[:k], 1)]
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from pinpoint.data import load_issues
    from pinpoint.embed import ENDPOINT_MODEL, EndpointEmbedder
    from pinpoint.split import recheck_ids

    issues = load_issues()
    issues = issues[issues.instance_id.isin(recheck_ids())]
    candidates = candidates_from(pd.read_parquet("data/silver/snapshot_membership.parquet"),
                                 pd.read_parquet("data/silver/functions.parquet"))
    candidates = candidates[candidates.instance_id.isin(issues.instance_id)]
    embedder = EndpointEmbedder(ENDPOINT_MODEL)
    embedder.encode(candidates.source.drop_duplicates().tolist())  # only hashes not cached yet
    embedder.save()
    vectors = {h: embedder.cache[h] for h in candidates.hash.unique()}
    query_vecs = {i: embedder.encode_query(t) for i, t in zip(issues.instance_id, issues.problem_statement)}
    ranked = rankings(issues, candidates, vectors, query_vecs, pd.read_parquet("data/silver/edges.parquet"))
    table = results_table(ranked, pd.read_parquet("data/gold/ground_truth.parquet"))
    print(table[table.repo == "all"].to_string(index=False))
