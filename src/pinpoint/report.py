"""The final results table: every strategy on one slice, each number with a 95% bootstrap interval.

    uv run python -m pinpoint.report heldout      # the final score (run once, after everything is frozen)
    uv run python -m pinpoint.report recheck      # same table on a tune slice, for comparison

Inputs, all already computed elsewhere (this module only scores):
    non-LLM rankings  gold.eval_rankings in Databricks (the job's evaluate task: top-50 per strategy per issue)
    agent rankings    data/gold/agent_runs.parquet (agent.py --slice=<slice> --seeds, H3 fixes)
    gold              data/gold/ground_truth.parquet (pipeline.py --slice=<slice>)

Top-50 only: an issue whose gold sits below rank 50 scores 0 instead of < 1/50, so MRR here can be
lower than gold.eval_results by at most .02 per such issue. The script prints both to show the gap.
"""
import sys

import pandas as pd

from pinpoint.metrics import bootstrap_ci, files_from_functions, paired_bootstrap, per_issue

H3 = frozenset({"labels", "prompt", "options", "graph"})
# Display order and names: (strategy key, level, label)
ROWS = [("dense_file", "file", "Dense (bge-large), function chunks -> file"),
        ("bm25_file", "file", "BM25 over function source -> file"),
        ("graph_file", "file", "Dense + call-graph expansion -> file"),
        ("agent_file", "file", "Agent H3 (dense seeds + tools, Llama 3.3 70B) -> file"),
        ("dense_fn", "fn", "Dense (bge-large), function chunks"),
        ("bm25_fn", "fn", "BM25 over function source"),
        ("rrf_fn", "fn", "RRF(BM25, dense)"),
        ("graph_fn", "fn", "Dense + call-graph expansion"),
        ("agent_fn", "fn", "Agent H3 (dense seeds + tools, Llama 3.3 70B)")]
METRICS = ("acc@1", "acc@5", "mrr")


def cloud_rankings(ids: set[str]) -> dict[str, dict[str, list[str]]]:
    """strategy -> instance_id -> top-50 ranking, from the latest evaluate run that scored each issue."""
    from pinpoint.cloud import CATALOG, PREFIX, query

    in_list = ", ".join(f"'{i}'" for i in sorted(ids))
    df = query(f"SELECT strategy, instance_id, rank, item, run_at FROM {CATALOG}.{PREFIX}_gold.eval_rankings "
               f"WHERE instance_id IN ({in_list})")
    df = df.astype({"rank": int})
    latest = df.groupby(["strategy", "instance_id"]).run_at.transform("max")
    df = df[df.run_at == latest].sort_values(["strategy", "instance_id", "rank"])
    return {s: {i: g.item.tolist() for i, g in sg.groupby("instance_id")} for s, sg in df.groupby("strategy")}


def cloud_results(slice_name: str) -> pd.DataFrame:
    """gold.eval_results rows (full rankings, no top-50 cut) for the slice's latest run."""
    from pinpoint.cloud import CATALOG, PREFIX, query

    df = query(f"SELECT strategy, level, repo, n, mrr, run_at FROM {CATALOG}.{PREFIX}_gold.eval_results "
               f"WHERE slice = '{slice_name}' AND repo = 'all'")
    return df[df.run_at == df.run_at.max()].astype({"mrr": float})


def agent_rankings(ids: set[str]) -> dict[str, list[str]]:
    """H3 (dense seeds, all four fixes) function rankings from the agent cache."""
    from pinpoint.agent import CACHE, MODEL, prompt_version

    runs = pd.read_parquet(CACHE)
    version = prompt_version(True, H3, "dense")
    runs = runs[(runs.model == MODEL) & (runs.prompt_version == version) & runs.instance_id.isin(ids)]
    return {r.instance_id: list(r.ranking) for r in runs.itertuples()}


def table(ranked: dict[str, dict[str, list[str]]], gt: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """One row per strategy: n and each metric as mean [low, high]. Also returns the per-issue
    scores, for paired comparisons."""
    gold = {"file": dict(zip(gt.instance_id, gt.gold_files)), "fn": dict(zip(gt.instance_id, gt.gold_functions))}
    rows, scores = [], {}
    for key, level, label in ROWS:
        if key not in ranked:
            continue
        row = {"strategy": label, "level": level}
        for m in METRICS:
            s = per_issue(ranked[key], gold[level], m)
            scores[(key, m)] = s
            mean, lo, hi = bootstrap_ci(s)
            row["n"] = len(s)
            row[m] = f"{mean:.3f} [{lo:.3f}, {hi:.3f}]"
        rows.append(row)
    return pd.DataFrame(rows), scores


def markdown(df: pd.DataFrame) -> str:
    rows = [list(df.columns), ["---"] * len(df.columns), *df.astype(str).values.tolist()]
    return "\n".join("| " + " | ".join(map(str, r)) + " |" for r in rows)


def main(slice_name: str) -> None:
    from pinpoint.split import slice_ids

    ids = slice_ids(slice_name)
    gt = pd.read_parquet("data/gold/ground_truth.parquet")
    gt = gt[gt.instance_id.isin(ids)]
    if len(gt) != len(ids):
        sys.exit(f"ground truth has {len(gt)} of {len(ids)} {slice_name} issues: run pipeline.py --slice={slice_name}")
    ranked = cloud_rankings(ids)
    missing = {s: len(ids) - len(r) for s, r in ranked.items() if len(r) != len(ids)}
    if missing:
        print(f"warning: issues missing from eval_rankings (they score 0): {missing}")
    agent = agent_rankings(ids)
    if agent:
        if len(agent) != len(ids):
            print(f"warning: agent ran on {len(agent)} of {len(ids)} issues (the rest score 0)")
        ranked["agent_fn"] = agent
        ranked["agent_file"] = {i: files_from_functions(r) for i, r in agent.items()}

    out, scores = table(ranked, gt)
    print(f"\n{slice_name}: {len(ids)} issues, {gt.repo.nunique()} repos. Mean [95% bootstrap interval], "
          f"10,000 resamples. Function rows skip issues with no gold function.\n")
    print(markdown(out))

    print("\nPaired differences (same issues resampled for both; p = share of resamples with no gain):")
    pairs = [("agent_fn", "dense_fn"), ("agent_file", "dense_file"), ("rrf_fn", "dense_fn"), ("bm25_fn", "dense_fn")]
    for a, b in pairs:
        for m in ("mrr", "acc@1"):
            if (a, m) in scores and (b, m) in scores:
                d, lo, hi, p = paired_bootstrap(scores[(a, m)], scores[(b, m)])
                print(f"  {a} - {b} {m}: {d:+.3f} [{lo:+.3f}, {hi:+.3f}], p = {p:.4f}")

    try:
        full = cloud_results(slice_name).set_index("strategy").mrr
        top50 = {k: sum(scores[(k, "mrr")].values()) / len(scores[(k, "mrr")]) for k, _, _ in ROWS
                 if (k, "mrr") in scores and k in full.index}
        print("\nTop-50 cut check (MRR here vs gold.eval_results): "
              + ", ".join(f"{k} {v:.3f}/{full[k]:.3f}" for k, v in top50.items()))
    except Exception as e:  # informational only
        print(f"\n(eval_results check skipped: {e})")
    out.to_csv(f"data/gold/report_{slice_name}.csv", index=False)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "recheck")
