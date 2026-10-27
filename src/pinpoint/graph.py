"""Strategy 6: call-graph expansion. Re-rank by giving the callers and callees of the top hits a bonus.

The call graph is name-only (silver/edges: body hash -> called name), so an edge "f calls `run`"
points at every function in the snapshot whose short name is `run`. An edge resolved to m targets
gets weight 1/m: a call to `__init__` (145 targets) says almost nothing, a call to
`is_rewrite_disabled` (1 target) is near-certain.

Rank-based, so it works on any retriever's ranking:
    base(f)  = 1 / (60 + rank(f))
    bonus(f) = alpha * max over seeds s adjacent to f of  weight(s, f) / (60 + rank(s))
    score(f) = base(f) + bonus(f)        seeds = the top `k` of the ranking

With alpha=1 an unambiguous neighbour of seed s can outrank s itself (helps when the top hit is the
failing frame and the fix is its caller; hurts when the top hit is already right). cap=True keeps
every neighbour just below the seed that lifted it.

    uv run python -m pinpoint.graph        # pytest-dev/pytest tune issues
"""
import sys
from collections import defaultdict

import pandas as pd

K_RRF = 60


def snapshot_graph(fns: pd.DataFrame, edges: pd.DataFrame, weighted: bool = True) -> dict[str, dict[str, dict[str, float]]]:
    """{"callees": {f: {g: w}}, "callers": {g: {f: w}}} for one snapshot's functions.
    `fns` needs function_id, qualname, hash; `edges` is silver/edges (hash, callee)."""
    by_short = defaultdict(list)
    for fid, qualname in zip(fns.function_id, fns.qualname):
        by_short[qualname.split(".")[-1]].append(fid)
    calls_by_hash = edges[edges.hash.isin(set(fns.hash))].groupby("hash").callee.apply(set)

    callees, callers = defaultdict(dict), defaultdict(dict)
    for fid, h in zip(fns.function_id, fns.hash):
        for name in calls_by_hash.get(h, ()):
            targets = by_short.get(name, [])
            for target in targets:
                if target == fid:
                    continue
                w = 1 / len(targets) if weighted else 1.0
                callees[fid][target] = max(w, callees[fid].get(target, 0.0))
                callers[target][fid] = max(w, callers[target].get(fid, 0.0))
    return {"callees": callees, "callers": callers}


def expand(ranking: list[str], graph: dict, k: int = 5, alpha: float = 1.0, direction: str = "both",
           cap: bool = False) -> list[str]:
    """Re-rank: every function keeps its base score; neighbours of the top-k seeds get a bonus.
    direction: "callees" (what the seeds call), "callers" (what calls the seeds), or "both".
    cap: a neighbour never scores above the seed that lifted it (it lands right after it)."""
    rank = {f: r for r, f in enumerate(ranking, start=1)}
    score = {f: 1 / (K_RRF + r) for f, r in rank.items()}
    bonus: dict[str, float] = {}
    ceiling: dict[str, float] = {}
    directions = ("callees", "callers") if direction == "both" else (direction,)
    for seed in ranking[:k]:
        seed_score = 1 / (K_RRF + rank[seed])
        for d in directions:
            for neighbour, w in graph[d].get(seed, {}).items():
                if alpha * w * seed_score > bonus.get(neighbour, 0.0):
                    bonus[neighbour], ceiling[neighbour] = alpha * w * seed_score, seed_score
    for f, b in bonus.items():
        if f in ranking[:k] and cap:
            continue   # a seed keeps its own place
        score[f] = score.get(f, 0.0) + b
        if cap:
            score[f] = min(score[f], ceiling[f] - 1e-9)
    return sorted(score, key=lambda f: -score[f])   # stable: ties keep the original order


if __name__ == "__main__":
    from pinpoint.data import load_issues
    from pinpoint.dense_functions import load_candidates
    from pinpoint.fusion import side_rankings
    from pinpoint.metrics import evaluate
    from pinpoint.split import tune_ids
    from pinpoint.tracking import log_run

    repo = sys.argv[1] if len(sys.argv) > 1 else "pytest-dev/pytest"
    issues = load_issues(repo)
    issues = issues[issues.instance_id.isin(tune_ids())]  # never score held-out here
    gt = pd.read_parquet("data/gold/ground_truth.parquet")
    gold = {i: g for i, g in zip(gt.instance_id, gt.gold_functions) if i in set(issues.instance_id)}
    candidates = load_candidates()
    edges = pd.read_parquet("data/silver/edges.parquet")
    sides = side_rankings(issues, candidates)
    graphs = {iid: {w: snapshot_graph(fns, edges, weighted=w) for w in (True, False)}
              for iid, fns in candidates[candidates.instance_id.isin(issues.instance_id)].groupby("instance_id")}

    def gold_rank(ranking, g):
        g = set(g)
        return next((r for r, f in enumerate(ranking, start=1) if f in g), None)

    # Pre-registered setting: k=5, alpha=1, both directions, ambiguity-weighted. The rest is sensitivity.
    MAIN = dict(k=5, alpha=1.0, direction="both", weighted=True, cap=False)
    variants = [MAIN] + [dict(MAIN, **change) for change in (
        {"weighted": False}, {"direction": "callers"}, {"direction": "callees"},
        {"k": 10}, {"alpha": 0.5}, {"alpha": 2.0}, {"cap": True}, {"cap": True, "direction": "callers"})]
    rows = []
    for seed in ("bm25_fn", "dense_fn"):
        base = evaluate(sides[seed], gold)
        rows.append({"seeds": seed, "variant": "no expansion", **{m: round(base[m], 3) for m in ("acc@1", "acc@5", "acc@10", "mrr")}})
        for v in variants:
            ranked = {i: expand(sides[seed][i], graphs[i][v["weighted"]], v["k"], v["alpha"], v["direction"], v["cap"]) for i in sides[seed]}
            r = evaluate(ranked, gold)
            label = "MAIN" if v is MAIN else ", ".join(f"{a}={b}" for a, b in v.items() if MAIN[a] != b)
            rows.append({"seeds": seed, "variant": label, **{m: round(r[m], 3) for m in ("acc@1", "acc@5", "acc@10", "mrr")}})
            if v is MAIN:
                log_run(f"graph-{seed}", {"strategy": f"graph-expansion+{seed}", "level": "function", **v,
                                         "slice": f"{repo} tune"}, ranked, gold)
                moves = {i.split("-")[-1]: (gold_rank(sides[seed][i], g), gold_rank(ranked[i], g))
                         for i, g in gold.items() if len(g)}
                print(f"{seed} MAIN, gold rank before -> after: {moves}")
    print(pd.DataFrame(rows).to_string(index=False))
