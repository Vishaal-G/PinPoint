"""Strategy 7: cross-encoder reranking of a first-stage ranking's top N.

A cross-encoder reads (query, function source) TOGETHER in one forward pass and outputs one
relevance score, so it can match words across the two texts; the bi-encoder (embed.py) encodes
each side alone and compares vectors. It costs one model pass per pair at query time, so it only
reranks the first stage's top N; everything below N keeps its first-stage order.

The pair must fit in 512 tokens together, so a raw 1,800-token issue would crowd out the code.
The pre-registered query is the issue title + the LLM summary (Strategy 5b), ~50 tokens.

Scores are cached by sha256(model, query, document): reruns and overlapping variants are free.

    uv run python -m pinpoint.rerank        # pytest-dev/pytest tune issues
"""
import hashlib
import sys
from pathlib import Path

import pandas as pd

MODEL = "BAAI/bge-reranker-base"
CACHE_DIR = Path("data/silver")


def pair_key(model: str, query: str, doc: str) -> str:
    return hashlib.sha256("\0".join((model, query, doc)).encode()).hexdigest()


class Reranker:
    def __init__(self, model_name: str = MODEL, cache_dir: Path = CACHE_DIR):
        from sentence_transformers import CrossEncoder  # slow import; only when used

        self.name = model_name
        self.model = CrossEncoder(model_name, device="cpu")
        self.cache_path = cache_dir / f"rerank_{model_name.split('/')[-1]}.parquet"
        self.cache: dict[str, float] = {}
        if self.cache_path.exists():
            df = pd.read_parquet(self.cache_path)
            self.cache = dict(zip(df.key, df.score))

    def score(self, query: str, docs: list[str]) -> list[float]:
        keys = [pair_key(self.name, query, d) for d in docs]
        todo = {k: d for k, d in zip(keys, docs) if k not in self.cache}
        if todo:
            scores = self.model.predict([(query, d) for d in todo.values()], batch_size=16)
            self.cache.update(zip(todo, map(float, scores)))
        return [self.cache[k] for k in keys]

    def save(self) -> None:
        pd.DataFrame({"key": list(self.cache), "score": list(self.cache.values())}).to_parquet(self.cache_path, index=False)


def doc_text(fn, header: bool = False) -> str:
    return f"{fn.path}\n{fn.qualname}\n{fn.source}" if header else fn.source


def rerank(ranking: list[str], fns: pd.DataFrame, query: str, scorer, n: int = 50, header: bool = False) -> list[str]:
    """Reorder the top n of `ranking` by cross-encoder score (ties keep first-stage order); the rest
    follows unchanged. `fns` is indexed by function_id with path, qualname, source. `scorer(query, docs)`."""
    top = ranking[:n]
    scores = scorer(query, [doc_text(fns.loc[f], header) for f in top])
    order = sorted(range(len(top)), key=lambda i: -scores[i])
    return [top[i] for i in order] + ranking[n:]


if __name__ == "__main__":
    from pinpoint.data import load_issues
    from pinpoint.dense_functions import load_candidates
    from pinpoint.fusion import side_rankings
    from pinpoint.graph import expand, snapshot_graph
    from pinpoint.llm_query import llm_queries
    from pinpoint.llm_query import rewrite as llm_rewrite
    from pinpoint.metrics import evaluate
    from pinpoint.split import tune_ids
    from pinpoint.tracking import log_run

    repo = sys.argv[1] if len(sys.argv) > 1 else "pytest-dev/pytest"
    issues = load_issues(repo)
    issues = issues[issues.instance_id.isin(tune_ids())]  # never score held-out here
    gt = pd.read_parquet("data/gold/ground_truth.parquet")
    gold = {i: g for i, g in zip(gt.instance_id, gt.gold_functions) if i in set(issues.instance_id)}
    candidates = load_candidates()
    fns_by_issue = {i: g.drop_duplicates("function_id").set_index("function_id")
                    for i, g in candidates[candidates.instance_id.isin(issues.instance_id)].groupby("instance_id")}
    sides = side_rankings(issues, candidates)
    edges = pd.read_parquet("data/silver/edges.parquet")
    first_stage = {
        "dense": sides["dense_fn"],
        "bm25": sides["bm25_fn"],
        "dense+graph": {i: expand(r, snapshot_graph(fns_by_issue[i].reset_index(), edges)) for i, r in sides["dense_fn"].items()},
    }
    answers = llm_queries(issues).set_index("instance_id")   # cached from Strategy 5b: no endpoint calls
    queries = {
        "title+llm": {i: llm_rewrite(t, answers.loc[i], "title+llm") for i, t in zip(issues.instance_id, issues.problem_statement)},
        "raw": dict(zip(issues.instance_id, issues.problem_statement)),
    }

    def recall_at(rankings, n):
        return sum(bool(set(g) & set(rankings[i][:n])) for i, g in gold.items() if len(g)) / sum(len(g) > 0 for g in gold.values())

    MAIN = dict(model=MODEL, stage="dense", query="title+llm", n=50, header=False)
    variants = [MAIN] + [dict(MAIN, **c) for c in (
        {"query": "raw"}, {"header": True}, {"n": 20}, {"model": "cross-encoder/ms-marco-MiniLM-L6-v2"},
        {"stage": "bm25"}, {"stage": "dense+graph"})]
    rerankers: dict[str, Reranker] = {}
    rows = []
    for stage, rankings in first_stage.items():
        r = evaluate(rankings, gold)
        rows.append({"variant": f"{stage} (no rerank)", "ceiling@50": round(recall_at(rankings, 50), 2),
                     **{m: round(r[m], 3) for m in ("acc@1", "acc@5", "acc@10", "mrr")}})
    for v in variants:
        rr = rerankers.setdefault(v["model"], Reranker(v["model"]))
        ranked = {i: rerank(first_stage[v["stage"]][i], fns_by_issue[i], queries[v["query"]][i], rr.score, v["n"], v["header"])
                  for i in gold}
        rr.save()
        r = evaluate(ranked, gold)
        label = "MAIN" if v is MAIN else ", ".join(f"{a}={b}" for a, b in v.items() if MAIN[a] != b)
        rows.append({"variant": label, "ceiling@50": round(recall_at(first_stage[v["stage"]], v["n"]), 2),
                     **{m: round(r[m], 3) for m in ("acc@1", "acc@5", "acc@10", "mrr")}})
        print(f"  {label}: {r['mrr']:.3f}", flush=True)
        if v is MAIN:
            log_run("rerank-dense", {"strategy": "cross-encoder-rerank", "level": "function", **v, "slice": f"{repo} tune"},
                    ranked, gold)
            ranks = {i.split("-")[-1]: (next((k for k, f in enumerate(first_stage["dense"][i], 1) if f in set(g)), None),
                                        next((k for k, f in enumerate(ranked[i], 1) if f in set(g)), None))
                     for i, g in gold.items() if len(g)}
            print(f"  MAIN gold rank before -> after: {ranks}")
    print(pd.DataFrame(rows).to_string(index=False))
