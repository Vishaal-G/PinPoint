"""Retrieval metrics. A ranking is a list of ids, best first; gold is the set of correct ids.

Ids are file paths at file level and "path::qualname" at function level; the math is the same.
Every function dedupes the ranking first, so a repeated id can't take up several top-k slots.
"""
from collections.abc import Iterable, Mapping


def dedupe(ranking: Iterable[str]) -> list[str]:
    """Drop repeats, keeping each id at its first (best) position."""
    return list(dict.fromkeys(ranking))


def accuracy_at_k(ranking: Iterable[str], gold: Iterable[str], k: int) -> float:
    """1.0 if any gold id is in the top k, else 0.0."""
    gold = set(gold)
    return float(any(r in gold for r in dedupe(ranking)[:k]))


def recall_at_k(ranking: Iterable[str], gold: Iterable[str], k: int) -> float:
    """Fraction of gold ids found in the top k."""
    gold = set(gold)
    if not gold:
        raise ValueError("recall is undefined with no gold ids")
    return len(gold & set(dedupe(ranking)[:k])) / len(gold)


def reciprocal_rank(ranking: Iterable[str], gold: Iterable[str]) -> float:
    """1 / position of the first gold id (positions start at 1); 0.0 if none is ranked."""
    gold = set(gold)
    for position, r in enumerate(dedupe(ranking), start=1):
        if r in gold:
            return 1 / position
    return 0.0


def files_from_functions(ranking: Iterable[str]) -> list[str]:
    """Turn a function ranking ("path::qualname") into a file ranking: each file at the
    position of its best-ranked function."""
    return dedupe(fid.split("::", 1)[0] for fid in ranking)


def evaluate(
    rankings: Mapping[str, list[str]],
    gold: Mapping[str, Iterable[str]],
    ks: Iterable[int] = (1, 5, 10),
) -> dict[str, float]:
    """Average every metric over the issues in `gold`.

    Issues with no gold ids are skipped (e.g. no gold function), and the count used is
    reported as "n". An issue with no ranking scores 0, so a strategy that fails on an
    issue can't look better by leaving it out.
    """
    # set() first: gold lists read from parquet are numpy arrays, and `if array` is an error.
    scored = {iid: set(g) for iid, g in gold.items()}
    scored = {iid: g for iid, g in scored.items() if g}
    if not scored:
        raise ValueError("no issues with gold ids to evaluate")

    totals: dict[str, float] = {}
    for iid, g in scored.items():
        ranking = rankings.get(iid, [])
        for k in ks:
            totals[f"acc@{k}"] = totals.get(f"acc@{k}", 0) + accuracy_at_k(ranking, g, k)
            totals[f"recall@{k}"] = totals.get(f"recall@{k}", 0) + recall_at_k(ranking, g, k)
        totals["mrr"] = totals.get("mrr", 0) + reciprocal_rank(ranking, g)

    results = {name: total / len(scored) for name, total in totals.items()}
    results["n"] = len(scored)
    return results


def per_issue(rankings: Mapping[str, list[str]], gold: Mapping[str, Iterable[str]],
              metric: str = "mrr") -> dict[str, float]:
    """Each scored issue's value of one metric ("mrr" or "acc@k"), same skipping rules as evaluate()."""
    scored = {iid: set(g) for iid, g in gold.items()}
    out = {}
    for iid, g in scored.items():
        if not g:
            continue
        ranking = rankings.get(iid, [])
        out[iid] = (reciprocal_rank(ranking, g) if metric == "mrr"
                    else accuracy_at_k(ranking, g, int(metric.removeprefix("acc@"))))
    return out


def bootstrap_ci(scores: Mapping[str, float], n_boot: int = 10_000, level: float = 0.95,
                 seed: int = 0) -> tuple[float, float, float]:
    """(mean, low, high): percentile bootstrap. Resample the issues with replacement n_boot times,
    take the mean each time; the middle `level` of those means is the interval."""
    import numpy as np

    x = np.fromiter(scores.values(), dtype=float)
    idx = np.random.default_rng(seed).integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    tail = (1 - level) / 2 * 100
    return float(x.mean()), float(np.percentile(means, tail)), float(np.percentile(means, 100 - tail))


def paired_bootstrap(a: Mapping[str, float], b: Mapping[str, float], n_boot: int = 10_000,
                     level: float = 0.95, seed: int = 0) -> tuple[float, float, float, float]:
    """(mean of a - b, low, high, share of resamples where a <= b) over the issues both scored.
    Paired: each resample draws the same issues for both, so issue difficulty cancels out; this
    is much tighter than comparing two separate intervals. The last value is a one-sided p-value
    for "a beats b"."""
    import numpy as np

    ids = sorted(a.keys() & b.keys())
    d = np.array([a[i] - b[i] for i in ids])
    idx = np.random.default_rng(seed).integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    tail = (1 - level) / 2 * 100
    return (float(d.mean()), float(np.percentile(means, tail)), float(np.percentile(means, 100 - tail)),
            float((means <= 0).mean()))
