"""Strategy 5a: rewrite an issue into a search query with regexes (no LLM).

Issues mix the useful part (names of code, a traceback, an error) with noise (`pip list`
output, templates, prose). The extractor keeps only code-looking terms:

    traceback frames   File "x/rewrite.py", line 760, in is_rewrite_disabled  /  rewrite.py:760: in run
    error names        TypeError, ImportError, PytestUnknownMarkWarning
    code identifiers   from backticks and code blocks, only if they look like code:
                       snake_case, dotted.names, camelCase, --options, call()
    .py paths          src/_pytest/python.py

    uv run python -m pinpoint.query      # score raw vs rewritten queries on pytest tune issues
"""
import re
import sys
from dataclasses import dataclass, field

from pinpoint.text import tokenize

FRAME = re.compile(r'File "(?P<p1>[^"]+)", line \d+, in (?P<f1>\w+)|(?P<p2>[\w/.\-]+\.py):\d+: in (?P<f2>\w+)')
ERROR = re.compile(r"\b[A-Z]\w*(?:Error|Exception|Warning)\b")
PATH = re.compile(r"[\w/.\-]*\w\.py\b")
CODE_BLOCK = re.compile(r"```.*?```", re.S)
BACKTICK = re.compile(r"`([^`\n]+)`")
TERM = re.compile(r"--[A-Za-z][\w-]*|[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*(?:\(\))?")


def looks_like_code(term: str) -> bool:
    """`attrs`, `the`, `version` are words; `is_rewrite_disabled`, `os.path`, `getBasetemp`,
    `--fixtures` and `run()` are code. `____` (a pytest separator line) is not."""
    if not re.search(r"[A-Za-z]{2}", term):
        return False
    return (term.startswith("--") or "_" in term or "." in term or term.endswith("()")
            or re.search(r"[a-z][A-Z]", term) is not None)


@dataclass
class Extracted:
    frames: list[str] = field(default_factory=list)       # function names from traceback frames
    errors: list[str] = field(default_factory=list)
    identifiers: list[str] = field(default_factory=list)
    paths: list[str] = field(default_factory=list)

    def terms(self) -> list[str]:
        """All terms, de-duplicated, frames first (the most specific evidence)."""
        return list(dict.fromkeys(self.frames + self.errors + self.identifiers + self.paths))


def _unique(items) -> list[str]:
    return list(dict.fromkeys(items))


def extract(text: str) -> Extracted:
    frames = [(m["p1"] or m["p2"], m["f1"] or m["f2"]) for m in FRAME.finditer(text)]
    code = " ".join(BACKTICK.findall(text) + CODE_BLOCK.findall(text))
    identifiers = [t.removesuffix("()") for t in TERM.findall(code) if looks_like_code(t)]
    return Extracted(
        # A path like /usr/lib/python3.8/site-packages/_pytest/x.py: keep from the package on.
        frames=_unique(f for _, f in frames if f != "<module>"),
        errors=_unique(ERROR.findall(text)),
        identifiers=_unique(i for i in identifiers if not PATH.fullmatch(i)),
        paths=_unique(p.split("site-packages/")[-1] for p in PATH.findall(text)),
    )


def title(text: str) -> str:
    """SWE-bench problem statements start with the issue title on its own line."""
    return text.strip().split("\n", 1)[0]


def in_vocab(terms: list[str], vocab: set[str]) -> list[str]:
    """Terms that occur in the repo: any of their identifier tokens is in `vocab`.
    Drops the reporter's own names (test_all_even, MyModel) that no repo function contains."""
    return [t for t in terms if any(tok in vocab for tok in tokenize(t, "whole"))]


def rewrite(text: str, mode: str, vocab: set[str] | None = None) -> str:
    """The query text sent to a retriever.

    raw          the issue as written (the baseline)
    terms        only the extracted terms; falls back to raw when nothing was extracted
    raw+terms    the issue, then the terms again: for BM25 a repeated term counts twice
    title+terms  the title and the terms: short enough that a 512-token embedder sees all of it
    repo-...     the same, keeping only terms that occur in the repo (`vocab` required)
    """
    terms = extract(text).terms()
    if mode.startswith("repo-"):
        terms, mode = in_vocab(terms, vocab), mode.removeprefix("repo-")
    terms = " ".join(terms)
    if mode == "raw" or (mode == "terms" and not terms):
        return text
    return {"terms": terms, "raw+terms": f"{text}\n{terms}", "title+terms": f"{title(text)}\n{terms}"}[mode]


if __name__ == "__main__":
    import pandas as pd

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
    gold = {"file": dict(zip(gt.instance_id, gt.gold_files)), "fn": dict(zip(gt.instance_id, gt.gold_functions))}
    gold = {level: {i: g[i] for i in issues.instance_id} for level, g in gold.items()}
    candidates = load_candidates()

    # Repo vocabulary per snapshot: every identifier token in its candidate functions.
    vocab = {iid: {tok for src in fns.source for tok in tokenize(src, "whole")}
             for iid, fns in candidates[candidates.instance_id.isin(issues.instance_id)].groupby("instance_id")}
    for _, issue in issues.iterrows():
        iid, terms = issue["instance_id"], extract(issue["problem_statement"]).terms()
        print(f"  {iid}: kept {in_vocab(terms, vocab[iid])[:10]}  dropped {[t for t in terms if t not in in_vocab(terms, vocab[iid])][:6]}")

    rows = []
    for mode in ("raw", "terms", "raw+terms", "title+terms", "repo-terms", "repo-raw+terms", "repo-title+terms"):
        queries = {i: rewrite(t, mode, vocab[i]) for i, t in zip(issues.instance_id, issues.problem_statement)}
        sides = side_rankings(issues, candidates, queries)
        for name, level in (("bm25_file", "file"), ("dense_file", "file"), ("bm25_fn", "fn"), ("dense_fn", "fn")):
            r = evaluate(sides[name], gold[level])
            rows.append({"query": mode, "retriever": name, **{m: round(r[m], 3) for m in ("acc@1", "acc@5", "acc@10", "mrr")}})
            retriever, lvl = name.split("_")
            log_run(f"query-{mode}-{name}", {"strategy": f"query-rewrite-regex+{retriever}", "query": mode,
                                              "level": "function" if lvl == "fn" else "file",
                                              "slice": f"{repo} tune"}, sides[name], gold[level])
    print(pd.DataFrame(rows).sort_values(["retriever", "query"]).to_string(index=False))
