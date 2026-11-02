"""The live path: any GitHub repo at its current commit + a new issue -> where to look.

Everything else in pinpoint scores SWE-bench issues from precomputed tables. This is what the bot
runs: check the repo out, parse it, embed only function bodies the cache hasn't seen (sha256 of
the source, the same key as silver, so the first index of a repo is slow and later ones aren't),
rank functions by dense similarity, and hand the top 10 to the agent as seeds (Strategy H3:
labels + prompt + options + call_graph, the best row on the recheck slice).

    uv run python -m pinpoint.live pytest-dev/pytest --issue 12345        # a GitHub issue, at HEAD
    uv run python -m pinpoint.live pytest-dev/pytest --text "..." [--ref <sha>]
    uv run python -m pinpoint.live path/to/checkout --text "..."          # a local checkout, as is
"""
import argparse
import json
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from pinpoint.agent import TOP_K, run_agent
from pinpoint.dense import search
from pinpoint.dense_functions import rank_functions
from pinpoint.ground_truth import function_id
from pinpoint.parse import parse_file
from pinpoint.repos import source_files

LIVE_DIR = Path("data/live")
FIXES = frozenset({"labels", "prompt", "options", "graph"})  # H3
SAVE_EVERY = 2000  # embed this many bodies between cache saves, so a cold index can resume


def checkout(repo: str, ref: str | None = None, root: Path = LIVE_DIR) -> tuple[Path, str]:
    """A checkout of `repo` ("owner/name", or a local directory used as is) at `ref` (default: the
    remote's default branch, freshly fetched). Returns the directory and the commit sha."""
    from git import Repo

    if Path(repo).is_dir():
        return Path(repo), Repo(repo, search_parent_directories=True).head.commit.hexsha
    path = root / "repos" / repo.replace("/", "__")
    if path.exists():
        r = Repo(path)
        r.remotes.origin.fetch()
    else:
        r = Repo.clone_from(f"https://github.com/{repo}.git", path)
    r.git.checkout("--detach", ref or "origin/HEAD")
    return path, r.head.commit.hexsha


def index(snapshot: Path) -> pd.DataFrame:
    """One row per source function (tests and docs excluded, as in every strategy):
    function_id, path, qualname, start_line, end_line, source, hash."""
    rows = [{"function_id": function_id(rel, f.qualname), "path": rel, "qualname": f.qualname,
             "start_line": f.start_line, "end_line": f.end_line, "source": f.source, "hash": f.hash}
            for rel in source_files(snapshot) for f in parse_file((snapshot / rel).read_bytes(), rel)]
    return pd.DataFrame(rows, columns=["function_id", "path", "qualname", "start_line", "end_line", "source", "hash"])


def embed(fns: pd.DataFrame, embedder, save_every: int = SAVE_EVERY) -> np.ndarray:
    """Vectors for every function, in row order. Only bodies missing from the cache are sent."""
    texts = fns.source.tolist()
    new = sum(h not in embedder.cache for h in set(fns.hash))
    print(f"  {len(fns)} functions, {fns.hash.nunique()} distinct bodies, {new} not embedded yet", flush=True)
    t0, before = time.perf_counter(), embedder.misses
    for i in range(0, len(texts), save_every):
        embedder.encode(texts[i:i + save_every])
        if embedder.misses > before:
            embedder.save()
            done = embedder.misses - before
            print(f"  embedded {done}/{new} ({done / (time.perf_counter() - t0):.1f}/s)", flush=True)
            before, t0 = embedder.misses, time.perf_counter()
    return embedder.encode(texts)


def triage(snapshot: Path, repo: str, issue_text: str, embedder, call_llm, k: int = TOP_K) -> dict:
    """Rank where the fix for `issue_text` likely goes. Returns the agent's functions (ids that name
    real functions, with line ranges), files in that order, the dense seeds, and the run's cost."""
    fns = index(snapshot)
    scores = search(embed(fns, embedder), embedder.encode_query(issue_text))
    # An id can repeat (overloads share a qualname): keep its best rank and its first definition's lines.
    dense = list(dict.fromkeys(rank_functions(fns, scores)))
    lines = {f: (int(a), int(b)) for f, a, b in
             fns.sort_values("start_line").drop_duplicates("function_id")[["function_id", "start_line", "end_line"]].values}
    seeds = [(f, *lines[f]) for f in dense[:k]]
    try:
        run = run_agent(call_llm, snapshot, repo, issue_text, seeds=seeds, fixes=FIXES)
    except Exception as e:  # endpoint down or over quota: the dense seeds are still an answer
        run = {"ranking": [], "stop": f"llm error: {e}", "steps": [], "prompt_tokens": 0,
               "completion_tokens": 0, "seconds": 0.0}
    agent = [f for f in dict.fromkeys(run["ranking"]) if f in lines]  # drop invented ids
    ranked = agent or dense[:k]  # nothing usable submitted: fall back to the seeds
    return {"functions": [{"id": f, "start_line": lines[f][0], "end_line": lines[f][1]} for f in ranked],
            "files": list(dict.fromkeys(f.split("::", 1)[0] for f in ranked)),
            "source": "agent" if agent else "dense", "seeds": [f for f, _, _ in seeds],
            "invented": [f for f in run["ranking"] if f not in lines], "stop": run["stop"],
            "steps": len(run["steps"]), "prompt_tokens": run["prompt_tokens"],
            "completion_tokens": run["completion_tokens"], "seconds": run["seconds"]}


def report(snapshot: Path, sha: str, repo: str, issue_text: str, embedder, call_llm, k: int = 5) -> dict:
    """What both front-ends (api.py, bot.py) return: triage + an owner per suspect + a grounded
    explanation. Degrades instead of failing when the LLM is down (warnings say what's missing);
    only an embedding failure raises, since then there is no ranking at all."""
    from pinpoint.blame import owner
    from pinpoint.explain import explain

    out, warnings = triage(snapshot, repo, issue_text, embedder, call_llm), []
    if out["source"] == "dense":
        warnings.append(f"agent unavailable ({out['stop']}); showing dense retrieval only")
    suspects = [{"function_id": f["id"], "path": f["id"].split("::", 1)[0], "start_line": f["start_line"],
                 "end_line": f["end_line"],
                 "owner": owner(snapshot, f["id"].split("::", 1)[0], f["start_line"], f["end_line"], rev=sha)}
                for f in out["functions"][:k]]
    explanation, tokens = None, [out["prompt_tokens"], out["completion_tokens"]]
    try:
        ex = explain(call_llm, issue_text, snapshot, out["functions"])
        explanation = {"text": ex["text"], "citations": ex["citations"]}
        tokens = [tokens[0] + ex["prompt_tokens"], tokens[1] + ex["completion_tokens"]]
        if any(not c["valid"] for c in ex["citations"]):
            warnings.append("explanation cites lines outside the code it was shown")
    except Exception as e:
        warnings.append(f"no explanation: {e}")
    return {"repo": repo, "commit": sha, "source": out["source"], "suspects": suspects,
            "files": list(dict.fromkeys(s["path"] for s in suspects)), "explanation": explanation,
            "warnings": warnings, "prompt_tokens": tokens[0], "completion_tokens": tokens[1]}


def github_issue(repo: str, number: int) -> str:
    """Title + body of a public GitHub issue (unauthenticated REST: 60 requests/hour)."""
    with urllib.request.urlopen(f"https://api.github.com/repos/{repo}/issues/{number}", timeout=30) as r:
        issue = json.load(r)
    return f"{issue['title']}\n{issue.get('body') or ''}"


def main(argv: list[str] | None = None) -> None:
    from pinpoint.agent import databricks_llm
    from pinpoint.embed import ENDPOINT_MODEL, EndpointEmbedder

    p = argparse.ArgumentParser()
    p.add_argument("repo", help="owner/name on GitHub, or a local checkout")
    p.add_argument("--issue", type=int, help="issue number on GitHub")
    p.add_argument("--text", help="issue text, instead of --issue")
    p.add_argument("--ref", help="commit or branch (default: the remote's default branch)")
    p.add_argument("--cache-dir", default="data/silver", help="embedding cache (shared with the SWE-bench runs)")
    args = p.parse_args(argv)
    if (args.issue is None) == (args.text is None):
        p.error("give exactly one of --issue and --text")

    t0 = time.perf_counter()
    snapshot, sha = checkout(args.repo, args.ref)
    name = args.repo if args.issue or not Path(args.repo).is_dir() else Path(args.repo).resolve().name
    text = github_issue(args.repo, args.issue) if args.issue else args.text
    print(f"{name} @ {sha[:10]}: {text.splitlines()[0][:100]}", flush=True)
    embedder = EndpointEmbedder(ENDPOINT_MODEL, cache_dir=Path(args.cache_dir))
    out = triage(snapshot, name, text, embedder, databricks_llm())
    print(f"\n{out['source']} ranking ({out['stop']}, {out['steps']} steps, {out['seconds']} s LLM, "
          f"{time.perf_counter() - t0:.0f} s total):")
    for i, f in enumerate(out["functions"], 1):
        print(f"  {i:>2}. {f['id']}  (lines {f['start_line']}-{f['end_line']})")
    if out["invented"]:
        print(f"  dropped ids that name no function: {out['invented']}")


if __name__ == "__main__":
    main()
