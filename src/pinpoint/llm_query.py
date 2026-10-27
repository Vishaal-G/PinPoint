"""Strategy 5b: an LLM rewrites the issue into a search query (structured output via tool calling).

The model fills a fixed JSON schema through a forced "tool" call, so the answer is data, not prose:
a one-line summary in code terms, plus guessed function names, file paths and other identifiers.
Unlike the regex extractor (5a), it can infer names that the issue never mentions.

Runs on Databricks Foundation Model endpoints (free on Free Edition). Answers are cached in
data/gold/llm_queries.parquet, keyed by model + prompt version, so reruns cost nothing and are stable.

    uv run python -m pinpoint.llm_query            # pytest-dev/pytest tune issues
"""
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

MODEL = "databricks-meta-llama-3-3-70b-instruct"   # gpt-oss-120b ignores forced tool calls
CACHE = Path("data/gold/llm_queries.parquet")

SYSTEM = """You help search the source code of the Python repository {repo} for the code that must \
change to fix a GitHub issue. Do not fix the bug. Guess where the fix goes, using names exactly as \
they would appear in the repository's source (function, method, class and file names), not the \
reporter's own code. Call submit_query once."""

TOOL = {"type": "function", "function": {
    "name": "submit_query",
    "description": "Submit a search query for finding the code to change.",
    "parameters": {"type": "object", "additionalProperties": False, "required": ["summary", "functions", "files", "terms"],
                   "properties": {
                       "summary": {"type": "string", "description": "One sentence: what the code does wrong, in code terms."},
                       "functions": {"type": "array", "items": {"type": "string"},
                                     "description": "Up to 5 function or method names in the repository likely to need the fix."},
                       "files": {"type": "array", "items": {"type": "string"},
                                 "description": "Up to 3 repository file paths likely to contain the fix."},
                       "terms": {"type": "array", "items": {"type": "string"},
                                 "description": "Up to 10 other identifiers, options or config names from the repository."},
                   }}}}
FIELDS = ("summary", "functions", "files", "terms")
PROMPT_VERSION = hashlib.sha256((SYSTEM + json.dumps(TOOL, sort_keys=True)).encode()).hexdigest()[:12]


def parse_arguments(raw: str) -> dict:
    """The tool call's JSON arguments, checked against the schema. Raises ValueError if invalid."""
    args = json.loads(raw)
    if not isinstance(args, dict) or not isinstance(args.get("summary"), str):
        raise ValueError(f"bad tool arguments: {raw[:200]}")
    for key in ("functions", "files", "terms"):
        if not isinstance(args.get(key, []), list) or not all(isinstance(x, str) for x in args.get(key, [])):
            raise ValueError(f"bad {key}: {raw[:200]}")
    return {"summary": args["summary"], **{k: args.get(k, []) for k in ("functions", "files", "terms")}}


def ask(w, repo: str, issue_text: str, model: str = MODEL, attempts: int = 2) -> tuple[dict, dict]:
    """(parsed arguments, usage) for one issue. Retries once on an invalid answer."""
    body = {"messages": [{"role": "system", "content": SYSTEM.format(repo=repo)},
                         {"role": "user", "content": issue_text}],
            "tools": [TOOL], "tool_choice": {"type": "function", "function": {"name": "submit_query"}},
            "max_tokens": 1000, "temperature": 0}
    for attempt in range(attempts):
        r = w.api_client.do("POST", f"/serving-endpoints/{model}/invocations", body=body)
        calls = r["choices"][0]["message"].get("tool_calls") or []
        try:
            if not calls:
                raise ValueError("no tool call")
            return parse_arguments(calls[0]["function"]["arguments"]), r.get("usage", {})
        except (ValueError, json.JSONDecodeError) as e:
            if attempt == attempts - 1:
                raise ValueError(f"no valid answer after {attempts} attempts: {e}") from e


def llm_queries(issues: pd.DataFrame, model: str = MODEL) -> pd.DataFrame:
    """One row per issue: instance_id, model, prompt_version, the four fields, token usage.
    Cached rows are reused; only missing issues call the endpoint."""
    stored = pd.read_parquet(CACHE) if CACHE.exists() else pd.DataFrame(columns=["instance_id", "model", "prompt_version"])
    cache = stored[(stored.model == model) & (stored.prompt_version == PROMPT_VERSION)]
    todo = issues[~issues.instance_id.isin(cache.instance_id)]
    if len(todo):
        from databricks.sdk import WorkspaceClient  # local group only

        w = WorkspaceClient(profile="pinpoint")
        rows = []
        for _, issue in todo.iterrows():
            args, usage = ask(w, issue["repo"], issue["problem_statement"], model)
            rows.append({"instance_id": issue["instance_id"], "model": model, "prompt_version": PROMPT_VERSION,
                         **args, "prompt_tokens": usage.get("prompt_tokens"),
                         "completion_tokens": usage.get("completion_tokens")})
            print(f"  {issue['instance_id']}: {args['functions']}", flush=True)
        new = pd.DataFrame(rows)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        pd.concat([stored, new], ignore_index=True).to_parquet(CACHE, index=False)
        cache = pd.concat([cache, new], ignore_index=True)
    return cache.set_index("instance_id").loc[issues.instance_id].reset_index()


def rewrite(text: str, answer: dict, mode: str) -> str:
    """Query text from the issue and the model's answer.

    llm        summary + guessed names + terms (replaces the issue)
    raw+llm    the issue, then the guessed names and terms (for BM25: the guesses count twice)
    title+llm  issue title + summary + names: short and natural, for a 512-token embedder
    names      guessed function names and files only
    """
    from pinpoint.query import title

    names = " ".join(list(answer["functions"]) + list(answer["files"]))
    terms = " ".join(answer["terms"])
    return {"llm": f"{answer['summary']}\n{names}\n{terms}",
            "raw+llm": f"{text}\n{names}\n{terms}",
            "title+llm": f"{title(text)}\n{answer['summary']}\n{names}",
            "names": names or text}[mode]


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
    gold = {"file": dict(zip(gt.instance_id, gt.gold_files)), "fn": dict(zip(gt.instance_id, gt.gold_functions))}
    gold = {level: {i: g[i] for i in issues.instance_id} for level, g in gold.items()}

    answers = llm_queries(issues)
    print(f"\n{MODEL}: {answers.prompt_tokens.sum()} prompt + {answers.completion_tokens.sum()} completion tokens "
          f"for {len(answers)} issues")
    # How often did the model name the gold function outright? (It may have seen pytest in training.)
    for a in answers.itertuples():
        g = [f.split("::")[1] for f in gold["fn"][a.instance_id]]
        hit = any(name.split(".")[-1] == x.split(".")[-1] for name in a.functions for x in g)
        print(f"  {a.instance_id.split('-')[-1]:>5} {'NAMED' if hit else '     '} guessed {list(a.functions)[:4]}  gold {g}")

    candidates = load_candidates()
    by_id = answers.set_index("instance_id")
    rows = []
    for mode in ("llm", "raw+llm", "title+llm", "names"):
        queries = {i: rewrite(t, by_id.loc[i], mode) for i, t in zip(issues.instance_id, issues.problem_statement)}
        sides = side_rankings(issues, candidates, queries)
        for name, level in (("bm25_file", "file"), ("dense_file", "file"), ("bm25_fn", "fn"), ("dense_fn", "fn")):
            r = evaluate(sides[name], gold[level])
            rows.append({"query": mode, "retriever": name, **{m: round(r[m], 3) for m in ("acc@1", "acc@5", "acc@10", "mrr")}})
            retriever, lvl = name.split("_")
            log_run(f"llmquery-{mode}-{name}", {"strategy": f"query-rewrite-llm+{retriever}", "query": mode, "model": MODEL,
                                                "prompt_version": PROMPT_VERSION, "level": "function" if lvl == "fn" else "file",
                                                "slice": f"{repo} tune"}, sides[name], gold[level])
    print(pd.DataFrame(rows).sort_values(["retriever", "query"]).to_string(index=False))
