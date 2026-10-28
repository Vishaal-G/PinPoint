"""Week 7, Strategy A: a tool-calling agent that explores the snapshot and ranks functions.

The loop, written by hand (no framework):
    send messages + tool schemas -> the model replies with tool calls -> we run them (tools.py)
    -> append each result as a role="tool" message -> send again ... until the model calls
    submit_ranking or the step limit is hit. On the last step submit_ranking is forced, so every
    run ends with a ranked list of `path::qualname` ids (the same ids as gold/ground_truth).

The model is stateless: each step resends the whole conversation, so prompt tokens grow every step.

Runs are cached in data/gold/agent_runs.parquet (ranking, step log, transcript), keyed by
model + prompt version, so reruns cost nothing and are stable.

    uv run python -m pinpoint.agent [repo]            # Strategy A on a repo's tune issues (default pytest)
    uv run python -m pinpoint.agent [repo] --seeds    # Strategy H: seeded with Pinpoint's top-10
    uv run python -m pinpoint.agent <instance_id> [--seeds]   # one issue live, traced in MLflow

Add --fixes=labels,prompt,options (any subset) for the Week 7 failure-analysis fixes:
    labels   tools label every line with its enclosing function id (tools.py)
    prompt   advice from the failures: fix is often in the caller, the trigger isn't the bug, submit 10
    options  where each --option the issue mentions is defined, appended to the issue
"""
import hashlib
import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path

from pinpoint.tools import MAX_LINES, TOOLS, option_definitions, run_tool, tool_schemas

MODEL = "databricks-meta-llama-3-3-70b-instruct"
CACHE = Path("data/gold/agent_runs.parquet")
PRICE_PER_M = (4.0, 20.0)  # reference $ per 1M prompt/completion tokens (Databricks Free Edition is free)
MAX_STEPS = 15
TOP_K = 10

SYSTEM = """You are locating the code that must change to fix a GitHub issue in the Python repository \
{repo}. The repository is checked out at the commit the issue was reported against. Do not fix the bug.

Use the tools to explore: grep for names, messages and options from the issue, then open the files \
around the matches to read the functions (at most {max_lines} lines per open_file call). Be efficient: \
you have {max_steps} steps in total, and each step can contain several tool calls.

When you know where the fix goes, call submit_ranking with up to {top_k} functions, most likely first, \
as 'path::qualname': the path relative to the repository root, and the qualified name with classes and \
outer functions joined by dots, e.g. 'src/pkg/mod.py::MyClass.my_method' or 'src/pkg/mod.py::helper'."""

SUBMIT = {"type": "function", "function": {
    "name": "submit_ranking",
    "description": "Finish: submit the functions most likely to need the fix, most likely first.",
    "parameters": {"type": "object", "additionalProperties": False, "required": ["functions"],
                   "properties": {"functions": {"type": "array", "items": {"type": "string"},
                                                "description": f"Up to {TOP_K} ids 'path::qualname', most likely first."}}}}}
PROMPT_VERSION = hashlib.sha256(
    (SYSTEM + json.dumps(TOOLS + [SUBMIT], sort_keys=True) + f"{MAX_STEPS}/{MAX_LINES}/{TOP_K}").encode()
).hexdigest()[:12]

# Strategy H: the same prompt, plus Pinpoint's top-10 appended to the issue.
SEEDS = """

A retriever ranked these functions as the most likely places for the fix. It is often wrong, so check \
them with the tools before trusting them; you may also submit functions that are not on this list.
{seeds}"""
PROMPT_VERSION_H = hashlib.sha256((PROMPT_VERSION + SEEDS).encode()).hexdigest()[:12]


# Week 7 fixes, from the failure analysis in NOTES.md. Each one is a switch, so they can be scored one by one.
FIXES = ("labels", "prompt", "options")
PROMPT_FIX = """

Advice from past mistakes:
- Where an exception is raised is often not where the fix goes: also check the callers of that function \
and the code that produced the bad value, and rank them too.
- Code the reporter runs (e.g. os.chdir in their test) is the trigger, not the bug: look for the \
repository code that produces the wrong behaviour or output.
- Read whole functions, not a few lines, and use your step budget before deciding.
- Submit {top_k} functions or methods if you can (never classes, constants or modules), most likely first."""
OPTIONS = """

Command-line options mentioned in the issue are defined here:
{options}"""


def prompt_version(seeded: bool = False, fixes: frozenset = frozenset()) -> str:
    """Cache key for a prompt variant. No fixes -> the original A/H versions, so old runs stay valid."""
    base = PROMPT_VERSION_H if seeded else PROMPT_VERSION
    if not fixes:
        return base
    extra = json.dumps(sorted(fixes)) + PROMPT_FIX + OPTIONS + json.dumps(tool_schemas(True), sort_keys=True)
    return hashlib.sha256((base + extra).encode()).hexdigest()[:12]


def tag(seeded: bool, fixes: frozenset) -> str:
    """Run name, e.g. agentH+labels+prompt."""
    return "+".join(["agentH" if seeded else "agent", *sorted(fixes)])


def seed_text(seeds: list[tuple[str, int, int]]) -> str:
    """(function_id, start_line, end_line) rows -> the numbered list shown to the model."""
    return SEEDS.format(seeds="\n".join(f"{i}. {fid} (lines {a}-{b})" for i, (fid, a, b) in enumerate(seeds, 1)))


def normalize(fid: str) -> str:
    """Tidy an id the model wrote: './a.py::f' and 'a.py:f' -> 'a.py::f'."""
    fid = fid.strip().strip("`'\"")
    if "::" not in fid and ".py:" in fid:
        fid = fid.replace(".py:", ".py::", 1)
    return fid.removeprefix("./")


def parse_ranking(raw: str) -> list[str]:
    """submit_ranking's JSON arguments -> deduped, normalized ids (top TOP_K). Raises ValueError if invalid."""
    args = json.loads(raw)
    fns = args.get("functions") if isinstance(args, dict) else None
    if not isinstance(fns, list) or not all(isinstance(f, str) for f in fns):
        raise ValueError(f"bad submit_ranking arguments: {raw[:200]}")
    return list(dict.fromkeys(normalize(f) for f in fns))[:TOP_K]


class _NoSpan:
    """Stands in for an MLflow span when tracing is off, so the loop has one code path."""
    def set_inputs(self, *a): pass
    def set_outputs(self, *a): pass
    def set_attributes(self, *a): pass


@contextmanager
def _span(trace: bool, name: str, span_type: str, inputs=None, attributes=None):
    """An MLflow tracing span (a no-op when trace=False). The first span opened becomes the
    trace's root; spans opened inside it become its children."""
    if not trace:
        yield _NoSpan()
        return
    import mlflow

    with mlflow.start_span(name=name, span_type=span_type, attributes=attributes) as span:
        if inputs is not None:
            span.set_inputs(inputs)
        yield span


def run_agent(call_llm, snapshot: Path, repo: str, issue_text: str, max_steps: int = MAX_STEPS,
              seeds: list[tuple[str, int, int]] | None = None, trace: bool = False,
              tags: dict | None = None, fixes: frozenset = frozenset()) -> dict:
    """Run the loop on one issue. `call_llm(body) -> response dict` sends one chat request
    (injected, so tests can script the model). `seeds` (Strategy H) are appended to the issue.
    With trace=True, the run is recorded as one MLflow trace: an AGENT root span, a RETRIEVER span
    for the seeds, and one CHAT_MODEL span per step with a TOOL span per tool call under it.
    `fixes` turns on any of FIXES. Returns the ranking, the step log and the transcript."""
    labels = "labels" in fixes
    system = SYSTEM.format(repo=repo, max_lines=MAX_LINES, max_steps=max_steps, top_k=TOP_K)
    if "prompt" in fixes:
        system += PROMPT_FIX.format(top_k=TOP_K)
    user = issue_text + (seed_text(seeds) if seeds else "")
    options = option_definitions(snapshot, issue_text) if "options" in fixes else []
    if options:
        user += OPTIONS.format(options="\n".join(options))
    tools = tool_schemas(labels) + [SUBMIT]
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    steps, ranking, stop = [], [], "max_steps"
    with _span(trace, "agent", "AGENT", inputs={"repo": repo, "issue": issue_text}) as root:
        if trace and tags:
            import mlflow

            mlflow.update_current_trace(tags=tags)
        if seeds:
            # Computed before the loop (pinpoint_seeds); recorded here so the trace shows what the agent was given.
            with _span(trace, "pinpoint_seeds", "RETRIEVER", inputs={"query": issue_text}) as span:
                span.set_outputs([{"id": f, "page_content": f, "metadata": {"rank": i, "lines": f"{a}-{b}"}}
                                  for i, (f, a, b) in enumerate(seeds, 1)])
        for step in range(1, max_steps + 1):
            last = step == max_steps
            body = {"messages": messages, "tools": tools, "max_tokens": 1000, "temperature": 0,
                    # Last step: no more exploring, the model must answer.
                    "tool_choice": {"type": "function", "function": {"name": "submit_ranking"}} if last else "auto"}
            with _span(trace, f"step {step}", "CHAT_MODEL", inputs={"messages": messages, "tool_choice": body["tool_choice"]},
                       attributes={"step": step}) as span:
                t0 = time.perf_counter()
                r = call_llm(body)
                seconds = time.perf_counter() - t0
                msg = r["choices"][0]["message"]
                calls = msg.get("tool_calls") or []
                usage = r.get("usage", {})
                span.set_outputs(msg)
                span.set_attributes({"prompt_tokens": usage.get("prompt_tokens"),
                                     "completion_tokens": usage.get("completion_tokens"), "llm_seconds": round(seconds, 2)})
                steps.append({"step": step, "seconds": round(seconds, 2), "prompt_tokens": usage.get("prompt_tokens"),
                              "completion_tokens": usage.get("completion_tokens"),
                              "calls": [(c["function"]["name"], c["function"]["arguments"]) for c in calls]})
                messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls} if calls
                                else {"role": "assistant", "content": msg.get("content") or ""})
                if not calls:
                    # The model answered in prose instead of calling a tool: nudge it back on track.
                    messages.append({"role": "user", "content": "Use the tools, or call submit_ranking to finish."})
                    continue
                submitted = False
                for c in calls:
                    name, raw = c["function"]["name"], c["function"]["arguments"]
                    with _span(trace, name, "TOOL", inputs={"arguments": raw}) as tool_span:
                        if name == "submit_ranking":
                            try:
                                ranking, submitted = parse_ranking(raw), True
                                result = "ranking received"
                            except (ValueError, json.JSONDecodeError) as e:
                                result = f"error: {e}; call submit_ranking again with a list of 'path::qualname' strings"
                        else:
                            result = run_tool(snapshot, name, raw, labels)
                        tool_span.set_outputs(result)
                        tool_span.set_attributes({"chars": len(result), "error": result.startswith("error")})
                    messages.append({"role": "tool", "tool_call_id": c["id"], "content": result})
                if submitted:
                    stop = "submitted"
                    break
        out = {"ranking": ranking, "stop": stop, "steps": steps, "messages": messages,
               "prompt_tokens": sum(s["prompt_tokens"] or 0 for s in steps),
               "completion_tokens": sum(s["completion_tokens"] or 0 for s in steps),
               "seconds": round(sum(s["seconds"] for s in steps), 1)}
        root.set_outputs({"ranking": ranking, "stop": stop})
        root.set_attributes({"steps": len(steps), "prompt_tokens": out["prompt_tokens"],
                             "completion_tokens": out["completion_tokens"]})
    return out


def databricks_llm(model: str = MODEL):
    """call_llm for a Databricks Foundation Model endpoint (profile `pinpoint`)."""
    from databricks.sdk import WorkspaceClient  # local group only

    w = WorkspaceClient(profile="pinpoint")
    return lambda body: w.api_client.do("POST", f"/serving-endpoints/{model}/invocations", body=body)


def file_ranking(functions: list[str]) -> list[str]:
    """Files in the order their first function appears."""
    return list(dict.fromkeys(f.split("::")[0] for f in functions))


def agent_runs(issues, model: str = MODEL, seeds: dict | None = None, fixes: frozenset = frozenset()) -> "pd.DataFrame":
    """One cached row per issue: ranking, stop reason, steps, tokens, seconds, transcript (JSON).
    `seeds` maps instance_id -> seed rows (Strategy H, cached under its own prompt version).
    Only issues missing from the cache call the endpoint."""
    import pandas as pd

    from pinpoint.repos import get_snapshot

    stored = pd.read_parquet(CACHE) if CACHE.exists() else pd.DataFrame(columns=["instance_id", "model", "prompt_version"])
    version = prompt_version(bool(seeds), fixes)
    cache = stored[(stored.model == model) & (stored.prompt_version == version)]
    todo = issues[~issues.instance_id.isin(cache.instance_id)]
    if len(todo):
        call_llm = databricks_llm(model)
        rows = []
        for _, issue in todo.iterrows():
            r = run_agent(call_llm, get_snapshot(issue), issue["repo"], issue["problem_statement"],
                          seeds=seeds[issue["instance_id"]] if seeds else None, fixes=fixes)
            rows.append({"instance_id": issue["instance_id"], "model": model, "prompt_version": version,
                         "ranking": r["ranking"], "stop": r["stop"], "n_steps": len(r["steps"]),
                         "n_calls": sum(len(s["calls"]) for s in r["steps"]),
                         "prompt_tokens": r["prompt_tokens"], "completion_tokens": r["completion_tokens"],
                         "seconds": r["seconds"], "steps": json.dumps(r["steps"]), "messages": json.dumps(r["messages"])})
            print(f"  {issue['instance_id']}: {r['stop']} in {len(r['steps'])} steps, {r['seconds']} s -> {r['ranking'][:3]}",
                  flush=True)
            # Save after every issue: a crash halfway keeps the paid-for runs.
            stored = pd.concat([stored, pd.DataFrame(rows[-1:])], ignore_index=True)
            CACHE.parent.mkdir(parents=True, exist_ok=True)
            stored.to_parquet(CACHE, index=False)
        cache = pd.concat([cache, pd.DataFrame(rows)], ignore_index=True)
    return cache.set_index("instance_id").loc[issues.instance_id].reset_index()


def print_transcript(result: dict) -> None:
    for m in result["messages"][2:]:
        if m["role"] == "assistant":
            if m["content"]:
                print(f"\n[model] {m['content'][:500]}")
            for c in m.get("tool_calls", []):
                print(f"\n[model -> {c['function']['name']}] {c['function']['arguments']}")
        elif m["role"] == "tool":
            lines = m["content"].splitlines()
            print(f"[tool] {len(lines)} lines, {len(m['content'])} chars")
            print("\n".join("    " + l for l in lines[:8]) + ("\n    ..." if len(lines) > 8 else ""))
        else:
            print(f"\n[nudge] {m['content']}")


def pinpoint_seeds(issues, k: int = TOP_K) -> dict[str, list[tuple[str, int, int]]]:
    """Pinpoint's best function ranking (dense Strategy 3 + call-graph expansion, Strategy 6 MAIN),
    top k per issue with line ranges."""
    import pandas as pd

    from pinpoint.dense_functions import load_candidates
    from pinpoint.fusion import side_rankings
    from pinpoint.graph import expand, snapshot_graph

    candidates = load_candidates()
    candidates = candidates[candidates.instance_id.isin(issues.instance_id)]
    edges = pd.read_parquet("data/silver/edges.parquet")
    dense = side_rankings(issues, candidates)["dense_fn"]
    out = {}
    for iid, fns in candidates.groupby("instance_id"):
        top = expand(dense[iid], snapshot_graph(fns, edges), k=5, alpha=1.0, direction="both")[:k]
        lines = dict(zip(fns.function_id, zip(fns.start_line, fns.end_line)))
        out[iid] = [(f, int(lines[f][0]), int(lines[f][1])) for f in top]
    return out


def single(iid: str, seeded: bool = False, fixes: frozenset = frozenset()) -> None:
    """Run one issue live (no cache), record it as an MLflow trace, and print its transcript."""
    import mlflow
    import pandas as pd

    from pinpoint.data import load_issues
    from pinpoint.repos import get_snapshot
    from pinpoint.tracking import EXPERIMENT, TRACKING_URI

    issues = load_issues()
    issue = issues.set_index("instance_id").loc[iid]
    seeds = pinpoint_seeds(issues[issues.instance_id == iid])[iid] if seeded else None
    gt = pd.read_parquet("data/gold/ground_truth.parquet").set_index("instance_id")
    gold = list(gt.loc[iid, "gold_functions"])
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    tags = {"instance_id": iid, "strategy": tag(seeded, fixes), "model": MODEL,
            "prompt_version": prompt_version(seeded, fixes)}
    result = run_agent(databricks_llm(), get_snapshot({"instance_id": iid, **issue}), issue["repo"],
                       issue["problem_statement"], seeds=seeds, trace=True, tags=tags,
                       fixes=fixes)
    print_transcript(result)
    print(f"\nstop: {result['stop']} after {len(result['steps'])} steps, {result['seconds']} s, "
          f"{result['prompt_tokens']} prompt + {result['completion_tokens']} completion tokens")
    print("prompt tokens per step:", [s["prompt_tokens"] for s in result["steps"]])
    print("ranking:")
    for i, f in enumerate(result["ranking"], 1):
        print(f"  {i:>2}. {f}{'   <-- GOLD' if f in gold else ''}")
    print("gold:", gold)
    print(f"trace: {mlflow.get_last_active_trace_id()}  (uv run mlflow ui --backend-store-uri {TRACKING_URI} -> Traces)")


def parse_fixes(argv: list[str]) -> frozenset:
    """--fixes=labels,prompt -> frozenset({"labels", "prompt"})."""
    given = [a.split("=", 1)[1] for a in argv if a.startswith("--fixes=")]
    fixes = frozenset(f for g in given for f in g.split(",") if f)
    if fixes - set(FIXES):
        raise SystemExit(f"unknown fixes {sorted(fixes - set(FIXES))}; choose from {FIXES}")
    return fixes


if __name__ == "__main__":
    fixes = parse_fixes(sys.argv)
    if len(sys.argv) > 1 and "__" in sys.argv[1]:
        single(sys.argv[1], seeded="--seeds" in sys.argv, fixes=fixes)
        raise SystemExit

    import pandas as pd

    from pinpoint.data import load_issues
    from pinpoint.metrics import evaluate
    from pinpoint.split import tune_ids
    from pinpoint.tracking import log_run

    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    seeded = "--seeds" in sys.argv
    repo = args[0] if args else "pytest-dev/pytest"
    issues = load_issues(repo)
    issues = issues[issues.instance_id.isin(tune_ids())]  # never score held-out here
    gt = pd.read_parquet("data/gold/ground_truth.parquet")
    gold = {"file": dict(zip(gt.instance_id, gt.gold_files)), "fn": dict(zip(gt.instance_id, gt.gold_functions))}
    gold = {level: {i: g[i] for i in issues.instance_id} for level, g in gold.items()}

    seeds = pinpoint_seeds(issues) if seeded else None
    runs = agent_runs(issues, seeds=seeds, fixes=fixes)
    fn = {r.instance_id: list(r.ranking) for r in runs.itertuples()}
    rankings = {"fn": fn, "file": {i: file_ranking(f) for i, f in fn.items()}}

    # Do the submitted ids name real functions in that snapshot? (Invented names score 0.)
    members = pd.read_parquet("data/silver/snapshot_membership.parquet", columns=["instance_id", "function_id"])
    real = set(zip(members.instance_id, members.function_id))
    ids = [(i, f) for i, fs in fn.items() for f in fs]
    valid = sum(x in real for x in ids) / max(len(ids), 1)

    cost = (runs.prompt_tokens * PRICE_PER_M[0] + runs.completion_tokens * PRICE_PER_M[1]) / 1e6
    name, version = tag(seeded, fixes), prompt_version(seeded, fixes)
    print(f"\n{name}: {MODEL}, prompt {version}, n={len(runs)}")
    print(f"  stop: {runs.stop.value_counts().to_dict()}; steps mean {runs.n_steps.mean():.1f}, "
          f"tool calls mean {runs.n_calls.mean():.1f}; ranked ids {len(ids)}, real functions {valid:.0%}")
    print(f"  latency mean {runs.seconds.mean():.1f} s (max {runs.seconds.max():.1f}); tokens mean "
          f"{runs.prompt_tokens.mean():.0f} prompt + {runs.completion_tokens.mean():.0f} completion; "
          f"${cost.mean():.4f}/query at ${PRICE_PER_M[0]:g}/${PRICE_PER_M[1]:g} per 1M")
    if seeded:
        seed_ids = {i: [f for f, _, _ in s] for i, s in seeds.items()}
        from_seeds = sum(f in seed_ids[i] for i, f in ids) / max(len(ids), 1)
        r = evaluate(seed_ids, gold["fn"])
        print(f"  seeds alone (dense + graph top-{TOP_K}): fn " + " ".join(f"{m} {r[m]:.3f}" for m in ("acc@1", "acc@5", "acc@10", "mrr")))
        print(f"  submitted ids taken from the seed list: {from_seeds:.0%}")
    params = {"strategy": "agent+seeds" if seeded else "agent", "model": MODEL, "prompt_version": version,
              "max_steps": MAX_STEPS, "fixes": ",".join(sorted(fixes)) or "none",
              "slice": f"{repo} tune"}
    for level in ("file", "fn"):
        r = evaluate(rankings[level], gold[level])
        print(f"  {level:>4}: n={r['n']} " + " ".join(f"{m} {r[m]:.3f}" for m in ("acc@1", "acc@5", "acc@10", "mrr")))
        log_run(f"{name}-{level}", {**params, "level": "function" if level == "fn" else "file"},
                rankings[level], gold[level])
    import mlflow

    run_stats = {"latency_s": runs.seconds.mean(), "prompt_tokens": runs.prompt_tokens.mean(),
                 "completion_tokens": runs.completion_tokens.mean(), "usd_per_query": cost.mean(),
                 "steps": runs.n_steps.mean(), "valid_id_rate": valid}
    with mlflow.start_run(run_name=f"{name}-cost"):
        mlflow.log_params(params)
        mlflow.log_metrics(run_stats)
    print("\nper issue (gold function rank, '-' = not ranked):")
    for r in runs.itertuples():
        g = set(gold["fn"][r.instance_id])
        rank = next((k for k, f in enumerate(r.ranking, 1) if f in g), "-")
        if seeded:
            rank = f"{next((k for k, f in enumerate(seed_ids[r.instance_id], 1) if f in g), '-')}->{rank}"
        frank = next((k for k, f in enumerate(rankings["file"][r.instance_id], 1) if f in set(gold["file"][r.instance_id])), "-")
        print(f"  {r.instance_id.split('-')[-1]:>5} fn {rank!s:>4} file {frank!s:>2} {r.stop:>9} {int(r.n_steps):>2} steps "
              f"top1 {r.ranking[0] if len(r.ranking) else None}  gold {sorted(g)}")
