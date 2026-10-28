"""Week 7: the tools the agent can call, sandboxed to one issue's snapshot directory.

Each tool takes plain arguments and returns a string, because whatever it returns is pasted back
into the conversation as the tool's result. Errors are returned as text too (not raised), so the
model can read what went wrong and try again instead of the loop crashing.

    grep(pattern)                 regex over the snapshot's source files -> "path:line: text" rows
    open_file(path, start, end)   numbered lines of one file, at most MAX_LINES per call
    call_graph(function_id)       callers and callees of one function (graph=True, scale-up fix 4)

With labels=True (Week 7 fix 1) every result also names the function each line sits in, as the
`path::qualname` id the agent must submit; without it the agent sees lines but not function names.
"""
import json
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

from pinpoint.ground_truth import function_id
from pinpoint.parse import parse_file
from pinpoint.repos import source_files

MAX_MATCHES = 50     # grep rows per call; the rest are counted, not shown
MAX_LINE_CHARS = 200
MAX_LINES = 200      # open_file lines per call

# OpenAI-style tool schemas, the format Databricks endpoints accept (same as llm_query.TOOL).
GREP = {"type": "function", "function": {
    "name": "grep",
    "description": "Search the repository's Python source files (tests and docs excluded) with a Python "
                   f"regular expression. Returns up to {MAX_MATCHES} matches as 'path:line: text'.",
    "parameters": {"type": "object", "additionalProperties": False, "required": ["pattern"],
                   "properties": {"pattern": {"type": "string", "description": "Python regex, e.g. 'def assertrepr_compare'."}}}}}
OPEN_FILE = {"type": "function", "function": {
    "name": "open_file",
    "description": f"Show numbered lines start..end (1-based, inclusive) of one file, at most {MAX_LINES} lines.",
    "parameters": {"type": "object", "additionalProperties": False, "required": ["path", "start", "end"],
                   "properties": {"path": {"type": "string", "description": "Path relative to the repository root."},
                                  "start": {"type": "integer"}, "end": {"type": "integer"}}}}}
TOOLS = [GREP, OPEN_FILE]
CALL_GRAPH = {"type": "function", "function": {
    "name": "call_graph",
    "description": "List the functions that call a function and the functions it calls, matched by name "
                   "(like Pinpoint's call graph): use it to follow a bug from where it shows up to its cause. "
                   f"At most {MAX_MATCHES} callers.",
    "parameters": {"type": "object", "additionalProperties": False, "required": ["function_id"],
                   "properties": {"function_id": {"type": "string",
                                                  "description": "Id 'path::qualname', e.g. 'src/pkg/mod.py::MyClass.run'."}}}}}
LABEL_NOTE = (" Each line is labelled with the id 'path::qualname' of the innermost function containing it; "
              "use these ids when you submit.")


def tool_schemas(labels: bool = False, graph: bool = False) -> list[dict]:
    """The tool schemas, with descriptions that mention the function labels when they are on."""
    tools = TOOLS + [CALL_GRAPH] if graph else TOOLS
    if not labels:
        return tools
    out = json.loads(json.dumps(tools))
    for t in out[:len(TOOLS)]:  # call_graph always speaks in ids, so it needs no note
        t["function"]["description"] += LABEL_NOTE
    return out


@lru_cache(maxsize=4096)
def _spans(file: str, rel: str) -> tuple[tuple[int, int, str], ...]:
    """(start, end, qualname) of every function in a file. Snapshots never change, so cache by path."""
    return tuple((f.start_line, f.end_line, f.qualname) for f in parse_file(Path(file).read_bytes(), rel))


def enclosing(snapshot: Path, rel: str, line: int) -> tuple[str, int, int] | None:
    """(function id, start, end) of the innermost function containing `line`, or None (module level).
    Same rule as ground truth: nested functions start last."""
    if not rel.endswith(".py"):
        return None
    containing = [sp for sp in _spans(str(snapshot.resolve() / rel), rel) if sp[0] <= line <= sp[1]]
    if not containing:
        return None
    start, end, qualname = max(containing)
    return function_id(rel, qualname), start, end


@lru_cache(maxsize=2)
def _graph(root: str) -> tuple[dict, dict, dict]:
    """One snapshot's name-only call graph, parsed on first use (a django snapshot takes seconds):
    id -> (start, end, calls), short name -> ids defined with it, short name -> ids that call it."""
    by_id, defined, callers = {}, defaultdict(list), defaultdict(list)
    for rel in source_files(Path(root)):
        for f in parse_file((Path(root) / rel).read_bytes(), rel):
            fid = function_id(rel, f.qualname)
            by_id[fid] = (f.start_line, f.end_line, f.calls)
            defined[f.qualname.split(".")[-1]].append(fid)
            for name in set(f.calls):
                callers[name].append(fid)
    return by_id, defined, callers


def call_graph(snapshot: Path, fid: str) -> str:
    by_id, defined, callers = _graph(str(snapshot.resolve()))
    if fid not in by_id:
        return f"error: no function {fid!r}; ids look like 'path::qualname' (grep and open_file show them)"
    start, end, calls = by_id[fid]
    name = fid.split("::", 1)[1].split(".")[-1]
    same = len(defined[name])
    rows = [f"{fid} (lines {start}-{end})"]
    by = sorted(set(callers[name]) - {fid})
    note = f"; {same} functions are named {name!r}, so some of these may call another one" if same > 1 else ""
    rows.append(f"called by {len(by)} functions (calls to {name!r}, matched by name{note}):")
    rows += [f"  {c} (lines {by_id[c][0]}-{by_id[c][1]})" for c in by[:MAX_MATCHES]]
    if len(by) > MAX_MATCHES:
        rows.append(f"  ... {len(by) - MAX_MATCHES} more not shown")
    rows.append("calls:")
    outside = 0
    for callee in sorted(set(calls)):
        targets = [t for t in defined.get(callee, []) if t != fid]
        if not targets:
            outside += 1
        elif len(targets) == 1:
            rows.append(f"  {callee} -> {targets[0]} (lines {by_id[targets[0]][0]}-{by_id[targets[0]][1]})")
        else:
            shown = ", ".join(targets[:3]) + (", ..." if len(targets) > 3 else "")
            rows.append(f"  {callee} -> one of {len(targets)} functions with that name: {shown}")
    if outside:
        rows.append(f"  ({outside} more names not defined in the repository, e.g. builtins or libraries)")
    return "\n".join(rows)


def _inside(snapshot: Path, path: str) -> Path | None:
    """The file `path` names, or None if it resolves outside the snapshot (../, absolute, symlink)."""
    root = snapshot.resolve()
    target = (root / path).resolve()
    return target if target.is_relative_to(root) else None


def grep(snapshot: Path, pattern: str, labels: bool = False) -> str:
    try:
        rx = re.compile(pattern)
    except re.error as e:
        return f"error: invalid regex {pattern!r}: {e}"
    rows, total = [], 0
    for rel in source_files(snapshot):
        text = (snapshot / rel).read_text(encoding="utf-8", errors="replace")
        for no, line in enumerate(text.splitlines(), 1):
            if rx.search(line):
                total += 1
                if len(rows) < MAX_MATCHES:
                    where = ""
                    if labels:
                        fn = enclosing(snapshot, rel, no)
                        where = f" [{fn[0] if fn else 'module level'}]"
                    rows.append(f"{rel}:{no}{where}: {line.strip()[:MAX_LINE_CHARS]}")
    if not rows:
        return f"no matches for {pattern!r}"
    if total > len(rows):
        rows.append(f"... {total - len(rows)} more matches not shown; use a more specific pattern")
    return "\n".join(rows)


def open_file(snapshot: Path, path: str, start: int, end: int, labels: bool = False) -> str:
    target = _inside(snapshot, path)
    if target is None:
        return f"error: {path!r} is outside the repository"
    if not target.is_file():
        return f"error: no such file {path!r}"
    lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
    start = max(1, start)
    end = min(end, len(lines), start + MAX_LINES - 1)
    if start > len(lines):
        return f"error: {path} has only {len(lines)} lines"
    rel = target.relative_to(snapshot.resolve()).as_posix()
    shown, current = [], "unset"
    for no in range(start, end + 1):
        if labels:
            # A header line whenever the innermost function changes, instead of a label on every line.
            fn = enclosing(snapshot, rel, no)
            if fn != current:
                shown.append(f"      ## {fn[0]} (lines {fn[1]}-{fn[2]})" if fn else "      ## module level")
                current = fn
        shown.append(f"{no:>5}| {lines[no - 1]}")
    return f"{path} lines {start}-{end} of {len(lines)}\n" + "\n".join(shown)


def run_tool(snapshot: Path, name: str, raw_args: str, labels: bool = False) -> str:
    """Run one tool call from the model (name + JSON argument string); always returns text."""
    try:
        args = json.loads(raw_args or "{}")
        if name == "grep":
            return grep(snapshot, str(args["pattern"]), labels)
        if name == "open_file":
            return open_file(snapshot, str(args["path"]), int(args["start"]), int(args["end"]), labels)
        if name == "call_graph":
            return call_graph(snapshot, str(args["function_id"]))
        return f"error: unknown tool {name!r}; available: grep, open_file, call_graph"
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        return f"error: bad arguments for {name}: {e!r}"


OPTION = re.compile(r"(?<![\w-])--[a-zA-Z][\w-]+")


def option_definitions(snapshot: Path, issue_text: str, limit: int = 10) -> list[str]:
    """For each command-line option the issue mentions (e.g. --fixtures), where the repository
    defines it: the first source line with the option as a string literal (Week 7 fix 3)."""
    wanted = list(dict.fromkeys(OPTION.findall(issue_text)))[:limit]
    found = {}
    for rel in source_files(snapshot):
        if len(found) == len(wanted):
            break
        text = (snapshot / rel).read_text(encoding="utf-8", errors="replace")
        for opt in wanted:
            if opt in found:
                continue
            m = re.search(rf"[\"']{re.escape(opt)}[\"']", text)
            if m:
                no = text.count("\n", 0, m.start()) + 1
                fn = enclosing(snapshot, rel, no)
                found[opt] = f"{opt}: {rel}:{no}" + (f" [{fn[0]}]" if fn else "")
    return [found[o] for o in wanted if o in found]
