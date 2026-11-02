"""Generation: a short explanation of where the fix likely goes, grounded in the suspects' code.

Retrieval (dense + agent) picks the functions; this shows the model their numbered source and asks
for 2-4 sentences citing `path:line`. Every citation is then checked against the excerpts the
model was actually shown: a cited line outside them is flagged invalid (a faithfulness check that
needs no labels).
"""
import re
from pathlib import Path

from pinpoint.tools import open_file

N_SUSPECTS = 3        # functions shown to the model
MAX_FN_LINES = 80     # lines per function excerpt (long functions are cut)

SYSTEM = (
    "You help maintainers triage GitHub issues. You are given an issue and excerpts of the functions "
    "most likely responsible, with line numbers. In 2-4 sentences, say where the fix most likely goes "
    "and why. Use only the code shown. Cite code as path:line (e.g. src/pkg/mod.py:42), using the line "
    "numbers from the excerpts. If the excerpts do not explain the issue, say so instead of guessing."
)
CITATION = re.compile(r"([\w./-]+\.py):(\d+)")


def excerpts(snapshot: Path, suspects: list[dict], n: int = N_SUSPECTS) -> list[tuple[str, int, int, str]]:
    """(path, first line, last line, numbered text) for the top-n suspects."""
    out = []
    for s in suspects[:n]:
        path = s["id"].split("::", 1)[0]
        start, end = s["start_line"], min(s["end_line"], s["start_line"] + MAX_FN_LINES - 1)
        out.append((path, start, end, open_file(snapshot, path, start, end)))
    return out


def check_citations(text: str, shown: list[tuple[str, int, int, str]]) -> list[dict]:
    """Every path:line in `text`, with valid = the line lies inside an excerpt of that file."""
    return [{"path": p, "line": int(ln),
             "valid": any(p == path and a <= int(ln) <= b for path, a, b, _ in shown)}
            for p, ln in dict.fromkeys(CITATION.findall(text))]


def explain(call_llm, issue_text: str, snapshot: Path, suspects: list[dict]) -> dict:
    """One chat call. Returns text, citations (each with valid), and token usage."""
    shown = excerpts(snapshot, suspects)
    context = "\n\n".join(text for *_, text in shown)
    body = {"messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": f"Issue:\n{issue_text}\n\nCode:\n{context}"}],
            "max_tokens": 400, "temperature": 0}
    r = call_llm(body)
    text = (r["choices"][0]["message"].get("content") or "").strip()
    usage = r.get("usage") or {}
    return {"text": text, "citations": check_citations(text, shown),
            "prompt_tokens": usage.get("prompt_tokens") or 0, "completion_tokens": usage.get("completion_tokens") or 0}
