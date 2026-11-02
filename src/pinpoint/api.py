"""The HTTP API: POST /triage with a repo and an issue -> suspects, owners, and an explanation.

    uv run uvicorn pinpoint.api:app --port 8000        # docs at http://127.0.0.1:8000/docs

The pipeline is live.triage (checkout -> parse -> embed -> dense seeds -> H3 agent), then
explain.explain (grounded generation) and blame.owner per suspect. The embedder and LLM come in
through FastAPI dependencies, so tests swap in fakes with app.dependency_overrides.
Requests are synchronous: a repo whose functions aren't embedded yet takes minutes (see NOTES.md).
"""
import threading
import time
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from pinpoint import live
from pinpoint.blame import owner
from pinpoint.explain import explain

app = FastAPI(title="Pinpoint", description="Points to the functions most likely responsible for an issue.")


class TriageRequest(BaseModel):
    repo: Annotated[str, Field(pattern=r"^[\w.-]+/[\w.-]+$", examples=["pytest-dev/pytest"])]
    ref: str | None = Field(None, description="commit or branch; default: the default branch head")
    title: Annotated[str, Field(min_length=1)]
    body: str = ""
    k: Annotated[int, Field(ge=1, le=10)] = 5


class Owner(BaseModel):
    name: str
    email: str
    lines: int
    share: float  # of the function's (non-bot) lines


class Suspect(BaseModel):
    function_id: str  # path::qualname
    path: str
    start_line: int
    end_line: int
    owner: Owner | None


class Citation(BaseModel):
    path: str
    line: int
    valid: bool  # the line lies inside code the model was shown


class Explanation(BaseModel):
    text: str
    citations: list[Citation]


class TriageResponse(BaseModel):
    repo: str
    commit: str
    source: str  # "agent", or "dense" when the agent gave nothing usable / the LLM failed
    suspects: list[Suspect]
    files: list[str]
    explanation: Explanation | None
    warnings: list[str]
    seconds: float
    prompt_tokens: int
    completion_tokens: int


@lru_cache(maxsize=1)
def get_embedder():
    from pinpoint.embed import ENDPOINT_MODEL, EndpointEmbedder

    return EndpointEmbedder(ENDPOINT_MODEL, cache_dir=Path("data/silver"))


@lru_cache(maxsize=1)
def get_llm():
    from pinpoint.agent import databricks_llm

    return databricks_llm()


# One request per repo at a time: they share one checkout, and `git checkout` would move it underneath
# a running request.
_repo_locks: defaultdict[str, threading.Lock] = defaultdict(threading.Lock)


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.post("/triage", response_model=TriageResponse)
def triage(req: TriageRequest, embedder=Depends(get_embedder), call_llm=Depends(get_llm)) -> dict:
    # A plain `def` (not async): FastAPI runs it in a worker thread, so blocking git / HTTP calls
    # don't stall the server's event loop.
    from git import GitCommandError

    t0, warnings = time.perf_counter(), []
    text = f"{req.title}\n{req.body}".strip()
    with _repo_locks[req.repo]:
        try:
            snapshot, sha = live.checkout(req.repo, req.ref)
        except GitCommandError as e:
            raise HTTPException(404, f"cannot check out {req.repo} at {req.ref or 'default branch'}") from e
        try:
            out = live.triage(snapshot, req.repo, text, embedder, call_llm)
        except Exception as e:  # embedding endpoint down: no ranking at all
            raise HTTPException(503, f"embedding failed: {e}") from e
        if out["source"] == "dense":
            warnings.append(f"agent unavailable ({out['stop']}); showing dense retrieval only")
        top = out["functions"][:req.k]
        suspects = [{"function_id": f["id"], "path": f["id"].split("::", 1)[0], "start_line": f["start_line"],
                     "end_line": f["end_line"],
                     "owner": owner(snapshot, f["id"].split("::", 1)[0], f["start_line"], f["end_line"], rev=sha)}
                    for f in top]
        explanation, tokens = None, (out["prompt_tokens"], out["completion_tokens"])
        try:
            ex = explain(call_llm, text, snapshot, out["functions"])
            explanation = {"text": ex["text"], "citations": ex["citations"]}
            tokens = (tokens[0] + ex["prompt_tokens"], tokens[1] + ex["completion_tokens"])
            if any(not c["valid"] for c in ex["citations"]):
                warnings.append("explanation cites lines outside the code it was shown")
        except Exception as e:
            warnings.append(f"no explanation: {e}")
    return {"repo": req.repo, "commit": sha, "source": out["source"], "suspects": suspects,
            "files": list(dict.fromkeys(s["path"] for s in suspects)), "explanation": explanation,
            "warnings": warnings, "seconds": round(time.perf_counter() - t0, 1),
            "prompt_tokens": tokens[0], "completion_tokens": tokens[1]}
