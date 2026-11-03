"""The HTTP API: POST /triage with a repo and an issue -> suspects, owners, and an explanation.

    uv run uvicorn pinpoint.api:app --port 8000        # docs at http://127.0.0.1:8000/docs

The pipeline is live.report: live.triage (checkout -> parse -> embed -> dense seeds -> H3 agent),
then explain.explain (grounded generation) and blame.owner per suspect; bot.py runs the same core.
The embedder and LLM come in through FastAPI dependencies, so tests swap in fakes with
app.dependency_overrides.
Requests are synchronous: a repo whose functions aren't embedded yet takes minutes (see NOTES.md).
"""
import time
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from pinpoint import live
from pinpoint.providers import get_embedder, get_llm, llm_label, repo_lock  # shared with app.py

app = FastAPI(title="Pinpoint", description="Points to the functions most likely responsible for an issue.")


class TriageRequest(BaseModel):
    repo: Annotated[str, Field(pattern=live.REPO_PATTERN, examples=["pytest-dev/pytest"])]
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


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.post("/triage", response_model=TriageResponse)
def triage(req: TriageRequest, embedder=Depends(get_embedder), call_llm=Depends(get_llm)) -> dict:
    # A plain `def` (not async): FastAPI runs it in a worker thread, so blocking git / HTTP calls
    # don't stall the server's event loop.
    from git import GitCommandError

    t0 = time.perf_counter()
    text = f"{req.title}\n{req.body}".strip()
    with repo_lock(req.repo):
        try:
            snapshot, sha = live.checkout(req.repo, req.ref)
        except GitCommandError as e:
            raise HTTPException(404, f"cannot check out {req.repo} at {req.ref or 'default branch'}") from e
        try:
            out = live.report(snapshot, sha, req.repo, text, embedder, call_llm, k=req.k, llm_label=llm_label())
        except Exception as e:  # embedding endpoint down: no ranking at all
            raise HTTPException(503, f"embedding failed: {e}") from e
    return {**out, "seconds": round(time.perf_counter() - t0, 1)}
