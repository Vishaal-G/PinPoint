# PinPoint

A GitHub bot that reads a new issue and points to the functions most likely responsible.
Benchmarked on SWE-bench Lite.

When an issue is opened, a GitHub Actions job parses the repo at its current commit, embeds every
function, ranks them against the issue, and comments a short table: function, file, line range (as
permalinks), and who wrote most of those lines. With an LLM configured, an agent then reads the top
candidates (grep, open files, follow calls) and re-ranks them, and the comment gets a "Why" paragraph
whose `path:line` citations are checked against the code it was shown.

## Install on your repo

Copy [`bot/pinpoint.yml`](bot/pinpoint.yml) to `.github/workflows/pinpoint.yml`. That's it: as shipped it
needs **no secrets** (it runs on the job's own `GITHUB_TOKEN`). The first run embeds every function
(a few minutes on a mid-size repo); later runs reuse the Actions cache and only embed changed code.

Pick how much ranking you want in the Triage step's `env`:

| Setup | Needs | What you get |
|---|---|---|
| `PINPOINT_LLM: none` (default) | nothing | embedding ranking only, no explanation |
| `PINPOINT_LLM: openai` + `PINPOINT_LLM_URL`, `PINPOINT_LLM_MODEL`, secret `PINPOINT_LLM_KEY` | any OpenAI-compatible chat API with tool calling | agent ranking + explanation |
| Databricks: `DATABRICKS_HOST`, `DATABRICKS_TOKEN` secrets, no `PINPOINT_LLM` | a Databricks workspace | agent on Llama 3.3 70B: the setup the numbers below were measured on |

Rate the comment with 👍 / 👎. [`bot/pinpoint-feedback.yml`](bot/pinpoint-feedback.yml) collects the
reactions weekly into a CSV.

## How well it works

Measured on SWE-bench Lite (a fixed 200-issue tune split; the 100 held-out issues are kept for the final
score). "Function MRR" = mean of 1/rank of the function the real fix changed.

| Ranking | Function MRR | Function acc@1 | File MRR | Issues |
|---|---|---|---|---|
| Embeddings only (bge-large) | .295 | .207 | .567 | 184 / 200 tune |
| Agent + embedding seeds + call-graph tool (Llama 3.3 70B) | .614 | .529 | .804 | 52 (4 per repo, 12 repos) |

The no-secrets default embeds with bge-small on the runner, which is weaker than bge-large; other LLMs
behind `PINPOINT_LLM=openai` are unmeasured. The agent costs about $0.11 per issue at Databricks list
prices (free on Databricks Free Edition, within its limits).

## Run it locally

```bash
uv sync
uv run python -m pinpoint.live pytest-dev/pytest --issue 12345      # any public repo + issue
uv run streamlit run src/pinpoint/app.py                            # paste an issue in a page
uv run uvicorn pinpoint.api:app                                     # POST /triage
uv run pytest
```

The same `PINPOINT_EMBEDDER=local` and `PINPOINT_LLM=...` switches work everywhere.
