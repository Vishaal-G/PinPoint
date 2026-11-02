# Learning plan

> Your 7-week schedule is fine for someone who already knows the tools. Since you're learning them,
> plan on **~10 weeks**. The schedule below has a working baseline by week 3, just like your original.

---

## Week 0 — Setup (a few hours)

**Learn:** `uv` (environments), git/GitHub, `pytest` basics.

- [x] Install `uv`; create a project pinned to **Python 3.12**. You have 3.14, but some libraries
      (tree-sitter grammars, PyTorch for sentence-transformers) can lag on brand-new Python versions,
      and Databricks serverless runs 3.11/3.12. Match it.
- [x] `git init`, push to a new GitHub repo, add `.gitignore` (ignore `data/`, `.venv/`, `scratch/`).
- [x] Folder layout: `src/pinpoint/`, `tests/`, `scratch/`, `data/`, `NOTES.md`.
- [x] Write one trivial function and one pytest test. Run `pytest`. See it pass, then make it fail.

**Check yourself:** Can you explain what a virtual environment is and why the test file name matters?

## Week 1 — Get the data

**Learn:** Hugging Face `datasets`, `GitPython`, `unidiff` (parsing patches), pandas → parquet.

- [x] Load `princeton-nlp/SWE-bench_Lite` (split `test`). Explore it in a notebook: what are
      `instance_id`, `repo`, `base_commit`, `patch`, `test_patch`, `problem_statement`?
- [x] Count issues per repo. Pick your small starting repo (a smaller one beats Django).
- [x] With GitPython: clone the repo once, then check out one issue's `base_commit`.
      Then learn `git worktree`, which lets you have several commits checked out without re-cloning.
- [x] Parse one gold `patch` with `unidiff`: list the files changed and line numbers of the
      **source (before-fix) side** of each hunk. Write down why it's the source side.
- [x] Save the issues table as `data/bronze/issues.parquet`.

**Milestone:** for one issue you can print the bug report, open the buggy file at the base commit,
and point at the lines the fix changed.

## Week 2 — Parsing and ground truth (the most important correctness work)

**Learn:** `tree-sitter` (py-tree-sitter + tree-sitter-python), `hashlib`, pytest fixtures.

- [x] Toy: parse a 10-line Python string, print the syntax tree, find every `function_definition`.
- [x] Extract functions and classes: qualified name (`Class.method`), file, start/end line, source text.
- [x] Extract call edges: inside each function, find `call` nodes and record the called name.
      (This will be imperfect. Name matching only, no type resolution. That's OK; note it.)
- [x] Content hash per function (`sha256` of the source text): the key for "embed only once".
- [x] Ground truth: overlap gold-patch source line ranges with function line ranges.
      Think about edge cases: a fix that only *adds* a new function, a change outside any function
      (module-level code), a file that's new.
- [x] pytest tests on a small fixture file you write by hand, where you know the right answer.
- [x] Tables: `silver/functions`, `silver/edges`, `silver/snapshot_membership`, `gold/ground_truth`.

**Check yourself:** What % of your issues have ≥1 ground-truth function? (If it's low, your
mapping has a bug or those issues should be excluded. Write down which.)

## Week 3 — Evaluation harness + BM25 baseline (first real result)

**Learn:** `rank_bm25`, `mlflow` (tracking, local UI), writing metric functions.

- [x] Write `accuracy_at_k`, `recall_at_k`, `mrr` **yourself**, with pytest tests on hand-made
      rankings. This harness is what makes the whole project credible.
- [x] Code tokenizer: split `snake_case` and `camelCase`, lowercase. Compare with/without.
- [x] Strategy 1: BM25 over whole files at each issue's base commit. Score at file level.
- [x] Log it as an MLflow run (params + metrics). Open `mlflow ui` and look at it.
- [x] Split issues into tune (200) / held-out (100) **now**, with a fixed seed, and save the split.
      Never look at held-out numbers until week 9.

**Milestone:** row 1 of your results table.

## Week 4 — Embeddings and chunking

**Learn:** `sentence-transformers` (`bge-small`), numpy cosine similarity, what embeddings are.

- [x] Toy: embed 5 sentences, compute a similarity matrix, see which pairs are close and why.
- [x] Strategy 2: fixed ~500-token chunks → embed → brute-force numpy search (no vector DB yet).
- [x] Strategy 3: one chunk per function (from silver). Embed by content hash, so duplicates are skipped.
      Record how many embeddings the hash saved; that's a resume number.
- [x] Both logged to MLflow, evaluated at file and function level.

**Check yourself:** Why does AST chunking help? Find one issue where it fixed a miss, and one where it didn't.

## Week 5 — Move to Databricks

**Learn:** Databricks Free Edition, Unity Catalog (`catalog.schema.table`), Delta Lake, PySpark,
Spark SQL, Jobs, AI Search (Delta Sync index, hybrid queries, filters).

- [x] Toy: in a notebook, make a Spark DataFrame from a pandas one, run the same query in
      PySpark and in SQL, write it as a Delta table, read it back.
- [x] Upload your parquet files → bronze/silver/gold Delta tables. Rewrite one pandas step in PySpark.
- [ ] Check the Free Edition limits yourself (compute quotas, outbound internet). If cloning
      repos is blocked or slow there, keep parsing local and upload results. That's a fine design.
- [x] Create an AI Search endpoint + Delta Sync index on gold functions. Query it from a notebook.
- [x] Strategy 4: hybrid search. Over-fetch, then filter to the issue's snapshot via membership.
- [x] Turn your pipeline into a Databricks Job.

**Milestone:** row 4, plus confirmation that the cloud setup reproduces your local row 3.

## Week 6 — LLMs, graph expansion, reranking

**Learn:** LLM APIs via Databricks Foundation Model endpoints (free; was: Anthropic Python SDK) (messages, structured output via tool use, token usage),
`CrossEncoder`.

- [x] Toy: one API call; print `usage` and compute cost by hand.
- [x] Strategy 5: query rewriting. First a regex extractor (identifiers, stack-trace frames), then an
      LLM cleanup. Measure each separately: you'll know which part actually helped.
- [x] Strategy 6: call-graph expansion. Add callers/callees of the top hits from `silver/edges`.
      Test how many neighbours to add and how to score them.
- [x] Strategy 7: cross-encoder reranking on the top ~50.

## Week 7 — Agent baseline

**Learn:** tool calling (writing the agent loop yourself, no framework), MLflow tracing.

- [x] Tools: `grep(pattern)` and `open_file(path, start, end)`, sandboxed to the snapshot dir.
- [x] Write the loop: send messages → model asks for a tool → run it → send result → repeat, with a
      step limit. Ask for a final ranked list of functions as structured output.
- [x] Strategy A on the tune set: accuracy, latency, $/query. Strategy H: seed the agent with
      Pinpoint's top-10.
- [x] Add MLflow tracing to the retrieval steps; inspect 5 failures and write down why they failed.

## Week 8 — Product: API and bot

**Learn:** FastAPI, Pydantic, GitHub Actions YAML, PyGithub, `git blame` via GitPython.

- [x] Toy: a FastAPI "hello" endpoint, then one with a Pydantic request/response model; test with
      `TestClient` in pytest.
- [x] `/triage` endpoint: best strategy + generation (explanation citing file:line) + owner from blame.
- [ ] **Design decision to make here (we'll discuss):** where the API runs so GitHub Actions can
      reach it, and how it authenticates.
- [ ] Workflow on `issues: opened` → call API → post comment. Try it on your own test repo.
- [ ] Log 👍/👎 reactions as feedback.

## Week 9 — Demo UI and final results

**Learn:** Streamlit, Databricks Apps.

- [ ] Streamlit app: repo picker, onboarding chat, "paste an issue" box.
- [ ] Deploy as a Databricks App.
- [ ] Run **held-out** numbers once for all strategies. Build the final results table.

## Week 10 — Ship

- [ ] README: problem, architecture diagram, results table, honest limitations (e.g. name-only call graph).
- [ ] 60-second demo video, LinkedIn post, resume bullets with real numbers.
- [ ] Optional: offer the bot to a small open-source project (with maintainer permission).

## Week 11 (optional, after shipping) — Port to LangChain / LangGraph

**Learn:** LangChain retrievers, `EnsembleRetriever`, LangGraph state, nodes, conditional edges, checkpointing.

- [ ] Retrieval in LangChain: wrap our embeddings + BM25 as retrievers, fuse with `EnsembleRetriever`.
      Done = reproduces our dense / BM25 / RRF rows on the same tune issues (explain any gap).
- [ ] Agent in LangGraph: LLM node, tool node, conditional edge on submit_ranking, forced submit at the
      step limit. Done = matches H2 numbers within T=0 noise.
- [ ] Write-up: what the frameworks saved, what we had to fight (function chunks, hash cache, snapshot
      filter), and which version the bot should run.

---

## Library checklist (tick when you could explain it to a friend)

| Library | Week | Done |
|---|---|---|
| uv, pytest | 0 | n/a (Claude handles) |
| datasets, GitPython, unidiff, pandas/parquet | 1 | [x] |
| tree-sitter, hashlib | 2 | [x] |
| rank_bm25, MLflow tracking | 3 | [x] |
| sentence-transformers, numpy | 4 | [ ] |
| Databricks, Unity Catalog, Delta, PySpark, Spark SQL, AI Search, Jobs | 5 | [ ] |
| Anthropic SDK, CrossEncoder | 6 | [ ] |
| Tool calling, MLflow tracing | 7 | [ ] |
| FastAPI, Pydantic, GitHub Actions, PyGithub | 8 | [ ] |
| Streamlit, Databricks Apps | 9 | [ ] |
