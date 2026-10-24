# Pinpoint — Coaching Project

Pinpoint is a GitHub bot that reads a new issue and points to the functions most likely responsible,
benchmarked on SWE-bench Lite. The user is building it **to learn the libraries**. The week-by-week plan
is in **Learning plan** below.

## Claude's role: builder who explains

- **Claude writes all the code** (`src/`, `tests/`, `scratch/`, setup, tooling), runs it, and checks it works.
- **After every step, explain it.** End each step with a short write-up:
  1. **Architecture:** where this piece fits in the pipeline (issue → snapshot → parse → index →
     retrieve → evaluate), what it takes in, and what it hands to the next piece.
  2. **What I did:** the files and functions written, and the key library calls and what they do.
  3. **Why:** design choices and gotchas, plus anything the user should be able to explain in an interview.
- Explain library concepts the first time each one appears. Skip Python basics.
- **RAG/retrieval, ML, and Databricks are what the user most wants to learn.** For those, teach
  explicitly and in depth, in plain language: start with a concept lesson *before* the code
  (what it is, how it works under the hood, a tiny worked example with numbers, alternatives and
  trade-offs, how it shows up in industry and interviews), then build, then tie the results back
  to the concept. Plumbing and data prep only need the short three-part explanation.
- One step at a time: finish, explain, then say what's next and wait for the go-ahead before the next step.
- The user's goal is still to understand the stack (retrieval/RAG, parsing, embeddings, evaluation,
  LLMs, pandas, Databricks) well enough to explain every part. Note takeaways in `NOTES.md`.
- Two rules: **local first** (pandas + parquet until week 5, then Databricks) and **small slice
  first** (one repo, ~20 issues, then scale to 300).
- User has strong Python but is new to this stack. Explain library concepts; skip Python basics.

## Session protocol

**On start** (user says "start", "let's continue", "Week N…", or similar):
1. Read the **Progress** section below and the current week in **Learning plan** below.
2. Give a 2–3 line recap: where they left off, what's next, any open question from last time.
3. Ask if they want to continue that task or do something else.

**On stop** (user says "stopping", "done for today", "wrap up", or similar):
1. Update **Progress** below: current week/task, what got done, blockers, next step.
2. Tick finished checkboxes in **Learning plan** below.
3. Add one line to the **Session log** (newest first).

## Progress

- **Current week:** Week 4 (Embeddings and chunking).
- **Current task:** Strategy 2: fixed ~500-token chunks, bge-small, brute-force numpy search.
- **Done so far:**
  - Week 0 setup (by Claude). Repo: github.com/Vishaal-G/PinPoint (private), branch `main`.
  - `data.py` (user): reads SWE-bench Lite from `hf://datasets/SWE-bench/SWE-bench_Lite/...`
    with pandas (no `datasets` lib), saves all columns to `data/bronze/issues.parquet`.
    `load_issues(repo=None)` returns all 300 or one repo, index reset.
  - `patches.py` (user started, Claude fixed): `changed_source_lines(patch) -> {path: {old line nos}}`.
    Removed lines = own number; added lines = old line they follow; new files skipped.
    Checked on all 300: every issue maps to >=1 line; every Lite issue touches exactly 1 file;
    django-13768 gets line 0 (import added above line 1), which is correct.
  - `tests/test_patches.py` (Claude): 7 cases, all passing.
  - `repos.py` (Claude): `get_snapshot(issue)` clones once into `data/repos/<name>`, then one
    `git worktree add --detach` per issue into `data/worktrees/<instance_id>`. 7 tests (fake local repo).
    `source_files()` / `is_source_file()` drop tests, docs, examples, conftest (gold never touches them).
  - `scratch/week1_demo.py` (Claude): Week 1 milestone. Prints the report and the buggy file with `>>` on changed lines.
  - `parse.py` (Claude): tree-sitter → one row per function: path, qualname (`Cls.meth`,
    `outer.inner`), start/end line (1-based, decorators included), source, sha256 hash, calls.
  - `ground_truth.py` (Claude): changed line → innermost containing function, id `path::qualname`.
    Lines outside functions are ignored; `has_function` marks issues with no gold function.
  - `pipeline.py` (Claude, `uv run python -m pinpoint.pipeline [repo]`): writes
    `silver/functions` (hash, source), `silver/edges` (hash, callee), `silver/snapshot_membership`,
    `gold/ground_truth` (gold_files, gold_functions, has_function).
  - Week 3 (Claude, 2026-10-21): `metrics.py` (accuracy@k, recall@k, MRR, `evaluate()`, dedupes
    rankings, missing ranking scores 0); `text.py` tokenizer (`tokenize(text, mode)` with
    whole / split / both); `bm25.py` (rank_bm25 `BM25Okapi` over source files per snapshot,
    `uv run python -m pinpoint.bm25 [repo]`); `tracking.py` (`log_run` → MLflow in
    `sqlite:///data/mlflow.db`, logs params, metrics, per_issue.csv). Scratch: random baseline,
    BM25 per-issue look.
  - BM25 file-level results, pytest slice (n=17, k1=1.5, b=0.75):
    whole acc@1 .412 / acc@5 .529 / acc@10 .647 / MRR .491;
    split .294 / .529 / .588 / .403; both .294 / .529 / .647 / .410.
    Whole-identifier tokens won on this slice (n=17, so the gap is noisy).
  - `split.py` (Claude, 2026-10-22): `splits/split.csv` (committed), 200 tune / 100 held-out,
    stratified by repo (largest-remainder counts), seed 20261022. `tune_ids()` for runners.
    `bm25.py` main now scores tune only and `run()` limits gold to ranked issues.
    Row 1 (pytest tune, n=11): whole acc@1 .455 / acc@5 .636 / MRR .539. Table in NOTES.md.
  - Week 2 results (pytest): 16/17 issues have a gold function (5227 = module constant only).
    73,389 function rows → 14,312 unique hashes (80% dedup). 54,502 call edges; on one snapshot
    27% of call names match no repo function, 35% exactly one, 37% several (`__init__` → 145).
  - Week 4 toy (Claude, 2026-10-22): `scratch/week4_embedding_toy.py`, bge-small on 5 sentences.
    Paraphrase cosine .74 with word overlap .12; a word-sharing decoy scored .76. Notes in NOTES.md.
  - Starting slice: `pytest-dev/pytest` (17 issues). Example: pytest-dev__pytest-11143 -> rewrite.py {678}.
- **Blockers / open questions:** none. `git worktree add` on `/mnt/c` takes ~12s per issue
  (~1h for 300); consider moving the project to `~/` before scaling. Optional: drop unused `datasets` dependency.
- **Next step:** Week 4: embeddings concept lesson + toy, then function-level chunks embedded by hash.
  Week 6 note: down-weight ambiguous callee names.

## Decisions made

- Python 3.12 (not 3.14) to match Databricks serverless and library support.
- ~10-week timeline instead of 7.
- Tune/held-out split (200/100) locked 2026-10-22 in `splits/split.csv`, stratified by repo; never regenerate it.
- Claude handles setup/tooling; the user learns the code and the RAG stack (2026-10-09).
- Ground truth: innermost function containing each changed line; lines outside functions ignored;
  function-less issues kept with `has_function=False` and scored at file level only (2026-10-20).
- Claude writes all the code and explains the architecture and what it did after every step (2026-10-20).

## Session log

- 2026-10-22: Synced Progress. Locked 200/100 split, re-scored BM25 on pytest tune (MRR .539). Week 3 done.
- 2026-10-21: Built metrics.py, tokenizer, BM25 file baseline, MLflow tracking. BM25 whole-token MRR .491 on pytest.
- 2026-10-20: Switched to Claude-writes-code mode. Built repos.py, Week 1 demo, parse.py, ground truth, call edges, pipeline. Weeks 1–2 done.
- 2026-10-09: Claude did Week 0 setup. User wrote data.py and patches.py (reviewed and fixed), Claude wrote tests.
  User prefers direct implementation guides, no warm-ups. Next: repos.py.
- 2026-10-03: Wrote the project description and the learning plan. Assigned Week 0.

---

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
- [ ] Strategy 2: fixed ~500-token chunks → embed → brute-force numpy search (no vector DB yet).
- [ ] Strategy 3: one chunk per function (from silver). Embed by content hash, so duplicates are skipped.
      Record how many embeddings the hash saved; that's a resume number.
- [ ] Both logged to MLflow, evaluated at file and function level.

**Check yourself:** Why does AST chunking help? Find one issue where it fixed a miss, and one where it didn't.

## Week 5 — Move to Databricks

**Learn:** Databricks Free Edition, Unity Catalog (`catalog.schema.table`), Delta Lake, PySpark,
Spark SQL, Jobs, AI Search (Delta Sync index, hybrid queries, filters).

- [ ] Toy: in a notebook, make a Spark DataFrame from a pandas one, run the same query in
      PySpark and in SQL, write it as a Delta table, read it back.
- [ ] Upload your parquet files → bronze/silver/gold Delta tables. Rewrite one pandas step in PySpark.
- [ ] Check the Free Edition limits yourself (compute quotas, outbound internet). If cloning
      repos is blocked or slow there, keep parsing local and upload results. That's a fine design.
- [ ] Create an AI Search endpoint + Delta Sync index on gold functions. Query it from a notebook.
- [ ] Strategy 4: hybrid search. Over-fetch, then filter to the issue's snapshot via membership.
- [ ] Turn your pipeline into a Databricks Job.

**Milestone:** row 4, plus confirmation that the cloud setup reproduces your local row 3.

## Week 6 — LLMs, graph expansion, reranking

**Learn:** Anthropic Python SDK (messages, structured output via tool use, token usage),
`CrossEncoder`.

- [ ] Toy: one API call; print `usage` and compute cost by hand.
- [ ] Strategy 5: query rewriting. First a regex extractor (identifiers, stack-trace frames), then an
      LLM cleanup. Measure each separately: you'll know which part actually helped.
- [ ] Strategy 6: call-graph expansion. Add callers/callees of the top hits from `silver/edges`.
      Test how many neighbours to add and how to score them.
- [ ] Strategy 7: cross-encoder reranking on the top ~50.

## Week 7 — Agent baseline

**Learn:** tool calling (writing the agent loop yourself, no framework), MLflow tracing.

- [ ] Tools: `grep(pattern)` and `open_file(path, start, end)`, sandboxed to the snapshot dir.
- [ ] Write the loop: send messages → model asks for a tool → run it → send result → repeat, with a
      step limit. Ask for a final ranked list of functions as structured output.
- [ ] Strategy A on the tune set: accuracy, latency, $/query. Strategy H: seed the agent with
      Pinpoint's top-10.
- [ ] Add MLflow tracing to the retrieval steps; inspect 5 failures and write down why they failed.

## Week 8 — Product: API and bot

**Learn:** FastAPI, Pydantic, GitHub Actions YAML, PyGithub, `git blame` via GitPython.

- [ ] Toy: a FastAPI "hello" endpoint, then one with a Pydantic request/response model; test with
      `TestClient` in pytest.
- [ ] `/triage` endpoint: best strategy + generation (explanation citing file:line) + owner from blame.
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
