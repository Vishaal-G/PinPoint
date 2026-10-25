# Pinpoint — Coaching Project

Pinpoint is a GitHub bot that reads a new issue and points to the functions most likely responsible,
benchmarked on SWE-bench Lite. The user is building it **to learn the libraries**. The week-by-week plan
is in `LEARNING_PLAN.md`.

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
1. Read the **Progress** section below and the current week in `LEARNING_PLAN.md`.
2. Give a 2–3 line recap: where they left off, what's next, any open question from last time.
3. Ask if they want to continue that task or do something else.

**On stop** (user says "stopping", "done for today", "wrap up", or similar):
1. Update **Progress** below: current week/task, what got done, blockers, next step.
2. Tick finished checkboxes in `LEARNING_PLAN.md`.
3. Add one line to the **Session log** (newest first).

## Progress

- **Current week:** Week 5 (Move to Databricks).
- **Current task:** Week 5 concept lesson, then the Spark/Delta toy notebook.
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
  - Strategy 2 (Claude, 2026-10-24): `chunking.py` (line-aligned ~500-token windows, 50 overlap),
    `embed.py` (`Embedder`: bge-small, query prefix, sha256 cache in `silver/embeddings_bge-small-en-v1.5.parquet`),
    `dense.py` (`uv run python -m pinpoint.dense [repo]`; file = max chunk, function = max overlapping chunk).
    Pytest tune: file acc@1 .364 / acc@5 .727 / MRR .520; function MRR .143 (chunk ties).
    6,202 chunks → 4,489 embedded (28% saved). 32 min on CPU. Per-issue: `scratch/week4_dense_vs_bm25.py`.
  - Strategy 3 (Claude, 2026-10-24): `dense_functions.py` (`uv run python -m pinpoint.dense_functions [repo] [--header]`),
    one chunk per silver function, cache key = silver hash. 16,545 chunks -> 5,843 unique (65% saved vs 28%).
    Pytest tune: no header file MRR .617 (best file row), function MRR .194;
    `--header` (path + qualname) file MRR .556, function acc@5 .500 / MRR .241 (best function row).
    5.8% of functions exceed 512 tokens. Per-issue: `scratch/week4_functions_vs_fixed.py`. Week 4 done.
  - Starting slice: `pytest-dev/pytest` (17 issues). Example: pytest-dev__pytest-11143 -> rewrite.py {678}.
- **Blockers / open questions:** none. `git worktree add` on `/mnt/c` takes ~12s per issue
  (~1h for 300); consider moving the project to `~/` before scaling. Optional: drop unused `datasets` dependency.
- **Next step:** Week 5 Databricks toy: pandas -> Spark DataFrame, same query in PySpark and SQL, Delta write/read.
  Known: issues over 512 tokens get truncated (10/17 pytest); Week 6 query rewriting.
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

- 2026-10-24: Strategy 2 (fixed chunks) and Strategy 3 (function chunks, 65% hash savings, header variant). Week 4 done.
- 2026-10-22: Synced Progress. Locked 200/100 split, re-scored BM25 on pytest tune (MRR .539). Week 3 done.
- 2026-10-21: Built metrics.py, tokenizer, BM25 file baseline, MLflow tracking. BM25 whole-token MRR .491 on pytest.
- 2026-10-20: Switched to Claude-writes-code mode. Built repos.py, Week 1 demo, parse.py, ground truth, call edges, pipeline. Weeks 1–2 done.
- 2026-10-09: Claude did Week 0 setup. User wrote data.py and patches.py (reviewed and fixed), Claude wrote tests.
  User prefers direct implementation guides, no warm-ups. Next: repos.py.
- 2026-10-03: Wrote the project description and the learning plan. Assigned Week 0.
