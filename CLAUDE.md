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

- **Current week:** Week 7 (Agent baseline). Weeks 0-4 and 6 done; Week 5 done except the quota check.
- **Current task:** Week 7 concept lesson (tool-calling agent loop), then tools grep/open_file.
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
  - Week 5 (2026-10-26, Databricks Free Edition, serverless). Notebooks in `scratch/` (gitignored), Databricks
    source format, imported via Workspace -> Import: `week5_spark_toy.py`, `week5_upload_delta.py`,
    `week5_vector_index.py`, `week5_hybrid.py`. UC tables: `workspace.pinpoint_{bronze,silver,gold}.*`, raw files in
    volume `/Volumes/workspace/pinpoint_bronze/raw`. Serverless has outbound internet (git clone works).
    `%pip` broke the env; use preinstalled `databricks-sdk`. Vector search endpoint `pinpoint`, Delta Sync index
    `workspace.pinpoint_gold.function_vectors_index` (5,843 bodies, self-managed vectors, score = 1/(1+L2^2)).
    `scratch/week5_export_queries.py` writes `gold/issue_queries.parquet` + `gold/candidates.parquet` for upload.
    Cloud ANN + snapshot post-filter reproduces local row 3 (file MRR .617). Limits: HYBRID max 200 results,
    ANN returned 2,097 of 5,843 silently. Databricks HYBRID file MRR .357 (weak keyword side).
  - `fusion.py` (Claude, 2026-10-26, `uv run python -m pinpoint.fusion`): RRF of BM25 + dense. File RRF MRR .561
    (acc@5 .727). BM25 over function source = best function row (acc@1 .200, MRR .257). Rows in NOTES.md.
  - Databricks Job (Claude, 2026-10-26): `databricks.yml` bundle + `jobs.py` (`ingest`, `build-tables` entry points),
    serverless wheel tasks. CLI at ~/.local/bin, profile `pinpoint`. Run reproduces local tables exactly, idempotent.
    Wheel deps slimmed; heavy libs in uv group `local`. Next job tasks: embed (new hashes only), sync_index, evaluate.
  - Week 6 (Claude, 2026-10-26): LLM toy `scratch/week6_llm_toy.py` (Databricks endpoints, usage, cost by hand).
    `query.py` (5a regex extractor + repo-vocab filter): mostly negative, best BM25 file .550. `llm_query.py` (5b,
    Llama 3.3 70B forced tool call, cached in data/gold/llm_queries.parquet): BM25 file .583, dense fn .231; hurts
    BM25 fn and dense file. Rewriting is per-retriever. `fusion.side_rankings` takes custom queries.
  - `graph.py` (Strategy 6, 2026-10-26): name-only call graph, 1/m ambiguity weights, rank-based expansion.
    Dense fn seeds MRR .194 -> .344 (best fn row; 3 up 2 down, n=10); BM25 seeds collapse (.038): neighbours
    overtake correct exact-match hits. Callers carry the gain. `cap=True` safe but small.
  - `rerank.py` (Strategy 7, 2026-10-26): cross-encoder on dense top-50, scores cached. bge-reranker-base hurts
    (MRR .194 -> .101; web-trained, bare code is out of domain); with path + qualname header .234. Week 6 done.
  - Best rows so far (pytest tune): file = dense Strategy 3 MRR .617 (BM25 + LLM query acc@5 .818);
    function = dense + call-graph expansion MRR .344 (acc@1 .3). All results in NOTES.md table.
  - Starting slice: `pytest-dev/pytest` (17 issues). Example: pytest-dev__pytest-11143 -> rewrite.py {678}.
- **Blockers / open questions:** TWO COPIES. `~/PinPoint` was made on 2026-10-22 (pushed cbc253c), but all later
  work (Weeks 4-5) happened in `/mnt/c/Users/Visha/Desktop/PinPoint`, which is the current one. `~/PinPoint` only has an
  uncommitted early `embed.py` draft (superseded by `dense_functions.py`). User to decide: delete it, or re-clone
  into `~/` and move `data/` there (worktree add ~12s/issue on `/mnt/c`, ~1h for 300).
  Week 5 quota check still open (user looks it up in the Databricks UI). All results are n=10/11 (pytest only):
  re-check the winners on the 200-issue tune set before trusting them.
- **Next step:** Week 7 agent baseline on Llama 3.3 70B (Databricks endpoint; it handles forced tool calls,
  gpt-oss doesn't). Before scaling to 300: add embed/sync/evaluate job tasks.
  Optional later: fine-tune a cross-encoder on (issue, gold function) pairs; LLM listwise reranking.

## Decisions made

- Python 3.12 (not 3.14) to match Databricks serverless and library support.
- ~10-week timeline instead of 7.
- Tune/held-out split (200/100) locked 2026-10-22 in `splits/split.csv`, stratified by repo; never regenerate it.
- Claude handles setup/tooling; the user learns the code and the RAG stack (2026-10-09).
- Ground truth: innermost function containing each changed line; lines outside functions ignored;
  function-less issues kept with `has_function=False` and scored at file level only (2026-10-20).
- Claude writes all the code and explains the architecture and what it did after every step (2026-10-20).
- LLMs via Databricks Foundation Model endpoints (free on Free Edition, no API key), not the paid
  Anthropic API (2026-10-26). Compare models by endpoint name; Claude/Gemini can be added later as rows.

## Session log

- 2026-10-26: Week 5: Spark/Delta toy, parquet -> UC Delta, vector index, Strategy 4 (Databricks hybrid + our RRF),
  Databricks CLI + Asset Bundle job (idempotent, reproduces local tables). Only the quota check is left in Week 5.
  Week 6 done: LLM toy on free Databricks endpoints, Strategy 5 (regex/LLM rewriting, mixed), 6 (call graph,
  dense fn MRR .344, best), 7 (cross-encoder, hurts unless names shown). Next: Week 7 agent.
- 2026-10-24: Strategy 2 (fixed chunks) and Strategy 3 (function chunks, 65% hash savings, header variant). Week 4 done.
- 2026-10-22: Synced Progress. Locked 200/100 split, re-scored BM25 on pytest tune (MRR .539). Week 3 done.
- 2026-10-21: Built metrics.py, tokenizer, BM25 file baseline, MLflow tracking. BM25 whole-token MRR .491 on pytest.
- 2026-10-20: Switched to Claude-writes-code mode. Built repos.py, Week 1 demo, parse.py, ground truth, call edges, pipeline. Weeks 1–2 done.
- 2026-10-09: Claude did Week 0 setup. User wrote data.py and patches.py (reviewed and fixed), Claude wrote tests.
  User prefers direct implementation guides, no warm-ups. Next: repos.py.
- 2026-10-03: Wrote the project description and the learning plan. Assigned Week 0.
