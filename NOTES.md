# Notes

Three takeaways per library, in my own words.

## uv, pytest (Week 0)

- Why does pytest find `test_something.py` but not `something_test_helper.py`?

## Results table (tune set only; held-out is untouched until Week 9)

Split: `src/pinpoint/split.csv` (was `splits/split.csv`), 200 tune / 100 held-out, stratified by repo, seed 20261022.
Slice so far: pytest-dev/pytest tune issues (n=11). Small n: one issue = 9 points of acc@1.

| # | Strategy | Level | acc@1 | acc@5 | acc@10 | MRR | n |
|---|---|---|---|---|---|---|---|
| 1 | BM25, whole-identifier tokens, k1=1.5 b=0.75 | file | .455 | .636 | .636 | .539 | 11 |
| 1b | BM25, split tokens | file | .273 | .636 | .636 | .412 | 11 |
| 1c | BM25, whole + split tokens | file | .273 | .636 | .727 | .422 | 11 |
| 2 | Dense, fixed 500-token line chunks (overlap 50), bge-small, max-pool | file | .364 | .727 | .727 | .520 | 11 |
| 2f | same | function | .100 | .100 | .300 | .143 | 10 |
| 3 | Dense, one chunk per function (silver), bge-small, file = best function | file | .545 | .636 | .727 | .617 | 11 |
| 3f | same | function | .100 | .300 | .500 | .194 | 10 |
| 3h | Strategy 3 + `path\nqualname` header on each function | file | .364 | .727 | .818 | .556 | 11 |
| 3hf | same | function | .100 | .500 | .600 | .241 | 10 |
| 3c | Strategy 3 in Databricks: ANN index (asked 5843, got 2097), post-filter to snapshot | file | .545 | .636 | .727 | .617 | 11 |
| 3cf | same (.191 vs local .194: ANN returned only 2,097 rows) | function | .100 | .300 | .500 | .191 | 10 |
| 4k | Databricks FULL_TEXT (whole issue text over function source), n=200, post-filter | file | .091 | .273 | .364 | .215 | 11 |
| 4kf | same | function | .000 | .200 | .300 | .114 | 10 |
| 4 | Databricks HYBRID (ANN + FULL_TEXT, rank-fused), n=200, post-filter | file | .091 | .727 | .727 | .357 | 11 |
| 4f | same | function | .000 | .400 | .500 | .153 | 10 |
| 1f | BM25 whole tokens over each function's source (silver), per snapshot | function | .200 | .300 | .300 | .257 | 10 |
| 4L | Our hybrid: RRF(BM25 whole file, dense Strategy 3 file), k=60, equal weights | file | .455 | .727 | .727 | .561 | 11 |
| 4Lf | Our hybrid: RRF(BM25 function, dense function), k=60, equal weights | function | .100 | .300 | .400 | .199 | 10 |
| 5a | BM25 file, query = raw issue + regex-extracted code terms (`pinpoint.query`, raw+terms) | file | .455 | .727 | .727 | .550 | 11 |
| 5ad | Dense file, query = title + regex terms kept only if in repo vocab (repo-title+terms) | file | .364 | .727 | .727 | .532 | 11 |
| 5b | BM25 file, query = LLM rewrite (Llama 3.3 70B tool call: summary + guessed names + terms) | file | .455 | .818 | .818 | .583 | 11 |
| 5bf | Dense function, query = title + LLM summary + guessed names (title+llm) | function | .200 | .300 | .300 | .231 | 10 |
| 6 | Dense function (Strategy 3) + call-graph expansion: top-5 seeds, callers+callees, 1/m ambiguity weight, alpha=1 | function | .300 | .400 | .400 | .344 | 10 |
| 6b | same expansion on BM25 function seeds | function | .000 | .000 | .200 | .038 | 10 |
| 7 | Dense fn top-50 reranked by bge-reranker-base, query = title + LLM summary | function | .000 | .200 | .300 | .101 | 10 |
| 7h | same, document = path + qualname + source | function | .100 | .400 | .500 | .234 | 10 |
| A | Agent: Llama 3.3 70B, grep + open_file, 15-step cap, submit_ranking (file = order of submitted functions) | file | .727 | .727 | .727 | .727 | 11 |
| Af | same agent, function level | function | .200 | .200 | .200 | .200 | 10 |
| H | Agent seeded with Pinpoint top-10 (dense + graph, Strategy 6), same prompt otherwise | file | .636 | .727 | .727 | .682 | 11 |
| Hf | same, function level | function | .300 | .600 | .600 | .450 | 10 |
| H+l | H + tools label each line with its enclosing function id (fix 1) | file | .727 | .818 | .818 | .773 | 11 |
| H+lf | same, function level | function | .400 | .600 | .600 | .500 | 10 |
| H+lp | H + labels + prompt advice (caller not crash site, trigger not bug, submit 10) | file | .636 | .818 | .818 | .712 | 11 |
| H+lpf | same, function level | function | .400 | .600 | .600 | .500 | 10 |
| H2 | H + labels + prompt + definitions of --options the issue mentions (all 3 fixes) | file | .727 | .818 | .818 | .773 | 11 |
| H2f | same, function level (.600 without the 11143 noise flip, see notes) | function | .600 | .700 | .700 | .650 | 10 |
| A2 | A + all 3 fixes (no seeds) | file | .636 | .818 | .818 | .727 | 11 |
| A2f | same, function level | function | .400 | .600 | .600 | .500 | 10 |
| 3L | Dense function chunks (Strategy 3, no header) with databricks-bge-large-en endpoint instead of local bge-small | file | .455 | .818 | .818 | .641 | 11 |
| 3Lf | same, function level | function | .000 | .400 | .500 | .180 | 10 |
| R3 | **Recheck slice (52 tune issues, all 12 repos, bge-large endpoint, Databricks job).** Dense function chunks | file | .442 | .769 | .808 | .587 | 52 |
| R3f | same, function level | function | .118 | .373 | .451 | .232 | 51 |
| R6 | recheck: dense + call graph (Strategy 6), file = best function | file | .308 | .654 | .788 | .460 | 52 |
| R6f | recheck: dense + call graph, function level | function | .059 | .176 | .275 | .147 | 51 |
| R1 | recheck: BM25 over function source, file = best function | file | .269 | .519 | .712 | .418 | 52 |
| R1f | recheck: BM25 over function source | function | .078 | .275 | .353 | .174 | 51 |
| R4f | recheck: RRF(BM25 fn, dense fn) | function | .039 | .373 | .471 | .162 | 51 |
| RA2 | recheck: agent A2 (no seeds, labels + prompt + options) | file | .654 | .731 | .731 | .692 | 52 |
| RA2f | same, function level | function | .451 | .569 | .569 | .510 | 51 |
| RH2 | recheck: agent H2, seeded with dense top-10 (bge-large) | file | .731 | .788 | .788 | .760 | 52 |
| RH2f | same, function level | function | .471 | .627 | .627 | .540 | 51 |
| RH3 | recheck: H2 dense seeds + call_graph tool (fix 4) | file | .731 | .885 | .885 | .804 | 52 |
| RH3f | same, function level | function | .529 | .725 | .745 | .614 | 51 |
| T3 | **Tune set (all 200 tune issues, bge-large endpoint, Databricks job, 2026-10-29).** Dense function chunks | file | .440 | .715 | .820 | .567 | 200 |
| T3f | same, function level | function | .207 | .380 | .495 | .295 | 184 |
| T4f | tune: RRF(BM25 fn, dense fn) | function | .158 | .391 | .522 | .262 | 184 |
| T1 | tune: BM25 over function source, file = best function | file | .345 | .560 | .690 | .463 | 200 |
| T1f | tune: BM25 over function source | function | .163 | .293 | .375 | .232 | 184 |
| T6 | tune: dense + call graph, file = best function | file | .315 | .595 | .735 | .448 | 200 |
| T6f | tune: dense + call graph, function level | function | .092 | .228 | .299 | .168 | 184 |

## Evaluation (Week 3)

- Lock the held-out set before tuning anything. Every time you look at a score and change
  something, you fit the test a little. Held-out stays honest only if it's scored once.
- Stratify by repo: django is 38% of Lite. A random split could put most of django in one side.
- Score only the issues you ranked. A gold issue with no ranking counts as a miss, which is
  right for a strategy that crashed but wrong for an issue that was never meant to be in the run.

## Embeddings (Week 4)

- An embedding model turns text into a fixed-length vector (bge-small: 384 floats, unit length).
  Similar meaning -> vectors point the same way. Cosine = dot product of unit vectors.
- Trained contrastively: (query, right passage) pairs pulled together, other passages in the
  batch pushed apart. So "pytest fails ... integer" lands near "test runner crashes ... number"
  with no shared content words (cosine .74, word overlap .12). BM25 would miss that.
- Not magic: a decoy sharing the words docstring/number scored .76, above the paraphrase.
  One vector per text blurs fine meaning (crash vs printed). That's what reranking fixes (Week 6).
- Scores are compressed: unrelated text still gets ~.38. Only the ordering means something;
  don't pick a fixed threshold like 0.5.

## Chunking + brute-force search (Week 4, Strategy 2)

- bge-small sees 512 tokens max. A pytest snapshot is 83 files / 313k tokens (median file 1,510),
  so whole-file embeddings would see a third of a typical file. Chunk = unit of retrieval.
- Brute-force search = one matrix-vector product (chunks x 384 @ 384). Exact, microseconds at
  ~700 chunks. ANN indexes (HNSW/IVF) trade a little recall for speed at millions of vectors.
- File score = max over its chunks (sum favours big files, mean punishes them).
- Dense vs BM25 (file): dense won where the issue describes behaviour in prose (8906 "skip at
  module level": BM25 rank 5 -> dense 1; 7373 skipif caching: 12 -> 3). It lost where the issue's
  first 512 tokens are noise: 11143 opens with a 400-token `pip list`, so the truncated query
  never reaches the traceback (BM25 1 -> dense 4). 10/17 issues are over 512 tokens.
- Function level is poor (MRR .143) by construction: a chunk holds a median 3 (max 18) functions,
  all with the same score, so ties are broken by file order. Motivates one chunk per function.
- Hash cache: 6,202 chunks across 11 snapshots, 4,489 embedded (28% saved). Fixed chunks dedupe
  badly: one inserted line shifts every later window in the file. 32 min on CPU.

## Function chunks (Week 4, Strategy 3)

- Chunk = one function from `silver/functions`; the embedding cache key is the same sha256 as the
  silver hash, so a function is embedded once across every snapshot it appears in.
- Hash savings: 16,545 function chunks across 11 pytest snapshots, 5,843 unique (65% saved,
  vs 28% for fixed chunks). Unchanged functions keep their hash across commits; fixed windows don't.
- File level is the best row so far (MRR .617 vs BM25 .539, fixed .520). Function MRR .143 -> .194.
- Fixed a miss: 5221 (`--fixtures` should show fixture scope): function rank 34 -> 1, because
  `_showfixtures_main` now gets its own vector instead of sharing a window with its neighbours.
  7220 also 362 -> 49; 11143 file 4 -> 1.
- Didn't help: 8906 (skip at module level) file 1 -> 8, function 1 -> 8. The fix is an error
  message inside `Module._importtestmodule`, but `outcomes.skip` (whose docstring talks about
  `allow_module_level`) is a closer paraphrase of the issue. Matching the topic is not the same as
  finding the code that needs to change.
- Function level is still weak (acc@1 .100): ~1,500 candidates per snapshot, short functions with
  generic names, and the 512-token query truncation from Strategy 2 is unchanged.
- Function length (unique pytest functions): median 118 tokens, p75 228, max 2,848. 5.8% are over
  510 and get their tail truncated; splitting those is a Week 6+ option, not worth it now.
- Header variant (`--header`, path + qualname above the source): function level up (MRR .194 ->
  .241, acc@5 .300 -> .500); file acc@1 down (.545 -> .364), acc@10 up (.727 -> .818).
  Wins: 11143 gold `AssertionRewriter.run` 21 -> 1 (the name says "rewrite", the issue is a rewrite
  crash); 5103 147 -> 10; tail issues 6116/7220/9359 all climb. Losses: 5221 file 1 -> 2 because
  `fixtures.py::fixture` now matches an issue about fixtures by name alone; 5495 the same with
  `rewrite.py`. The name is a strong prior: it rescues weak bodies and over-rewards on-topic files.
- Header dedupes slightly worse: 5,939 texts embedded vs 5,843 (same source at a different path
  is now a different text). Keep header for function ranking; file-level choice waits for hybrid.


## Spark + Delta toy (Week 5)

- Free Edition = serverless compute only; catalog `workspace`, tables are `catalog.schema.table`
  (`workspace.pinpoint_toy.functions`). Import a `.py` starting with `# Databricks notebook source`
  and it becomes a notebook (`# COMMAND ----------` = cell break, `# MAGIC` = markdown/SQL cell).
- Spark is lazy: transformations build a plan, `display()`/`count()`/`toPandas()` run it.
  `.explain()` shows `HashAggregate` + `Exchange` (the shuffle). PySpark and SQL compile to the same plan.
- Delta never edits a parquet file in place. Every write = new files + a new JSON entry in
  `_delta_log`; `DESCRIBE HISTORY` reads that log. Time travel: `VERSION AS OF n`.
- UPDATE on UC managed tables uses deletion vectors (merge-on-read: mark the row dead in a sidecar
  file, add a new file) instead of rewriting the whole file (copy-on-write).
- Surprise: a version 2 `OPTIMIZE` appeared 3s after the UPDATE. Auto-compaction merges small
  files (the small-file problem: per-file overhead dominates). Old files stay until `VACUUM`
  (default 7 days), which is what bounds time travel.
- `.toPandas()` pulls every row to the driver: fine for 8 rows, a crash at scale.

## Parquet -> Unity Catalog (Week 5)

- Route: upload files to a volume (`/Volumes/workspace/pinpoint_bronze/raw`, a governed folder for
  raw files), then `spark.read.parquet(...).write.saveAsTable(...)` -> managed Delta tables in
  schemas `pinpoint_bronze` / `pinpoint_silver` / `pinpoint_gold`. ~48 MB total, UI upload is fine.
- Managed table: Databricks owns the storage, DROP deletes the data. External: you own the path.
- Free Edition serverless HAS outbound internet (2026-10-26): `curl github.com` 200, `git ls-remote` works.
  So cloning + parsing could run in Databricks, not only locally.
- PySpark rewrite of `week2_edges_check.py`, all 17 snapshots: 236,078 call names, one .474 /
  none .302 / many .224, identical to pandas. Mapping: `.str.split().str[-1]` -> `element_at(split())`,
  `Series.map` -> left join, `fillna` -> `coalesce`. Small side of a join is broadcast
  (BroadcastHashJoin); two big sides need a shuffle + SortMergeJoin.

## Vector search index (Week 5)

- `%pip install databricks-vectorsearch` broke the serverless environment ("Failure starting repl");
  fix = Environment panel -> remove dependency -> Apply. Use the preinstalled `databricks-sdk`
  (`w.vector_search_endpoints`, `w.vector_search_indexes`) instead: same REST API, nothing to install.
- Endpoint `pinpoint` (STANDARD) + Delta Sync index `workspace.pinpoint_gold.function_vectors_index`
  on `function_vectors` (5,843 function bodies, self-managed bge-small vectors cast to float32,
  CDF on, TRIGGERED sync). Queries embedded locally with the same model + prefix.
- ANN recall@10 vs brute force: 1.0 on all 11 pytest tune issues, same top-1. Expected at 5.8k vectors.
- Score is NOT cosine: score = 1 / (1 + L2^2) (checked: cos .7813 -> .6957). Unit vectors make
  L2^2 = 2 - 2cos, so the ranking matches cosine but cosine thresholds don't transfer.
- Global index wastes slots: pytest-11143 top-10 has only 3 functions from its own snapshot; the rest
  are other commits' versions of the same functions (is_rewrite_disabled x2, pytest_runtest_protocol x3,
  pastebin.pytest_configure x3). Gold AssertionRewriter.run (cos .725) misses the top 10.
  -> Strategy 4 must over-fetch then filter to the snapshot (or filter inside the index).

## Hybrid + snapshot filter (Week 5, Strategy 4)

- Post-filter: fetch top n from the global index, keep this snapshot's functions. ~1 in 5 survives
  (n=100 -> 19 kept, n=500 -> 106). Over-fetch 5-10x what you need. Pre-filter (filters_json on an
  instance_id column) would need one vector per (snapshot, function): 16,545 vs 5,843, no dedup.
- Limits found: HYBRID max 200 results per query (hard error). ANN asked 5843 -> got 2097 with no
  error (1000 and 2000 came back exactly). Always check rows returned vs asked.
- Milestone: cloud ANN reproduces local row 3 exactly at file level (.545/.636/.727/.617).
- Hybrid (n=200): file acc@5 .636 -> .727, fn acc@5 .3 -> .4, but file acc@1 .545 -> .091, MRR .611 -> .357.
  Per issue (fn rank ANN -> HYBRID): 5495 7->2, 8906 8->3, 5692 4->3; 5221 1->5, 11143 21->44, 7373 3->7.
- Why: RRF weights both lists equally and assumes both are decent. Databricks' keyword side is weak
  here (file acc@1 .091 vs our BM25 .455): whole issue text as query, tokenizer not built for code.
  Fusing a strong retriever with a weak one helps recall@5 but wrecks the top spot.
- Next ideas: RRF of OUR BM25 + dense (isolates fusion vs keyword engine); identifiers-only query text
  for the keyword side (Week 6 query rewriting); weighted RRF.

## Our RRF: BM25 + dense (Week 5, Strategy 4L, `pinpoint.fusion`)

- Both sides reproduce their rows (BM25 file .539, dense file .617), so fusion is tested on correct inputs.
- File: RRF k=60 gives acc@5 .727 (best tie) but MRR .561 < dense .617. Same shape as Databricks
  HYBRID, much milder (.561 vs .357): our keyword side is decent, theirs wasn't. So fusion itself
  is fine; their keyword engine was most of the problem.
- Function: BM25 over function source ALONE is the best function row so far (acc@1 .200, MRR .257
  vs header-dense .241). Wins 5495 and 5692 at rank 1, 8906 at 2: issues that name identifiers.
- RRF rewards agreement, punishes lone winners. 11143: bm25 44, dense 21 -> fused 8 (both half-right).
  5221: dense 1, bm25 474 -> fused 42: 1/61 + 1/534 = .018 loses to anything ~rank 20 in both (2/80 = .025).
- Hard issues (5103, 6116, 7220, 9359) are bad in both lists; fusion can't rescue what neither finds.
- Sensitivity (not logged as rows; 11 issues is too few to pick from): file dense x2 weight
  .545/.727/.818/.626 (best file numbers), k=10 ~ k=60. Function dense x2 drops to MRR .116.
  Revisit weights on the full 200-issue tune set.

## Databricks Job via Asset Bundle (Week 5)

- `databricks.yml` + `src/pinpoint/jobs.py`: job `[dev vishaaltor] pinpoint-pipeline`, tasks
  `ingest -> build_tables`, python_wheel_task on serverless (environment_version 3), wheel built by
  `uv build`. `databricks bundle deploy` then `databricks bundle run pinpoint_pipeline`.
- Bundle variables (catalog, schema_prefix) = deploy time; job parameter `repo` = run time.
- Wheel deps kept minimal; mlflow/torch/sentence-transformers moved to a uv `local` group so the job
  doesn't pull 2 GB of CUDA torch or clash with the runtime's packages.
- Idempotent writes: functions/edges `MERGE ... WHEN NOT MATCHED THEN INSERT` on hash; membership and
  ground_truth `replaceWhere instance_id IN (...)`. First run: 14,312 functions built, 0 new; 54,502
  edges, 0 new: cloud parse == local parse. After a failed run + a good run, tables still
  14,312 / 54,502 / 73,389 / 17. Retries are safe because of this.
- Gotcha: the job env upgraded pandas to 3 (pyarrow-backed strings); after a concat they're chunked
  and pyspark's createDataFrame fails ("Cannot convert ChunkedArray to Array"). Fix: cast non-numeric
  columns to object first (`jobs.plain_dtypes`).
- CLI: installed to ~/.local/bin (no sudo); OAuth U2M login `databricks auth login --profile pinpoint`.

## LLM toy (Week 6, Databricks Foundation Model endpoints)

- Free Edition serves open models (llama-3.3-70b, llama-4-maverick, gpt-oss-120b/20b, qwen, deepseek,
  gemma) at `databricks-*` endpoints; `WorkspaceClient(profile="pinpoint").serving_endpoints.query(...)`,
  no API key. Response has `usage.prompt_tokens / completion_tokens` like any LLM API.
- Issue 11143 (5,312 chars): llama 1,815 tokens, gpt-oss 1,920 (2.9 chars/token: logs and code tokenize
  worse than prose's ~4). Different tokenizer + chat template = different count for the same text.
  gpt-oss adds a hidden template: a 10-word question cost 111 prompt tokens vs llama's 45.
- Output is billed, not just shown: gpt-oss 193 tokens for a 31-word answer (~150 hidden reasoning
  tokens); llama 57 for 38 words. At a reference $4/$20 per 1M: $0.008-0.012 per issue, ~$3 per 300.
- Both answers name the real bug (numeric first expression treated as docstring -> `"PYTEST_DONT_REWRITE"
  in docstring` TypeError) from the stack trace at the END of the issue. bge-small only saw the first 512
  tokens (a `pip list`), which is why 11143 was weak for dense. That's the case for query rewriting.

## Query rewriting with regexes (Week 6, Strategy 5a, `pinpoint.query`)

- Tune issues: Python tracebacks in only 14% (pytest 1/11); backticks 44%, code blocks 52%. So the
  extractor is mostly "code-looking identifiers" (snake_case, dotted, camelCase, --options, call()),
  plus error names, .py paths, and traceback frames when present.
- Mostly a negative result (n=11). Best: BM25 file raw+terms MRR .539 -> .550 (acc@5 .636 -> .727),
  i.e. noise level. Terms-only queries hurt everywhere (BM25 fn .257 -> .047, dense file .617 -> .219):
  the prose carries meaning ("fixture scope", "skip at module level") that a term list throws away,
  and an embedder given a bag of identifiers isn't seeing a sentence anymore.
- Many extracted terms are the REPORTER's code, not the repo's (test_all_even, GeoLocationTest,
  MacBook). Filter: keep a term only if one of its tokens occurs in the snapshot's functions.
  Helps dense (title+terms .395 -> .532) but changes nothing for BM25, because BM25 already ignores
  words that no document contains (zero term frequency everywhere). Dense has no such filter: every
  input token moves the vector.
- The fix is often not IN the issue: 5221 says only `--fixtures`; the gold is `_showfixtures_main`.
  Extraction can't add words that aren't there; an LLM can infer them. That is the case for 5b.
- 11143's trace ends in `is_rewrite_disabled` but the fix is one frame up in `run`: failing frame
  != fixed function. Callers of the top hit matter (Strategy 6).

## Query rewriting with an LLM (Week 6, Strategy 5b, `pinpoint.llm_query`)

- Structured output = a forced tool call with a JSON schema (summary, functions, files, terms).
  Databricks endpoints speak the OpenAI chat format; the typed SDK `query()` has no `tools`, so call
  `w.api_client.do("POST", "/serving-endpoints/<name>/invocations", body=...)`. Parse + validate the
  arguments, retry once. Llama 3.3 70B obeys; gpt-oss-120b ignored the forced tool and wrote markdown.
- The tool schema is hidden prompt: a one-line question with the tool = 702 prompt tokens.
- 11 issues: 15,550 prompt + 904 completion tokens (~1,400 in / 80 out per issue; ~$0.007 at $4/$20).
  Cached in data/gold/llm_queries.parquet keyed by model + prompt hash: reruns free and identical.
- Mixed result. Helps: BM25 file MRR .539 -> .583 (acc@5 .636 -> .818, best BM25 file row); dense
  function .194 -> .231 (acc@1 .100 -> .200). Hurts: BM25 function .257 -> .212 (raw+llm), dense file
  .617 -> .576. Raw dense file (.617) and BM25 function (.257) stay the best rows at their level.
- The model rarely knows the right names: exact gold name only for 7373, and that issue already
  names them. Near misses: 5221 guessed `show_fixtures` / `_show_fixtures` for `_showfixtures_main`;
  whole-identifier BM25 can't match those, split tokens could. Low evidence of memorised pytest fixes.
- 5a vs 5b: inferring names helps more than extracting them, but neither beats the raw issue
  everywhere. Rewriting is a per-retriever choice, not a global switch.

## Call-graph expansion (Week 6, Strategy 6, `pinpoint.graph`)

- Ceiling first: gold is 1 hop from the dense top-5 (and not in it) for 2/10 issues; 0/10 for BM25.
  Neighbourhood of 5 seeds ~25 functions (max 158), so the bonus must be selective.
- Rank-based: score = 1/(60+rank) + alpha * max over adjacent top-k seeds of w / (60+seed rank);
  name-only edges resolve to every function with that short name, weight 1/m for m targets.
- Pre-registered (k=5, alpha=1, both directions, weighted) on dense seeds: MRR .194 -> .344, acc@1
  .1 -> .3 (best function row). 11143 21 -> 1 (caller `run` of the failing frame), 5495 7 -> 1,
  5692 4 -> 1; but 5221 1 -> 26, 8906 8 -> 27. Three up, two down, n=10.
- On BM25 seeds it collapses (.257 -> .038): 5495 and 5692 were already rank 1 and neighbours jumped over them.
- Why: with alpha=1 a weight-1 neighbour scores above its own seed. That's right when the top hit is
  a near miss (dense: topically close, often the frame or a helper) and wrong when it is already the
  answer (BM25: exact identifier match). Expand the retriever whose errors are near misses.
- Callers carry the gain (callers only .330, callees only .135): the fix tends to be in what CALLS the
  topical hit. Ambiguity weighting barely mattered here (.344 vs .334 unweighted).
- cap=True (neighbour lands right after its seed, added after seeing results, so sensitivity only):
  safe but small: dense .230, BM25 .214. Leapfrogging is where the gain is, and where the risk is.

## Cross-encoder reranking (Week 6, Strategy 7, `pinpoint.rerank`)

- bge-reranker-base on CPU: ~0.3 s per pair (50 long pairs 14.6 s); MiniLM-L6 6x faster. Both cap the
  PAIR at 512 tokens, so the raw 1,800-token issue would crowd out the code: pre-registered query =
  title + LLM summary (5b). Scores cached by sha256(model, query, doc).
- Ceiling = first-stage recall@50: dense .7, dense+graph .6, BM25 .5. A reranker can't beat it.
- Negative result. Pre-registered (dense top-50, title+llm query, plain source): MRR .194 -> .101.
  5221 1 -> 10, 5495 7 -> 47, 7373 3 -> 12; 8906 8 -> 2, 7220 49 -> 24. Raw query .098, MiniLM .063,
  on BM25 or dense+graph stages ~.12: every variant hurts, except...
- ...header (path + qualname above the source): .234 > .194 (acc@5 .3 -> .4). The names give a
  web-trained model something language-like to match ("showfixtures", "rewrite"); bare code is out of
  its training distribution (MS MARCO: questions vs web passages).
- Why rerankers fail here: domain shift + the top 50 are all on-topic functions from the same
  subsystem (hard negatives), which is exactly where an untuned reranker is weakest.
- Fixes for later: fine-tune a cross-encoder on (issue, gold function) pairs from tune repos only;
  or an LLM as a listwise reranker (Week 7 territory). Off-the-shelf rerankers are not free wins.

## Agent tools (Week 7, `pinpoint.tools`)

- Agent = LLM in a loop: model replies with a tool call (name + JSON args) instead of an answer, our
  code runs it, appends the result as a `role: "tool"` message, calls the model again. Stops on a
  final answer or the step limit. The model never runs anything; it only asks. The loop is ~30 lines.
- No memory between calls: every request resends the whole history, so cost grows ~quadratically
  with steps. Tool output size is the main cost lever -> caps (grep 50 rows, open_file 200 lines).
- Tools return errors as text, never raise: the model reads "invalid regex" and retries.
- Sandbox = resolve the path, then `is_relative_to(snapshot)`; catches `../`, absolute paths, symlinks.
- grep searches the same source files the retrievers rank (tests/docs out), so the agent and
  Pinpoint compete over the same candidate set.
- Strategy A (prompt ea214022b16c, Llama 3.3 70B, 15-step cap, pytest tune n=11): all 11 submitted on
  their own, mean 8.9 steps / 10.9 s / 24.6k prompt + 217 completion tokens ($0.10/query at $4/$20
  reference: ~10x Strategy 5b, ~$30 per 300). **File acc@1 = MRR .727, best file row** (dense .617).
  **Function MRR .200** (6116, 7373 at rank 1), below dense + graph .344.
- Why fn is weak: it submits ~2 ids, not 10, so acc@1 = acc@10 (no recall tail). Only 60% of ids are
  real functions: it names classes (`AssertionRewritingHook`, `Package`) and constants
  (`DEFAULT_LOG_FORMAT`). Typical miss = right file, wrong method (11143 `is_rewrite_disabled` vs `run`,
  5495 `assertrepr_compare` vs its helper `_compare_eq_sequence`). It stops at the first plausible hit.
- Caveat: Llama has likely seen pytest's source in training (see 5b "NAMED"), so the file gain may be
  partly memory. Check on other repos before trusting it.
- Strategy H (prompt 47109164b8a8 = A + seed list of dense+graph top-10 with line ranges): **function
  MRR .450, acc@5 .600, best function row** (seeds alone .333 / .400, A .200). File MRR .682 (A .727).
  Cheaper than A: 5.7 steps, 6.9 s, 15.4k prompt tokens, $0.065/query. Real-function ids 60% -> 96%.
- Seed rank -> agent rank: 5103 -/2 and 6116 -/1 (found outside the seeds), 7373 3/2, 5495 and 5692
  1/1; one loss 11143 1/2 (put `is_rewrite_disabled` above `run` again). 62% of submitted ids come from
  the seed list. 5227 submitted in 1 step without checking anything (rubber-stamping the seeds).
- Lesson: retrieval gives recall + real names; the agent verifies and follows one edge further.
  Seeding also cuts exploring, so it is better AND cheaper. n=10, one issue = .1 MRR: re-check at scale.
- MLflow tracing (`run_agent(trace=True)`, on for `python -m pinpoint.agent <id> [--seeds]`): one trace per
  issue = AGENT root -> RETRIEVER (seeds) -> CHAT_MODEL per step -> TOOL per call, tagged with
  instance_id / strategy / prompt_version. A span = one timed unit of work with inputs, outputs and
  attributes; parent links make the tree. Traces are written async: `flush_trace_async_logging()`
  before reading one back. Batch runs are not traced (cached transcripts are the record).
- Live traced 5221 (H): 9.5 s total, LLM calls ~9.4 s, all tools ~0.16 s: latency is ~98% the model.
  One open_file returned 9,665 chars and the prompt jumped 2,087 -> 4,663 tokens, and every later step
  pays for it again. Same issue/prompt/temperature 0 as the cached run, but different path (8 steps
  vs 6) and ranking: endpoint LLMs are not deterministic at T=0, which is why runs are cached.

### Week 7 failure analysis (Strategy H transcripts, cached runs; A compared)

| issue | gold | what the agent did | failure type |
|---|---|---|---|
| 5221 | `python.py::_showfixtures_main` | grepped the reporter's words ("fixture scope", "pytest --fixtures" = help text), never the option's definition (`"--fixtures"` -> `showfixtures`). H: read 2 small windows, then submitted seeds 2-4 as-is. | vocabulary gap + anchoring on wrong seeds |
| 7220 | `nodes.py::Node._repr_failure_py` | fixated on `chdir`, which the *reporter's* code calls, not the code that prints the path (relative to cwd). Also grepped `FSCollector._recurse` (qualnames aren't in source text). | symptom trigger mistaken for fix site |
| 8906 | `python.py::Module._importtestmodule` (572-617) | found the exact message at python.py:611 and opened 600-620, *inside the gold function*, but submitted class names `Module` / `CollectError`. | right line, wrong id: can't map a line to its enclosing function |
| 9359 | `_code/source.py::get_statement_startend2` | grepped "lineno" (capped at 50) and "python3.9", stopped after 6 of 15 steps. Never reasoned about the mechanism (statement line range, ast end_lineno), never opened `_code/source.py`. Seeds missed the file too. | shallow exploration, wrong subsystem |
| 11143 | `rewrite.py::AssertionRewriter.run` | the TypeError is raised in `is_rewrite_disabled`, so it ranked that first. H had `run` at seed rank 1 and opened 650-700 (line 678, the fix line) and still demoted it. | crash site vs fix site (fix is in the caller) |

- Cross-cutting: never uses the budget (5-8 of 15 steps), reads 10-20 line peepholes, submits 1-3 ids,
  and has no way to see which function a line belongs to (8906, 11143). Same issues fail in A and H.
- Fix ideas (A2/H2, one change at a time): (1) tools annotate every line with its enclosing
  `path::qualname` (we have it in snapshot_membership) -> fixes 8906 by construction; (2) prompt: the
  fix is often in the caller of the function that raises, rank callers too; submit 10 functions, not
  classes; (3) list the repo's CLI options/config names up front (5221).

### Week 7 fixes, scored one at a time (pytest tune, cumulative on H)

- Fix 1, labels (`tools.enclosing`: innermost function per line, same rule as ground truth; grep rows get
  `[path::qualname]`, open_file gets a `## id (lines a-b)` header at each function change): fn MRR
  .450 -> .500, file .682 -> .773, real-function ids 96% -> 100%. 8906 not ranked -> 2 (as predicted from
  its transcript), 7373 2 -> 1; one loss, 5103 2 -> not ranked. Cost unchanged ($0.065).
- Fix 2, prompt advice: fn unchanged (.500), file down (.712), ids per issue 2 -> 4, +20% tokens. It did
  NOT fix the cases it was written for (11143 still 2, 7220 still missed). Telling the model about a
  mistake is weaker than changing what it can see.
- Fix 3, options (only 5221 and 6116 mention a --option): 5221 not ranked -> 1 directly. fn .650.
- **Noise:** for the 8 scored issues without --options, run 3's input was byte-identical to run 2's, yet
  11143 moved 2 -> 1. So the honest attribution for fix 3 is .600, and one rank flip in ~8 issues per
  rerun is the noise floor at T=0 (~.05 MRR here). Gains smaller than ~.1 on n=10 are not evidence.
- A2 (no seeds, all fixes): fn .200 -> .500 (5103 found at 1, 8906 at 2, 11143 at 2), file .727 unchanged.
  It ignored the --fixtures hint on 5221. Seeds + fixes (H2, .650 / .600 honest) is the best function row.
- Still unsolved: 7220 (trigger vs bug) and 9359 (wrong subsystem): neither seeds nor tools reach them.

## Embeddings at scale: Databricks endpoint (scale-up prep, `embed.EndpointEmbedder`)

- 200 tune issues = 12 repos (django 76, sympy 51): ~100k-300k unique bodies; local bge-small on CPU
  (~3/s) would take about a day. Free Edition serves `databricks-bge-large-en` (1024-dim, 512-token
  truncation, same BGE query prefix), plus gte-large-en and qwen3-embedding-0.6b.
- Rate limit, measured: max 150 inputs per request, but on Free Edition any request with >= 32 inputs
  gets HTTP 429 REQUEST_LIMIT_EXCEEDED ("workspace QPS"); 16 per request, sequential, sustains
  ~24 texts/s (~500k tokens/min) with zero 429s. Batch 4 = 4.6 texts/s: per-request overhead dominates.
- Gotcha: the SDK's `api_client.do` silently retries 429s for up to 5 minutes, so a batch that can
  never succeed looks like a hang. Probe with raw `requests` (or a short retry_timeout) to see codes.
- Same-slice swap check (pytest tune, no header): bge-large file MRR .641 vs bge-small .617, function
  .180 vs .194. Within noise, so bge-large is a fair stand-in for scale. 5,843 bodies in 3m40s (~26/s).

## Recheck slice: do the pytest winners hold across repos? (2026-10-27 job run)

- Slice: `split.recheck_ids()`, 4 tune issues per repo + all 11 pytest = 52 (51 with a gold function).
  Job: build_tables -> embed (bge-large endpoint) -> evaluate -> `gold.eval_results` (per repo + "all").
- Dense wins both levels: file MRR .587, fn .232. The pytest-only winner, dense + call graph
  (fn .344 on pytest with bge-small), comes **last** across repos (fn .147). Overfit to one small slice.
- BM25 fn (.174) and RRF (.162) also lose to dense alone. RRF with a weaker partner pulls dense down.
- Lesson: n=10 on one repo picked the wrong winner. Always recheck on a stratified slice before trusting.

## Tune set: all 200 tune issues (2026-10-29 job run)

- Dense still wins both levels on 200 issues: file MRR .567, fn .295. Call graph last again (fn .168):
  dropping it as a re-ranker holds up.
- RRF moves from last (recheck .162) to second at function level (.262), with the best acc@10 (.522):
  fusion pulls more gold functions into the top 10 but lower down. Matters for agent seeds (top-10).
- Tune is 38% django (76) and 26% sympy (51); recheck is ~4 per repo. Tune = "SWE-bench as a whole",
  recheck = "a typical repo". Report both; they answer different questions.
- Runtime: first attempt killed by Free Edition quota (`RESOURCE_USAGE_BLOCKED`) after 5.5 h of evaluate,
  at issue ~176, all lost. Cause: 2 Spark queries per issue (fixed cost ~98 s of ~100 s per issue) + a
  Python UDF filter + no checkpoints. Fix (`jobs.evaluate`): 3 queries per 20-issue chunk, filter in pandas,
  full rankings checkpointed to the bronze volume per chunk (resume skips done chunks). Recheck reproduced
  all 91 eval_results rows exactly, 26 min vs ~1.5 h. Tune: 3 h 27 min, no quota kill.
- Still slow: ranking on the serverless driver (sympy ~170 s/issue vs ~9 s on the laptop, mostly BM25).
  Fetch is now 4-23 s per chunk. Not profiled yet.
- Lessons: Spark wants one big set-based query, not many small ones in a loop; Python UDFs leave the JVM
  per row; long jobs on a quota'd platform must checkpoint; verify a refactor by exact parity on old results.

## Why the call graph loses on the recheck slice (`scratch/recheck_graph.py`, 2026-10-28)

- Method: cloud per-issue rankings (new `gold.eval_rankings`, top-50 per strategy) + local call graph,
  replay `graph.expand` on the cloud dense_fn ranking. Replay is approximate (dense tail past 50 is cut,
  so neighbours lose their base score: MAIN .175 here vs .147 in the cloud); compare rows within the replay.
- **Flooding:** on average 6 of the top 10 are graph neighbours that dense didn't put there. The top-5 seeds
  have a median of 14 unambiguous (weight 1) neighbours; scikit-learn-10508 has 282 (gold 13 -> 291).
- **It evicts correct #1s:** dense's #1 survives in only 34% of issues. flask-4045 1 -> 6, flask-5063 1 -> 11,
  sympy-19254 1 -> 14. Any weight-1 neighbour of seed s scores base + 1/(60+rank s) > seed's own score.
- **Not a big-repo problem:** pytest itself, with bge-large seeds, is 3 gained / 4 lost. The old pytest win
  (.194 -> .344) came with bge-small seeds and n=10: a property of those seeds plus noise, not of pytest.
- Real gains exist but are rare (sympy-16988 30 -> 1, pytest-5495 15 -> 4): 8 gained, 24 lost of 51.
- No variant beats dense alone (MRR .230): alpha .25 .204, k=1 cap .209, callers .207 (best acc@1 .137, +1 issue).
- Decision: drop graph expansion as a re-ranker. Keep the call graph as information *for the agent*
  (a callers/callees tool), and seed the agent with dense top-10, not graph top-10 (agent_seeds is graph_fn today).

## Agent beyond pytest (recheck slice, 2026-10-28)

- The agent holds up across 12 repos, unlike the call graph: A2 fn MRR .510 (pytest-only A2f was .500),
  4x dense acc@1 (.451 vs .118). Django is the weak repo (fn .25-.40) in every variant.
- Seeds switched to dense top-10 (graph expansion lost on this slice). Seeds add a little: fn .510 -> .540,
  file .692 -> .760, and fewer steps (8.4 -> 6.7) and tokens (36k -> 27k), so cheaper ($0.15 -> $0.11).
- call_graph tool (the graph as information, not as a re-ranker): fn .540 -> .614, acc@5 .627 -> .725,
  file .760 -> .804. Used in 34/52 issues (0.7 calls/issue). Per issue: 10 up, 4 down.
  Of the 10 ups, 6 called the tool; 4 didn't (a different tool list changes the T=0 trajectory). Sign test
  10 vs 4: p ~ .18, so promising, not proven. Same signal as Strategy 6's rare big wins, minus the flooding.
- Lesson: the graph signal is real but needs judgment. Blind re-ranking promoted every neighbour; the agent
  asks for neighbours of the one function it suspects and reads them before ranking.
- Gotcha: function ids are not unique per snapshot (sympy @dispatch overloads share a qualname); seeds dedupe.

## Live path (`live.py`, 2026-10-28): any repo at any commit -> ranked functions

- checkout (clone once into data/live/repos, fetch, detach) -> index (tree-sitter, same hashes as silver)
  -> embed only unseen bodies (saves every 2,000, resumable) -> dense top-10 -> H3 agent. Invented ids dropped,
  empty answer falls back to dense.
- Parity: on 3 SWE-bench snapshots the live dense top-10 equals the cloud's eval_rankings top-10, same order.
- First post-cutoff issue (pytest #15052, filed 2026-09, fixed in PR #15053; run at the fix's parent commit):
  cold index of pytest HEAD-ish = 2,127 functions, 1,539 new bodies, ~60 s embedding; 99 s total, 8 s LLM.
  **Miss**: gold compat.py::getlocation (shared helper with the off-by-one); agent ranked the
  --fixtures-per-test code that calls it (trigger vs bug, like 7220), stopping at 7 of 15 steps.
- Gotcha: a fresh GitHub clone can die with "RPC failed; curl 56 Connection reset"; a retry worked.
