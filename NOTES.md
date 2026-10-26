# Notes

Three takeaways per library, in my own words.

## uv, pytest (Week 0)

- Why does pytest find `test_something.py` but not `something_test_helper.py`?

## Results table (tune set only; held-out is untouched until Week 9)

Split: `splits/split.csv`, 200 tune / 100 held-out, stratified by repo, seed 20261022.
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
| 3cf | same | function | .100 | .300 | .500 | — | 10 |
| 4k | Databricks FULL_TEXT (whole issue text over function source), n=200, post-filter | file | .091 | .273 | .364 | .215 | 11 |
| 4kf | same | function | .000 | .200 | .300 | — | 10 |
| 4 | Databricks HYBRID (ANN + FULL_TEXT, rank-fused), n=200, post-filter | file | .091 | .727 | .727 | .357 | 11 |
| 4f | same | function | .000 | .400 | .500 | — | 10 |
| 1f | BM25 whole tokens over each function's source (silver), per snapshot | function | .200 | .300 | .300 | .257 | 10 |
| 4L | Our hybrid: RRF(BM25 whole file, dense Strategy 3 file), k=60, equal weights | file | .455 | .727 | .727 | .561 | 11 |
| 4Lf | Our hybrid: RRF(BM25 function, dense function), k=60, equal weights | function | .100 | .300 | .400 | .199 | 10 |

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
