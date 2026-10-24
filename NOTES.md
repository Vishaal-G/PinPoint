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
