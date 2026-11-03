"""Entry points for the Databricks job (databricks.yml). Each runs as one task on serverless.

    ingest        SWE-bench Lite from Hugging Face -> <catalog>.<prefix>_bronze.issues
    build-tables  a slice's issues: clone + parse -> silver functions/edges/membership, gold ground_truth
    embed         candidate bodies not embedded yet -> silver.embeddings_<model> (Databricks endpoint)
    evaluate      non-LLM strategies on the slice -> gold.eval_results, gold.eval_rankings, MLflow runs
    sync-index    Delta Sync vector index over the embeddings table (for the API, not for scoring)

A slice is `--slice repo --repo <name>` (one repo), `--slice recheck` (52 stratified tune issues)
`--slice tune` (all 200), or `--slice heldout` (the 100 held-out issues: the final score, run once).

Every task is idempotent, so a retry or a second run gives the same tables:
  - content-addressed tables (functions, edges; key = body hash) only INSERT hashes they don't have,
  - per-issue tables (membership, ground_truth) replace exactly the rows of the issues being built.
"""
import argparse
import os
import tempfile
import time
from pathlib import Path

import pandas as pd

from pinpoint import pipeline
from pinpoint.embed import ENDPOINT_MODEL
from pinpoint.repos import is_source_file

# The file data.HF_URL points at, over plain https: pandas reads it without huggingface_hub.
HF_HTTPS_URL = "https://huggingface.co/datasets/SWE-bench/SWE-bench_Lite/resolve/main/data/test-00000-of-00001.parquet"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", default="workspace")
    parser.add_argument("--schema-prefix", default="pinpoint")
    parser.add_argument("--repo", default="pytest-dev/pytest")
    parser.add_argument("--slice", default="repo", choices=["repo", "recheck", "tune", "heldout"])
    parser.add_argument("--model", default=ENDPOINT_MODEL)
    parser.add_argument("--vs-endpoint", default="pinpoint")
    return parser.parse_args(argv)


def select_issues(issues: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """The slice's rows of the 300 issues."""
    from pinpoint.split import slice_ids

    if args.slice == "repo":
        keep = issues.repo == args.repo  # all of the repo's issues, as build-tables always did
    else:
        keep = issues.instance_id.isin(slice_ids(args.slice))
    return issues[keep].reset_index(drop=True)


def slice_label(args: argparse.Namespace) -> str:
    return args.repo if args.slice == "repo" else args.slice


def embeddings_table(args: argparse.Namespace) -> str:
    """e.g. workspace.pinpoint_silver.embeddings_databricks_bge_large_en"""
    return table_name(args, "silver", "embeddings_" + args.model.replace("-", "_").replace("/", "_"))


def table_name(args: argparse.Namespace, layer: str, name: str) -> str:
    return f"{args.catalog}.{args.schema_prefix}_{layer}.{name}"


def in_list(values) -> str:
    """SQL IN list for ids we control (SWE-bench instance ids: letters, digits, _ and -)."""
    return ", ".join(f"'{v}'" for v in sorted(set(values)))


def _spark():
    from pyspark.sql import SparkSession  # only exists on Databricks, not in the local env

    return SparkSession.builder.getOrCreate()


def plain_dtypes(pdf: pd.DataFrame) -> pd.DataFrame:
    """Non-numeric columns as Python objects. pandas 3 keeps strings in pyarrow arrays, and a
    concat makes them chunked, which pyspark's createDataFrame can't convert (first job run)."""
    keep = lambda t: pd.api.types.is_numeric_dtype(t) or pd.api.types.is_bool_dtype(t)  # noqa: E731
    return pdf.astype({c: object for c, t in pdf.dtypes.items() if not keep(t)})


def _insert_new(spark, pdf: pd.DataFrame, table: str, key: str, schema: str | None = None) -> None:
    """Insert rows whose `key` the table doesn't have yet. For content-addressed rows a known
    key means identical content, so there is nothing to update. `schema` (a DDL string) pins
    column types that inference would get wrong, e.g. array<float> instead of array<double>."""
    pdf = plain_dtypes(pdf)
    if not spark.catalog.tableExists(table):
        spark.createDataFrame(pdf, schema).write.saveAsTable(table)
        print(f"{table}: created with {len(pdf)} rows")
        return
    spark.createDataFrame(pdf, schema).createOrReplaceTempView("incoming")
    result = spark.sql(f"""
        MERGE INTO {table} AS t USING incoming AS s ON t.{key} = s.{key}
        WHEN NOT MATCHED THEN INSERT *
    """).collect()[0].asDict()
    print(f"{table}: {len(pdf)} rows built, {result.get('num_inserted_rows')} new")


def _replace_issues(spark, pdf: pd.DataFrame, table: str) -> None:
    """Overwrite only the rows of these issues; other repos' rows are untouched."""
    pdf = plain_dtypes(pdf)
    writer = spark.createDataFrame(pdf).write
    if spark.catalog.tableExists(table):
        writer = writer.mode("overwrite").option("replaceWhere", f"instance_id IN ({in_list(pdf.instance_id)})")
    writer.saveAsTable(table)
    print(f"{table}: {len(pdf)} rows for {pdf.instance_id.nunique()} issues")


def ingest(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    spark = _spark()
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {args.catalog}.{args.schema_prefix}_bronze")
    issues = pd.read_parquet(HF_HTTPS_URL)
    table = table_name(args, "bronze", "issues")
    spark.createDataFrame(issues).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(table)
    print(f"{table}: {len(issues)} issues")


def build_tables(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    spark = _spark()
    for layer in ("silver", "gold"):
        spark.sql(f"CREATE SCHEMA IF NOT EXISTS {args.catalog}.{args.schema_prefix}_{layer}")
    issues = select_issues(spark.read.table(table_name(args, "bronze", "issues")).toPandas(), args)
    print(f"{slice_label(args)}: {len(issues)} issues in {issues.repo.nunique()} repos")

    # repos.py clones into relative data/ paths: give them the task's local disk, not a volume
    # (git worktrees need a real filesystem, and clones are cheap to rebuild).
    os.chdir(tempfile.mkdtemp(prefix="pinpoint-"))
    tables = pipeline.build_tables(issues)

    _insert_new(spark, tables["functions"], table_name(args, "silver", "functions"), key="hash")
    _insert_new(spark, tables["edges"], table_name(args, "silver", "edges"), key="hash")
    _replace_issues(spark, tables["snapshot_membership"], table_name(args, "silver", "snapshot_membership"))
    _replace_issues(spark, tables["ground_truth"], table_name(args, "gold", "ground_truth"))


def _candidate_sql(args: argparse.Namespace, ids) -> str:
    """Membership rows of these issues that are search candidates (source files only)."""
    return f"""SELECT m.* FROM {table_name(args, "silver", "snapshot_membership")} m
               WHERE m.instance_id IN ({in_list(ids)}) AND is_source_file(m.path)"""


def _register_udfs(spark) -> None:
    from pyspark.sql.types import BooleanType

    from pinpoint.repos import is_source_file

    spark.udf.register("is_source_file", is_source_file, BooleanType())


def _workspace():
    from databricks.sdk import WorkspaceClient  # preinstalled on serverless; auth comes from the task

    return WorkspaceClient()


EMBED_CHUNK = 1600  # bodies per MERGE: a crash loses at most one chunk (~70 s of endpoint calls)


def embed(argv: list[str] | None = None) -> None:
    """Embed every candidate body of the slice that the embeddings table doesn't have yet.
    Re-running only embeds what is missing, so a failed run resumes where it stopped."""
    from pinpoint.embed import EndpointEmbedder

    args = parse_args(argv)
    spark = _spark()
    _register_udfs(spark)
    issues = select_issues(spark.read.table(table_name(args, "bronze", "issues")).toPandas(), args)
    table = embeddings_table(args)
    spark.sql(_candidate_sql(args, issues.instance_id)).createOrReplaceTempView("cands")
    known = f"AND f.hash NOT IN (SELECT hash FROM {table})" if spark.catalog.tableExists(table) else ""
    todo = spark.sql(f"""
        SELECT DISTINCT f.hash, f.source FROM {table_name(args, "silver", "functions")} f
        JOIN cands c ON c.hash = f.hash WHERE true {known}""").toPandas()
    total = spark.sql("SELECT COUNT(DISTINCT hash) AS n FROM cands").collect()[0].n
    print(f"{slice_label(args)}: {total} candidate bodies, {len(todo)} to embed with {args.model}", flush=True)

    embedder = EndpointEmbedder(args.model, cache_dir=Path(tempfile.mkdtemp()), client=_workspace())
    t0 = time.perf_counter()
    for i in range(0, len(todo), EMBED_CHUNK):
        chunk = todo.iloc[i:i + EMBED_CHUNK]
        vecs = embedder.encode(chunk.source.tolist())
        rows = pd.DataFrame({"hash": chunk.hash.values, "embedding": [v.tolist() for v in vecs]})
        _insert_new(spark, rows, table, key="hash", schema="hash string, embedding array<float>")
        embedder.cache.clear()  # the table is the cache here; keep the driver's memory flat
        done = i + len(chunk)
        print(f"  {done}/{len(todo)} embedded, {done / (time.perf_counter() - t0):.1f}/s", flush=True)
    if spark.catalog.tableExists(table):
        # Delta Sync vector indexes read the table's change feed (sync-index task).
        spark.sql(f"ALTER TABLE {table} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")


EVAL_CHUNK = 20  # issues per Spark round trip and per checkpoint file


def eval_chunks(issues: pd.DataFrame, size: int = EVAL_CHUNK) -> list[pd.DataFrame]:
    """The slice in a fixed order, at most `size` issues per chunk and one repo per chunk (a
    repo's snapshots share most bodies, so a chunk's bodies dedupe well). Same slice, same chunks:
    that is what lets a re-run skip the chunks a killed run already checkpointed."""
    issues = issues.sort_values("instance_id")
    return [group.iloc[i:i + size] for _, group in issues.groupby("repo", sort=True)
            for i in range(0, len(group), size)]


def chunk_candidates(members: pd.DataFrame, bodies: pd.DataFrame) -> pd.DataFrame:
    """Source-file membership rows joined with their bodies (hash, source), in the row order the
    per-issue rankings depend on (stable sorts break ties by it). Inner join, like the SQL it replaced."""
    return members.merge(bodies, on="hash").sort_values(["instance_id", "path", "start_line"]).reset_index(drop=True)


def save_checkpoint(directory: Path, chunk_id: str, ranked: dict[str, dict[str, list[str]]]) -> None:
    """Full rankings of one chunk, long format. Written under a temp name and renamed, so a
    kill mid-write never leaves a file that looks finished."""
    from pinpoint.evaluate import rankings_table

    directory.mkdir(parents=True, exist_ok=True)
    part = directory / f"{chunk_id}.parquet.part"
    rankings_table(ranked, k=None).to_parquet(part, index=False)
    part.rename(directory / f"{chunk_id}.parquet")


def load_checkpoints(directory: Path) -> tuple[set[str], dict[str, dict[str, list[str]]]]:
    """(chunk ids already done, their rankings), the inverse of save_checkpoint."""
    from pinpoint.evaluate import STRATEGIES

    ranked = {name: {} for name in STRATEGIES}
    files = sorted(directory.glob("*.parquet")) if directory.exists() else []
    for f in files:
        long = pd.read_parquet(f).sort_values(["strategy", "instance_id", "rank"])
        for (name, iid), group in long.groupby(["strategy", "instance_id"], sort=False):
            ranked[name][iid] = group["item"].tolist()
    return {f.name.removesuffix(".parquet") for f in files}, ranked


def _checkpoint_dir(args: argparse.Namespace) -> Path:
    """One folder per (slice, model) in the bronze volume; deleted once the task succeeds."""
    name = f"evaluate-{slice_label(args)}-{args.model}".replace("/", "_")
    return Path(f"/Volumes/{args.catalog}/{args.schema_prefix}_bronze/raw/checkpoints/{name}")


def evaluate(argv: list[str] | None = None) -> None:
    """Score the non-LLM strategies on the slice, one chunk of issues at a time.

    Three Spark queries per chunk (membership, bodies + vectors, edges), not two per issue: each
    query is a distributed job with seconds of fixed cost, and per issue that cost was ~98% of the
    first tune run. Each finished chunk is checkpointed to the volume, so a run killed by the
    Free Edition quota resumes where it stopped instead of starting over."""
    import shutil

    import mlflow
    import numpy as np

    from pinpoint import evaluate as ev
    from pinpoint.embed import EndpointEmbedder
    from pinpoint.tracking import per_issue

    args = parse_args(argv)
    spark = _spark()
    issues = select_issues(spark.read.table(table_name(args, "bronze", "issues")).toPandas(), args)
    embedder = EndpointEmbedder(args.model, cache_dir=Path(tempfile.mkdtemp()), client=_workspace())
    checkpoints = _checkpoint_dir(args)
    spark.sql(f"CREATE VOLUME IF NOT EXISTS {args.catalog}.{args.schema_prefix}_bronze.raw")
    done, ranked = load_checkpoints(checkpoints)
    chunks = eval_chunks(issues)
    print(f"{slice_label(args)}: {len(issues)} issues in {len(chunks)} chunks, {len(done)} already done", flush=True)
    for chunk in chunks:
        chunk_id = chunk.instance_id.iloc[0]
        if chunk_id in done:
            continue
        t0 = time.perf_counter()
        members = spark.sql(f"""SELECT * FROM {table_name(args, "silver", "snapshot_membership")}
                                WHERE instance_id IN ({in_list(chunk.instance_id)})""").toPandas()
        members = members[members.path.map(is_source_file)]  # in pandas: a Python UDF is slow in Spark
        spark.createDataFrame(pd.DataFrame({"hash": members.hash.unique()})).createOrReplaceTempView("chunk_hashes")
        bodies = spark.sql(f"""
            SELECT f.hash, f.source, e.embedding FROM chunk_hashes h
            JOIN {table_name(args, "silver", "functions")} f ON f.hash = h.hash
            JOIN {embeddings_table(args)} e ON e.hash = h.hash""").toPandas()
        edges = spark.sql(f"""SELECT e.* FROM {table_name(args, "silver", "edges")} e
                              JOIN chunk_hashes h ON e.hash = h.hash""").toPandas()
        vectors = {h: np.asarray(v, dtype=np.float32) for h, v in zip(bodies.hash, bodies.embedding)}
        query_vecs = {i: embedder.encode_query(t) for i, t in zip(chunk.instance_id, chunk.problem_statement)}
        t1 = time.perf_counter()
        one = ev.rankings(chunk, chunk_candidates(members, bodies.drop(columns="embedding")), vectors,
                          query_vecs, edges)
        save_checkpoint(checkpoints, chunk_id, one)
        for name in ranked:
            ranked[name].update(one[name])
        print(f"chunk {chunk_id} ({len(chunk)} issues, {len(bodies)} bodies): fetch + queries {t1 - t0:.0f} s, "
              f"rank {time.perf_counter() - t1:.0f} s", flush=True)

    gt = spark.read.table(table_name(args, "gold", "ground_truth")).toPandas()
    table = ev.results_table(ranked, gt[gt.instance_id.isin(issues.instance_id)])
    run_at = pd.Timestamp.now(tz="UTC").isoformat()
    table = table.assign(slice=slice_label(args), model=args.model, run_at=run_at)
    print(table[table.repo == "all"].to_string(index=False))
    spark.createDataFrame(plain_dtypes(table)).write.mode("append").saveAsTable(table_name(args, "gold", "eval_results"))
    long = ev.rankings_table(ranked).assign(model=args.model, run_at=run_at)
    _replace_issues(spark, long, table_name(args, "gold", "eval_rankings"))

    me = _workspace().current_user.me().user_name
    mlflow.set_tracking_uri("databricks")
    mlflow.set_experiment(f"/Users/{me}/pinpoint")
    for name, level in ev.STRATEGIES.items():
        rows = table[table.strategy == name]
        overall = rows[rows.repo == "all"].iloc[0]
        with mlflow.start_run(run_name=f"{name}-{slice_label(args)}"):
            mlflow.log_params({"strategy": name, "level": level, "slice": slice_label(args), "model": args.model,
                               "n": int(overall.n)})
            mlflow.log_metrics({m.replace("@", "_at_"): float(overall[m]) for m in ev.METRICS})
            mlflow.log_text(rows.to_csv(index=False), "per_repo.csv")
            col = "gold_files" if level == "file" else "gold_functions"
            gold = {i: g for i, g in zip(gt.instance_id, gt[col]) if i in ranked[name]}
            mlflow.log_text(per_issue(ranked[name], gold).to_csv(index=False), "per_issue.csv")
    shutil.rmtree(checkpoints)  # results are in the tables now; the next run starts fresh


def sync_index(argv: list[str] | None = None) -> None:
    """Create the Delta Sync index over the embeddings table if it is missing, then sync it.
    Scoring doesn't use it (exact search per snapshot is cheap); the Week 8 API will."""
    from databricks.sdk.service.vectorsearch import (DeltaSyncVectorIndexSpecRequest, EmbeddingVectorColumn,
                                                     PipelineType, VectorIndexType)

    args = parse_args(argv)
    w = _workspace()
    source = embeddings_table(args)
    index = source + "_index"
    existing = {i.name for i in w.vector_search_indexes.list_indexes(endpoint_name=args.vs_endpoint)}
    if index not in existing:
        w.vector_search_indexes.create_index(
            name=index, endpoint_name=args.vs_endpoint, primary_key="hash", index_type=VectorIndexType.DELTA_SYNC,
            delta_sync_index_spec=DeltaSyncVectorIndexSpecRequest(
                source_table=source, pipeline_type=PipelineType.TRIGGERED,
                embedding_vector_columns=[EmbeddingVectorColumn(name="embedding", embedding_dimension=1024)]))
        print(f"{index}: created on endpoint {args.vs_endpoint} (first sync starts by itself)")
        return
    w.vector_search_indexes.sync_index(index_name=index)
    print(f"{index}: sync triggered")
