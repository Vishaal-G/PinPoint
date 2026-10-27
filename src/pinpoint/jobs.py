"""Entry points for the Databricks job (databricks.yml). Each runs as one task on serverless.

    ingest        SWE-bench Lite from Hugging Face -> <catalog>.<prefix>_bronze.issues
    build-tables  one repo's issues: clone + parse -> silver functions/edges/membership, gold ground_truth

Every task is idempotent, so a retry or a second run gives the same tables:
  - content-addressed tables (functions, edges; key = body hash) only INSERT hashes they don't have,
  - per-issue tables (membership, ground_truth) replace exactly the rows of the issues being built.
"""
import argparse
import os
import tempfile

import pandas as pd

from pinpoint import pipeline

# The file data.HF_URL points at, over plain https: pandas reads it without huggingface_hub.
HF_HTTPS_URL = "https://huggingface.co/datasets/SWE-bench/SWE-bench_Lite/resolve/main/data/test-00000-of-00001.parquet"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", default="workspace")
    parser.add_argument("--schema-prefix", default="pinpoint")
    parser.add_argument("--repo", default="pytest-dev/pytest")
    return parser.parse_args(argv)


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


def _insert_new(spark, pdf: pd.DataFrame, table: str, key: str) -> None:
    """Insert rows whose `key` the table doesn't have yet. For content-addressed rows a known
    key means identical content, so there is nothing to update."""
    pdf = plain_dtypes(pdf)
    if not spark.catalog.tableExists(table):
        spark.createDataFrame(pdf).write.saveAsTable(table)
        print(f"{table}: created with {len(pdf)} rows")
        return
    spark.createDataFrame(pdf).createOrReplaceTempView("incoming")
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
    issues = (spark.read.table(table_name(args, "bronze", "issues"))
                   .filter(f"repo = '{args.repo}'").toPandas())
    print(f"{args.repo}: {len(issues)} issues")

    # repos.py clones into relative data/ paths: give them the task's local disk, not a volume
    # (git worktrees need a real filesystem, and clones are cheap to rebuild).
    os.chdir(tempfile.mkdtemp(prefix="pinpoint-"))
    tables = pipeline.build_tables(issues)

    _insert_new(spark, tables["functions"], table_name(args, "silver", "functions"), key="hash")
    _insert_new(spark, tables["edges"], table_name(args, "silver", "edges"), key="hash")
    _replace_issues(spark, tables["snapshot_membership"], table_name(args, "silver", "snapshot_membership"))
    _replace_issues(spark, tables["ground_truth"], table_name(args, "gold", "ground_truth"))
