"""Read small Delta tables from the Databricks workspace into pandas, from the laptop.

Uses the SQL Statement Execution API on the workspace's SQL warehouse (Free Edition: the
"Serverless Starter Warehouse", which starts on the first query). Results come back inline as
JSON, so this is for small tables only (seeds, results), not for silver.

    uv run python -m pinpoint.cloud "SELECT * FROM workspace.pinpoint_gold.eval_results"
"""
import sys

import pandas as pd

CATALOG, PREFIX = "workspace", "pinpoint"


def _client():
    from databricks.sdk import WorkspaceClient  # local group only

    return WorkspaceClient(profile="pinpoint")


def query(sql: str, client=None) -> pd.DataFrame:
    """Run one SQL statement and return its rows. Waits for a cold warehouse to start."""
    from databricks.sdk.service.sql import Disposition, Format, StatementState

    w = client or _client()
    warehouse = next(iter(w.warehouses.list())).id
    r = w.statement_execution.execute_statement(statement=sql, warehouse_id=warehouse, wait_timeout="50s",
                                                disposition=Disposition.INLINE, format=Format.JSON_ARRAY)
    while r.status.state in (StatementState.PENDING, StatementState.RUNNING):
        r = w.statement_execution.get_statement(r.statement_id)
    if r.status.state != StatementState.SUCCEEDED:
        raise RuntimeError(f"{r.status.state}: {r.status.error.message if r.status.error else ''}")
    columns = [c.name for c in r.manifest.schema.columns]
    rows = list(r.result.data_array or [])
    for i in range(1, r.manifest.total_chunk_count or 1):  # results over one chunk come in pieces
        rows += w.statement_execution.get_statement_result_chunk_n(r.statement_id, i).data_array or []
    return pd.DataFrame(rows, columns=columns)


def agent_seeds(ids, strategy: str = "dense_fn", k: int = 10, client=None) -> dict[str, list[tuple[str, int, int]]]:
    """A strategy's top-k from gold.eval_rankings (the job's evaluate task), as agent.pinpoint_seeds
    returns them. Line ranges come from the local silver membership table (the agent runs locally)."""
    in_list = ", ".join(f"'{i}'" for i in sorted(set(ids)))
    df = query(f"SELECT instance_id, rank, item AS function_id FROM {CATALOG}.{PREFIX}_gold.eval_rankings "
               f"WHERE strategy = '{strategy}' AND rank <= {int(k)} AND instance_id IN ({in_list})", client)
    df = df.astype({"rank": int}).sort_values(["instance_id", "rank"])
    lines = pd.read_parquet("data/silver/snapshot_membership.parquet",
                            columns=["instance_id", "function_id", "start_line", "end_line"])
    df = df.merge(lines, on=["instance_id", "function_id"], how="left", validate="one_to_one")
    return {i: list(zip(g.function_id, g.start_line.astype(int), g.end_line.astype(int)))
            for i, g in df.groupby("instance_id")}


if __name__ == "__main__":
    print(query(sys.argv[1]).to_string(index=False))
