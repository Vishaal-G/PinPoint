import pandas as pd
from pathlib import Path

HF_URL = "hf://datasets/SWE-bench/SWE-bench_Lite/data/test-00000-of-00001.parquet"


BRONZE_PATH = Path("data/bronze/issues.parquet")

def download_issues():
    """Download the issues dataset from the Hugging Face Hub and save it as a Parquet file."""
    df = pd.read_parquet(HF_URL)
    BRONZE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(BRONZE_PATH, index=False)

def load_issues(repo : str | None = None) -> pd.DataFrame:
    """Load the issues dataset from the local Parquet file."""
    if not BRONZE_PATH.exists():
        download_issues()

    df = pd.read_parquet(BRONZE_PATH)
    if repo is not None:
        return df[df["repo"] == repo].reset_index(drop=True)
    return df

if __name__ == "__main__":
    download_issues()