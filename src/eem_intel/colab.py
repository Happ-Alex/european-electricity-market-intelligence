from __future__ import annotations

from pathlib import Path
import io
import zipfile

import pandas as pd
import requests


def download_github_artifact_zip(url: str, destination: str | Path = "entsoe_artifact.zip") -> Path:
    """Download a GitHub Actions artifact ZIP from a browser-accessible URL.

    For private repositories or expiring API artifact URLs, use a GitHub token or
    manually upload the ZIP to Colab. This helper is intentionally credential-free.
    """
    destination = Path(destination)
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    destination.write_bytes(response.content)
    return destination


def extract_artifact(zip_path: str | Path, destination: str | Path = "data") -> Path:
    zip_path = Path(zip_path)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(destination)
    return destination


def load_master(root: str | Path, features: bool = False) -> pd.DataFrame:
    root = Path(root)
    filename = (
        "entsoe_features_hourly_2019_2026.parquet"
        if features
        else "entsoe_master_hourly_2019_2026.parquet"
    )
    candidates = list(root.rglob(filename))
    if not candidates:
        raise FileNotFoundError(f"Could not find {filename} below {root}")
    return pd.read_parquet(candidates[0])


def load_zone(root: str | Path, zone: str, features: bool = False) -> pd.DataFrame:
    df = load_master(root, features=features)
    return df.loc[df["zone"].eq(zone)].reset_index(drop=True)
