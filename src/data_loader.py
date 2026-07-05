"""
data_loader.py
==============
Responsible for locating, validating and loading the raw
Credit Card Fraud Detection dataset (Time, Amount, V1-V28, Class).

The dataset ("Credit Card Fraud Detection", ULB Machine Learning
Group, 284,807 transactions) is distributed via Kaggle and is subject
to Kaggle's terms of use, so it is NOT bundled with this repository.
This module gives you two ways to obtain it locally:

1. Manual download (recommended, no credentials needed):
   https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
   -> place ``creditcard.csv`` inside ``data/raw/``.

2. Programmatic download via the official ``kaggle`` CLI/API, provided
   you have a valid ``~/.kaggle/kaggle.json`` API token configured.
   See ``download_dataset()`` below.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

from src.utils import DATA_RAW_DIR, get_logger

logger = get_logger(__name__)

RAW_FILENAME = "creditcard.csv"
EXPECTED_COLUMNS = [
    "Time",
    *[f"V{i}" for i in range(1, 29)],
    "Amount",
    "Class",
]


def download_dataset(destination: Path = DATA_RAW_DIR) -> Path:
    """
    Attempt to download the dataset via the Kaggle API.

    This requires the ``kaggle`` package to be installed and a valid
    API token at ``~/.kaggle/kaggle.json``. If either is missing, an
    informative error is raised instructing the user to download the
    file manually instead.

    Parameters
    ----------
    destination : Path, optional
        Directory to download and unzip the dataset into,
        by default ``data/raw/``.

    Returns
    -------
    Path
        Path to the downloaded ``creditcard.csv`` file.
    """
    destination.mkdir(parents=True, exist_ok=True)
    target_csv = destination / RAW_FILENAME

    if target_csv.exists():
        logger.info("Dataset already present at %s — skipping download.", target_csv)
        return target_csv

    try:
        logger.info("Attempting download via Kaggle API...")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "kaggle",
                "datasets",
                "download",
                "-d",
                "mlg-ulb/creditcardfraud",
                "-p",
                str(destination),
                "--unzip",
            ],
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise RuntimeError(
            "Automatic download failed. Please download the dataset manually "
            "from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud and "
            f"place 'creditcard.csv' inside '{destination}'."
        ) from exc

    if not target_csv.exists():
        raise FileNotFoundError(
            f"Expected file not found at {target_csv} after download attempt."
        )

    logger.info("Dataset downloaded successfully to %s", target_csv)
    return target_csv


def validate_schema(df: pd.DataFrame) -> None:
    """
    Validate that a loaded dataframe matches the expected fraud
    dataset schema (Time, V1-V28, Amount, Class).

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to validate.

    Raises
    ------
    ValueError
        If any expected column is missing.
    """
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(
            f"Dataset is missing expected columns: {sorted(missing)}. "
            "Make sure you are using the official 'creditcard.csv' file."
        )


def load_raw_data(path: Path | str | None = None) -> pd.DataFrame:
    """
    Load the raw ``creditcard.csv`` dataset into a pandas DataFrame.

    Parameters
    ----------
    path : Path | str | None, optional
        Explicit path to the CSV file. If ``None``, defaults to
        ``data/raw/creditcard.csv``.

    Returns
    -------
    pd.DataFrame
        Raw dataset with columns Time, V1-V28, Amount, Class.

    Raises
    ------
    FileNotFoundError
        If the CSV file cannot be found at the resolved path.
    ValueError
        If the file exists but doesn't match the expected schema.
    """
    csv_path = Path(path) if path is not None else DATA_RAW_DIR / RAW_FILENAME

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Could not find dataset at {csv_path}.\n"
            "Download it from "
            "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud "
            f"and place 'creditcard.csv' inside '{DATA_RAW_DIR}', or call "
            "'download_dataset()' with a configured Kaggle API token."
        )

    logger.info("Loading raw dataset from %s", csv_path)
    df = pd.read_csv(csv_path)
    validate_schema(df)
    logger.info("Loaded dataset with shape %s", df.shape)
    return df


if __name__ == "__main__":
    # Allow `python -m src.data_loader` for a quick sanity check.
    try:
        data = load_raw_data()
        print(data.head())
        print(f"\nShape: {data.shape}")
    except FileNotFoundError as e:
        print(e)
