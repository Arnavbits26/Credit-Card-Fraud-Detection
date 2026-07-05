"""
preprocessing.py
=================
Data-cleaning and preprocessing utilities: missing-value handling,
duplicate removal, feature scaling, and stratified train/test
splitting for the fraud detection dataset.
"""

from __future__ import annotations

from typing import Tuple

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler

from src.utils import DATA_PROCESSED_DIR, MODELS_DIR, RANDOM_SEED, get_logger

logger = get_logger(__name__)


def check_missing_values(df: pd.DataFrame) -> pd.Series:
    """
    Report missing values per column.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    Returns
    -------
    pd.Series
        Count of missing values per column (only columns with > 0
        shown, sorted descending).
    """
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        logger.info("No missing values found in the dataset.")
    else:
        logger.warning("Missing values detected:\n%s", missing)
    return missing


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect and drop duplicate rows.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe, potentially containing duplicate rows.

    Returns
    -------
    pd.DataFrame
        Dataframe with duplicate rows removed (index reset).
    """
    n_before = len(df)
    n_duplicates = df.duplicated().sum()
    if n_duplicates > 0:
        logger.info("Found %d duplicate rows — removing them.", n_duplicates)
        df = df.drop_duplicates().reset_index(drop=True)
    else:
        logger.info("No duplicate rows found.")
    n_after = len(df)
    logger.info("Rows before: %d | after: %d", n_before, n_after)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the full cleaning pipeline: missing-value check + duplicate
    removal. Rows with missing values are dropped (this dataset is
    documented to have none, but the guard is kept for robustness).

    Parameters
    ----------
    df : pd.DataFrame
        Raw input dataframe.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe.
    """
    missing = check_missing_values(df)
    if not missing.empty:
        df = df.dropna().reset_index(drop=True)
    df = remove_duplicates(df)
    return df


def scale_features(
    df: pd.DataFrame,
    fit: bool = True,
    scaler_filename: str = "scaler.joblib",
) -> pd.DataFrame:
    """
    Scale the ``Amount`` and ``Time`` columns using RobustScaler
    (robust to the heavy outliers present in transaction amounts).
    The V1-V28 columns are already PCA-transformed / standardized by
    the dataset publisher and are left untouched.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing at least ``Amount`` and ``Time`` columns.
    fit : bool, optional
        If True, fits a new scaler and persists it to ``models/``.
        If False, loads the previously fitted scaler and only
        transforms (used at inference time to avoid data leakage).
    scaler_filename : str, optional
        File name for persisting/loading the fitted scaler.

    Returns
    -------
    pd.DataFrame
        Dataframe with ``Amount`` and ``Time`` replaced by their
        scaled versions (``scaled_amount``, ``scaled_time``), and the
        original raw columns dropped.
    """
    df = df.copy()
    scaler_path = MODELS_DIR / scaler_filename

    cols_to_scale = ["Amount", "Time"]

    if fit:
        scaler = RobustScaler()
        scaled = scaler.fit_transform(df[cols_to_scale])
        joblib.dump(scaler, scaler_path)
        logger.info("Fitted new RobustScaler and saved to %s", scaler_path)
    else:
        if not scaler_path.exists():
            raise FileNotFoundError(
                f"No fitted scaler found at {scaler_path}. Run preprocessing "
                "with fit=True first (i.e. during training)."
            )
        scaler = joblib.load(scaler_path)
        scaled = scaler.transform(df[cols_to_scale])
        logger.info("Loaded existing scaler from %s", scaler_path)

    df[["scaled_amount", "scaled_time"]] = scaled
    df = df.drop(columns=cols_to_scale)
    return df


def split_data(
    df: pd.DataFrame,
    target_col: str = "Class",
    test_size: float = 0.2,
    random_state: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Perform a stratified train/test split, preserving the original
    fraud-to-legitimate ratio in both splits — critical for a highly
    imbalanced dataset such as this one (~0.17% fraud).

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned, scaled dataframe including the target column.
    target_col : str, optional
        Name of the binary target column, by default ``"Class"``.
    test_size : float, optional
        Fraction of data held out for testing, by default 0.2.
    random_state : int, optional
        Seed for reproducibility, by default ``RANDOM_SEED``.

    Returns
    -------
    tuple
        ``(X_train, X_test, y_train, y_test)``.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    logger.info(
        "Split data -> train: %s (fraud rate %.4f%%) | test: %s (fraud rate %.4f%%)",
        X_train.shape,
        y_train.mean() * 100,
        X_test.shape,
        y_test.mean() * 100,
    )
    return X_train, X_test, y_train, y_test


def save_processed_splits(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> None:
    """
    Persist processed train/test splits to ``data/processed/`` as CSV
    files so downstream scripts (train.py, evaluate.py) don't need to
    repeat the cleaning/scaling/splitting steps.

    Parameters
    ----------
    X_train, X_test : pd.DataFrame
        Feature splits.
    y_train, y_test : pd.Series
        Target splits.

    Returns
    -------
    None
    """
    X_train.to_csv(DATA_PROCESSED_DIR / "X_train.csv", index=False)
    X_test.to_csv(DATA_PROCESSED_DIR / "X_test.csv", index=False)
    y_train.to_csv(DATA_PROCESSED_DIR / "y_train.csv", index=False)
    y_test.to_csv(DATA_PROCESSED_DIR / "y_test.csv", index=False)
    logger.info("Saved processed train/test splits to %s", DATA_PROCESSED_DIR)


def load_processed_splits() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Load previously saved processed splits from ``data/processed/``.

    Returns
    -------
    tuple
        ``(X_train, X_test, y_train, y_test)``.
    """
    X_train = pd.read_csv(DATA_PROCESSED_DIR / "X_train.csv")
    X_test = pd.read_csv(DATA_PROCESSED_DIR / "X_test.csv")
    y_train = pd.read_csv(DATA_PROCESSED_DIR / "y_train.csv").squeeze("columns")
    y_test = pd.read_csv(DATA_PROCESSED_DIR / "y_test.csv").squeeze("columns")
    return X_train, X_test, y_train, y_test
