"""
feature_engineering.py
=======================
Feature engineering and class-imbalance handling.

The V1-V28 features are already anonymized PCA components, so
domain-driven feature engineering is limited — but we still derive a
handful of statistically motivated features from ``Amount``/``Time``
and provide the three standard resampling strategies used to combat
extreme class imbalance:

- Random Undersampling
- Random Oversampling
- SMOTE (Synthetic Minority Oversampling Technique)

Class imbalance matters because with ~99.83% of transactions being
legitimate, a naive classifier that predicts "not fraud" for every
transaction achieves ~99.83% accuracy while catching zero fraud —
useless in production. Resampling (or cost-sensitive learning) forces
the model to pay attention to the minority (fraud) class.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from src.utils import RANDOM_SEED, get_logger

logger = get_logger(__name__)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a small set of derived features on top of the scaled
    Time/Amount columns:

    - ``amount_log``: log1p transform of the (unscaled) amount proxy,
      which compresses the heavy right-skew typical of transaction
      amounts and helps linear models such as Logistic Regression.
    - ``hour_of_day``: hour extracted from the ``Time`` feature
      (seconds elapsed since the first transaction), which can
      capture cyclical patterns in fraud occurrence.

    Note: this function expects the *unscaled* columns to still be
    present, so call it BEFORE ``preprocessing.scale_features`` in the
    pipeline (see ``train.py`` for exact ordering).

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing raw ``Amount`` and ``Time`` columns.

    Returns
    -------
    pd.DataFrame
        Dataframe with two additional engineered columns.
    """
    df = df.copy()
    df["amount_log"] = np.log1p(df["Amount"])
    # Time is seconds elapsed since the first transaction in the dataset.
    df["hour_of_day"] = (df["Time"] // 3600) % 24
    logger.info("Added engineered features: amount_log, hour_of_day")
    return df


def apply_random_undersampling(
    X: pd.DataFrame, y: pd.Series, random_state: int = RANDOM_SEED
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Randomly discard majority-class (legitimate) samples until the
    class distribution is balanced 1:1.

    Trade-off: fast and simple, but throws away a large amount of
    potentially useful legitimate-transaction information, which can
    hurt generalization and increase false positives.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Binary target.
    random_state : int, optional
        Seed for reproducibility.

    Returns
    -------
    tuple
        Resampled ``(X_resampled, y_resampled)``.
    """
    sampler = RandomUnderSampler(random_state=random_state)
    X_res, y_res = sampler.fit_resample(X, y)
    logger.info(
        "Random undersampling -> new class distribution: %s",
        y_res.value_counts().to_dict(),
    )
    return X_res, y_res


def apply_random_oversampling(
    X: pd.DataFrame, y: pd.Series, random_state: int = RANDOM_SEED
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Randomly duplicate minority-class (fraud) samples until the class
    distribution is balanced 1:1.

    Trade-off: retains all majority-class information, but duplicate
    fraud samples can cause the model to overfit to those exact
    points rather than learning generalizable fraud patterns.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Binary target.
    random_state : int, optional
        Seed for reproducibility.

    Returns
    -------
    tuple
        Resampled ``(X_resampled, y_resampled)``.
    """
    sampler = RandomOverSampler(random_state=random_state)
    X_res, y_res = sampler.fit_resample(X, y)
    logger.info(
        "Random oversampling -> new class distribution: %s",
        y_res.value_counts().to_dict(),
    )
    return X_res, y_res


def apply_smote(
    X: pd.DataFrame, y: pd.Series, random_state: int = RANDOM_SEED, k_neighbors: int = 5
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Apply SMOTE (Synthetic Minority Oversampling Technique), which
    generates *synthetic* fraud samples by interpolating between real
    fraud samples and their nearest minority-class neighbors, rather
    than duplicating existing points.

    Trade-off: generally produces better generalization than plain
    oversampling since it doesn't just clone points, but it can
    generate unrealistic synthetic samples in regions where the
    minority class is sparse or noisy, and is more computationally
    expensive than random resampling.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Binary target.
    random_state : int, optional
        Seed for reproducibility.
    k_neighbors : int, optional
        Number of nearest neighbors used to generate synthetic
        samples, by default 5.

    Returns
    -------
    tuple
        Resampled ``(X_resampled, y_resampled)``.
    """
    sampler = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
    X_res, y_res = sampler.fit_resample(X, y)
    logger.info(
        "SMOTE -> new class distribution: %s", y_res.value_counts().to_dict()
    )
    return X_res, y_res


RESAMPLING_STRATEGIES = {
    "none": lambda X, y: (X, y),
    "undersample": apply_random_undersampling,
    "oversample": apply_random_oversampling,
    "smote": apply_smote,
}


def resample(
    X: pd.DataFrame, y: pd.Series, strategy: str = "smote"
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Dispatch to the requested resampling strategy.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (training data only — never resample test data).
    y : pd.Series
        Binary target.
    strategy : str, optional
        One of ``"none"``, ``"undersample"``, ``"oversample"``,
        ``"smote"``. Default is ``"smote"``.

    Returns
    -------
    tuple
        Resampled ``(X, y)``.

    Raises
    ------
    ValueError
        If an unknown strategy name is passed.
    """
    if strategy not in RESAMPLING_STRATEGIES:
        raise ValueError(
            f"Unknown strategy '{strategy}'. Choose from {list(RESAMPLING_STRATEGIES)}"
        )
    return RESAMPLING_STRATEGIES[strategy](X, y)
