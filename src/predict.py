"""
predict.py
==========
Inference-time pipeline: takes raw, unlabeled transaction data (same
schema as the training data minus ``Class``) and returns fraud
probability + binary prediction using a previously trained model and
the persisted scaler — WITHOUT ever re-fitting anything, to avoid
data leakage.

Usage
-----
    python -m src.predict --input path/to/new_transactions.csv --model random_forest
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.data_loader import EXPECTED_COLUMNS
from src.feature_engineering import add_engineered_features
from src.preprocessing import scale_features
from src.utils import MODELS_DIR, get_logger, load_model

logger = get_logger(__name__)

DEFAULT_MODEL_NAME = "random_forest"


def validate_input_columns(df: pd.DataFrame) -> None:
    """
    Validate that incoming data has the required raw columns
    (Time, V1-V28, Amount) — ``Class`` is optional since it won't be
    present for genuinely unseen data.

    Parameters
    ----------
    df : pd.DataFrame
        Raw input dataframe to validate.

    Raises
    ------
    ValueError
        If required columns are missing.
    """
    required = [c for c in EXPECTED_COLUMNS if c != "Class"]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Input data is missing required columns: {sorted(missing)}")


def prepare_inference_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the exact same feature engineering + scaling used at
    training time, but with ``fit=False`` so the previously fitted
    scaler is reused (never refit at inference time).

    Parameters
    ----------
    df : pd.DataFrame
        Raw input dataframe with Time, V1-V28, Amount columns.

    Returns
    -------
    pd.DataFrame
        Fully prepared feature matrix ready to feed into a trained model.
    """
    validate_input_columns(df)
    df = df.copy()

    if "Class" in df.columns:
        df = df.drop(columns=["Class"])

    df = add_engineered_features(df)
    df = scale_features(df, fit=False)
    return df


def predict_fraud(
    df: pd.DataFrame,
    model_name: str = DEFAULT_MODEL_NAME,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Run the full inference pipeline on a batch of raw transactions
    and return predictions alongside fraud probability.

    Parameters
    ----------
    df : pd.DataFrame
        Raw transaction data (Time, V1-V28, Amount [, Class]).
    model_name : str, optional
        Name of the persisted model to load (without extension),
        by default ``"random_forest"``.
    threshold : float, optional
        Probability threshold above which a transaction is flagged as
        fraud, by default 0.5. Lower this to increase recall at the
        cost of more false positives (common in fraud triage systems
        where missing fraud is costlier than a false alarm).

    Returns
    -------
    pd.DataFrame
        Original dataframe with two new columns:
        ``fraud_probability`` and ``predicted_class``.
    """
    model = load_model(f"{model_name}.joblib")
    features = prepare_inference_features(df)

    proba = model.predict_proba(features)[:, 1]
    predictions = (proba >= threshold).astype(int)

    result = df.copy()
    result["fraud_probability"] = np.round(proba, 6)
    result["predicted_class"] = predictions

    n_flagged = int(predictions.sum())
    logger.info(
        "Scored %d transactions using '%s' — %d flagged as fraud (threshold=%.2f)",
        len(df),
        model_name,
        n_flagged,
        threshold,
    )
    return result


def main() -> None:
    """CLI entry point for scoring a CSV file of new transactions."""
    parser = argparse.ArgumentParser(description="Score transactions for fraud probability.")
    parser.add_argument("--input", type=str, required=True, help="Path to input CSV file.")
    parser.add_argument(
        "--output", type=str, default=None, help="Path to save scored CSV (default: alongside input)."
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL_NAME, help="Model name (without .joblib)."
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="Fraud probability threshold."
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    df = pd.read_csv(input_path)

    scored = predict_fraud(df, model_name=args.model, threshold=args.threshold)

    output_path: Optional[Path] = (
        Path(args.output) if args.output else input_path.with_name(input_path.stem + "_scored.csv")
    )
    scored.to_csv(output_path, index=False)
    logger.info("Saved scored predictions to %s", output_path)


if __name__ == "__main__":
    main()
