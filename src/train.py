"""
train.py
========
End-to-end training entry point for the fraud detection pipeline.

Running this script will:

1. Load and clean the raw dataset.
2. Engineer features and scale Amount/Time.
3. Split into stratified train/test sets.
4. Resample the TRAINING set only (SMOTE by default) to combat class
   imbalance — the test set is left untouched so evaluation reflects
   real-world class distribution.
5. Train Logistic Regression, Random Forest, Gradient Boosting, and
   (if installed) XGBoost, each with cross-validation.
6. Optionally run RandomizedSearchCV for hyperparameter tuning.
7. Persist fitted models to ``models/`` and metrics to ``reports/``.

Usage
-----
    python -m src.train
    python -m src.train --tune            # enable hyperparameter search
    python -m src.train --resampling none # disable resampling
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, Tuple

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score

from src.data_loader import load_raw_data
from src.feature_engineering import add_engineered_features, add_kmeans_cluster_features, resample
from src.preprocessing import clean_data, save_processed_splits, scale_features, split_data
from src.utils import RANDOM_SEED, get_logger, measure_time, save_json, save_model, set_global_seed

logger = get_logger(__name__)

try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except ImportError:  # pragma: no cover
    XGBOOST_AVAILABLE = False
    logger.warning("xgboost is not installed — skipping XGBoost model.")


# --------------------------------------------------------------------------- #
# Model registry
# --------------------------------------------------------------------------- #
def get_model_registry(random_state: int = RANDOM_SEED) -> Dict[str, Any]:
    """
    Build the dictionary of models to train, each with sensible,
    documented default hyperparameters.

    Parameters
    ----------
    random_state : int, optional
        Seed forwarded to every model that accepts one.

    Returns
    -------
    dict
        Mapping of model name -> unfitted estimator instance.
    """
    registry: Dict[str, Any] = {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            C=1.0,  # inverse regularization strength; 1.0 is a balanced default
            class_weight=None,  # resampling already balances classes
            random_state=random_state,
            n_jobs=-1,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,  # enough trees for stable estimates without excessive cost
            max_depth=12,  # cap depth to reduce overfitting on synthetic SMOTE samples
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=3,  # shallow trees are standard for boosting to avoid overfitting
            random_state=random_state,
        ),
    }

    if XGBOOST_AVAILABLE:
        registry["xgboost"] = XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        )

    return registry


# --------------------------------------------------------------------------- #
# Hyperparameter search spaces (for RandomizedSearchCV)
# --------------------------------------------------------------------------- #
PARAM_DISTRIBUTIONS: Dict[str, Dict[str, list]] = {
    "logistic_regression": {
        "C": [0.01, 0.1, 1.0, 10.0, 100.0],
        "penalty": ["l2"],
        "solver": ["lbfgs", "liblinear"],
    },
    "random_forest": {
        "n_estimators": [100, 200, 300],
        "max_depth": [8, 12, 16, None],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    },
    "gradient_boosting": {
        "n_estimators": [100, 150, 200],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [2, 3, 4],
    },
}
if XGBOOST_AVAILABLE:
    PARAM_DISTRIBUTIONS["xgboost"] = {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [3, 5, 7],
        "subsample": [0.6, 0.8, 1.0],
    }


def tune_model(
    name: str,
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_iter: int = 15,
    cv_folds: int = 3,
    random_state: int = RANDOM_SEED,
) -> Any:
    """
    Run RandomizedSearchCV over the predefined parameter distribution
    for the given model, scoring on average precision (a.k.a. PR AUC)
    which is far more informative than accuracy on imbalanced data.

    Parameters
    ----------
    name : str
        Model key matching ``PARAM_DISTRIBUTIONS``.
    model : Any
        Unfitted estimator instance.
    X_train, y_train : pd.DataFrame, pd.Series
        Training data (already resampled if applicable).
    n_iter : int, optional
        Number of random parameter combinations to try, by default 15.
    cv_folds : int, optional
        Number of stratified CV folds, by default 3.
    random_state : int, optional
        Seed for reproducibility.

    Returns
    -------
    Any
        The best estimator found (already refit on the full training set).
    """
    if name not in PARAM_DISTRIBUTIONS:
        logger.info("No parameter distribution defined for '%s' — skipping tuning.", name)
        return model

    logger.info("Running RandomizedSearchCV for '%s'...", name)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=PARAM_DISTRIBUTIONS[name],
        n_iter=n_iter,
        scoring="average_precision",
        cv=cv,
        random_state=random_state,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_train, y_train)
    logger.info(
        "Best params for '%s': %s (PR-AUC=%.4f)",
        name,
        search.best_params_,
        search.best_score_,
    )
    return search.best_estimator_


def cross_validate_model(
    model: Any, X_train: pd.DataFrame, y_train: pd.Series, cv_folds: int = 5
) -> Dict[str, float]:
    """
    Evaluate a model with stratified k-fold cross-validation on the
    training set using average precision (PR AUC) as the scoring
    metric, appropriate for highly imbalanced classification.

    Parameters
    ----------
    model : Any
        Unfitted (or fitted — refit internally by CV) estimator.
    X_train, y_train : pd.DataFrame, pd.Series
        Training data.
    cv_folds : int, optional
        Number of folds, by default 5.

    Returns
    -------
    dict
        ``{"cv_mean_pr_auc": float, "cv_std_pr_auc": float}``.
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_SEED)
    scores = cross_val_score(
        model, X_train, y_train, cv=cv, scoring="average_precision", n_jobs=-1
    )
    logger.info(
        "CV PR-AUC for %s: mean=%.4f std=%.4f",
        model.__class__.__name__,
        scores.mean(),
        scores.std(),
    )
    return {"cv_mean_pr_auc": float(scores.mean()), "cv_std_pr_auc": float(scores.std())}


def build_training_data(
    resampling_strategy: str = "smote",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Run the full data preparation pipeline: load -> clean -> engineer
    features -> scale -> split -> resample (train only).

    Parameters
    ----------
    resampling_strategy : str, optional
        One of ``"none"``, ``"undersample"``, ``"oversample"``,
        ``"smote"``. Default ``"smote"``.

    Returns
    -------
    tuple
        ``(X_train_resampled, X_test, y_train_resampled, y_test)``.
    """
    df = load_raw_data()
    df = clean_data(df)
    df = add_engineered_features(df)
    df = scale_features(df, fit=True)
    df = add_kmeans_cluster_features(df, n_clusters=5, fit=True)  # adds cluster_id + dist_to_centroid

    X_train, X_test, y_train, y_test = split_data(df)
    save_processed_splits(X_train, X_test, y_train, y_test)

    X_train_res, y_train_res = resample(X_train, y_train, strategy=resampling_strategy)
    return X_train_res, X_test, y_train_res, y_test


def train_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    tune: bool = False,
    cv_folds: int = 5,
) -> Dict[str, Dict[str, Any]]:
    """
    Train every model in the registry, optionally tuning
    hyperparameters first, run cross-validation, persist each fitted
    model, and record training time + CV metrics.

    Parameters
    ----------
    X_train, y_train : pd.DataFrame, pd.Series
        (Resampled) training data.
    tune : bool, optional
        If True, run RandomizedSearchCV before final fit.
    cv_folds : int, optional
        Number of cross-validation folds.

    Returns
    -------
    dict
        Mapping of model name -> {"model": fitted estimator,
        "train_time_seconds": float, "cv_metrics": dict}.
    """
    set_global_seed()
    registry = get_model_registry()
    results: Dict[str, Dict[str, Any]] = {}

    for name, model in registry.items():
        logger.info("=" * 70)
        logger.info("Training model: %s", name)

        if tune:
            model = tune_model(name, model, X_train, y_train)

        cv_metrics = cross_validate_model(model, X_train, y_train, cv_folds=cv_folds)

        fitted_model, train_time = measure_time(model.fit, X_train, y_train)
        save_model(fitted_model, f"{name}.joblib")

        results[name] = {
            "model": fitted_model,
            "train_time_seconds": train_time,
            "cv_metrics": cv_metrics,
        }
        logger.info("Finished training '%s' in %.2fs", name, train_time)

    return results


def main() -> None:
    """
    CLI entry point: parses arguments, builds training data, trains
    all models, and saves a summary of training metadata to
    ``reports/training_summary.json``.
    """
    parser = argparse.ArgumentParser(description="Train fraud detection models.")
    parser.add_argument(
        "--resampling",
        type=str,
        default="smote",
        choices=["none", "undersample", "oversample", "smote"],
        help="Resampling strategy applied to the training set only.",
    )
    parser.add_argument(
        "--tune", action="store_true", help="Run RandomizedSearchCV hyperparameter tuning."
    )
    parser.add_argument(
        "--cv-folds", type=int, default=5, help="Number of cross-validation folds."
    )
    args = parser.parse_args()

    set_global_seed()

    logger.info("Building training data with resampling strategy: %s", args.resampling)
    X_train, X_test, y_train, y_test = build_training_data(args.resampling)

    results = train_all_models(X_train, y_train, tune=args.tune, cv_folds=args.cv_folds)

    summary = {
        name: {
            "train_time_seconds": res["train_time_seconds"],
            "cv_metrics": res["cv_metrics"],
        }
        for name, res in results.items()
    }
    save_json(summary, "training_summary.json")
    logger.info("Training complete. Summary saved to reports/training_summary.json")


if __name__ == "__main__":
    main()
