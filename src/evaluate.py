"""
evaluate.py
===========
Model evaluation utilities: classification metrics, confusion matrix,
ROC / Precision-Recall curves, feature importance (native + permutation),
and a multi-model comparison table.

Why ROC AUC alone is insufficient here
---------------------------------------
With ~99.83% of transactions legitimate, the negative class dominates
the ROC curve's False Positive Rate axis, so ROC AUC can look
excellent (e.g. > 0.95) even when the model catches very few actual
frauds relative to how many false alarms it raises. Precision-Recall
AUC (PR AUC) focuses only on the positive (fraud) class and how well
the model trades precision against recall, which is what actually
matters for a fraud team triaging alerts. Both are reported, but
PR AUC and Recall/Precision should drive model selection.
"""

from __future__ import annotations

from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.utils import OUTPUTS_DIR, get_logger, measure_time, save_json

logger = get_logger(__name__)
sns.set_theme(style="whitegrid")


def compute_metrics(
    y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray
) -> Dict[str, float]:
    """
    Compute the full suite of classification metrics appropriate for
    an imbalanced binary classification problem.

    Parameters
    ----------
    y_true : pd.Series
        Ground-truth binary labels.
    y_pred : np.ndarray
        Predicted binary labels (0/1).
    y_proba : np.ndarray
        Predicted probability of the positive (fraud) class.

    Returns
    -------
    dict
        Dictionary with accuracy, precision, recall, f1, roc_auc,
        pr_auc.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
    }
    logger.info("Computed metrics: %s", {k: round(v, 4) for k, v in metrics.items()})
    return metrics


def print_classification_report(y_true: pd.Series, y_pred: np.ndarray) -> str:
    """
    Generate and log the full sklearn classification report
    (per-class precision/recall/F1/support).

    Parameters
    ----------
    y_true : pd.Series
        Ground-truth binary labels.
    y_pred : np.ndarray
        Predicted binary labels.

    Returns
    -------
    str
        The classification report as a string.
    """
    report = classification_report(y_true, y_pred, target_names=["Legitimate", "Fraud"])
    logger.info("Classification report:\n%s", report)
    return report


def plot_confusion_matrix(
    y_true: pd.Series, y_pred: np.ndarray, model_name: str, save: bool = True
) -> plt.Figure:
    """
    Plot (and optionally save) a confusion matrix heatmap.

    Parameters
    ----------
    y_true : pd.Series
        Ground-truth binary labels.
    y_pred : np.ndarray
        Predicted binary labels.
    model_name : str
        Used in the plot title and output filename.
    save : bool, optional
        If True, saves the figure to ``outputs/confusion_matrix_<model>.png``.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Legitimate", "Fraud"])
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(f"Confusion Matrix — {model_name}")
    fig.tight_layout()

    if save:
        path = OUTPUTS_DIR / f"confusion_matrix_{model_name}.png"
        fig.savefig(path, dpi=150)
        logger.info("Saved confusion matrix to %s", path)
    return fig


def plot_roc_curve(
    y_true: pd.Series, y_proba: np.ndarray, model_name: str, save: bool = True
) -> plt.Figure:
    """
    Plot (and optionally save) the ROC curve for a single model.

    Parameters
    ----------
    y_true : pd.Series
        Ground-truth binary labels.
    y_proba : np.ndarray
        Predicted probability of the positive class.
    model_name : str
        Used in the plot title and output filename.
    save : bool, optional
        If True, saves to ``outputs/roc_curve_<model>.png``.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_true, y_proba, ax=ax, name=model_name)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random baseline")
    ax.set_title(f"ROC Curve — {model_name}")
    ax.legend(loc="lower right")
    fig.tight_layout()

    if save:
        path = OUTPUTS_DIR / f"roc_curve_{model_name}.png"
        fig.savefig(path, dpi=150)
        logger.info("Saved ROC curve to %s", path)
    return fig


def plot_precision_recall_curve(
    y_true: pd.Series, y_proba: np.ndarray, model_name: str, save: bool = True
) -> plt.Figure:
    """
    Plot (and optionally save) the Precision-Recall curve — the more
    informative curve for this imbalanced problem.

    Parameters
    ----------
    y_true : pd.Series
        Ground-truth binary labels.
    y_proba : np.ndarray
        Predicted probability of the positive class.
    model_name : str
        Used in the plot title and output filename.
    save : bool, optional
        If True, saves to ``outputs/precision_recall_curve_<model>.png``.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_predictions(y_true, y_proba, ax=ax, name=model_name)
    ax.set_title(f"Precision-Recall Curve — {model_name}")
    fig.tight_layout()

    if save:
        path = OUTPUTS_DIR / f"precision_recall_curve_{model_name}.png"
        fig.savefig(path, dpi=150)
        logger.info("Saved Precision-Recall curve to %s", path)
    return fig


def plot_feature_importance(
    model: Any,
    feature_names: List[str],
    model_name: str,
    top_n: int = 15,
    save: bool = True,
) -> pd.DataFrame:
    """
    Plot the top-N native feature importances for tree-based models
    (``feature_importances_`` attribute). Skips gracefully for models
    without that attribute (e.g. Logistic Regression — use its
    coefficients instead via ``plot_logistic_coefficients``).

    Parameters
    ----------
    model : Any
        Fitted tree-based estimator with a ``feature_importances_``
        attribute.
    feature_names : list of str
        Column names matching the model's input feature order.
    model_name : str
        Used in the plot title and output filename.
    top_n : int, optional
        Number of top features to display, by default 15.
    save : bool, optional
        If True, saves to ``outputs/feature_importance_<model>.png``.

    Returns
    -------
    pd.DataFrame
        Sorted dataframe of feature importances.
    """
    if not hasattr(model, "feature_importances_"):
        logger.info("Model '%s' has no native feature_importances_ — skipping.", model_name)
        return pd.DataFrame()

    importances = pd.DataFrame(
        {"feature": feature_names, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)

    top_features = importances.head(top_n)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(data=top_features, x="importance", y="feature", ax=ax, color="steelblue")
    ax.set_title(f"Top {top_n} Feature Importances — {model_name}")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    fig.tight_layout()

    if save:
        path = OUTPUTS_DIR / f"feature_importance_{model_name}.png"
        fig.savefig(path, dpi=150)
        logger.info("Saved feature importance plot to %s", path)

    return importances


def compute_permutation_importance(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_repeats: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Compute permutation importance on the held-out test set: measures
    the drop in PR AUC when each feature's values are randomly
    shuffled, giving a model-agnostic importance measure (unlike
    native tree importances, this also works for Logistic Regression).

    Parameters
    ----------
    model : Any
        Fitted estimator.
    X_test, y_test : pd.DataFrame, pd.Series
        Held-out test data.
    n_repeats : int, optional
        Number of shuffles per feature, by default 5.
    random_state : int, optional
        Seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Sorted dataframe with columns [feature, importance_mean, importance_std].
    """
    result = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring="average_precision",
        n_jobs=-1,
    )
    df = pd.DataFrame(
        {
            "feature": X_test.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    return df


def build_comparison_table(all_results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Build the final model comparison table with accuracy, precision,
    recall, F1, ROC AUC, training time, prediction time and memory
    usage, then rank models by PR AUC (most informative metric for
    this imbalanced task) as a tiebreaker-aware proxy for F1/recall
    balance.

    Parameters
    ----------
    all_results : dict
        Mapping model_name -> dict containing at least the keys
        ``metrics`` (dict from ``compute_metrics``),
        ``train_time_seconds``, ``predict_time_seconds``, and
        ``memory_usage_mb``.

    Returns
    -------
    pd.DataFrame
        Comparison table sorted by rank (best model first), with a
        ``rank`` column added.
    """
    rows = []
    for name, res in all_results.items():
        m = res["metrics"]
        rows.append(
            {
                "Model": name,
                "Accuracy": round(m["accuracy"], 4),
                "Precision": round(m["precision"], 4),
                "Recall": round(m["recall"], 4),
                "F1": round(m["f1_score"], 4),
                "ROC_AUC": round(m["roc_auc"], 4),
                "PR_AUC": round(m["pr_auc"], 4),
                "Training_Time_s": round(res.get("train_time_seconds", 0.0), 4),
                "Prediction_Time_s": round(res.get("predict_time_seconds", 0.0), 4),
                "Memory_Usage_MB": round(res.get("memory_usage_mb", 0.0), 2),
            }
        )

    table = pd.DataFrame(rows)
    table = table.sort_values(by=["PR_AUC", "F1"], ascending=False).reset_index(drop=True)
    table.insert(0, "Rank", range(1, len(table) + 1))
    return table


def recommend_model(comparison_table: pd.DataFrame) -> str:
    """
    Recommend the best model from the comparison table: rank #1 by
    PR AUC / F1 (already sorted by ``build_comparison_table``).

    Parameters
    ----------
    comparison_table : pd.DataFrame
        Output of ``build_comparison_table``.

    Returns
    -------
    str
        Name of the recommended model.
    """
    best = comparison_table.iloc[0]
    logger.info(
        "Recommended model: %s (PR_AUC=%.4f, F1=%.4f, Recall=%.4f)",
        best["Model"],
        best["PR_AUC"],
        best["F1"],
        best["Recall"],
    )
    return str(best["Model"])


def evaluate_model(
    model: Any,
    model_name: str,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    generate_plots: bool = True,
) -> Dict[str, Any]:
    """
    Full single-model evaluation: predict, compute metrics + timing +
    memory footprint, optionally generate and save all plots.

    Parameters
    ----------
    model : Any
        Fitted estimator.
    model_name : str
        Identifier used for plot titles/filenames.
    X_test, y_test : pd.DataFrame, pd.Series
        Held-out test data.
    generate_plots : bool, optional
        If True, generate and save confusion matrix, ROC, PR curve,
        and feature importance plots.

    Returns
    -------
    dict
        ``{"metrics": {...}, "predict_time_seconds": float,
        "memory_usage_mb": float, "classification_report": str}``.
    """
    import sys

    y_pred, predict_time = measure_time(model.predict, X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = compute_metrics(y_test, y_pred, y_proba)
    report = print_classification_report(y_test, y_pred)

    # Rough memory footprint of the fitted estimator object itself.
    memory_usage_mb = sys.getsizeof(model) / (1024 * 1024)
    try:
        import joblib
        import io

        buf = io.BytesIO()
        joblib.dump(model, buf)
        memory_usage_mb = len(buf.getvalue()) / (1024 * 1024)
    except Exception:  # pragma: no cover
        pass

    if generate_plots:
        plot_confusion_matrix(y_test, y_pred, model_name)
        plot_roc_curve(y_test, y_proba, model_name)
        plot_precision_recall_curve(y_test, y_proba, model_name)
        plot_feature_importance(model, list(X_test.columns), model_name)

    return {
        "metrics": metrics,
        "predict_time_seconds": predict_time,
        "memory_usage_mb": memory_usage_mb,
        "classification_report": report,
    }


if __name__ == "__main__":
    # Quick standalone evaluation entry point:
    # loads persisted models + processed test split and prints a
    # comparison table. Assumes `train.py` has already been run.
    from src.preprocessing import load_processed_splits
    from src.utils import MODELS_DIR, load_model

    X_train, X_test, y_train, y_test = load_processed_splits()

    all_results: Dict[str, Dict[str, Any]] = {}
    for model_path in sorted(MODELS_DIR.glob("*.joblib")):
        if model_path.name == "scaler.joblib":
            continue
        name = model_path.stem
        logger.info("Evaluating model: %s", name)
        model = load_model(model_path.name)
        all_results[name] = evaluate_model(model, name, X_test, y_test)

    if all_results:
        comparison = build_comparison_table(all_results)
        print(comparison.to_string(index=False))
        best_model = recommend_model(comparison)
        print(f"\nRecommended model: {best_model}")
        save_json({"comparison_table": comparison.to_dict(orient="records"),
                    "recommended_model": best_model}, "model_comparison.json")
    else:
        logger.warning("No trained models found in %s. Run train.py first.", MODELS_DIR)
