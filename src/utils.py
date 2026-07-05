"""
utils.py
========
Shared utility functions used across the fraud-detection pipeline:
logging configuration, reproducibility helpers, timing decorators,
and generic I/O helpers (saving/loading models and JSON metrics).

This module has no dependency on any other module in ``src`` so it
can be safely imported everywhere without circular imports.
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict

import joblib
import numpy as np

# --------------------------------------------------------------------------- #
# Global constants
# --------------------------------------------------------------------------- #
RANDOM_SEED: int = 42

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DATA_RAW_DIR: Path = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"
MODELS_DIR: Path = PROJECT_ROOT / "models"
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"

for _dir in (DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR, OUTPUTS_DIR, REPORTS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def get_logger(name: str, log_file: str = "pipeline.log") -> logging.Logger:
    """
    Create (or fetch) a configured logger that writes to both the
    console and a rotating log file under ``reports/``.

    Parameters
    ----------
    name : str
        Name of the logger, typically ``__name__`` of the caller module.
    log_file : str, optional
        File name (relative to ``reports/``) to persist logs to.

    Returns
    -------
    logging.Logger
        A logger instance ready to use.
    """
    logger = logging.getLogger(name)
    if logger.handlers:  # avoid duplicate handlers on repeated calls
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(REPORTS_DIR / log_file)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def set_global_seed(seed: int = RANDOM_SEED) -> None:
    """
    Fix the random seed across ``random``, ``numpy`` and the ``PYTHONHASHSEED``
    environment variable so results are reproducible across runs.

    Parameters
    ----------
    seed : int, optional
        Seed value to apply everywhere, by default ``RANDOM_SEED`` (42).

    Returns
    -------
    None
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# --------------------------------------------------------------------------- #
# Timing decorator
# --------------------------------------------------------------------------- #
def timeit(func: Callable) -> Callable:
    """
    Decorator that logs and returns the wall-clock execution time (in
    seconds) of the wrapped function. The wrapped function's return
    value is untouched; timing is only logged, not injected into the
    return value, so this decorator is transparent to callers.

    Parameters
    ----------
    func : Callable
        The function to time.

    Returns
    -------
    Callable
        Wrapped function with timing/logging side effects.
    """
    logger = get_logger(func.__module__)

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info("Function '%s' completed in %.4f seconds", func.__name__, elapsed)
        return result

    return wrapper


def measure_time(func: Callable, *args: Any, **kwargs: Any) -> tuple[Any, float]:
    """
    Run ``func`` once and return both its result and the elapsed
    wall-clock time in seconds. Useful for benchmarking training /
    prediction time per model without polluting the function itself
    with a decorator.

    Parameters
    ----------
    func : Callable
        Function to execute.
    *args, **kwargs
        Forwarded to ``func``.

    Returns
    -------
    tuple[Any, float]
        ``(result, elapsed_seconds)``.
    """
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, elapsed


# --------------------------------------------------------------------------- #
# Persistence helpers
# --------------------------------------------------------------------------- #
def save_model(model: Any, filename: str) -> Path:
    """
    Persist a fitted model/estimator to ``models/`` using joblib.

    Parameters
    ----------
    model : Any
        Fitted scikit-learn compatible estimator.
    filename : str
        File name, e.g. ``"random_forest.joblib"``.

    Returns
    -------
    Path
        Full path where the model was saved.
    """
    path = MODELS_DIR / filename
    joblib.dump(model, path)
    return path


def load_model(filename: str) -> Any:
    """
    Load a previously persisted model from ``models/``.

    Parameters
    ----------
    filename : str
        File name of the saved model, e.g. ``"random_forest.joblib"``.

    Returns
    -------
    Any
        The deserialized model object.
    """
    path = MODELS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"No model found at {path}")
    return joblib.load(path)


def save_json(data: Dict[str, Any], filename: str, directory: Path = REPORTS_DIR) -> Path:
    """
    Save a dictionary as a pretty-printed JSON file.

    Parameters
    ----------
    data : dict
        Serializable dictionary (e.g. metrics).
    filename : str
        File name, e.g. ``"metrics.json"``.
    directory : Path, optional
        Target directory, by default ``REPORTS_DIR``.

    Returns
    -------
    Path
        Full path of the written JSON file.
    """
    path = directory / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, default=str)
    return path


def load_json(filename: str, directory: Path = REPORTS_DIR) -> Dict[str, Any]:
    """
    Load a JSON file previously written by ``save_json``.

    Parameters
    ----------
    filename : str
        File name to load.
    directory : Path, optional
        Directory to load from, by default ``REPORTS_DIR``.

    Returns
    -------
    dict
        Parsed JSON content.
    """
    path = directory / filename
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
