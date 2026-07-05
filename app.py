"""
app.py
======
Streamlit application that lets a user upload a CSV of raw
transactions (Time, V1-V28, Amount) and get back fraud probabilities
and predictions from a previously trained model, entirely in-browser.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent))

from src.data_loader import EXPECTED_COLUMNS  # noqa: E402
from src.predict import predict_fraud  # noqa: E402
from src.utils import MODELS_DIR  # noqa: E402

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
)


def get_available_models() -> list[str]:
    """
    Scan ``models/`` for persisted ``.joblib`` model files (excluding
    the feature scaler) and return their names for the dropdown.

    Returns
    -------
    list of str
        Available model names (without file extension).
    """
    if not MODELS_DIR.exists():
        return []
    return sorted(
        p.stem for p in MODELS_DIR.glob("*.joblib") if p.stem != "scaler"
    )


def render_header() -> None:
    """Render the app title, description, and disclaimer banner."""
    st.title("💳 Credit Card Fraud Detection")
    st.markdown(
        """
        Upload a CSV of transactions (columns: `Time`, `V1`-`V28`, `Amount`,
        optionally `Class`) and score them for fraud probability using a
        model trained on the public *Credit Card Fraud Detection* dataset.
        """
    )
    st.info(
        "This is a portfolio / educational project. Do not use these "
        "predictions for real financial decisions.",
        icon="ℹ️",
    )


def render_sidebar(available_models: list[str]) -> tuple[str, float]:
    """
    Render the sidebar controls: model selector and probability
    threshold slider.

    Parameters
    ----------
    available_models : list of str
        Model names discovered in ``models/``.

    Returns
    -------
    tuple
        ``(selected_model_name, threshold)``.
    """
    st.sidebar.header("⚙️ Settings")

    if not available_models:
        st.sidebar.error(
            "No trained models found in `models/`. Run `python -m src.train` first."
        )
        return "", 0.5

    model_name = st.sidebar.selectbox("Model", available_models, index=0)
    threshold = st.sidebar.slider(
        "Fraud probability threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01,
        help="Transactions with fraud probability at or above this value "
        "are flagged as fraud. Lower it to catch more fraud at the cost "
        "of more false alarms.",
    )
    return model_name, threshold


def render_results(scored_df: pd.DataFrame) -> None:
    """
    Render the scored results: summary metrics, flagged transactions
    table, full results table, and a CSV download button.

    Parameters
    ----------
    scored_df : pd.DataFrame
        Output of ``predict_fraud`` containing ``fraud_probability``
        and ``predicted_class`` columns.

    Returns
    -------
    None
    """
    n_total = len(scored_df)
    n_flagged = int(scored_df["predicted_class"].sum())

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Transactions", f"{n_total:,}")
    col2.metric("Flagged as Fraud", f"{n_flagged:,}")
    col3.metric("Flagged Rate", f"{(n_flagged / n_total * 100):.3f}%" if n_total else "0%")

    st.subheader("🚨 Flagged Transactions")
    flagged = scored_df[scored_df["predicted_class"] == 1].sort_values(
        "fraud_probability", ascending=False
    )
    if flagged.empty:
        st.success("No transactions were flagged as fraud at the current threshold.")
    else:
        st.dataframe(flagged, use_container_width=True)

    with st.expander("📄 View all scored transactions"):
        st.dataframe(
            scored_df.sort_values("fraud_probability", ascending=False),
            use_container_width=True,
        )

    csv_bytes = scored_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download scored CSV",
        data=csv_bytes,
        file_name="scored_transactions.csv",
        mime="text/csv",
    )


def main() -> None:
    """Main Streamlit app entry point."""
    render_header()

    available_models = get_available_models()
    model_name, threshold = render_sidebar(available_models)

    uploaded_file = st.file_uploader("Upload transactions CSV", type=["csv"])

    if uploaded_file is not None and model_name:
        try:
            raw_df = pd.read_csv(uploaded_file)
        except Exception as exc:
            st.error(f"Could not read the uploaded file: {exc}")
            return

        required_cols = [c for c in EXPECTED_COLUMNS if c != "Class"]
        missing = set(required_cols) - set(raw_df.columns)
        if missing:
            st.error(f"Uploaded CSV is missing required columns: {sorted(missing)}")
            return

        with st.spinner("Scoring transactions..."):
            try:
                scored_df = predict_fraud(raw_df, model_name=model_name, threshold=threshold)
            except FileNotFoundError as exc:
                st.error(str(exc))
                return

        render_results(scored_df)
    elif uploaded_file is not None and not model_name:
        st.warning("No model available. Train a model first with `python -m src.train`.")
    else:
        st.markdown("👆 Upload a CSV file to get started.")


if __name__ == "__main__":
    main()
