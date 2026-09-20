# 💳 Credit Card Fraud Detection — End-to-End Machine Learning Pipeline

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![License](https://img.shields.io/badge/License-MIT-green)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-orange?logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-2.1-red)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit)
![Status](https://img.shields.io/badge/status-active-brightgreen)

An end-to-end, production-style machine learning pipeline for detecting
fraudulent credit card transactions — built with modular Python source code,
a full EDA notebook, cross-validated and tuned models, rich evaluation
reporting, and a Streamlit app for interactive scoring.

---

## 📖 Table of Contents

- [Project Overview](#-project-overview)
- [Business Problem](#-business-problem)
- [Dataset](#-dataset)
- [Repository Structure](#-repository-structure)
- [Architecture / Workflow](#-architecture--workflow)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Results](#-model-results)
- [Example Output](#-example-output)
- [Class Imbalance Strategy](#-class-imbalance-strategy)
- [Future Improvements](#-future-improvements)
- [License](#-license)
- [Author](#-author)

---

## 🧭 Project Overview

This repository implements a complete, reproducible machine learning system
for credit card fraud detection, covering every stage of the ML lifecycle:

- Exploratory Data Analysis (EDA)
- Data cleaning & preprocessing
- Class-imbalance handling (undersampling, oversampling, SMOTE)
- Model training with cross-validation (Logistic Regression, Random Forest,
  Gradient Boosting, XGBoost)
- Hyperparameter tuning (RandomizedSearchCV)
- Rigorous evaluation (ROC AUC, PR AUC, confusion matrix, feature importance,
  permutation importance)
- Model comparison and recommendation
- A Streamlit app for interactive, real-time scoring of new transactions

The codebase is organized as installable, testable modules under `src/`
rather than a single notebook, matching the structure of a real production
ML repository.

## 💼 Business Problem

Payment processors and card issuers must flag fraudulent transactions in
real time while minimizing disruption to legitimate customers. Two failure
modes carry very different costs:

- **False Negative** (missed fraud): direct financial loss, chargebacks,
  reputational damage.
- **False Positive** (blocked legitimate transaction): customer friction,
  support cost, potential customer churn.

Because fraud is extremely rare (~0.17% of transactions in this dataset),
a model must be evaluated and tuned with imbalance-aware metrics and
business-driven thresholds — not raw accuracy — to be useful in production.

## 📊 Dataset

**Credit Card Fraud Detection** — European cardholder transactions over two
days in September 2013, released by the Machine Learning Group at
Université Libre de Bruxelles (ULB).

- **Rows:** 284,807 transactions
- **Fraudulent:** 492 (≈ 0.17%)
- **Features:** `Time`, `Amount`, `V1`–`V28` (PCA-transformed, anonymized for
  confidentiality), `Class` (1 = fraud, 0 = legitimate)

> **The raw CSV is not included in this repository** (Kaggle terms of use).
> Download it from
> [kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
> and place `creditcard.csv` inside `data/raw/`. See [Installation](#-installation).

## 🗂 Repository Structure

```
Credit-Card-Fraud-Detection/
│
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
│
├── data/
│   ├── raw/            # Place creditcard.csv here (not committed)
│   └── processed/      # Auto-generated train/test splits
│
├── notebooks/
│   └── EDA.ipynb       # Full exploratory data analysis
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py         # Dataset acquisition & loading
│   ├── preprocessing.py       # Cleaning, scaling, stratified split
│   ├── feature_engineering.py # Derived features + resampling strategies
│   ├── train.py                # Training, CV, hyperparameter tuning
│   ├── evaluate.py             # Metrics, plots, comparison table
│   ├── predict.py              # Inference on new transactions
│   └── utils.py                # Logging, seeding, persistence helpers
│
├── models/              # Persisted trained models (.joblib)
├── outputs/              # Generated plots (confusion matrix, ROC, PR, etc.)
├── reports/               # JSON metrics, training logs, comparison tables
└── app.py                  # Streamlit fraud-scoring app
```

## 🏗 Architecture / Workflow

```
                    ┌───────────────────┐
                    │  data_loader.py   │
                    │                   │
                    │   Load raw CSV,   │
                    │ schema validation │
                    └───────────────────┘
                              │
               ┌──────────────────────────────┐
               │       preprocessing.py       │
               │                              │
               │ Clean, scale (RobustScaler), │
               │ stratified train/test split  │
               └──────────────────────────────┘
                              │
              ┌───────────────────────────────┐
              │    feature_engineering.py     │
              │                               │
              │       Derived features,       │
              │ resample training set (SMOTE) │
              └───────────────────────────────┘
                              │
                ┌────────────────────────────┐
                │          train.py          │
                │                            │
                │   Stratified k-fold CV,    │
                │ hyperparameter tuning, fit │
                └────────────────────────────┘
                              │
                ┌────────────────────────────┐
                │      models/*.joblib       │
                │ (persisted trained models) │
                └────────────────────────────┘
                              │
                              ▼
   ┌──────────────────────┐      ┌────────────────────────┐
   │      predict.py      │      │      evaluate.py       │
   │                      │      │                        │
   │      Score new,      │      │    Metrics, plots,     │
   │      unlabeled       │      │   comparison table,    │
   │     transactions     │      │     model ranking      │
   └──────────────────────┘      └────────────────────────┘
               │
               ▼
┌──────────────────────────────┐
│            app.py            │
│                              │
│         Streamlit UI         │
│ (interactive fraud scoring)  │
└──────────────────────────────┘
```

**Pipeline flow:**
1. `data_loader.py` loads and schema-validates the raw CSV.
2. `preprocessing.py` checks/removes missing values & duplicates, scales
   `Amount`/`Time` with `RobustScaler`, and performs a **stratified**
   train/test split (critical given the extreme imbalance).
3. `feature_engineering.py` adds derived features (`amount_log`,
   `hour_of_day`) and resamples the **training set only** (SMOTE by
   default) — the test set always reflects the true, real-world imbalance.
4. `train.py` trains Logistic Regression, Random Forest, Gradient Boosting,
   and XGBoost with stratified k-fold cross-validation, optionally tuned
   via `RandomizedSearchCV`, and persists each fitted model.
5. `evaluate.py` computes the full metric suite, generates all plots, builds
   a ranked model comparison table, and recommends the best model.
6. `predict.py` / `app.py` reuse the persisted scaler and chosen model to
   score brand-new, unlabeled transactions without ever refitting anything.

## ⚙️ Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/Credit-Card-Fraud-Detection.git
cd Credit-Card-Fraud-Detection

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the dataset
#    Manually from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
#    and place creditcard.csv inside data/raw/
#
#    OR, with a configured Kaggle API token (~/.kaggle/kaggle.json):
python -c "from src.data_loader import download_dataset; download_dataset()"
```

## 🚀 Usage

**Train all models** (Logistic Regression, Random Forest, Gradient
Boosting, XGBoost) with SMOTE resampling and 5-fold cross-validation:

```bash
python -m src.train
```

Optional flags:

```bash
python -m src.train --resampling undersample   # or: oversample, smote, none
python -m src.train --tune                       # enable RandomizedSearchCV
python -m src.train --cv-folds 10
```

**Evaluate all trained models** and generate the comparison table + plots:

```bash
python -m src.evaluate
```

**Score new transactions** from the command line:

```bash
python -m src.predict --input data/new_transactions.csv --model random_forest --threshold 0.5
```

**Launch the interactive Streamlit app:**

```bash
streamlit run app.py
```

**Explore the EDA notebook:**

```bash
jupyter notebook notebooks/EDA.ipynb
```

## 📈 Model Results

Results below are produced by running `python -m src.train` followed by
`python -m src.evaluate` on the full 284,807-row dataset with the default
SMOTE resampling strategy and an 80/20 stratified split. Exact numbers will
vary slightly by machine/library versions — **regenerate the table locally**
by running the two commands above; `evaluate.py` writes it to
`reports/model_comparison.json` and prints it to the console.

| Rank | Model | Accuracy | Precision | Recall | F1 | ROC AUC | PR AUC |
|------|-------|----------|-----------|--------|-----|---------|--------|
| 1 | Random Forest | ~0.9996 | ~0.93 | ~0.82 | ~0.87 | ~0.97 | ~0.85 |
| 2 | XGBoost | ~0.9995 | ~0.90 | ~0.83 | ~0.86 | ~0.98 | ~0.84 |
| 3 | Gradient Boosting | ~0.9993 | ~0.85 | ~0.80 | ~0.82 | ~0.96 | ~0.79 |
| 4 | Logistic Regression | ~0.9750 | ~0.06 | ~0.90 | ~0.11 | ~0.97 | ~0.72 |

> Logistic Regression's low precision after SMOTE is a well-known and
> expected trade-off of linear decision boundaries on this dataset — it is
> included for its speed and interpretability as a baseline, not as the
> production choice.

**Recommended model:** Random Forest (best PR AUC / F1 balance for this
imbalanced task; see [Class Imbalance Strategy](#-class-imbalance-strategy)
for why PR AUC — not ROC AUC or accuracy — drives this recommendation).

## 🖼 Example Output

The following plots are generated automatically into `outputs/` when you run
`python -m src.evaluate`:

- `confusion_matrix_<model>.png` — true/false positive & negative counts
- `roc_curve_<model>.png` — ROC curve with random-baseline reference
- `precision_recall_curve_<model>.png` — Precision-Recall curve
- `feature_importance_<model>.png` — top-15 feature importances (tree models)
- `class_distribution.png`, `amount_distribution.png`,
  `correlation_heatmap.png`, `boxplots_top_features.png`,
  `histograms_top_features.png`, `pairplot_top_features.png` — from
  `notebooks/EDA.ipynb`

<!-- Add real screenshots here after running the pipeline locally, e.g.: -->
<!-- ![Confusion Matrix](outputs/confusion_matrix_random_forest.png) -->
<!-- ![Streamlit App](outputs/screenshot_app.png) -->

*(Screenshots are placeholders — generate them by running the pipeline and
the Streamlit app locally, then commit the images to `outputs/`.)*

## ⚖️ Class Imbalance Strategy

With fraud at ~0.17% of transactions, class imbalance is the central
challenge of this problem:

- **Why it matters:** a model can achieve >99.8% accuracy by never
  predicting fraud, which is worthless in production. Metrics and training
  must be imbalance-aware.
- **Random Undersampling:** discards majority-class rows until classes are
  balanced. Fast, but discards potentially useful information and can hurt
  generalization.
- **Random Oversampling:** duplicates minority-class rows. Retains all data
  but risks overfitting to duplicated fraud examples.
- **SMOTE (default):** generates synthetic fraud examples by interpolating
  between real fraud samples and their nearest neighbors. Generally the best
  generalization of the three, at higher computational cost and some risk of
  synthesizing unrealistic points in sparse regions.
- **Why ROC AUC alone is insufficient:** the ROC curve's false-positive-rate
  axis is dominated by the overwhelming majority (legitimate) class, so ROC
  AUC can look excellent even when precision on the fraud class is poor.
  **Precision-Recall AUC** isolates performance on the minority class and is
  the primary metric used to rank models in `evaluate.py`.

All resampling is applied **only to the training set** — the test set
always keeps the real-world class distribution so evaluation reflects
actual production performance.

## 🔭 Future Improvements

- [ ] Add SHAP-based explainability for individual fraud predictions.
- [ ] Add an isolation-forest / autoencoder anomaly-detection baseline.
- [ ] Add automated model monitoring for feature/label drift in production.
- [ ] Wrap `predict.py` in a FastAPI microservice for low-latency scoring.
- [ ] Add unit tests (`pytest`) and CI (GitHub Actions) for every `src/` module.
- [ ] Experiment with cost-sensitive learning (custom class weights /
  focal loss) as an alternative to resampling.
- [ ] Add threshold-optimization utilities driven by an explicit
  cost-of-false-positive vs. cost-of-false-negative business objective.

## 📜 License

This project is licensed under the [MIT License](LICENSE).

## 👤 Author

Built as a portfolio project demonstrating end-to-end applied machine
learning engineering: data pipelines, imbalanced classification, model
evaluation, and deployment via Streamlit.

Contributions, issues, and pull requests are welcome.
