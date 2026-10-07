# AI Cyber Threat Detection

A Python project for binary intrusion classification, attack-category classification, anomaly detection, and probability-based risk labels using NSL-KDD.

## What is included

- Binary models: Logistic Regression, Random Forest, and XGBoost.
- Multi-class workflow groups traffic into Normal, DoS, Probe, R2L, and U2R.
- Isolation Forest supplies an anomaly-detection workflow.
- Preprocessing handles categorical encoding and numeric scaling.
- Reports include model comparisons, classification reports, confusion matrices, and ROC curves.

## Getting started

From the repository root:

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

Use the menu to train before evaluating or predicting. Direct commands include `python src/train_model.py`, `python src/evaluate_model.py`, `python src/train_multiclass_model.py`, and `python src/train_anomaly_model.py`. Keep `data/KDDTrain+.txt` and `data/KDDTest+.txt` available.

## Repository guide

- `README.md`
- `check_dataset.py`
- `data/`
- `main.py`
- `notebooks_or_eda.py`
- `outputs/`
- `requirements.txt`
- `src/`

## Limitations and reproducibility

Binary model comparison uses an 80/20 stratified split of `KDDTrain+.txt`; the separate `evaluate_model.py` evaluates the saved binary model on `KDDTest+.txt`. Do not conflate their scores. Risk thresholds are High ≥ 0.75, Medium ≥ 0.40, and Low below 0.40; these are heuristic categories. Prediction scripts use sample records rather than live packet capture. Model files must be trained before inference. NSL-KDD results do not establish real-network detection performance.
