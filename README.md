# 🚀 AI Cyber Threat Detection System

An end-to-end machine learning cybersecurity project that detects suspicious network activity, classifies cyber attacks, and identifies anomalous behavior using network traffic data.

This project uses the NSL-KDD dataset to build:

- Binary Attack Detection
- Multi-Class Attack Classification
- Anomaly Detection
- Risk Scoring
- Model Evaluation Dashboard
- Production-Style ML Pipelines

---

# 📌 Project Overview

The goal of this project is to analyze network traffic and predict whether the activity is:

- Normal Traffic
- Cyber Attack

The system can also classify attacks into multiple categories:

- DoS
- Probe
- R2L
- U2R

and detect unknown suspicious behavior using anomaly detection.

---

# 🧠 Machine Learning Concepts Used

- Supervised Learning
- Binary Classification
- Multi-Class Classification
- Anomaly Detection
- Feature Engineering
- One-Hot Encoding
- Feature Scaling
- Pipeline Building
- Model Evaluation
- Risk Scoring

---

# 📂 Dataset

Dataset Used:

NSL-KDD Dataset

The dataset contains labeled network traffic records with multiple attack categories.

Example labels:

- normal
- neptune
- smurf
- portsweep
- ipsweep
- buffer_overflow
- warezclient

---

# ⚙️ Technologies Used

## Programming

- Python

## Libraries

- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn
- Joblib

---

# 📊 Features

## 1. Binary Classification

Detects whether traffic is:

- Normal
- Attack

Models Used:

- Logistic Regression
- Random Forest
- XGBoost

---

## 2. Multi-Class Classification

Classifies attacks into:

- Normal
- DoS
- Probe
- R2L
- U2R

---

## 3. Anomaly Detection

Uses Isolation Forest to identify unknown suspicious behavior.

This is important because real-world cyber attacks continuously evolve.

---

## 4. Risk Scoring

Attack probability is converted into:

- Low Risk
- Medium Risk
- High Risk

---

# 🏗️ Project Architecture

```text
Raw Network Data
        ↓
Data Preprocessing
        ↓
Feature Encoding & Scaling
        ↓
ML Models
        ↓
Threat Prediction
        ↓
Risk Scoring
        ↓
Evaluation Dashboard
```

---

# 📁 Project Structure

```text
ai-cyber-threat-detection/
│
├── data/
│   ├── KDDTrain+.txt
│   └── KDDTest+.txt
│
├── models/
│   ├── cyber_threat_model.pkl
│   ├── multiclass_cyber_threat_model.pkl
│   ├── anomaly_detection_model.pkl
│   └── attack_label_encoder.pkl
│
├── outputs/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── model_results.csv
│   ├── classification_report.csv
│   └── anomaly_detection_report.txt
│
├── src/
│   ├── data_preprocessing.py
│   ├── eda_visualization.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── predict.py
│   ├── risk_scoring.py
│   ├── train_multiclass_model.py
│   ├── predict_multiclass.py
│   └── train_anomaly_model.py
│
├── main.py
├── requirements.txt
└── README.md
```

---

# 📈 Model Performance

## Binary Classification Results

| Model | Accuracy | Recall | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 97% | 96% | 99% |
| Random Forest | 99.9% | 99.8% | 100% |
| XGBoost | 99.9% | 99.9% | 100% |

---

# 🔍 Why Recall Matters

In cybersecurity:

> Missing attacks is dangerous.

So recall is one of the most important metrics.

The models achieved extremely high recall while minimizing false negatives.

---

# ⚠️ Dataset Limitations

The NSL-KDD dataset is a benchmark cybersecurity dataset with relatively clean and separable attack patterns.

While the models achieved near-perfect performance, real-world network traffic is significantly more complex, noisy, encrypted, and continuously evolving.

Future improvements could include testing on:

- CICIDS2017
- UNSW-NB15
- TON_IoT

for more realistic cybersecurity evaluation.

---

# ▶️ How to Run the Project

## 1. Clone Repository

```bash
git clone https://github.com/BhavyaPatamsetti/AI_Cyber_Threat_Detection.git
cd ai-cyber-threat-detection
```

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## 3. Run Main System

```bash
python main.py
```

---

# 🖥️ Example Prediction

## Input

```text
protocol_type: tcp
service: http
flag: SF
src_bytes: 181
dst_bytes: 5450
```

## Output

```text
Prediction: Normal Traffic
Risk Level: Low
Confidence: 99.80%
```

---

# 📊 Outputs Generated

- Confusion Matrix
- ROC Curve
- Classification Report
- Model Comparison CSV
- Anomaly Detection Report
- Threat Predictions

---

# 💡 Future Improvements

- Deploy with Streamlit
- Real-time packet monitoring
- Live network sniffing using Scapy
- Deep Learning Models
- LSTM-based intrusion detection
- Docker deployment
- Cloud deployment
- SIEM integration

---

# 👩‍💻 Author

Radhi Sri Bhavya Patamsetti

- LinkedIn: https://www.linkedin.com/in/bhavyapatamsetti/
- GitHub: https://github.com/BhavyaPatamsetti
