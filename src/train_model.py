import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# ---------------------------------------------------
# Create models folder
# ---------------------------------------------------

os.makedirs("models", exist_ok=True)

# ---------------------------------------------------
# Column Names
# ---------------------------------------------------

columns = [
    'duration','protocol_type','service','flag','src_bytes','dst_bytes','land',
    'wrong_fragment','urgent','hot','num_failed_logins','logged_in',
    'num_compromised','root_shell','su_attempted','num_root',
    'num_file_creations','num_shells','num_access_files','num_outbound_cmds',
    'is_host_login','is_guest_login','count','srv_count','serror_rate',
    'srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate',
    'diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count',
    'dst_host_same_srv_rate','dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate','dst_host_srv_diff_host_rate',
    'dst_host_serror_rate','dst_host_srv_serror_rate',
    'dst_host_rerror_rate','dst_host_srv_rerror_rate',
    'label','difficulty'
]

# ---------------------------------------------------
# Load Dataset
# ---------------------------------------------------

df = pd.read_csv(
    "data/KDDTrain+.txt",
    names=columns
)

# ---------------------------------------------------
# Binary Target
# ---------------------------------------------------

df["target"] = df["label"].apply(
    lambda x: 0 if x == "normal" else 1
)

# ---------------------------------------------------
# Features and Target
# ---------------------------------------------------

X = df.drop(
    ["label", "difficulty", "target"],
    axis=1
)

y = df["target"]

# ---------------------------------------------------
# Categorical Columns
# ---------------------------------------------------

categorical_cols = [
    "protocol_type",
    "service",
    "flag"
]

# ---------------------------------------------------
# Numerical Columns
# ---------------------------------------------------

numerical_cols = [
    col for col in X.columns
    if col not in categorical_cols
]

# ---------------------------------------------------
# Preprocessing
# ---------------------------------------------------

numeric_transformer = Pipeline(
    steps=[
        ("scaler", StandardScaler())
    ]
)

categorical_transformer = Pipeline(
    steps=[
        (
            "onehot",
            OneHotEncoder(handle_unknown="ignore")
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numerical_cols),
        ("cat", categorical_transformer, categorical_cols)
    ]
)

# ---------------------------------------------------
# Train Test Split
# ---------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ---------------------------------------------------
# Models
# ---------------------------------------------------

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),

    "XGBoost": XGBClassifier(
        eval_metric="logloss",
        random_state=42
    )
}

# ---------------------------------------------------
# Train Models
# ---------------------------------------------------

results = []

best_model = None
best_recall = 0

for name, model in models.items():

    print(f"\nTraining {name}...")

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    pipeline.fit(X_train, y_train)

    # Predictions
    y_pred = pipeline.predict(X_test)

    # Probabilities
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    # Save Results
    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC AUC": roc_auc
    })

    # Print Results
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC AUC  : {roc_auc:.4f}")

    # Save Best Model
    if recall > best_recall:
        best_recall = recall
        best_model = pipeline

# ---------------------------------------------------
# Save Best Model
# ---------------------------------------------------

joblib.dump(
    best_model,
    "models/cyber_threat_model.pkl"
)

# ---------------------------------------------------
# Save Results
# ---------------------------------------------------

results_df = pd.DataFrame(results)

results_df.to_csv(
    "outputs/model_results.csv",
    index=False
)

print("\nBest model saved successfully.")

print("\nModel comparison saved in outputs folder.")