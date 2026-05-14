import os
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

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

df = pd.read_csv("data/KDDTrain+.txt", names=columns)

df["target"] = df["label"].apply(lambda x: 0 if x == "normal" else 1)

X = df.drop(["label", "difficulty", "target"], axis=1)
y = df["target"]

normal_data = df[df["target"] == 0]

X_normal = normal_data.drop(["label", "difficulty", "target"], axis=1)

categorical_cols = ["protocol_type", "service", "flag"]
numerical_cols = [col for col in X.columns if col not in categorical_cols]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ]
)

anomaly_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", IsolationForest(
            n_estimators=100,
            contamination=0.10,
            random_state=42
        ))
    ]
)

print("Training Isolation Forest on normal traffic only...")

anomaly_pipeline.fit(X_normal)

predictions = anomaly_pipeline.predict(X)

# Isolation Forest output:
# 1 = normal
# -1 = anomaly

y_pred = [0 if p == 1 else 1 for p in predictions]

print("\nAnomaly Detection Classification Report:")
report = classification_report(
    y,
    y_pred,
    target_names=["Normal", "Attack"]
)

print(report)

print("\nConfusion Matrix:")
print(confusion_matrix(y, y_pred))

with open("outputs/anomaly_detection_report.txt", "w") as f:
    f.write(report)

joblib.dump(anomaly_pipeline, "models/anomaly_detection_model.pkl")

print("\nAnomaly detection model saved successfully.")