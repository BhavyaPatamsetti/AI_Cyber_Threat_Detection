import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

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

dos = ['back','land','neptune','pod','smurf','teardrop','apache2','udpstorm','processtable','worm']
probe = ['satan','ipsweep','nmap','portsweep','mscan','saint']
r2l = ['guess_passwd','ftp_write','imap','phf','multihop','warezmaster','warezclient','spy','xlock','xsnoop','snmpguess','snmpgetattack','httptunnel','sendmail','named']
u2r = ['buffer_overflow','loadmodule','rootkit','perl','sqlattack','xterm','ps']

def map_attack(label):
    if label == "normal":
        return "Normal"
    elif label in dos:
        return "DoS"
    elif label in probe:
        return "Probe"
    elif label in r2l:
        return "R2L"
    elif label in u2r:
        return "U2R"
    else:
        return "Unknown"

df = pd.read_csv("data/KDDTrain+.txt", names=columns)

df["attack_category"] = df["label"].apply(map_attack)

df = df[df["attack_category"] != "Unknown"]

X = df.drop(["label", "difficulty", "attack_category"], axis=1)
y = df["attack_category"]

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

categorical_cols = ["protocol_type", "service", "flag"]
numerical_cols = [col for col in X.columns if col not in categorical_cols]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ]
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Multi-Class Model Accuracy:")
print(f"{accuracy:.4f}")

print("\nClassification Report:")
report = classification_report(
    y_test,
    y_pred,
    target_names=label_encoder.classes_
)

print(report)

with open("outputs/multiclass_classification_report.txt", "w") as f:
    f.write(report)

joblib.dump(pipeline, "models/multiclass_cyber_threat_model.pkl")
joblib.dump(label_encoder, "models/attack_label_encoder.pkl")

print("\nMulti-class model saved successfully.")