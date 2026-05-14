import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

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
# Create Binary Target
# ---------------------------------------------------

df["target"] = df["label"].apply(
    lambda x: 0 if x == "normal" else 1
)

# ---------------------------------------------------
# Drop Unnecessary Columns
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

# ---------------------------------------------------
# Combine Transformers
# ---------------------------------------------------

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
# Fit and Transform
# ---------------------------------------------------

X_train_processed = preprocessor.fit_transform(X_train)

X_test_processed = preprocessor.transform(X_test)

# ---------------------------------------------------
# Print Information
# ---------------------------------------------------

print("Training Data Shape:")
print(X_train_processed.shape)

print("\nTesting Data Shape:")
print(X_test_processed.shape)

print("\nTarget Distribution:")
print(y.value_counts())

print("\nPreprocessing completed successfully.")