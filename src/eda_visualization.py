import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create outputs folder if not exists
os.makedirs("outputs", exist_ok=True)

# Column names
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

# Load dataset
df = pd.read_csv("data/KDDTrain+.txt", names=columns)

# Binary target
df["target"] = df["label"].apply(
    lambda x: "Normal" if x == "normal" else "Attack"
)

# Set style
sns.set_style("whitegrid")

# ---------------------------------------------------
# 1. Normal vs Attack
# ---------------------------------------------------

plt.figure(figsize=(6,5))

sns.countplot(x="target", data=df)

plt.title("Normal vs Attack Traffic")
plt.savefig("outputs/normal_vs_attack.png")
plt.close()

# ---------------------------------------------------
# 2. Protocol Distribution
# ---------------------------------------------------

plt.figure(figsize=(7,5))

sns.countplot(x="protocol_type", data=df)

plt.title("Protocol Type Distribution")
plt.savefig("outputs/protocol_distribution.png")
plt.close()

# ---------------------------------------------------
# 3. Top Attack Labels
# ---------------------------------------------------

top_attacks = df["label"].value_counts().head(10)

plt.figure(figsize=(10,6))

sns.barplot(
    x=top_attacks.values,
    y=top_attacks.index
)

plt.title("Top 10 Attack Types")
plt.xlabel("Count")

plt.savefig("outputs/top_attack_types.png")
plt.close()

# ---------------------------------------------------
# 4. Correlation Heatmap
# ---------------------------------------------------

numeric_df = df.select_dtypes(include=['int64', 'float64'])

plt.figure(figsize=(15,10))

sns.heatmap(
    numeric_df.corr(),
    cmap="coolwarm"
)

plt.title("Feature Correlation Heatmap")

plt.savefig("outputs/correlation_heatmap.png")
plt.close()

print("EDA charts saved successfully in outputs folder.")