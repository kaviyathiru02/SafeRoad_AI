import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# ============================================================
# 1. LOAD DATASET
# ============================================================

FILE_PATH = "dataset/India_ML_Dataset.csv"

df = pd.read_csv(FILE_PATH)

print("=" * 60)
print("INDIA K-MEANS ACCIDENT HOTSPOT DETECTION")
print("=" * 60)

print("\nOriginal dataset shape:", df.shape)


# ============================================================
# 2. BASIC CLEANING
# ============================================================

# Remove unwanted spaces from column names
df.columns = df.columns.str.strip()

# Clean State names
df["State"] = df["State"].astype(str).str.strip()

# Remove Total/All India records if present
df = df[
    ~df["State"].str.lower().str.contains(
        "total|all india|india total",
        na=False
    )
]

print("After removing Total/All India:", df.shape)


# ============================================================
# 3. NUMERIC FEATURES
# ============================================================

features = [
    "Accidents",
    "Deaths",
    "Injuries",
    "Fatality_Rate",
    "Injury_Rate",
    "Accident_Density",
    "Accident_Change_Percent",
    "State_Average_Accidents",
    "Risk_Score"
]

# Check that all required columns exist
available_features = [
    col for col in features
    if col in df.columns
]

print("\nFeatures used for K-Means:")
print(available_features)


# Convert selected columns to numeric
for col in available_features:
    df[col] = pd.to_numeric(df[col], errors="coerce")


# ============================================================
# 4. HANDLE MISSING VALUES
# ============================================================

print("\nMissing values before cleaning:")

print(df[available_features].isnull().sum())

# Fill missing numerical values using median
for col in available_features:
    df[col] = df[col].fillna(df[col].median())


print("\nMissing values after cleaning:")

print(df[available_features].isnull().sum())


# ============================================================
# 5. CREATE ONE RECORD PER STATE
# ============================================================

# Average yearly values for each state
state_data = df.groupby("State")[available_features].mean().reset_index()

print("\nState-level dataset shape:", state_data.shape)

print("\nNumber of unique states:", state_data["State"].nunique())


# ============================================================
# 6. PREPARE DATA FOR K-MEANS
# ============================================================

X = state_data[available_features]

# Standardization is important because features have
# different scales.
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# ============================================================
# 7. K-MEANS CLUSTERING
# ============================================================

print("\nRunning K-Means clustering...")

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=20
)

state_data["Cluster"] = kmeans.fit_predict(X_scaled)


# ============================================================
# 8. IDENTIFY LOW / MEDIUM / HIGH RISK CLUSTERS
# ============================================================

# Calculate average Risk Score for each cluster
cluster_risk = (
    state_data
    .groupby("Cluster")["Risk_Score"]
    .mean()
    .sort_values()
)

print("\nAverage Risk Score by Cluster:")
print(cluster_risk)


# Lowest Risk Score = Low
# Middle Risk Score = Medium
# Highest Risk Score = High

risk_mapping = {}

sorted_clusters = cluster_risk.index.tolist()

risk_mapping[sorted_clusters[0]] = "Low"
risk_mapping[sorted_clusters[1]] = "Medium"
risk_mapping[sorted_clusters[2]] = "High"

state_data["Hotspot_Level"] = state_data["Cluster"].map(risk_mapping)


# ============================================================
# 9. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 60)
print("STATE-WISE HOTSPOT RESULTS")
print("=" * 60)

result = state_data[
    [
        "State",
        "Risk_Score",
        "Cluster",
        "Hotspot_Level"
    ]
].sort_values("Risk_Score", ascending=False)

print(result.to_string(index=False))


# ============================================================
# 10. PCA FOR VISUALIZATION
# ============================================================

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

state_data["PCA1"] = X_pca[:, 0]
state_data["PCA2"] = X_pca[:, 1]


# ============================================================
# 11. CLEAN K-MEANS GRAPH
# ============================================================

plt.figure(figsize=(12, 8))

colors = {
    "Low": "green",
    "Medium": "orange",
    "High": "red"
}

for level in ["Low", "Medium", "High"]:

    subset = state_data[
        state_data["Hotspot_Level"] == level
    ]

    plt.scatter(
        subset["PCA1"],
        subset["PCA2"],
        s=100,
        alpha=0.75,
        label=f"{level} Risk",
        color=colors[level],
        edgecolors="black"
    )


# Label only important/high-risk states
high_risk = state_data[
    state_data["Hotspot_Level"] == "High"
].sort_values(
    "Risk_Score",
    ascending=False
)

# Label up to 10 high-risk states
for _, row in high_risk.head(10).iterrows():

    plt.annotate(
        row["State"],
        (row["PCA1"], row["PCA2"]),
        xytext=(6, 6),
        textcoords="offset points",
        fontsize=9
    )


plt.title(
    "K-Means Clustering of Indian Road Accident Risk",
    fontsize=18,
    fontweight="bold"
)

plt.xlabel(
    "Principal Component 1",
    fontsize=12
)

plt.ylabel(
    "Principal Component 2",
    fontsize=12
)

plt.legend(
    title="Hotspot Level"
)

plt.grid(
    alpha=0.25,
    linestyle="--"
)

plt.tight_layout()

plt.show()


# ============================================================
# 12. SAVE HOTSPOT DATASET
# ============================================================

output_columns = [
    "State",
    "Accidents",
    "Deaths",
    "Injuries",
    "Fatality_Rate",
    "Injury_Rate",
    "Accident_Density",
    "Accident_Change_Percent",
    "State_Average_Accidents",
    "Risk_Score",
    "Cluster",
    "Hotspot_Level"
]

output = state_data[output_columns].copy()

output = output.sort_values(
    "Risk_Score",
    ascending=False
)

OUTPUT_FILE = "dataset/India_Accident_Hotspots.csv"

output.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 60)
print("K-MEANS COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nOutput file created:")
print(OUTPUT_FILE)

print("\nHotspot distribution:")

print(
    output["Hotspot_Level"]
    .value_counts()
)

print("\nTop 10 High-Risk States:")

print(
    output[
        output["Hotspot_Level"] == "High"
    ][
        ["State", "Risk_Score", "Hotspot_Level"]
    ].head(10).to_string(index=False)
)