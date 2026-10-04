import pandas as pd
import numpy as np
from pathlib import Path

INPUT_FILE = Path(
    "dataset/India_Integrated_Road_Accidents.csv"
)

OUTPUT_FILE = Path(
    "dataset/India_Leakage_Safe_ML_Dataset.csv"
)

print("\n" + "=" * 70)
print("SAFE ROAD AI - STRICT TIME-BASED LEAKAGE-SAFE DATASET")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

print("\nOriginal dataset shape:", df.shape)

required_columns = [
    "State",
    "Year",
    "Accidents",
    "Deaths",
    "Injuries"
]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(
            f"Required column missing: {column}"
        )

numeric_columns = [
    "Year",
    "Accidents",
    "Deaths",
    "Injuries"
]

for column in numeric_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

df = df.dropna(
    subset=[
        "State",
        "Year"
    ]
)

df = df.sort_values(
    ["State", "Year"]
).reset_index(drop=True)

# ============================================================
# HISTORICAL FEATURES
# ============================================================

df["Previous_Year_Accidents"] = (
    df.groupby("State")["Accidents"]
    .shift(1)
)

df["Previous_Year_Deaths"] = (
    df.groupby("State")["Deaths"]
    .shift(1)
)

df["Previous_Year_Injuries"] = (
    df.groupby("State")["Injuries"]
    .shift(1)
)

# Previous historical accident change
df["Historical_Accident_Change"] = (
    df.groupby("State")["Accidents"]
    .pct_change()
    .shift(1)
    * 100
)

# ============================================================
# FUTURE TARGET
# ============================================================

df["Future_Year_Accidents"] = (
    df.groupby("State")["Accidents"]
    .shift(-1)
)

# ============================================================
# HISTORICAL MEDIAN
# IMPORTANT:
# For each prediction year, the threshold is calculated
# ONLY from accident data available up to that year.
# ============================================================

df["Historical_Median_Accidents"] = np.nan

years = sorted(
    df["Year"].unique()
)

for year in years:

    historical_values = df.loc[
        (df["Year"] <= year)
        &
        df["Accidents"].notna(),
        "Accidents"
    ]

    if not historical_values.empty:

        median_value = (
            historical_values.median()
        )

        df.loc[
            df["Year"] == year,
            "Historical_Median_Accidents"
        ] = median_value

# ============================================================
# FUTURE RISK SCORE
# ============================================================

df["Future_Risk_Score"] = (
    df["Future_Year_Accidents"]
    /
    df["Historical_Median_Accidents"]
)

# ============================================================
# FUTURE RISK LEVEL
# ============================================================

def assign_future_risk(score):

    if pd.isna(score):
        return np.nan

    if score >= 1.5:
        return "High"

    elif score >= 0.75:
        return "Medium"

    else:
        return "Low"


df["Future_Risk_Level"] = (
    df["Future_Risk_Score"]
    .apply(assign_future_risk)
)

# ============================================================
# KEEP PREDICTION ROWS
# ============================================================

safe_df = df[
    df["Previous_Year_Accidents"].notna()
    &
    df["Future_Year_Accidents"].notna()
    &
    df["Future_Risk_Level"].notna()
].copy()

# ============================================================
# FINAL COLUMNS
# ============================================================

final_columns = [
    "State",
    "Year",
    "Previous_Year_Accidents",
    "Previous_Year_Deaths",
    "Previous_Year_Injuries",
    "Historical_Accident_Change",
    "Future_Year_Accidents",
    "Historical_Median_Accidents",
    "Future_Risk_Score",
    "Future_Risk_Level"
]

safe_df = safe_df[
    final_columns
]

# ============================================================
# INFORMATION
# ============================================================

print("\nStrict leakage-safe dataset shape:")
print(safe_df.shape)

print("\nYears available:")
print(
    safe_df["Year"]
    .value_counts()
    .sort_index()
)

print("\nRows available for each prediction year:")

for year in sorted(
    safe_df["Year"].unique()
):

    count = (
        safe_df["Year"] == year
    ).sum()

    print(
        f"{year}: {count}"
    )

print("\nFuture risk distribution:")
print(
    safe_df["Future_Risk_Level"]
    .value_counts()
)

print("\nFinal columns:")
print(
    safe_df.columns.tolist()
)

# ============================================================
# SAVE
# ============================================================

safe_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 70)
print("STRICT LEAKAGE-SAFE DATASET CREATED")
print("=" * 70)

print("\nSaved file:")
print(OUTPUT_FILE)