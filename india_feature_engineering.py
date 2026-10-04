import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# SAFE ROAD AI - INDIA FEATURE ENGINEERING
# ============================================================

INPUT_FILE = Path(
    "dataset/India_Integrated_Road_Accidents.csv"
)

OUTPUT_FILE = Path(
    "dataset/India_ML_Dataset.csv"
)

print("\n" + "=" * 60)
print("SAFE ROAD AI - INDIA FEATURE ENGINEERING")
print("=" * 60)


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

print("\nOriginal Shape:", df.shape)


# ============================================================
# 2. CHECK REQUIRED COLUMNS
# ============================================================

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


# ============================================================
# 3. NUMERIC CONVERSION
# ============================================================

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


# ============================================================
# 4. REMOVE INVALID ROWS
# ============================================================

df = df.dropna(
    subset=[
        "State",
        "Year"
    ]
)


# ============================================================
# 5. CREATE ACCIDENT DENSITY
# ============================================================

# Accident count is used as the main indicator
# of accident risk at state-year level.

df["Accident_Density"] = (
    df["Accidents"]
)


# ============================================================
# 6. CREATE FATALITY RATE
# ============================================================

df["Fatality_Rate"] = np.nan

mask = (
    df["Accidents"].notna()
    &
    (df["Accidents"] > 0)
    &
    df["Deaths"].notna()
)

df.loc[
    mask,
    "Fatality_Rate"
] = (
    df.loc[mask, "Deaths"]
    /
    df.loc[mask, "Accidents"]
) * 100


# ============================================================
# 7. CREATE INJURY RATE
# ============================================================

df["Injury_Rate"] = np.nan

mask = (
    df["Accidents"].notna()
    &
    (df["Accidents"] > 0)
    &
    df["Injuries"].notna()
)

df.loc[
    mask,
    "Injury_Rate"
] = (
    df.loc[mask, "Injuries"]
    /
    df.loc[mask, "Accidents"]
) * 100


# ============================================================
# 8. CREATE YEAR-OVER-YEAR ACCIDENT CHANGE
# ============================================================

df = df.sort_values(
    ["State", "Year"]
)

df["Accident_Change_Percent"] = (
    df.groupby("State")["Accidents"]
    .pct_change()
    * 100
)


# ============================================================
# 9. CREATE STATE-WISE AVERAGE ACCIDENTS
# ============================================================

df["State_Average_Accidents"] = (
    df.groupby("State")["Accidents"]
    .transform("mean")
)


# ============================================================
# 10. CREATE RISK SCORE
# ============================================================

# We calculate a relative risk score
# using accident count compared with
# the overall median accident count.

median_accidents = df["Accidents"].median()

df["Risk_Score"] = (
    df["Accidents"] /
    median_accidents
)


# ============================================================
# 11. CREATE RISK LEVEL
# ============================================================

def assign_risk(score):

    if pd.isna(score):
        return "Unknown"

    elif score >= 1.5:
        return "High"

    elif score >= 0.75:
        return "Medium"

    else:
        return "Low"


df["Risk_Level"] = (
    df["Risk_Score"]
    .apply(assign_risk)
)


# ============================================================
# 12. DISPLAY RISK DISTRIBUTION
# ============================================================

print("\nRisk Level Distribution:")

print(
    df["Risk_Level"]
    .value_counts()
)


# ============================================================
# 13. DISPLAY FINAL COLUMNS
# ============================================================

print("\nFinal Columns:")

print(
    df.columns.tolist()
)


# ============================================================
# 14. DISPLAY SAMPLE
# ============================================================

print("\nFirst 10 rows:")

print(
    df.head(10).to_string(
        index=False
    )
)


# ============================================================
# 15. SAVE ML DATASET
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\n" + "=" * 60)
print("FEATURE ENGINEERING COMPLETED")
print("=" * 60)

print("\nFinal Shape:", df.shape)

print("\nSaved file:")

print(OUTPUT_FILE)