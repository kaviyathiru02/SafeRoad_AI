import pandas as pd
from pathlib import Path

# ============================================================
# SAFE ROAD AI - INTEGRATED DATASET CHECK
# ============================================================

file_path = Path(
    "dataset/India_Integrated_Road_Accidents.csv"
)

print("\n" + "=" * 60)
print("SAFE ROAD AI - INTEGRATED DATASET CHECK")
print("=" * 60)

# ------------------------------------------------------------
# Load dataset
# ------------------------------------------------------------

df = pd.read_csv(file_path)

print("\nDataset loaded successfully!")

print("\nShape:")
print(df.shape)

# ------------------------------------------------------------
# Display columns
# ------------------------------------------------------------

print("\nColumns:")
print(df.columns.tolist())

# ------------------------------------------------------------
# First 10 rows
# ------------------------------------------------------------

print("\nFirst 10 rows:")
print(df.head(10).to_string(index=False))

# ------------------------------------------------------------
# Data types
# ------------------------------------------------------------

print("\nData Types:")
print(df.dtypes)

# ------------------------------------------------------------
# Missing values
# ------------------------------------------------------------

print("\nMissing Values:")
print(df.isnull().sum())

# ------------------------------------------------------------
# Duplicate rows
# ------------------------------------------------------------

print("\nDuplicate Rows:")
print(df.duplicated().sum())

# ------------------------------------------------------------
# Number of states
# ------------------------------------------------------------

print("\nNumber of States/UTs:")
print(df["State"].nunique())

# ------------------------------------------------------------
# List states
# ------------------------------------------------------------

print("\nStates/UTs:")
print(
    sorted(df["State"].dropna().unique())
)

# ------------------------------------------------------------
# Available years
# ------------------------------------------------------------

print("\nAvailable Years:")
print(
    sorted(df["Year"].dropna().unique())
)

# ------------------------------------------------------------
# Year-wise records
# ------------------------------------------------------------

print("\nRecords by Year:")
print(
    df.groupby("Year").size()
)

# ------------------------------------------------------------
# Accident statistics
# ------------------------------------------------------------

print("\nAccident Statistics:")

if "Accidents" in df.columns:

    print(
        df["Accidents"].describe()
    )

# ------------------------------------------------------------
# Death statistics
# ------------------------------------------------------------

print("\nDeath Statistics:")

if "Deaths" in df.columns:

    print(
        df["Deaths"].describe()
    )

# ------------------------------------------------------------
# Injury statistics
# ------------------------------------------------------------

print("\nInjury Statistics:")

if "Injuries" in df.columns:

    print(
        df["Injuries"].describe()
    )

# ------------------------------------------------------------
# Check invalid negative values
# ------------------------------------------------------------

print("\nNegative Values Check:")

for column in [
    "Accidents",
    "Deaths",
    "Injuries"
]:

    if column in df.columns:

        negative_count = (
            df[column] < 0
        ).sum()

        print(
            column,
            ":",
            negative_count
        )

# ------------------------------------------------------------
# Save verified copy
# ------------------------------------------------------------

output_file = Path(
    "dataset/India_Final_Checked_Dataset.csv"
)

df.to_csv(
    output_file,
    index=False
)

print("\n" + "=" * 60)
print("DATASET CHECK COMPLETED")
print("=" * 60)

print("\nVerified dataset saved at:")

print(output_file)