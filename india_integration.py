import pandas as pd
from pathlib import Path
import re

# ============================================================
# SAFE ROAD AI
# INDIAN DATASET INTEGRATION
# ============================================================

BASE_FOLDER = Path("dataset/Indian Datasets/Cleaned")
OUTPUT_FOLDER = Path("dataset")

print("\n" + "=" * 60)
print("SAFE ROAD AI - INDIAN DATASET INTEGRATION")
print("=" * 60)


# ============================================================
# FILE PATHS
# ============================================================

ACCIDENT_FILE = BASE_FOLDER / "cleaned_india_accidents.csv"
DEATH_FILE = BASE_FOLDER / "cleaned_india_deaths.csv"
INJURY_FILE = BASE_FOLDER / "cleaned_india_injuries.csv"


# ============================================================
# LOAD DATASETS
# ============================================================

print("\nLoading datasets...")


accidents = pd.read_csv(ACCIDENT_FILE)
deaths = pd.read_csv(DEATH_FILE)
injuries = pd.read_csv(INJURY_FILE)


print("\nLoaded datasets:")
print("Accidents :", accidents.shape)
print("Deaths    :", deaths.shape)
print("Injuries  :", injuries.shape)


# ============================================================
# CLEAN COLUMN NAMES
# ============================================================

def clean_columns(df):

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.replace("\n", " ", regex=False)
        .str.replace(r"\s+", " ", regex=True)
    )

    return df


accidents = clean_columns(accidents)
deaths = clean_columns(deaths)
injuries = clean_columns(injuries)


# ============================================================
# FIND STATE COLUMN
# ============================================================

def standardize_state_column(df):

    possible = [
        "State",
        "States/UTs",
        "State/UT",
        "State / UT",
        "States / UTs"
    ]

    for col in possible:

        if col in df.columns:

            df = df.rename(
                columns={col: "State"}
            )

            break

    if "State" not in df.columns:

        raise ValueError(
            "State column not found. Available columns:\n"
            + str(df.columns.tolist())
        )

    df["State"] = (
        df["State"]
        .astype(str)
        .str.strip()
    )

    return df


accidents = standardize_state_column(accidents)
deaths = standardize_state_column(deaths)
injuries = standardize_state_column(injuries)


# ============================================================
# ACCIDENT DATA
# ============================================================

print("\n" + "-" * 60)
print("PROCESSING ACCIDENT DATA")
print("-" * 60)


def process_accidents(df):

    records = []

    years = [
        2018,
        2019,
        2020,
        2021,
        2022
    ]

    for _, row in df.iterrows():

        state = row["State"]

        for year in years:

            column = str(year)

            if column in df.columns:

                value = pd.to_numeric(
                    row[column],
                    errors="coerce"
                )

                if pd.isna(value):
                    value = 0

                records.append({
                    "State": state,
                    "Year": year,
                    "Accidents": value
                })

    result = pd.DataFrame(records)

    print(
        "Accident records created:",
        len(result)
    )

    return result


accidents_long = process_accidents(
    accidents
)


# ============================================================
# DEATH DATA
# ============================================================

print("\n" + "-" * 60)
print("PROCESSING DEATH DATA")
print("-" * 60)


def process_deaths(df):

    records = []

    for year in [
        2018,
        2019,
        2020,
        2021
    ]:

        # Find the correct column
        matching_column = None

        for column in df.columns:

            column_lower = column.lower()

            # Must contain the year
            # Must contain killed
            # Must NOT be rank/share/per lakh/per vehicle/per road

            if (
                str(year) in column
                and "killed" in column_lower
                and "rank" not in column_lower
                and "share" not in column_lower
                and "per lakh" not in column_lower
                and "per 10,000" not in column_lower
            ):

                matching_column = column
                break


        if matching_column is None:

            print(
                f"WARNING: Death column for {year} not found"
            )

            continue


        print(
            f"{year} -> {matching_column}"
        )


        for _, row in df.iterrows():

            state = row["State"]

            value = pd.to_numeric(
                row[matching_column],
                errors="coerce"
            )

            if pd.isna(value):
                value = 0

            records.append({
                "State": state,
                "Year": year,
                "Deaths": value
            })


    result = pd.DataFrame(records)

    print(
        "Death records created:",
        len(result)
    )

    return result


deaths_long = process_deaths(
    deaths
)


# ============================================================
# INJURY DATA
# ============================================================

print("\n" + "-" * 60)
print("PROCESSING INJURY DATA")
print("-" * 60)


def process_injuries(df):

    records = []

    for year in [
        2021,
        2022,
        2023,
        2024
    ]:

        matching_column = None

        for column in df.columns:

            column_lower = column.lower()

            if (
                str(year) in column
                and "injured" in column_lower
                and "rank" not in column_lower
                and "share" not in column_lower
                and "per lakh" not in column_lower
                and "per 10,000" not in column_lower
            ):

                matching_column = column
                break


        if matching_column is None:

            print(
                f"WARNING: Injury column for {year} not found"
            )

            continue


        print(
            f"{year} -> {matching_column}"
        )


        for _, row in df.iterrows():

            state = row["State"]

            value = pd.to_numeric(
                row[matching_column],
                errors="coerce"
            )

            if pd.isna(value):
                value = 0

            records.append({
                "State": state,
                "Year": year,
                "Injuries": value
            })


    result = pd.DataFrame(records)

    print(
        "Injury records created:",
        len(result)
    )

    return result


injuries_long = process_injuries(
    injuries
)


# ============================================================
# DISPLAY CONVERTED DATA
# ============================================================

print("\n" + "=" * 60)
print("CONVERTED DATA")
print("=" * 60)

print(
    "\nAccidents:",
    accidents_long.shape
)

print(
    "Deaths:",
    deaths_long.shape
)

print(
    "Injuries:",
    injuries_long.shape
)


# ============================================================
# MERGE ACCIDENTS + DEATHS
# ============================================================

print("\nMerging Accident + Death data...")


integrated = pd.merge(
    accidents_long,
    deaths_long,
    on=["State", "Year"],
    how="outer"
)


# ============================================================
# MERGE INJURIES
# ============================================================

print("Merging Injury data...")


integrated = pd.merge(
    integrated,
    injuries_long,
    on=["State", "Year"],
    how="outer"
)


# ============================================================
# SORT
# ============================================================

integrated = integrated.sort_values(
    by=[
        "State",
        "Year"
    ]
)


integrated = integrated.reset_index(
    drop=True
)


# ============================================================
# NUMERIC CONVERSION
# ============================================================

for column in [
    "Accidents",
    "Deaths",
    "Injuries"
]:

    if column in integrated.columns:

        integrated[column] = pd.to_numeric(
            integrated[column],
            errors="coerce"
        )


# ============================================================
# IMPORTANT:
# DO NOT ASSUME MISSING DATA = ZERO
# ============================================================

print("\nMissing values before filling:")

print(
    integrated[
        [
            "Accidents",
            "Deaths",
            "Injuries"
        ]
    ].isnull().sum()
)


# ============================================================
# FATALITY RATE
# ============================================================

integrated["Fatality_Rate"] = (
    integrated["Deaths"]
    /
    integrated["Accidents"]
) * 100


# ============================================================
# INJURY RATE
# ============================================================

integrated["Injury_Rate"] = (
    integrated["Injuries"]
    /
    integrated["Accidents"]
) * 100


# ============================================================
# SAVE FINAL DATASET
# ============================================================

output_file = (
    OUTPUT_FOLDER /
    "India_Integrated_Road_Accidents.csv"
)


integrated.to_csv(
    output_file,
    index=False
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 60)
print("INTEGRATION COMPLETED SUCCESSFULLY")
print("=" * 60)


print(
    "\nFinal Dataset Shape:",
    integrated.shape
)


print("\nFinal Columns:")

for column in integrated.columns:

    print(
        "-",
        column
    )


print("\nFirst 15 rows:")

print(
    integrated.head(15).to_string(
        index=False
    )
)


print("\nFinal missing values:")

print(
    integrated.isnull().sum()
)


print("\nSaved file:")

print(
    output_file
)


print("\n" + "=" * 60)
print("INDIAN DATASET READY FOR NEXT STAGE")
print("=" * 60)