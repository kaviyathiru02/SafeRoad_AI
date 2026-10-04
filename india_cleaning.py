import pandas as pd
from pathlib import Path

# ============================================================
# SAFE ROAD AI - INDIA DATASET CLEANING
# ============================================================

BASE_FOLDER = Path("dataset/Indian Datasets")

OUTPUT_FOLDER = BASE_FOLDER / "Cleaned"

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)


print("\n==============================================")
print("SAFE ROAD AI - INDIA DATASET CLEANING")
print("==============================================\n")


# ============================================================
# FILE NAMES
# ============================================================

ACCIDENT_FILE = BASE_FOLDER / "India_Statewise_Road_Accidents.csv"

DEATH_FILE = BASE_FOLDER / "India_Statewise_Road_Accident_Deaths.csv"

INJURY_FILE = BASE_FOLDER / "India_Statewise_Road_Accident_Injuries.csv"


# ============================================================
# FUNCTION TO READ CSV
# ============================================================

def read_dataset(file_path):

    if not file_path.exists():

        print("ERROR: File not found:")
        print(file_path)

        return None

    try:

        df = pd.read_csv(file_path)

        return df

    except Exception as e:

        print("Error reading file:")
        print(file_path)

        print(e)

        return None


# ============================================================
# FUNCTION TO CLEAN COLUMN NAMES
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


# ============================================================
# FUNCTION TO FIND STATE COLUMN
# ============================================================

def find_state_column(df):

    possible_columns = [
        "States/UTs",
        "State/UT",
        "State / UT",
        "States / UTs",
        "State"
    ]

    for column in possible_columns:

        if column in df.columns:

            return column

    return None


# ============================================================
# FUNCTION TO CLEAN DATASET
# ============================================================

def clean_dataset(df, dataset_name):

    print("\n----------------------------------------------")
    print("Cleaning:", dataset_name)
    print("----------------------------------------------")

    if df is None:

        print("Dataset could not be loaded.")

        return None


    # --------------------------------------------
    # Clean column names
    # --------------------------------------------

    df = clean_columns(df)


    print("Original shape:", df.shape)


    # --------------------------------------------
    # Find State column
    # --------------------------------------------

    state_column = find_state_column(df)


    if state_column is None:

        print("ERROR: State column not found.")

        print("\nAvailable columns:")

        for column in df.columns:

            print("-", column)

        return None


    # --------------------------------------------
    # Rename State column
    # --------------------------------------------

    df = df.rename(
        columns={
            state_column: "State"
        }
    )


    # --------------------------------------------
    # Clean State names
    # --------------------------------------------

    df["State"] = (
        df["State"]
        .astype(str)
        .str.strip()
    )


    # --------------------------------------------
    # Remove unwanted rows
    # --------------------------------------------

    unwanted_states = [
        "",
        "nan",
        "NaN",
        "Total",
        "TOTAL",
        "Total India",
        "India",
        "Grand Total"
    ]


    df = df[
        ~df["State"].isin(unwanted_states)
    ]


    # --------------------------------------------
    # Remove duplicate rows
    # --------------------------------------------

    before = len(df)

    df = df.drop_duplicates()

    after = len(df)

    print(
        "Duplicate rows removed:",
        before - after
    )


    # --------------------------------------------
    # Convert numeric columns
    # --------------------------------------------

    for column in df.columns:

        if column == "State":

            continue

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )


    # --------------------------------------------
    # Reset index
    # --------------------------------------------

    df = df.reset_index(
        drop=True
    )


    # --------------------------------------------
    # Missing values BEFORE filling
    # --------------------------------------------

    print("\nMissing values:")

    missing = df.isnull().sum()

    print(
        missing[missing > 0]
    )


    # --------------------------------------------
    # Fill numeric missing values
    # --------------------------------------------

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns


    for column in numeric_columns:

        df[column] = df[column].fillna(
            df[column].median()
        )


    # --------------------------------------------
    # Sort by State
    # --------------------------------------------

    df = df.sort_values(
        by="State"
    )


    # --------------------------------------------
    # Reset index again
    # --------------------------------------------

    df = df.reset_index(
        drop=True
    )


    # --------------------------------------------
    # Final shape
    # --------------------------------------------

    print("\nFinal shape:", df.shape)


    # --------------------------------------------
    # Save cleaned dataset
    # --------------------------------------------

    output_file = (
        OUTPUT_FOLDER /
        f"cleaned_india_{dataset_name}.csv"
    )


    df.to_csv(
        output_file,
        index=False
    )


    print(
        "Saved successfully:"
    )

    print(
        output_file
    )


    return df


# ============================================================
# 1. LOAD ACCIDENT DATASET
# ============================================================

print("\nLoading Accident Dataset...")

accidents = read_dataset(
    ACCIDENT_FILE
)


# ============================================================
# 2. LOAD DEATH DATASET
# ============================================================

print("\nLoading Death Dataset...")

deaths = read_dataset(
    DEATH_FILE
)


# ============================================================
# 3. LOAD INJURY DATASET
# ============================================================

print("\nLoading Injury Dataset...")

injuries = read_dataset(
    INJURY_FILE
)


# ============================================================
# 4. CLEAN ACCIDENT DATA
# ============================================================

accidents = clean_dataset(
    accidents,
    "accidents"
)


# ============================================================
# 5. CLEAN DEATH DATA
# ============================================================

deaths = clean_dataset(
    deaths,
    "deaths"
)


# ============================================================
# 6. CLEAN INJURY DATA
# ============================================================

injuries = clean_dataset(
    injuries,
    "injuries"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n==============================================")
print("CLEANING COMPLETED")
print("==============================================\n")


if accidents is not None:

    print(
        "Accidents:",
        accidents.shape
    )

else:

    print(
        "Accidents: FAILED"
    )


if deaths is not None:

    print(
        "Deaths:",
        deaths.shape
    )

else:

    print(
        "Deaths: FAILED"
    )


if injuries is not None:

    print(
        "Injuries:",
        injuries.shape
    )

else:

    print(
        "Injuries: FAILED"
    )


print("\nCleaned datasets are saved in:")

print(
    OUTPUT_FOLDER
)


print("\n==============================================")
print("FILES CREATED")
print("==============================================")

for file in OUTPUT_FOLDER.glob("*.csv"):

    print(
        "-",
        file.name
    )


print("\nCleaning process finished successfully!")