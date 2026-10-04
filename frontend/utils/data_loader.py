from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT / "dataset"
USA_FILE = DATASET_DIR / "US Datasets" / "cleaned_US_Accidents.csv.gz"
INDIA_FILES = [
    DATASET_DIR / "India_Final_Checked_Dataset.csv",
    DATASET_DIR / "India_Integrated_Road_Accidents.csv",
    DATASET_DIR / "India_ML_Dataset.csv",
    DATASET_DIR / "Indian Datasets" / "india_accidents_eda.csv",
]

def load_usa():
    if not USA_FILE.exists():
        return pd.DataFrame()
    return pd.read_csv(USA_FILE, low_memory=False)

def load_india():
    for path in INDIA_FILES:
        if path.exists():
            return pd.read_csv(path, low_memory=False)
    return pd.DataFrame()

def find_col(df, names):
    if df.empty:
        return None

    exact = {str(c).strip().lower(): c for c in df.columns}

    for name in names:
        if name.lower() in exact:
            return exact[name.lower()]

    for col in df.columns:
        low = str(col).lower()
        for name in names:
            if name.lower() in low:
                return col

    return None

def country_states(country):
    df = load_usa() if country == "USA" else load_india()

    col = find_col(
        df,
        ["State", "State/UT", "State_Name", "state_name"]
    )

    if col is None:
        return ["Other"]

    values = (
        df[col]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )

    return sorted(values) if values else ["Other"]