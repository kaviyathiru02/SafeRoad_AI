import pandas as pd

files = {
    "Deaths": "dataset/Indian Datasets/India_Statewise_Road_Accident_Deaths.csv",
    "Accidents": "dataset/Indian Datasets/India_Statewise_Road_Accidents.csv",
    "Injuries": "dataset/Indian Datasets/India_Statewise_Road_Accident_Injuries.csv"
}

for name, file in files.items():

    print("\n" + "=" * 60)
    print(name.upper(), "DATASET")
    print("=" * 60)

    df = pd.read_csv(file)

    print("\nShape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nMissing values:")
    print(df.isnull().sum())