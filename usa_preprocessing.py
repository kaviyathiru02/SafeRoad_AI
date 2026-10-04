import pandas as pd
from sklearn.preprocessing import LabelEncoder
df = pd.read_csv("dataset/cleaned_US_Accidents.csv")
print("Missing Values:")
print(df.isnull().sum())
numerical_columns = [
    "Temperature(F)",
    "Humidity(%)",
    "Visibility(mi)",
    "Wind_Speed(mph)"
]

for col in numerical_columns:
    df[col] = df[col].fillna(df[col].median())
    categorical_columns = [
    "Weather_Condition",
    "Sunrise_Sunset",
    "City",
    "County",
    "State"
]

for col in categorical_columns:
    df[col] = df[col].fillna(df[col].mode()[0])
print("\nMissing Values After Filling:")
print(df.isnull().sum())
encoder = LabelEncoder()

for col in categorical_columns:
    df[col] = encoder.fit_transform(df[col])
print("\nData Types:")
print(df.dtypes)
df.to_csv("dataset/preprocessed_US_Accidents.csv", index=False)

print("\n✅ Preprocessed dataset saved successfully!")