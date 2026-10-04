import pandas as pd
import numpy as np
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ============================================================
# SAFE ROAD AI - USA RANDOM FOREST (LEAKAGE-FREE)
# ============================================================

INPUT_CANDIDATES = [
    Path("outputs/features_usa.csv"),
    Path("dataset/US Datasets/preprocessed_US_Accidents.csv"),
    Path("dataset/preprocessed_US_Accidents.csv")
]

MODEL_OUTPUT = Path("models/usa_risk_model.pkl")

print("\n" + "=" * 65)
print("SAFE ROAD AI - USA RANDOM FOREST (TARGET LEAKAGE REMOVED)")
print("=" * 65)

# 1. LOAD DATASET
input_file = None
for p in INPUT_CANDIDATES:
    if p.exists():
        input_file = p
        break

if input_file is None:
    raise FileNotFoundError("Could not locate USA features dataset.")

print(f"\nLoading dataset from: {input_file}")
df = pd.read_csv(input_file)
print("Dataset Shape:", df.shape)

# Ensure High_Severity target is present
if "High_Severity" not in df.columns:
    if "Severity" in df.columns:
        print("Deriving 'High_Severity' binary target from (Severity >= 3)...")
        df["High_Severity"] = (df["Severity"] >= 3).astype(int)
    else:
        raise ValueError("Neither 'High_Severity' nor 'Severity' found in dataset!")

target = "High_Severity"

# 2. DEFINE LEGITIMATE PRE-CRASH FEATURES (NO TARGET LEAKAGE)
# CRITICAL FIX: 'Severity' is EXCLUDED because High_Severity is derived directly from Severity >= 3.
features = [
    "Distance(mi)",
    "Temperature(F)",
    "Humidity(%)",
    "Visibility(mi)",
    "Wind_Speed(mph)",
    "Is_Weekend",
    "Is_Night",
    "Low_Visibility",
    "Extreme_Temperature",
    "Long_Distance_Accident",
    "Amenity",
    "Bump",
    "Crossing",
    "Junction",
    "Traffic_Signal",
    "Stop",
    "Give_Way",
    "Railway",
    "Roundabout",
    "No_Exit",
    "Traffic_Calming"
]

# Academic integrity check: Verify 'Severity' is NOT in features
assert "Severity" not in features, "Target leakage detected! 'Severity' must not be in features."

# Keep only existing features
features = [f for f in features if f in df.columns]

print(f"\nFeatures selected ({len(features)} variables):")
for f in features:
    print(f" - {f}")

print("\nTarget variable:", target)
print("Target distribution:")
print(df[target].value_counts())
print("Target proportions:")
print(df[target].value_counts(normalize=True))

# 3. PREPARE X AND y
X = df[features].copy()
y = df[target].copy()

# Handle missing values if any
for col in X.columns:
    if X[col].isnull().any():
        X[col] = X[col].fillna(X[col].median())

# 4. STRATIFIED TRAIN-TEST SPLIT (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"\nTraining samples: {len(X_train):,}")
print(f"Testing samples : {len(X_test):,}")

# 5. TRAIN RANDOM FOREST MODEL
print("\nTraining Random Forest Model (Leakage-Safe)...")
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

rf.fit(X_train, y_train)
print("Training completed successfully!")

# 6. EVALUATION ON UNSEEN TEST SET
y_pred = rf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
macro_p = precision_score(y_test, y_pred, average="macro", zero_division=0)
macro_r = recall_score(y_test, y_pred, average="macro", zero_division=0)
macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

weighted_p = precision_score(y_test, y_pred, average="weighted", zero_division=0)
weighted_r = recall_score(y_test, y_pred, average="weighted", zero_division=0)
weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

binary_p = precision_score(y_test, y_pred, average="binary", zero_division=0)
binary_r = recall_score(y_test, y_pred, average="binary", zero_division=0)
binary_f1 = f1_score(y_test, y_pred, average="binary", zero_division=0)

cm = confusion_matrix(y_test, y_pred)

print("\n" + "=" * 65)
print("USA MODEL EVALUATION RESULTS (LEAKAGE-FREE)")
print("=" * 65)
print(f"Accuracy          : {acc * 100:.2f}%")
print(f"Weighted Precision: {weighted_p * 100:.2f}%")
print(f"Weighted Recall   : {weighted_r * 100:.2f}%")
print(f"Weighted F1-score : {weighted_f1 * 100:.2f}%")
print(f"Macro Precision   : {macro_p * 100:.2f}%")
print(f"Macro Recall      : {macro_r * 100:.2f}%")
print(f"Macro F1-score    : {macro_f1 * 100:.2f}%")
print(f"High-Severity Precision (Class 1): {binary_p * 100:.2f}%")
print(f"High-Severity Recall    (Class 1): {binary_r * 100:.2f}%")
print(f"High-Severity F1-score  (Class 1): {binary_f1 * 100:.2f}%")

print("\nConfusion Matrix:")
print(cm)

print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, digits=4, zero_division=0))

# 7. FEATURE IMPORTANCES
importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": rf.feature_importances_
}).sort_values(by="Importance", ascending=False).reset_index(drop=True)

print("\nFeature Importances (Top to Bottom):")
print(importance_df.to_string(index=False))

# 8. SAVE RETRAINED MODEL IN EXISTING FORMAT
MODEL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
save_dict = {
    "model": rf,
    "features": features
}
joblib.dump(save_dict, MODEL_OUTPUT)
print(f"\n[OK] Retrained model saved successfully to: {MODEL_OUTPUT}")
print("Verified format: dictionary containing 'model' and 'features'.")
