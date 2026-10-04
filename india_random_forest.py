import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt


# ============================================================
# SAFE ROAD AI - INDIA RANDOM FOREST
# ============================================================

INPUT_FILE = Path(
    "dataset/India_ML_Dataset.csv"
)

MODEL_OUTPUT = Path(
    "dataset/India_RandomForest_Predictions.csv"
)


print("\n" + "=" * 60)
print("SAFE ROAD AI - INDIA RANDOM FOREST")
print("=" * 60)


# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv(INPUT_FILE)

print("\nDataset Shape:", df.shape)


# ============================================================
# 2. CHECK TARGET
# ============================================================

target = "Risk_Level"

if target not in df.columns:

    raise ValueError(
        "Risk_Level column not found!"
    )


# ============================================================
# 3. REMOVE UNKNOWN TARGETS
# ============================================================

df = df[
    df[target].isin(
        ["Low", "Medium", "High"]
    )
].copy()


print(
    "\nRisk distribution:"
)

print(
    df[target].value_counts()
)


# ============================================================
# 4. SELECT FEATURES
# ============================================================

features = [
    "Year",
    "Accidents",
    "Deaths",
    "Injuries",
    "Fatality_Rate",
    "Injury_Rate",
    "Accident_Change_Percent",
    "State_Average_Accidents"
]


# Keep only features that actually exist
features = [
    column
    for column in features
    if column in df.columns
]


print("\nFeatures used:")

print(features)


# ============================================================
# 5. PREPARE X AND Y
# ============================================================

X = df[features].copy()

y = df[target].copy()


# ============================================================
# 6. HANDLE MISSING VALUES
# ============================================================

for column in X.columns:

    if X[column].isnull().any():

        X[column] = X[column].fillna(
            X[column].median()
        )


# ============================================================
# 7. ENCODE TARGET
# ============================================================

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)


print("\nTarget classes:")

for number, label in enumerate(
    label_encoder.classes_
):

    print(
        number,
        "->",
        label
    )


# ============================================================
# 8. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)


print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))


# ============================================================
# 9. RANDOM FOREST MODEL
# ============================================================

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)


print("\nTraining Random Forest...")

model.fit(
    X_train,
    y_train
)


print("Training completed!")


# ============================================================
# 10. PREDICTION
# ============================================================

y_pred = model.predict(
    X_test
)


# ============================================================
# 11. ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)


print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)


print(
    f"\nAccuracy: {accuracy * 100:.2f}%"
)


# ============================================================
# 12. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_,
        zero_division=0
    )
)


# ============================================================
# 13. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)


print("\nConfusion Matrix:")

print(cm)


# ============================================================
# 14. CONFUSION MATRIX GRAPH
# ============================================================

plt.figure(
    figsize=(7, 6)
)

plt.imshow(cm)

plt.title(
    "Random Forest - India Risk Prediction"
)

plt.xlabel(
    "Predicted Risk"
)

plt.ylabel(
    "Actual Risk"
)

plt.xticks(
    range(len(label_encoder.classes_)),
    label_encoder.classes_
)

plt.yticks(
    range(len(label_encoder.classes_)),
    label_encoder.classes_
)


# Add numbers
for i in range(cm.shape[0]):

    for j in range(cm.shape[1]):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )


plt.colorbar()

plt.tight_layout()

plt.show()


# ============================================================
# 15. FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})


importance = importance.sort_values(
    by="Importance",
    ascending=False
)


print("\nFeature Importance:")

print(
    importance.to_string(
        index=False
    )
)


# ============================================================
# 16. FEATURE IMPORTANCE GRAPH
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.barh(
    importance["Feature"],
    importance["Importance"]
)

plt.xlabel(
    "Importance"
)

plt.ylabel(
    "Feature"
)

plt.title(
    "Random Forest Feature Importance - India"
)

plt.gca().invert_yaxis()

plt.tight_layout()

plt.show()


# ============================================================
# 17. GENERATE PREDICTIONS FOR COMPLETE DATASET
# ============================================================

df["Predicted_Risk"] = label_encoder.inverse_transform(
    model.predict(X)
)


# ============================================================
# 18. SAVE RESULTS
# ============================================================

df.to_csv(
    MODEL_OUTPUT,
    index=False
)


print("\n" + "=" * 60)
print("RANDOM FOREST COMPLETED")
print("=" * 60)

print(
    "\nPrediction file saved:"
)

print(
    MODEL_OUTPUT
)