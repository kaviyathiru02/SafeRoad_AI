import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


# ============================================================
# SAFE ROAD AI - RANDOM FOREST EVALUATION
# ============================================================

INPUT_FILE = Path("dataset/India_ML_Dataset.csv")

print("\n" + "=" * 70)
print("SAFE ROAD AI - RANDOM FOREST EVALUATION")
print("=" * 70)


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

print("\nOriginal dataset shape:", df.shape)


# ============================================================
# 2. TARGET
# ============================================================

target = "Risk_Level"

if target not in df.columns:
    raise ValueError(
        f"{target} column not found in dataset."
    )


# Keep only valid classes
df = df[
    df[target].isin(
        ["Low", "Medium", "High"]
    )
].copy()


print("\nRisk distribution:")
print(
    df[target].value_counts()
)


# ============================================================
# 3. FEATURES
# ============================================================

features = [
    "Year",
    "Accidents",
    "Deaths",
    "Injuries",
    "Fatality_Rate",
    "Injury_Rate",
    "Accident_Change_Percent",
    "State_Average_Accidents",
]

features = [
    column
    for column in features
    if column in df.columns
]


print("\nFeatures:")
print(features)


# ============================================================
# 4. PREPARE X / Y
# ============================================================

X = df[features].copy()
y = df[target].copy()


# ============================================================
# 5. HANDLE MISSING VALUES
# ============================================================

for column in X.columns:

    if X[column].isnull().any():

        X[column] = X[column].fillna(
            X[column].median()
        )


# ============================================================
# 6. ENCODE TARGET
# ============================================================

encoder = LabelEncoder()

y_encoded = encoder.fit_transform(y)


# ============================================================
# 7. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded,
)


# ============================================================
# 8. TRAIN RANDOM FOREST
# ============================================================

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1,
)


model.fit(
    X_train,
    y_train,
)


# ============================================================
# 9. TEST SET PREDICTION
# ============================================================

y_pred = model.predict(
    X_test
)


# ============================================================
# 10. METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred,
)

precision = precision_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0,
)

recall = recall_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0,
)

f1 = f1_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0,
)


print("\n" + "=" * 70)
print("TEST SET PERFORMANCE")
print("=" * 70)

print(f"\nAccuracy          : {accuracy * 100:.2f}%")
print(f"Macro Precision   : {precision * 100:.2f}%")
print(f"Macro Recall      : {recall * 100:.2f}%")
print(f"Macro F1-score    : {f1 * 100:.2f}%")


# ============================================================
# 11. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=encoder.classes_,
        zero_division=0,
    )
)


# ============================================================
# 12. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred,
)

print("\nConfusion Matrix:")
print(cm)


# ============================================================
# 13. CROSS VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("5-FOLD CROSS-VALIDATION")
print("=" * 70)


cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)


cv_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1,
)


cv_scores = cross_val_score(
    cv_model,
    X,
    y_encoded,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1,
)


print(
    "\nFold accuracies:"
)

for index, score in enumerate(
    cv_scores,
    start=1
):

    print(
        f"Fold {index}: {score * 100:.2f}%"
    )


print(
    f"\nCross-validation mean: "
    f"{cv_scores.mean() * 100:.2f}%"
)

print(
    f"Cross-validation std : "
    f"{cv_scores.std() * 100:.2f}%"
)


# ============================================================
# 14. FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame(
    {
        "Feature": features,
        "Importance": model.feature_importances_,
    }
).sort_values(
    "Importance",
    ascending=False,
)


print("\n" + "=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)

print(
    importance.to_string(
        index=False
    )
)


# ============================================================
# 15. SIMPLE LEAKAGE WARNING
# ============================================================

print("\n" + "=" * 70)
print("DATA-LEAKAGE REVIEW")
print("=" * 70)

possible_risk_features = [
    "Accidents",
    "Deaths",
    "Injuries",
    "Fatality_Rate",
    "Injury_Rate",
    "Accident_Change_Percent",
    "State_Average_Accidents",
]

used_risk_features = [
    feature
    for feature in possible_risk_features
    if feature in features
]


print(
    "\nThe model uses these risk-related variables:"
)

for feature in used_risk_features:
    print(
        f"- {feature}"
    )


print(
    """
IMPORTANT:
If Risk_Level was originally created using thresholds or formulas
based directly on any of the above variables, the model may have
target leakage. In that situation, the accuracy can be unusually
high because the target is partly encoded inside the input features.

Check how Risk_Level was created before claiming the 97.37% result
as the final real-world model performance.
"""
)


print("\nEvaluation completed.")