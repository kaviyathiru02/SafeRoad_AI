import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# ============================================================
# SAFE ROAD AI - FINAL TIME-BASED LEAKAGE-SAFE EVALUATION
# ============================================================

INPUT_FILE = Path(
    "dataset/India_Leakage_Safe_ML_Dataset.csv"
)

OUTPUT_FILE = Path(
    "dataset/India_Time_Based_Final_Evaluation.csv"
)

print("\n" + "=" * 70)
print("SAFE ROAD AI - FINAL TIME-BASED LEAKAGE-SAFE EVALUATION")
print("=" * 70)

# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

print("\nDataset shape:", df.shape)

# ============================================================
# 2. FEATURES AND TARGET
# ============================================================

features = [
    "Year",
    "Previous_Year_Accidents",
    "Previous_Year_Deaths",
    "Historical_Accident_Change"
]

target = "Future_Risk_Level"

print("\nFeatures used:")
for feature in features:
    print("-", feature)

print("\nTarget:")
print("-", target)

# ============================================================
# 3. REMOVE MISSING VALUES
# ============================================================

model_df = df.dropna(
    subset=features + [target]
).copy()

print("\nRows available for evaluation:", len(model_df))

# ============================================================
# 4. AVAILABLE YEARS
# ============================================================

years = sorted(
    model_df["Year"].unique()
)

print("\nYears available:")
print(
    model_df["Year"]
    .value_counts()
    .sort_index()
)

# ============================================================
# 5. TIME-BASED EVALUATION
# ============================================================

results = []

for test_year in years:

    training_years = [
        year for year in years
        if year < test_year
    ]

    if len(training_years) == 0:
        continue

    train_df = model_df[
        model_df["Year"].isin(training_years)
    ]

    test_df = model_df[
        model_df["Year"] == test_year
    ]

    if train_df.empty or test_df.empty:
        continue

    X_train = train_df[features]
    y_train = train_df[target]

    X_test = test_df[features]
    y_test = test_df[target]

    print("\n" + "-" * 70)
    print("TRAIN YEARS :", training_years)
    print("TEST YEAR   :", test_year)

    print(
        "Training rows:",
        len(train_df)
    )

    print(
        "Testing rows :",
        len(test_df)
    )

    # ========================================================
    # RANDOM FOREST
    # ========================================================

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(
        X_train,
        y_train
    )

    y_pred = model.predict(
        X_test
    )

    # ========================================================
    # METRICS
    # ========================================================

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    print("\nPerformance:")

    print(
        f"Accuracy        : {accuracy * 100:.2f}%"
    )

    print(
        f"Macro Precision : {precision * 100:.2f}%"
    )

    print(
        f"Macro Recall    : {recall * 100:.2f}%"
    )

    print(
        f"Macro F1-score  : {f1 * 100:.2f}%"
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )

    results.append({
        "test_year": test_year,
        "training_years": ",".join(
            map(str, training_years)
        ),
        "training_rows": len(train_df),
        "testing_rows": len(test_df),
        "accuracy": accuracy * 100,
        "precision": precision * 100,
        "recall": recall * 100,
        "f1": f1 * 100
    })

# ============================================================
# 6. RESULTS SUMMARY
# ============================================================

results_df = pd.DataFrame(
    results
)

print("\n" + "=" * 70)
print("FINAL TIME-BASED LEAKAGE-SAFE SUMMARY")
print("=" * 70)

print("\nResults by test year:")

print(
    results_df.to_string(
        index=False
    )
)

# ============================================================
# 7. OVERALL AVERAGE
# ============================================================

if not results_df.empty:

    mean_accuracy = results_df[
        "accuracy"
    ].mean()

    mean_precision = results_df[
        "precision"
    ].mean()

    mean_recall = results_df[
        "recall"
    ].mean()

    mean_f1 = results_df[
        "f1"
    ].mean()

    print("\nAverage across time-based tests:")

    print(
        f"Mean Accuracy        : {mean_accuracy:.2f}%"
    )

    print(
        f"Mean Macro Precision : {mean_precision:.2f}%"
    )

    print(
        f"Mean Macro Recall    : {mean_recall:.2f}%"
    )

    print(
        f"Mean Macro F1-score  : {mean_f1:.2f}%"
    )

# ============================================================
# 8. SAVE RESULTS
# ============================================================

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nEvaluation summary saved:")
print(OUTPUT_FILE)

print("\n" + "=" * 70)
print("FINAL EVALUATION COMPLETED")
print("=" * 70)