import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 1. LOAD INDIAN ACCIDENT DATASET
# ==========================================

file_path = "dataset/Indian Datasets/India_Statewise_Road_Accidents.csv"

df = pd.read_csv(file_path)

print("Dataset Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())


# ==========================================
# 2. CLEAN COLUMN NAMES
# ==========================================

df.columns = df.columns.str.strip()

# Remove unnecessary Sl. No. column if available
if "Sl. No." in df.columns:
    df = df.drop(columns=["Sl. No."])


# ==========================================
# 3. AVAILABLE YEARS
# ==========================================

years = [2018, 2019, 2020, 2021, 2022]

accidents = []

for year in years:

    if str(year) in df.columns:
        total = pd.to_numeric(
            df[str(year)],
            errors="coerce"
        ).sum()

        accidents.append(total)

    elif year in df.columns:
        total = pd.to_numeric(
            df[year],
            errors="coerce"
        ).sum()

        accidents.append(total)

    else:
        accidents.append(0)


# ==========================================
# 4. DISPLAY TOTAL ACCIDENTS
# ==========================================

print("\nTotal Road Accidents by Year:")

for year, total in zip(years, accidents):
    print(year, ":", int(total))


# ==========================================
# 5. GRAPH – TOTAL ACCIDENTS BY YEAR
# ==========================================

plt.figure(figsize=(10, 6))

bars = plt.bar(years, accidents)

plt.title(
    "Total Road Accidents in India by Year",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Year", fontsize=12)
plt.ylabel("Number of Accidents", fontsize=12)

plt.xticks(years)

# Add values on top of bars
for bar, value in zip(bars, accidents):

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{int(value):,}",
        ha="center",
        va="bottom",
        fontsize=10
    )

plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.4
)

plt.tight_layout()

plt.show()


# ==========================================
# 6. STATE-WISE ACCIDENT ANALYSIS
# ==========================================

state_column = None

for col in df.columns:

    if "State" in col or "States" in col:
        state_column = col
        break

if state_column is not None:

    # Calculate total accidents for each state
    df["Total_Accidents"] = 0

    for year in years:

        if str(year) in df.columns:

            df["Total_Accidents"] += pd.to_numeric(
                df[str(year)],
                errors="coerce"
            ).fillna(0)

        elif year in df.columns:

            df["Total_Accidents"] += pd.to_numeric(
                df[year],
                errors="coerce"
            ).fillna(0)

    # ==========================================
    # TOP 10 STATES
    # ==========================================

    top10 = df.sort_values(
        "Total_Accidents",
        ascending=False
    ).head(10)

    print("\nTop 10 States with Highest Road Accidents:")

    print(
        top10[
            [state_column, "Total_Accidents"]
        ].to_string(index=False)
    )


    # ==========================================
    # TOP 10 STATES GRAPH
    # ==========================================

    plt.figure(figsize=(12, 6))

    plt.bar(
        top10[state_column],
        top10["Total_Accidents"]
    )

    plt.title(
        "Top 10 States with Highest Road Accidents",
        fontsize=16,
        fontweight="bold"
    )

    plt.xlabel("State / UT")
    plt.ylabel("Total Accidents")

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.grid(
        axis="y",
        linestyle="--",
        alpha=0.4
    )

    plt.tight_layout()

    plt.show()


# ==========================================
# 7. SAVE CLEANED DATA
# ==========================================

output_file = "dataset/Indian Datasets/india_accidents_eda.csv"

df.to_csv(
    output_file,
    index=False
)

print("\nEDA dataset saved successfully:")
print(output_file)