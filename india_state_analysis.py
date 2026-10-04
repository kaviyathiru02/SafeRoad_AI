import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 1. LOAD DATASET
# ==========================================

file_path = "dataset/Indian Datasets/India_Statewise_Road_Accidents.csv"

df = pd.read_csv(file_path)

# Clean column names
df.columns = df.columns.str.strip()

print("Dataset Shape:", df.shape)


# ==========================================
# 2. FIND STATE COLUMN
# ==========================================

state_column = "States/UTs"

if state_column not in df.columns:
    state_column = "State/UT"

print("State Column:", state_column)


# ==========================================
# 3. CALCULATE TOTAL ACCIDENTS
# ==========================================

years = [2018, 2019, 2020, 2021, 2022]

df["Total_Accidents"] = 0

for year in years:

    year_column = str(year)

    if year_column in df.columns:

        df[year_column] = pd.to_numeric(
            df[year_column],
            errors="coerce"
        ).fillna(0)

        df["Total_Accidents"] += df[year_column]


# ==========================================
# 4. TOP 10 STATES
# ==========================================

top10 = df.sort_values(
    by="Total_Accidents",
    ascending=False
).head(10)

print("\n================================")
print("TOP 10 HIGH-ACCIDENT STATES")
print("================================")

print(
    top10[
        [state_column, "Total_Accidents"]
    ].to_string(index=False)
)


# ==========================================
# 5. GRAPH
# ==========================================

plt.figure(figsize=(12, 6))

bars = plt.bar(
    top10[state_column],
    top10["Total_Accidents"]
)

plt.title(
    "Top 10 States/UTs with Highest Road Accidents (2018–2022)",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel(
    "State / Union Territory",
    fontsize=12
)

plt.ylabel(
    "Total Number of Accidents",
    fontsize=12
)

plt.xticks(
    rotation=45,
    ha="right"
)

# Add values above bars
for bar in bars:

    value = bar.get_height()

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value,
        f"{int(value):,}",
        ha="center",
        va="bottom",
        fontsize=9
    )

plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.4
)

plt.tight_layout()

plt.show()


# ==========================================
# 6. SAVE RESULT
# ==========================================

top10.to_csv(
    "dataset/Indian Datasets/top10_high_accident_states.csv",
    index=False
)

print("\nTop 10 state data saved successfully!")