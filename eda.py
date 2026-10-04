import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("dataset/cleaned_US_Accidents.csv")

# Graph 1
plt.figure(figsize=(6,4))
df["Severity"].value_counts().sort_index().plot(kind="bar")
plt.title("Accident Severity Distribution")
plt.xlabel("Severity Level")
plt.ylabel("Number of Accidents")
plt.show()

# Graph 2
plt.figure(figsize=(10,5))
df["Weather_Condition"].value_counts().head(10).plot(kind="bar")
plt.title("Top 10 Weather Conditions")
plt.xlabel("Weather")
plt.ylabel("Number of Accidents")
plt.show()

plt.figure(figsize=(10,5))

df["Temperature(F)"].plot(kind="hist", bins=30)

plt.title("Temperature Distribution")
plt.xlabel("Temperature (F)")
plt.ylabel("Frequency")

plt.show()

plt.figure(figsize=(10,5))

df["State"].value_counts().head(10).plot(kind="bar")

plt.title("Top 10 States with Most Accidents")
plt.xlabel("State")
plt.ylabel("Number of Accidents")

plt.show()

plt.figure(figsize=(10,5))

pd.crosstab(df["Weather_Condition"], df["Severity"]).head(10).plot(kind="bar")

plt.title("Severity by Weather Condition")
plt.xlabel("Weather")
plt.ylabel("Number of Accidents")

plt.show()

plt.figure(figsize=(10,5))

df["City"].value_counts().head(10).plot(kind="bar")

plt.title("Top 10 Cities with Most Accidents")
plt.xlabel("City")
plt.ylabel("Number of Accidents")

plt.show()

plt.figure(figsize=(7,7))

df["Severity"].value_counts().plot(
    kind="pie",
    autopct="%1.1f%%",
    startangle=90
)

plt.title("Accident Severity Percentage")
plt.ylabel("")

plt.show()

plt.figure(figsize=(10,5))

df["Visibility(mi)"].plot(kind="hist", bins=30)

plt.title("Visibility Distribution")
plt.xlabel("Visibility (Miles)")
plt.ylabel("Frequency")

plt.show()

plt.figure(figsize=(10,5))

df["Wind_Speed(mph)"].plot(kind="hist", bins=30)

plt.title("Wind Speed Distribution")
plt.xlabel("Wind Speed (mph)")
plt.ylabel("Frequency")

plt.show()

plt.figure(figsize=(10,5))

df["Humidity(%)"].plot(kind="hist", bins=30)

plt.title("Humidity Distribution")
plt.xlabel("Humidity (%)")
plt.ylabel("Frequency")

plt.show()