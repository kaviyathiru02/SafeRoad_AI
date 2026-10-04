import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
df = pd.read_csv("dataset/preprocessed_US_Accidents.csv")
location = df[["Start_Lat", "Start_Lng"]]

print(location.head())

print("\nShape:")
print(location.shape)
print("Training K-Means Model...")

kmeans = KMeans(
    n_clusters=10,
    random_state=42,
    n_init=10
)

kmeans.fit(location)

print("✅ K-Means Model Trained Successfully!")
df["Cluster"] = kmeans.labels_

print(df["Cluster"].head())
df.to_csv(
    "dataset/hotspot_US_Accidents.csv",
    index=False
)

print("✅ Hotspot dataset saved successfully!")
print(df["Cluster"].value_counts())
plt.figure(figsize=(10, 8))

plt.scatter(
    df["Start_Lng"],
    df["Start_Lat"],
    c=df["Cluster"],
    cmap="tab10",
    s=1
)

plt.title("Accident Hotspot Clusters")
plt.xlabel("Longitude")
plt.ylabel("Latitude")

plt.show()