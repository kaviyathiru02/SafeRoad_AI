import pandas as pd
import folium

df = pd.read_csv("dataset/hotspot_US_Accidents.csv")

# Create map centered on USA
accident_map = folium.Map(
    location=[39.5, -98.35],
    zoom_start=4
)

sample = df.head(5000)

colors = [
    "red", "blue", "green", "purple", "orange",
    "darkred", "cadetblue", "darkgreen", "pink", "black"
]

for _, row in sample.iterrows():

    folium.CircleMarker(
        location=[row["Start_Lat"], row["Start_Lng"]],
        radius=2,
        color=colors[int(row["Cluster"])],
        fill=True,
        fill_opacity=0.7
    ).add_to(accident_map)

accident_map.save("hotspot_map.html")

print("✅ Hotspot map created successfully!")