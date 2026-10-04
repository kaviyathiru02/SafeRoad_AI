import pandas as pd
import folium
from folium.plugins import MarkerCluster

# ============================================================
# SAFE ROAD AI - INDIA ACCIDENT HOTSPOT MAP
# ============================================================

print("=" * 60)
print("SAFE ROAD AI - INDIA ACCIDENT HOTSPOT MAP")
print("=" * 60)

# ------------------------------------------------------------
# 1. Load hotspot dataset
# ------------------------------------------------------------

file_path = "dataset/India_Accident_Hotspots.csv"

print("\nLoading hotspot dataset...")

df = pd.read_csv(file_path)

print("Dataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

# ------------------------------------------------------------
# 2. State coordinates
# ------------------------------------------------------------

state_coordinates = {
    "Andhra Pradesh": (15.9129, 79.7400),
    "Arunachal Pradesh": (28.2180, 94.7278),
    "Assam": (26.2006, 92.9376),
    "Bihar": (25.0961, 85.3131),
    "Chhattisgarh": (21.2787, 81.8661),
    "Goa": (15.2993, 74.1240),
    "Gujarat": (22.2587, 71.1924),
    "Haryana": (29.0588, 76.0856),
    "Himachal Pradesh": (31.1048, 77.1734),
    "Jharkhand": (23.6102, 85.2799),
    "Karnataka": (15.3173, 75.7139),
    "Kerala": (10.8505, 76.2711),
    "Madhya Pradesh": (22.9734, 78.6569),
    "Maharashtra": (19.7515, 75.7139),
    "Manipur": (24.6637, 93.9063),
    "Meghalaya": (25.4670, 91.3662),
    "Mizoram": (23.1645, 92.9376),
    "Nagaland": (26.1584, 94.5624),
    "Odisha": (20.9517, 85.0985),
    "Punjab": (31.1471, 75.3412),
    "Rajasthan": (27.0238, 74.2179),
    "Sikkim": (27.5330, 88.5122),
    "Tamil Nadu": (11.1271, 78.6569),
    "Telangana": (18.1124, 79.0193),
    "Tripura": (23.9408, 91.9882),
    "Uttar Pradesh": (26.8467, 80.9462),
    "Uttarakhand": (30.0668, 79.0193),
    "West Bengal": (22.9868, 87.8550),

    "Delhi": (28.7041, 77.1025),
    "Jammu and Kashmir": (33.7782, 76.5762),
    "Ladakh": (34.1526, 77.5771),
    "Puducherry": (11.9416, 79.8083),
    "Chandigarh": (30.7333, 76.7794),

    "Andaman and Nicobar Islands": (11.7401, 92.6586),
    "Dadra and Nagar Haveli": (20.1809, 73.0169),
    "Daman and Diu": (20.4283, 72.8397),
    "Lakshadweep": (10.5667, 72.6417)
}

# ------------------------------------------------------------
# 3. Check states
# ------------------------------------------------------------

print("\nNumber of states:", len(df))

missing_coordinates = []

for state in df["State"]:
    if state not in state_coordinates:
        missing_coordinates.append(state)

if missing_coordinates:
    print("\nStates without coordinates:")
    for state in missing_coordinates:
        print("-", state)

# ------------------------------------------------------------
# 4. Create India map
# ------------------------------------------------------------

print("\nCreating India hotspot map...")

india_map = folium.Map(
    location=[22.5, 80.0],
    zoom_start=5,
    tiles="CartoDB positron"
)

# ------------------------------------------------------------
# 5. Add title
# ------------------------------------------------------------

title_html = """
<div style="
position: fixed;
top: 10px;
left: 50%;
transform: translateX(-50%);
z-index:9999;
background-color:white;
padding:10px 20px;
border-radius:8px;
box-shadow:0px 2px 8px rgba(0,0,0,0.3);
font-size:20px;
font-weight:bold;
">
SafeRoad AI - India Accident Hotspot Map
</div>
"""

india_map.get_root().html.add_child(
    folium.Element(title_html)
)

# ------------------------------------------------------------
# 6. Marker cluster
# ------------------------------------------------------------

marker_cluster = MarkerCluster(
    name="Accident Hotspots"
).add_to(india_map)

# ------------------------------------------------------------
# 7. Add accident hotspot markers
# ------------------------------------------------------------

for _, row in df.iterrows():

    state = row["State"]

    if state not in state_coordinates:
        continue

    latitude, longitude = state_coordinates[state]

    hotspot = str(row["Hotspot_Level"]).strip()

    risk_score = row["Risk_Score"]

    accidents = row["Accidents"]

    deaths = row["Deaths"]

    injuries = row["Injuries"]

    # --------------------------------------------------------
    # Choose marker color
    # --------------------------------------------------------

    if hotspot.lower() == "high":
        color = "red"

    elif hotspot.lower() == "medium":
        color = "orange"

    else:
        color = "green"

    # --------------------------------------------------------
    # Popup information
    # --------------------------------------------------------

    popup_html = f"""
    <div style="font-family:Arial; width:250px;">

        <h3 style="margin-bottom:8px;">
            {state}
        </h3>

        <b>Hotspot Level:</b>
        <span style="color:{color};">
            {hotspot}
        </span>

        <br><br>

        <b>Risk Score:</b>
        {risk_score:.3f}

        <br>

        <b>Accidents:</b>
        {accidents:.0f}

        <br>

        <b>Deaths:</b>
        {deaths:.0f}

        <br>

        <b>Injuries:</b>
        {injuries:.0f}

    </div>
    """

    # --------------------------------------------------------
    # Add circle marker
    # --------------------------------------------------------

    folium.CircleMarker(
        location=[latitude, longitude],
        radius=10 if hotspot.lower() == "high" else 7,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.75,
        popup=folium.Popup(
            popup_html,
            max_width=300
        ),
        tooltip=f"{state} - {hotspot} Risk"
    ).add_to(marker_cluster)

# ------------------------------------------------------------
# 8. Add legend
# ------------------------------------------------------------

legend_html = """
<div style="
position: fixed;
bottom: 30px;
left: 30px;
width: 190px;
z-index:9999;
background-color:white;
border:2px solid grey;
border-radius:8px;
padding:12px;
font-size:14px;
box-shadow:0px 2px 8px rgba(0,0,0,0.3);
">

<b>Accident Risk Level</b>

<br><br>

<span style="
display:inline-block;
width:14px;
height:14px;
background:red;
border-radius:50%;
margin-right:8px;
"></span>

High Risk

<br><br>

<span style="
display:inline-block;
width:14px;
height:14px;
background:orange;
border-radius:50%;
margin-right:8px;
"></span>

Medium Risk

<br><br>

<span style="
display:inline-block;
width:14px;
height:14px;
background:green;
border-radius:50%;
margin-right:8px;
"></span>

Low Risk

</div>
"""

india_map.get_root().html.add_child(
    folium.Element(legend_html)
)

# ------------------------------------------------------------
# 9. Add layer control
# ------------------------------------------------------------

folium.LayerControl().add_to(india_map)

# ------------------------------------------------------------
# 10. Save map
# ------------------------------------------------------------

output_file = "India_Accident_Hotspot_Map.html"

india_map.save(output_file)

print("\n" + "=" * 60)
print("MAP CREATED SUCCESSFULLY!")
print("=" * 60)

print("\nOutput file:")
print(output_file)

print("\nHigh Risk states:")
print(
    df[df["Hotspot_Level"].str.lower() == "high"]
    [["State", "Risk_Score"]]
    .to_string(index=False)
)

print("\nMedium Risk states:",
      (df["Hotspot_Level"].str.lower() == "medium").sum())

print("Low Risk states:",
      (df["Hotspot_Level"].str.lower() == "low").sum())

print("\nOpen the HTML file in your browser to view the map.")