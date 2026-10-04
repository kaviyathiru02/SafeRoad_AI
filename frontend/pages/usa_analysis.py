import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path


# ============================================================
# SAFE ROAD AI - USA ACCIDENT ANALYSIS
# ============================================================

st.set_page_config(
    page_title="USA Accident Analysis | SafeRoad AI",
    page_icon="🇺🇸",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# DATASET PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

USA_DATASET = (
    PROJECT_ROOT
    / "dataset"
    / "US Datasets"
    / "cleaned_US_Accidents.csv.gz"
)


# ============================================================
# PAGE STYLE
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(
                circle at 85% 5%,
                rgba(31, 78, 121, 0.30),
                transparent 35%
            ),
            linear-gradient(
                135deg,
                #06111f 0%,
                #091827 48%,
                #102d4b 100%
            );
    }

    .block-container {
        max-width: 1450px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3, h4 {
        color: #ffffff !important;
    }

    p, label {
        color: #cbd8e8 !important;
    }

    .page-subtitle {
        color: #9fb5cb;
        font-size: 16px;
        margin-bottom: 18px;
    }

    [data-testid="stMetric"] {
        background: rgba(14, 34, 57, 0.90);
        border: 1px solid rgba(110, 180, 240, 0.17);
        border-radius: 14px;
        padding: 12px;
    }

    [data-testid="stMetricLabel"] {
        color: #a9bfd4 !important;
    }

    [data-testid="stMetricValue"] {
        color: #ffffff !important;
    }

    .footer {
        margin-top: 30px;
        padding-top: 20px;
        border-top: 1px solid rgba(255,255,255,0.08);
        text-align: center;
        color: #7891aa;
        font-size: 13px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD USA DATASET
# ============================================================

@st.cache_data(show_spinner="Loading USA accident dataset...")
def load_usa_data():
    if not USA_DATASET.exists():
        return pd.DataFrame()

    required_columns = [
        "Severity",
        "Start_Time",
        "Start_Lat",
        "Start_Lng",
        "Distance(mi)",
        "City",
        "County",
        "State",
        "Temperature(F)",
        "Humidity(%)",
        "Visibility(mi)",
        "Wind_Speed(mph)",
        "Weather_Condition",
        "Sunrise_Sunset",
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
        "Traffic_Calming",
    ]

    try:
        df = pd.read_csv(
            USA_DATASET,
            usecols=lambda column: column in required_columns,
            low_memory=False,
        )
    except Exception as exc:
        st.error(f"Unable to read the USA dataset: {exc}")
        return pd.DataFrame()

    df.columns = [str(c).strip() for c in df.columns]

    df["Start_Time"] = pd.to_datetime(
        df["Start_Time"],
        format="%d-%m-%Y %H:%M",
        errors="coerce",
    )

    # Fallback parser for any records that do not match the
    # expected format exactly.
    invalid_time = df["Start_Time"].isna()

    if invalid_time.any():
        df.loc[invalid_time, "Start_Time"] = pd.to_datetime(
            df.loc[invalid_time, "Start_Time"],
            errors="coerce",
            dayfirst=True,
        )

    df["Year"] = df["Start_Time"].dt.year
    df["Month"] = df["Start_Time"].dt.month
    df["Hour"] = df["Start_Time"].dt.hour

    df = df.dropna(
        subset=[
            "Severity",
            "Start_Time",
            "State",
        ]
    ).copy()

    numeric_columns = [
        "Severity",
        "Start_Lat",
        "Start_Lng",
        "Distance(mi)",
        "Temperature(F)",
        "Humidity(%)",
        "Visibility(mi)",
        "Wind_Speed(mph)",
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    return df


df = load_usa_data()


# ============================================================
# DATASET ERROR
# ============================================================

if df.empty:
    st.error("❌ USA accident dataset could not be loaded.")

    st.write("Expected dataset location:")
    st.caption(str(USA_DATASET))

    st.info(
    "Please make sure that cleaned_US_Accidents.csv.gz is inside "
    "dataset/US Datasets/"
    )

    st.stop()


# ============================================================
# PAGE HEADER
# ============================================================

st.title("🇺🇸 USA Accident Analysis")

st.markdown(
    '<div class="page-subtitle">'
    "Detailed analysis of road accidents across the United States "
    "using the connected USA accident dataset."
    "</div>",
    unsafe_allow_html=True,
)

st.success(
    f"✅ USA dataset connected successfully — "
    f"{len(df):,} accident records loaded."
)


# ============================================================
# SIDEBAR FILTERS
# ============================================================

st.sidebar.header("🔎 USA Analysis Filters")

available_years = sorted(
    df["Year"]
    .dropna()
    .astype(int)
    .unique()
)

selected_year = st.sidebar.selectbox(
    "Select Year",
    ["All Years"] + available_years,
)

available_states = sorted(
    df["State"]
    .dropna()
    .astype(str)
    .unique()
)

selected_state = st.sidebar.selectbox(
    "Select State",
    ["All States"] + available_states,
)


# ============================================================
# APPLY FILTERS
# ============================================================

filtered_df = df.copy()

if selected_year != "All Years":
    filtered_df = filtered_df[
        filtered_df["Year"] == selected_year
    ]

if selected_state != "All States":
    filtered_df = filtered_df[
        filtered_df["State"] == selected_state
    ]


# ============================================================
# KEY METRICS
# ============================================================

st.subheader("📊 Accident Overview")

total_accidents = len(filtered_df)

average_severity = (
    filtered_df["Severity"].mean()
    if not filtered_df.empty
    else 0
)

states_count = (
    filtered_df["State"].nunique()
    if not filtered_df.empty
    else 0
)

average_distance = (
    filtered_df["Distance(mi)"].mean()
    if not filtered_df.empty
    else 0
)

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Accidents",
    f"{total_accidents:,}",
)

col2.metric(
    "Average Severity",
    f"{average_severity:.2f}",
)

col3.metric(
    "States / Regions",
    f"{states_count:,}",
)

col4.metric(
    "Avg. Accident Distance",
    f"{average_distance:.2f} mi",
)


# ============================================================
# ACCIDENT TREND
# ============================================================

st.subheader("📈 Accident Trend Over Years")

# The trend follows the selected state filter.
# A year selection is not used to collapse the entire trend.
if selected_state != "All States":
    trend_source = filtered_df.copy()
else:
    trend_source = df.copy()

yearly_accidents = (
    trend_source
    .groupby("Year")
    .size()
    .reset_index(name="Accidents")
    .dropna(subset=["Year"])
    .sort_values("Year")
)

if not yearly_accidents.empty:

    fig = px.line(
        yearly_accidents,
        x="Year",
        y="Accidents",
        markers=True,
        title=(
            "USA Accident Trend"
            if selected_state == "All States"
            else f"{selected_state} Accident Trend"
        ),
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=420,
        margin=dict(l=20, r=20, t=55, b=20),
        xaxis_title="Year",
        yaxis_title="Accident Records",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

else:
    st.warning("No yearly accident data available.")


# ============================================================
# SEVERITY ANALYSIS
# ============================================================

st.subheader("⚠️ Accident Severity Distribution")

severity_data = (
    filtered_df["Severity"]
    .value_counts()
    .sort_index()
    .rename_axis("Severity")
    .reset_index(name="Accidents")
)

if not severity_data.empty:

    fig = px.bar(
        severity_data,
        x="Severity",
        y="Accidents",
        text="Accidents",
        title=(
            f"{selected_state} Severity Distribution"
            if selected_state != "All States"
            else "USA Severity Distribution"
        ),
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=380,
        margin=dict(l=20, r=20, t=55, b=20),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

else:
    st.warning("No severity data available for the selected filters.")

st.caption(
    "Severity levels are based on the Severity column available "
    "in the connected USA dataset."
)


# ============================================================
# STATE-WISE ANALYSIS
# ============================================================

st.subheader("🗺️ State-wise Accident Analysis")

state_accidents = (
    filtered_df["State"]
    .value_counts()
    .head(15)
    .rename_axis("State")
    .reset_index(name="Accidents")
)

if not state_accidents.empty:

    fig = px.bar(
        state_accidents.sort_values("Accidents"),
        x="Accidents",
        y="State",
        orientation="h",
        text="Accidents",
        title="Top 15 States by Accident Records",
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=500,
        margin=dict(l=20, r=20, t=55, b=20),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

else:
    st.warning("No state-wise data available.")


# ============================================================
# WEATHER ANALYSIS
# ============================================================

st.subheader("🌦️ Top Weather Conditions")

weather_data = (
    filtered_df["Weather_Condition"]
    .fillna("Unknown")
    .astype(str)
    .value_counts()
    .head(10)
    .sort_values()
    .rename_axis("Weather Condition")
    .reset_index(name="Accidents")
)

if not weather_data.empty:

    fig = px.bar(
        weather_data,
        x="Accidents",
        y="Weather Condition",
        orientation="h",
        text="Accidents",
        title="Top 10 Weather Conditions",
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=430,
        margin=dict(l=20, r=20, t=55, b=20),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

else:
    st.warning("No weather information available.")


# ============================================================
# ENVIRONMENTAL CONDITIONS
# ============================================================

st.subheader("🌡️ Environmental Conditions")

env_col1, env_col2, env_col3 = st.columns(3)

avg_temperature = (
    filtered_df["Temperature(F)"].mean()
    if not filtered_df.empty
    else 0
)

avg_visibility = (
    filtered_df["Visibility(mi)"].mean()
    if not filtered_df.empty
    else 0
)

avg_wind = (
    filtered_df["Wind_Speed(mph)"].mean()
    if not filtered_df.empty
    else 0
)

env_col1.metric(
    "Avg. Temperature",
    f"{avg_temperature:.1f} °F",
)

env_col2.metric(
    "Avg. Visibility",
    f"{avg_visibility:.1f} mi",
)

env_col3.metric(
    "Avg. Wind Speed",
    f"{avg_wind:.1f} mph",
)


# ============================================================
# DAY / NIGHT ANALYSIS
# ============================================================

st.subheader("🌙 Day vs Night Accident Analysis")

day_night = (
    filtered_df["Sunrise_Sunset"]
    .fillna("Unknown")
    .astype(str)
    .value_counts()
    .rename_axis("Period")
    .reset_index(name="Accidents")
)

if not day_night.empty:

    fig = px.bar(
        day_night,
        x="Period",
        y="Accidents",
        text="Accidents",
        title="Accidents by Day / Night",
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=350,
        margin=dict(l=20, r=20, t=55, b=20),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

else:
    st.warning("No day/night information available.")


# ============================================================
# ROAD FEATURES
# ============================================================

st.subheader("🛣️ Road Feature Analysis")

road_features = [
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
    "Traffic_Calming",
]

road_feature_counts = {}

for feature in road_features:

    if feature in filtered_df.columns:

        values = (
            filtered_df[feature]
            .fillna(False)
            .astype(str)
            .str.lower()
        )

        road_feature_counts[feature] = int(
            values.isin(
                [
                    "true",
                    "1",
                    "yes",
                ]
            ).sum()
        )

road_feature_series = (
    pd.Series(road_feature_counts)
    .sort_values()
    .rename_axis("Road Feature")
    .reset_index(name="Accidents")
)

if not road_feature_series.empty:

    fig = px.bar(
        road_feature_series,
        x="Accidents",
        y="Road Feature",
        orientation="h",
        text="Accidents",
        title="Accident Records Associated with Road Features",
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=480,
        margin=dict(l=20, r=20, t=55, b=20),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

else:
    st.warning("Road feature information is unavailable.")


# ============================================================
# TOP ACCIDENT CITIES
# ============================================================

st.subheader("🏙️ Top Cities by Accident Count")

city_data = (
    filtered_df["City"]
    .fillna("Unknown")
    .astype(str)
    .value_counts()
    .head(15)
    .sort_values()
    .rename_axis("City")
    .reset_index(name="Accidents")
)

if not city_data.empty:

    fig = px.bar(
        city_data,
        x="Accidents",
        y="City",
        orientation="h",
        text="Accidents",
        title="Top 15 Cities by Accident Records",
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=500,
        margin=dict(l=20, r=20, t=55, b=20),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

else:
    st.warning("City information is unavailable.")


# ============================================================
# ACCIDENT LOCATION OVERVIEW
# ============================================================

st.subheader("🔥 Accident Location Overview")

location_df = filtered_df[
    [
        "Start_Lat",
        "Start_Lng",
    ]
].dropna()

if not location_df.empty:

    map_data = location_df.sample(
        min(5000, len(location_df)),
        random_state=42,
    ).rename(
        columns={
            "Start_Lat": "lat",
            "Start_Lng": "lon",
        }
    )

    st.map(
        map_data,
        latitude="lat",
        longitude="lon",
        size=8,
    )

    st.caption(
        "The map displays a representative sample of accident "
        "coordinates from the selected USA data."
    )

else:
    st.warning(
        "Latitude and longitude information is unavailable."
    )


# ============================================================
# DATA PREVIEW
# ============================================================

st.subheader("📋 USA Accident Dataset Preview")

preview_columns = [
    "Severity",
    "Start_Time",
    "City",
    "County",
    "State",
    "Temperature(F)",
    "Visibility(mi)",
    "Weather_Condition",
]

available_preview = [
    column
    for column in preview_columns
    if column in filtered_df.columns
]

st.dataframe(
    filtered_df[available_preview].head(100),
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# DATASET INFORMATION
# ============================================================

st.subheader("ℹ️ Dataset Information")

info_col1, info_col2 = st.columns(2)

with info_col1:

    st.write("**Dataset file:**")
    st.caption(
    "dataset/US Datasets/cleaned_US_Accidents.csv.gz"
    )

    st.write(
        f"**Records loaded:** {len(df):,}"
    )

    st.write(
        f"**Columns used:** {len(df.columns):,}"
    )

with info_col2:

    st.write("**Current filters:**")

    st.write(
        f"Year: {selected_year}"
    )

    st.write(
        f"State: {selected_state}"
    )

    st.write(
        f"Records after filtering: {len(filtered_df):,}"
    )


# ============================================================
# IMPORTANT DATASET NOTE
# ============================================================

st.info(
    "Note: The connected USA dataset contains accident severity, "
    "location, weather, environmental and road-feature information. "
    "It does not contain direct Deaths or Injuries columns, so "
    "those values are not displayed or artificially generated."
)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        🇺🇸 SafeRoad AI &nbsp;|&nbsp;
        USA Accident Analysis
        <br>
        Intelligent Accident Risk Prediction & Hotspot Detection Platform
    </div>
    """,
    unsafe_allow_html=True,
)
