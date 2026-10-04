import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path


# ============================================================
# SAFE ROAD AI - MAP VIEW
# ============================================================
# India:
#   State-level accident/hotspot data is visualized using
#   approximate state-centre coordinates.
#
# USA:
#   Exact accident coordinates are available as Start_Lat /
#   Start_Lng, so the actual geographic points are shown.
# ============================================================


# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT_DIR / "dataset"

INDIA_DATA = DATASET_DIR / "India_ML_Dataset.csv"
INDIA_HOTSPOTS = DATASET_DIR / "India_Accident_Hotspots.csv"

USA_DATA = (
    DATASET_DIR
    / "US Datasets"
    / "cleaned_US_Accidents.csv"
)


# ------------------------------------------------------------
# PAGE STYLE
# ------------------------------------------------------------

st.markdown(
    """
    <style>

    .stApp {
        background:
            radial-gradient(
                circle at 85% 5%,
                rgba(29, 78, 121, 0.25),
                transparent 30%
            ),
            linear-gradient(
                135deg,
                #06111f 0%,
                #0a1d32 50%,
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
        color: #d2dfed !important;
    }

    [data-testid="stMetric"] {
        background: rgba(12, 32, 54, 0.95);
        border: 1px solid rgba(96, 176, 236, 0.18);
        border-radius: 15px;
        padding: 14px;
    }

    [data-testid="stMetricLabel"] {
        color: #aac0d4 !important;
    }

    [data-testid="stMetricValue"] {
        color: #ffffff !important;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

@st.cache_data
def load_csv(path):
    if not path.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(path, low_memory=False)
        df.columns = [str(c).strip() for c in df.columns]
        return df
    except Exception:
        return pd.DataFrame()


# ------------------------------------------------------------
# INDIA STATE COORDINATES
# ------------------------------------------------------------

INDIA_COORDINATES = {
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
    "Jammu & Kashmir": (33.7782, 76.5762),
    "Ladakh": (34.1526, 77.5771),
    "Puducherry": (11.9416, 79.8083),
    "Chandigarh": (30.7333, 76.7794),
    "Andaman and Nicobar Islands": (11.7401, 92.6586),
    "Lakshadweep": (10.5667, 72.6417),
}


# ------------------------------------------------------------
# HEADER
# ------------------------------------------------------------

st.title("🗺️ Accident Hotspot Map")

st.caption(
    "Interactive geographical visualization of accident hotspots "
    "for India and the United States."
)


# ------------------------------------------------------------
# DATASET SELECTOR
# ------------------------------------------------------------

country = st.selectbox(
    "Select Dataset",
    ["India", "USA"],
)


# ============================================================
# INDIA MAP
# ============================================================

if country == "India":

    india_df = load_csv(INDIA_DATA)
    hotspot_df = load_csv(INDIA_HOTSPOTS)

    if india_df.empty:

        st.error(
            "India_ML_Dataset.csv could not be loaded."
        )

        st.stop()


    if "State" not in india_df.columns:

        st.error(
            "The India dataset does not contain a State column."
        )

        st.stop()


    if "Accidents" not in india_df.columns:

        st.error(
            "The India dataset does not contain an Accidents column."
        )

        st.stop()


    # --------------------------------------------------------
    # STATE-LEVEL ACCIDENT SUMMARY
    # --------------------------------------------------------

    india_df["State"] = (
        india_df["State"]
        .astype(str)
        .str.strip()
    )

    india_df["Accidents"] = pd.to_numeric(
        india_df["Accidents"],
        errors="coerce"
    ).fillna(0)

    map_df = (
        india_df
        .groupby("State", as_index=False)["Accidents"]
        .sum()
    )


    # --------------------------------------------------------
    # MERGE EXISTING HOTSPOT RESULTS
    # --------------------------------------------------------

    if not hotspot_df.empty and "State" in hotspot_df.columns:

        hotspot_df["State"] = (
            hotspot_df["State"]
            .astype(str)
            .str.strip()
        )

        merge_cols = ["State"]

        if "Risk_Score" in hotspot_df.columns:

            hotspot_df["Risk_Score"] = pd.to_numeric(
                hotspot_df["Risk_Score"],
                errors="coerce"
            )

            merge_cols.append("Risk_Score")

        if "Hotspot_Level" in hotspot_df.columns:

            merge_cols.append("Hotspot_Level")

        hotspot_small = hotspot_df[
            merge_cols
        ].drop_duplicates(
            subset=["State"]
        )

        map_df = map_df.merge(
            hotspot_small,
            on="State",
            how="left"
        )


    # --------------------------------------------------------
    # CALCULATE FALLBACK RISK SCORE
    # --------------------------------------------------------

    max_accidents = map_df["Accidents"].max()

    if max_accidents > 0:

        fallback_score = (
            map_df["Accidents"] / max_accidents
        )

    else:

        fallback_score = pd.Series(
            0.0,
            index=map_df.index
        )


    if "Risk_Score" not in map_df.columns:

        map_df["Risk_Score"] = fallback_score

    else:

        map_df["Risk_Score"] = (
            map_df["Risk_Score"]
            .fillna(fallback_score)
        )


    # --------------------------------------------------------
    # CLASSIFICATION
    # --------------------------------------------------------

    def classify(score):

        if score >= 0.66:
            return "High"

        if score >= 0.33:
            return "Medium"

        return "Low"


    if "Hotspot_Level" not in map_df.columns:

        map_df["Hotspot_Level"] = (
            map_df["Risk_Score"]
            .apply(classify)
        )

    else:

        empty_levels = (
            map_df["Hotspot_Level"]
            .isna()
        )

        map_df.loc[
            empty_levels,
            "Hotspot_Level"
        ] = (
            map_df.loc[
                empty_levels,
                "Risk_Score"
            ]
            .apply(classify)
        )


    # --------------------------------------------------------
    # COORDINATES
    # --------------------------------------------------------

    map_df["Latitude"] = map_df["State"].map(
        lambda state:
        INDIA_COORDINATES.get(
            state,
            (np.nan, np.nan)
        )[0]
    )

    map_df["Longitude"] = map_df["State"].map(
        lambda state:
        INDIA_COORDINATES.get(
            state,
            (np.nan, np.nan)
        )[1]
    )

    map_df = map_df.dropna(
        subset=[
            "Latitude",
            "Longitude",
        ]
    )


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    st.subheader("🇮🇳 India Map Overview")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "States Mapped",
            f"{len(map_df):,}"
        )

    with c2:
        st.metric(
            "High-Risk States",
            int(
                (
                    map_df["Hotspot_Level"]
                    .astype(str)
                    .str.lower()
                    == "high"
                ).sum()
            )
        )

    with c3:
        st.metric(
            "Total Accident Records",
            f"{map_df['Accidents'].sum():,.0f}"
        )


    # --------------------------------------------------------
    # MAP
    # --------------------------------------------------------

    st.subheader("🔥 India State-Level Accident Hotspots")

    st.info(
        "The available India dataset is state-level and does not "
        "contain exact accident GPS coordinates. Therefore each "
        "state is represented using its approximate geographic "
        "centre. Marker size represents accident concentration."
    )


    if map_df.empty:

        st.warning(
            "No India state names could be matched with map coordinates."
        )

    else:

        fig = px.scatter_mapbox(
            map_df,
            lat="Latitude",
            lon="Longitude",
            size="Accidents",
            color="Hotspot_Level",
            hover_name="State",
            hover_data={
                "Accidents": ":,.0f",
                "Risk_Score": ":.3f",
                "Hotspot_Level": True,
                "Latitude": False,
                "Longitude": False,
            },
            center={
                "lat": 22.5,
                "lon": 79.0,
            },
            zoom=4.2,
            mapbox_style="open-street-map",
            title="India State-Level Accident Hotspot Map",
        )

        fig.update_layout(
            height=680,
            margin=dict(
                l=0,
                r=0,
                t=55,
                b=0
            ),
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # --------------------------------------------------------
    # INDIA TABLE
    # --------------------------------------------------------

    st.subheader("📋 Mapped State Details")

    display_cols = [
        "State",
        "Accidents",
        "Risk_Score",
        "Hotspot_Level",
    ]

    st.dataframe(
        map_df[display_cols].sort_values(
            "Risk_Score",
            ascending=False
        ),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# USA MAP
# ============================================================

else:

    usa_df = load_csv(USA_DATA)

    if usa_df.empty:

        st.error(
            "cleaned_US_Accidents.csv could not be loaded."
        )

        st.stop()


    required = [
        "Start_Lat",
        "Start_Lng",
    ]

    missing = [
        col
        for col in required
        if col not in usa_df.columns
    ]

    if missing:

        st.error(
            "USA dataset is missing: "
            + ", ".join(missing)
        )

        st.stop()


    usa_df["Start_Lat"] = pd.to_numeric(
        usa_df["Start_Lat"],
        errors="coerce"
    )

    usa_df["Start_Lng"] = pd.to_numeric(
        usa_df["Start_Lng"],
        errors="coerce"
    )

    usa_df = usa_df.dropna(
        subset=[
            "Start_Lat",
            "Start_Lng",
        ]
    )


    # Keep coordinates inside the continental/general US range
    usa_df = usa_df[
        usa_df["Start_Lat"].between(18, 72)
        & usa_df["Start_Lng"].between(-170, -60)
    ]


    if usa_df.empty:

        st.error(
            "No valid USA accident coordinates were found."
        )

        st.stop()


    # --------------------------------------------------------
    # OPTIONAL STATE SUMMARY
    # --------------------------------------------------------

    state_available = "State" in usa_df.columns

    state_summary = pd.DataFrame()

    if state_available:

        usa_df["State"] = (
            usa_df["State"]
            .astype(str)
            .str.strip()
        )

        state_summary = (
            usa_df["State"]
            .value_counts()
            .rename("Records")
            .reset_index()
        )

        state_summary.columns = [
            "State",
            "Records"
        ]


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    st.subheader("🇺🇸 USA Map Overview")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Mapped Accident Records",
            f"{len(usa_df):,}"
        )

    with c2:
        st.metric(
            "States / Regions",
            (
                f"{usa_df['State'].nunique():,}"
                if state_available
                else "N/A"
            )
        )

    with c3:
        st.metric(
            "Coordinates Available",
            "Yes"
        )


    # --------------------------------------------------------
    # USA MAP
    # --------------------------------------------------------

    st.subheader("🔥 USA Accident Locations")

    st.success(
        "The USA dataset contains Start_Lat and Start_Lng, "
        "so the map uses actual accident coordinates."
    )


    # For performance, display a representative sample when
    # the full dataset is very large.
    max_points = 30000

    if len(usa_df) > max_points:

        plot_df = usa_df.sample(
            n=max_points,
            random_state=42
        )

        st.caption(
            f"Showing {max_points:,} representative accident points "
            f"from {len(usa_df):,} valid records for responsive visualization."
        )

    else:

        plot_df = usa_df


    # --------------------------------------------------------
    # COLOR BY SEVERITY WHEN AVAILABLE
    # --------------------------------------------------------

    if "Severity" in plot_df.columns:

        severity_values = pd.to_numeric(
            plot_df["Severity"],
            errors="coerce"
        )

        plot_df = plot_df.copy()

        plot_df["Severity_Label"] = (
            severity_values
            .map(
                lambda x:
                f"Severity {int(x)}"
                if pd.notna(x)
                else "Unknown"
            )
        )

        color_column = "Severity_Label"

    else:

        color_column = None


    map_kwargs = {
        "lat": "Start_Lat",
        "lon": "Start_Lng",
        "hover_data": {
            "Start_Lat": ":.4f",
            "Start_Lng": ":.4f",
        },
        "zoom": 3.5,
        "center": {
            "lat": 39.5,
            "lon": -98.5,
        },
        "mapbox_style": "open-street-map",
        "title": "USA Accident Location Map",
    }

    if color_column:

        map_kwargs["color"] = color_column
        map_kwargs["hover_name"] = (
            "State"
            if state_available
            else None
        )

    fig = px.scatter_mapbox(
        plot_df,
        **map_kwargs
    )

    fig.update_layout(
        height=700,
        margin=dict(
            l=0,
            r=0,
            t=55,
            b=0
        ),
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # --------------------------------------------------------
    # TOP STATES
    # --------------------------------------------------------

    if not state_summary.empty:

        st.subheader("🏆 Top USA States by Accident Records")

        top_states = state_summary.head(10)

        fig2 = px.bar(
            top_states.sort_values(
                "Records",
                ascending=True
            ),
            x="Records",
            y="State",
            orientation="h",
            text="Records",
            title="Top 10 USA States by Accident Records",
        )

        fig2.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=430,
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

        st.dataframe(
            top_states,
            use_container_width=True,
            hide_index=True,
        )


# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------

st.markdown("---")

st.caption(
    "🗺️ SafeRoad AI — Interactive Accident Map | "
    "India state-level visualization + USA coordinate-based visualization"
)
