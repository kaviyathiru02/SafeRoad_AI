import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

try:
    from sklearn.cluster import KMeans
except ImportError:
    KMeans = None


# ============================================================
# SAFE ROAD AI - ACCIDENT HOTSPOT DETECTION
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT_DIR / "dataset"

INDIA_ML = DATASET_DIR / "India_ML_Dataset.csv"
INDIA_HOTSPOT = DATASET_DIR / "India_Accident_Hotspots.csv"

USA_CLEANED = DATASET_DIR / "US Datasets" / "cleaned_US_Accidents.csv.gz"


# ============================================================
# PAGE STYLE
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at 85% 5%, rgba(28,78,124,0.27), transparent 30%),
            linear-gradient(135deg, #06111f 0%, #0a1c30 50%, #102d4b 100%);
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
        background: rgba(12,32,54,0.95);
        border: 1px solid rgba(96,176,236,0.18);
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


# ============================================================
# LOADERS
# ============================================================

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


@st.cache_data
def load_usa_for_hotspots(path):
    if not path.exists():
        return pd.DataFrame()

    try:
        needed = ["Start_Lat", "Start_Lng", "State", "Severity"]
        df = pd.read_csv(
            path,
            usecols=lambda c: c in needed,
            low_memory=False,
        )
        df.columns = [str(c).strip() for c in df.columns]
        return df
    except Exception:
        return load_csv(path)


# ============================================================
# HEADER
# ============================================================

st.title("🔥 Accident Hotspot Detection")

st.caption(
    "Identify accident-concentration hotspots for India and the United States."
)


# ============================================================
# DATASET SELECTION
# ============================================================

country = st.selectbox(
    "Select Dataset",
    ["India", "USA"],
)


# ============================================================
# INDIA
# ============================================================

if country == "India":

    india_df = load_csv(INDIA_ML)
    existing_hotspots = load_csv(INDIA_HOTSPOT)

    if india_df.empty:
        st.error("India_ML_Dataset.csv could not be loaded.")
        st.stop()

    required = ["State", "Accidents"]
    missing = [c for c in required if c not in india_df.columns]

    if missing:
        st.error(
            "India dataset is missing: " + ", ".join(missing)
        )
        st.stop()

    india_df["State"] = india_df["State"].astype(str).str.strip()
    india_df["Accidents"] = pd.to_numeric(
        india_df["Accidents"], errors="coerce"
    ).fillna(0)

    # ALL states from the main India dataset are retained.
    state_summary = (
        india_df.groupby("State", as_index=False)["Accidents"]
        .sum()
        .sort_values("Accidents", ascending=False)
    )

    # Use previously generated hotspot results where available.
    if not existing_hotspots.empty and "State" in existing_hotspots.columns:

        existing_hotspots["State"] = (
            existing_hotspots["State"].astype(str).str.strip()
        )

        merge_cols = ["State"]

        if "Risk_Score" in existing_hotspots.columns:
            existing_hotspots["Risk_Score"] = pd.to_numeric(
                existing_hotspots["Risk_Score"],
                errors="coerce"
            )
            merge_cols.append("Risk_Score")

        if "Hotspot_Level" in existing_hotspots.columns:
            merge_cols.append("Hotspot_Level")

        previous = existing_hotspots[merge_cols].drop_duplicates("State")

        state_summary = state_summary.merge(
            previous,
            on="State",
            how="left",
        )

    # Fill missing relative score from accident concentration.
    max_accidents = state_summary["Accidents"].max()

    if max_accidents > 0:
        calculated_score = (
            state_summary["Accidents"] / max_accidents
        )
    else:
        calculated_score = pd.Series(
            np.zeros(len(state_summary)),
            index=state_summary.index,
        )

    if "Risk_Score" not in state_summary.columns:
        state_summary["Risk_Score"] = calculated_score
    else:
        state_summary["Risk_Score"] = (
            state_summary["Risk_Score"].fillna(calculated_score)
        )

    # Fill missing classifications only.
    def level_from_score(score):
        if score >= 0.66:
            return "High"
        if score >= 0.33:
            return "Medium"
        return "Low"

    if "Hotspot_Level" not in state_summary.columns:
        state_summary["Hotspot_Level"] = (
            state_summary["Risk_Score"].apply(level_from_score)
        )
    else:
        state_summary["Hotspot_Level"] = (
            state_summary["Hotspot_Level"]
            .astype("object")
        )

        missing_level = (
            state_summary["Hotspot_Level"]
            .isna()
            |
            (state_summary["Hotspot_Level"].astype(str).str.strip() == "")
        )

        state_summary.loc[missing_level, "Hotspot_Level"] = (
            state_summary.loc[missing_level, "Risk_Score"]
            .apply(level_from_score)
        )

    state_summary = state_summary.sort_values(
        ["Risk_Score", "Accidents"],
        ascending=False
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    st.subheader("🇮🇳 India Hotspot Overview")

    high = int(
        (
            state_summary["Hotspot_Level"]
            .astype(str)
            .str.lower()
            == "high"
        ).sum()
    )

    medium = int(
        (
            state_summary["Hotspot_Level"]
            .astype(str)
            .str.lower()
            == "medium"
        ).sum()
    )

    low = int(
        (
            state_summary["Hotspot_Level"]
            .astype(str)
            .str.lower()
            == "low"
        ).sum()
    )

    regions = int(state_summary["State"].nunique())

    m1, m2, m3, m4 = st.columns(4)

    m1.metric("🔴 High-Risk Hotspots", high)
    m2.metric("🟠 Medium-Risk Hotspots", medium)
    m3.metric("🟢 Low-Risk Hotspots", low)
    m4.metric("📍 Regions Analyzed", regions)

    # --------------------------------------------------------
    # TABS
    # --------------------------------------------------------

    overview, hotspot_table, map_tab, method = st.tabs(
        [
            "OVERVIEW",
            "HOTSPOT TABLE",
            "MAP VIEW",
            "DETECTION METHOD",
        ]
    )

    # --------------------------------------------------------
    # OVERVIEW
    # --------------------------------------------------------

    with overview:

        left, right = st.columns(2)

        with left:

            st.subheader("🔥 Hotspot Level Distribution")

            counts = (
                state_summary["Hotspot_Level"]
                .value_counts()
                .rename_axis("Hotspot Level")
                .reset_index(name="Regions")
            )

            fig = px.bar(
                counts,
                x="Hotspot Level",
                y="Regions",
                text="Regions",
                title="India Hotspot Classification",
            )

            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=380,
            )

            st.plotly_chart(fig, use_container_width=True)

        with right:

            st.subheader("📈 Risk Score Distribution")

            top = (
                state_summary[
                    ["State", "Risk_Score"]
                ]
                .head(10)
                .sort_values("Risk_Score", ascending=True)
            )

            fig = px.bar(
                top,
                x="Risk_Score",
                y="State",
                orientation="h",
                title="Highest-Risk Indian Regions",
            )

            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=380,
            )

            st.plotly_chart(fig, use_container_width=True)

        st.subheader("🏆 Highest-Risk Regions")

        st.dataframe(
            state_summary[
                ["State", "Accidents", "Risk_Score", "Hotspot_Level"]
            ].head(10),
            use_container_width=True,
            hide_index=True,
        )

    # --------------------------------------------------------
    # TABLE
    # --------------------------------------------------------

    with hotspot_table:

        st.subheader("📋 All India State-Level Hotspots")

        st.dataframe(
            state_summary[
                ["State", "Accidents", "Risk_Score", "Hotspot_Level"]
            ],
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "⬇️ Download India Hotspot Results",
            state_summary.to_csv(index=False).encode("utf-8"),
            "India_Hotspot_Results.csv",
            "text/csv",
            use_container_width=True,
        )

    # --------------------------------------------------------
    # MAP
    # --------------------------------------------------------

    with map_tab:

        st.subheader("🗺️ India Macro-Spatial Hotspot Map")

        st.info(
            "📍 **Macro-Spatial Administrative Map Visualization:**\n\n"
            "India's official accident-recording framework supports geospatial accident reporting, but the "
            "publicly available state-level dataset used in SafeRoad AI does not expose individual accident GPS records.\n\n"
            "To maintain strict academic data integrity without fabricating synthetic coordinates, India's accident "
            "hotspots are evaluated at the state and union territory administrative level using official aggregate road safety data. "
            "State-centre coordinates are used to geographically visualize relative accident concentrations and risk classifications "
            "across regions without claiming to represent individual incident crash sites."
        )

        coords = {
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

        map_df = state_summary.copy()

        map_df["Latitude"] = map_df["State"].map(
            lambda s: coords.get(s, (np.nan, np.nan))[0]
        )
        map_df["Longitude"] = map_df["State"].map(
            lambda s: coords.get(s, (np.nan, np.nan))[1]
        )

        map_df = map_df.dropna(
            subset=["Latitude", "Longitude"]
        )

        if map_df.empty:

            st.warning(
                "No state names in the dataset matched the map coordinates."
            )

        else:

            fig = px.scatter_mapbox(
                map_df,
                lat="Latitude",
                lon="Longitude",
                size="Risk_Score",
                color="Hotspot_Level",
                hover_name="State",
                hover_data={
                    "Accidents": True,
                    "Risk_Score": ":.3f",
                    "Hotspot_Level": True,
                    "Latitude": False,
                    "Longitude": False,
                },
                center={"lat": 22.5, "lon": 79.0},
                zoom=4.2,
                mapbox_style="open-street-map",
                title="India State-Level Accident Hotspots",
            )

            fig.update_layout(
                height=650,
                margin=dict(l=0, r=0, t=55, b=0),
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

    # --------------------------------------------------------
    # METHOD
    # --------------------------------------------------------

    with method:

        st.subheader("🧠 Geospatial Hotspot Methodology (Macro vs. Micro Spatial)")

        st.markdown(
            "SafeRoad AI implements a dual-scale spatial methodology tailored to the official data architecture of each nation:\n\n"
            "#### 🇮🇳 India: Macro-Spatial Administrative Clustering\n"
            "• **Data Framework & Availability:** India's official accident recording framework—specifically MoRTH's "
            "**Integrated Road Accident Database (iRAD)** and **e-DAR (Electronic Detailed Accident Report)**—records "
            "high-fidelity geo-tagged coordinates internally via law enforcement and transport agencies. However, this raw "
            "microdata is legally restricted to authorized government stakeholders for privacy and security reasons and is not "
            "available as an open public download. The publicly released MoRTH and Open Government Data (data.gov.in) datasets "
            "publish aggregated state/UT and district-level statistics.\n"
            "• **Analytical Method:** Macro-spatial multi-indicator clustering and risk scoring based on "
            "(`Accidents`, `Deaths`, `Injuries`, `Fatality_Rate`, `Injury_Rate`, and `Risk_Score`).\n"
            "• **Spatial Representation:** Macro-level administrative concentration mapped via state-centre coordinates. "
            "*(SafeRoad AI strictly adheres to academic honesty: no synthetic GPS coordinates are generated, and state centroids are never misrepresented as individual crash sites).*\n\n"
            "#### 🇺🇸 USA: Micro-Spatial Point-Source Clustering\n"
            "• **Data Framework & Availability:** The US Accidents open repository contains individual incident records with sensor-logged latitude and longitude (`Start_Lat`, `Start_Lng`).\n"
            "• **Analytical Method:** Micro-spatial K-Means clustering applied directly to physical coordinates to identify spatial accident density centroids.\n"
            "• **Spatial Representation:** Localized coordinate clusters and highway segment blackspots."
        )

        with st.expander("📋 India State-Level Data Pipeline Details"):
            st.write(
                "• All states present in `India_ML_Dataset.csv` are retained in the hotspot table."
            )
            st.write(
                "• Where `India_Accident_Hotspots.csv` already contains a `Risk_Score` and `Hotspot_Level`, those results are used."
            )
            st.write(
                "• For states without an existing classification, the dashboard calculates a relative concentration score "
                "from accident counts and assigns an analytical level (High, Medium, Low)."
            )


# ============================================================
# USA
# ============================================================

else:

    usa_df = load_usa_for_hotspots(USA_CLEANED)

    if usa_df.empty:

        st.error(
            "USA dataset could not be loaded."
        )

        st.stop()

    required = ["Start_Lat", "Start_Lng"]

    missing = [
        c for c in required
        if c not in usa_df.columns
    ]

    if missing:

        st.error(
            "USA dataset is missing: " + ", ".join(missing)
        )

        st.stop()

    if KMeans is None:

        st.error(
            "scikit-learn is required for USA K-Means hotspot detection."
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
        subset=["Start_Lat", "Start_Lng"]
    )

    usa_df = usa_df[
        usa_df["Start_Lat"].between(18, 72)
        & usa_df["Start_Lng"].between(-170, -60)
    ]

    if usa_df.empty:

        st.error(
            "No valid USA geographic coordinates were found."
        )

        st.stop()

    st.subheader("🇺🇸 USA Hotspot Overview")

    sample_size = min(
        30000,
        len(usa_df)
    )

    sample_df = usa_df.sample(
        n=sample_size,
        random_state=42,
    ).copy()

    coordinates = sample_df[
        ["Start_Lat", "Start_Lng"]
    ].to_numpy()

    n_clusters = 10 if sample_size >= 5000 else max(
        3,
        min(10, sample_size // 500)
    )

    model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10,
    )

    sample_df["Cluster"] = model.fit_predict(
        coordinates
    )

    cluster_counts = (
        sample_df["Cluster"]
        .value_counts()
        .rename("Accidents")
        .reset_index()
    )

    cluster_counts.columns = [
        "Cluster",
        "Accidents",
    ]

    cluster_counts = cluster_counts.sort_values(
        "Accidents",
        ascending=False,
    ).reset_index(drop=True)

    cluster_counts["Hotspot_Rank"] = (
        np.arange(len(cluster_counts)) + 1
    )

    cluster_counts["Risk_Score"] = (
        cluster_counts["Accidents"]
        / cluster_counts["Accidents"].max()
    )

    def classify_usa(rank, total):
        if rank <= max(1, int(np.ceil(total * 0.30))):
            return "High"
        if rank <= max(2, int(np.ceil(total * 0.70))):
            return "Medium"
        return "Low"

    cluster_counts["Hotspot_Level"] = [
        classify_usa(
            rank,
            len(cluster_counts)
        )
        for rank in cluster_counts["Hotspot_Rank"]
    ]

    centers = pd.DataFrame(
        model.cluster_centers_,
        columns=["Latitude", "Longitude"]
    )

    centers["Cluster"] = centers.index

    cluster_summary = cluster_counts.merge(
        centers,
        on="Cluster",
        how="left"
    )

    high = int(
        (cluster_summary["Hotspot_Level"] == "High").sum()
    )

    medium = int(
        (cluster_summary["Hotspot_Level"] == "Medium").sum()
    )

    low = int(
        (cluster_summary["Hotspot_Level"] == "Low").sum()
    )

    m1, m2, m3, m4 = st.columns(4)

    m1.metric("🔴 High-Risk Hotspots", high)
    m2.metric("🟠 Medium-Risk Hotspots", medium)
    m3.metric("🟢 Low-Risk Hotspots", low)
    m4.metric("📍 Clusters Analyzed", n_clusters)

    overview, hotspot_table, map_tab, method = st.tabs(
        [
            "OVERVIEW",
            "HOTSPOT TABLE",
            "MAP VIEW",
            "DETECTION METHOD",
        ]
    )

    with overview:

        left, right = st.columns(2)

        with left:

            st.subheader("🔥 Hotspot Level Distribution")

            levels = (
                cluster_summary["Hotspot_Level"]
                .value_counts()
                .rename_axis("Hotspot Level")
                .reset_index(name="Clusters")
            )

            fig = px.bar(
                levels,
                x="Hotspot Level",
                y="Clusters",
                text="Clusters",
                title="USA K-Means Hotspot Classification"
            )

            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=380,
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        with right:

            st.subheader("📈 Risk Score Distribution")

            risk = cluster_summary[
                ["Cluster", "Risk_Score"]
            ].sort_values(
                "Risk_Score",
                ascending=True,
            )

            fig = px.bar(
                risk,
                x="Risk_Score",
                y="Cluster",
                orientation="h",
                title="USA Cluster Risk Scores"
            )

            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=380,
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        st.subheader("🏆 Highest-Risk Clusters")

        st.dataframe(
            cluster_summary[
                [
                    "Cluster",
                    "Accidents",
                    "Risk_Score",
                    "Hotspot_Level",
                    "Latitude",
                    "Longitude",
                ]
            ].sort_values(
                "Risk_Score",
                ascending=False
            ),
            use_container_width=True,
            hide_index=True,
        )

    with hotspot_table:

        st.subheader("📋 USA K-Means Hotspot Clusters")

        st.dataframe(
            cluster_summary[
                [
                    "Cluster",
                    "Hotspot_Rank",
                    "Accidents",
                    "Risk_Score",
                    "Hotspot_Level",
                    "Latitude",
                    "Longitude",
                ]
            ].sort_values("Hotspot_Rank"),
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "⬇️ Download USA Hotspot Results",
            cluster_summary.to_csv(index=False).encode("utf-8"),
            "USA_Hotspot_Results.csv",
            "text/csv",
            use_container_width=True,
        )

    with map_tab:

        st.subheader("🗺️ USA K-Means Accident Hotspot Map")

        st.success(
            f"K-Means detected {n_clusters} geographic clusters "
            f"using {sample_size:,} representative records."
        )

        fig = px.scatter_mapbox(
            cluster_summary,
            lat="Latitude",
            lon="Longitude",
            size="Accidents",
            color="Hotspot_Level",
            hover_name="Cluster",
            hover_data={
                "Accidents": True,
                "Risk_Score": ":.3f",
                "Hotspot_Level": True,
                "Latitude": ":.4f",
                "Longitude": ":.4f",
            },
            center={"lat": 39.5, "lon": -98.5},
            zoom=3.5,
            mapbox_style="open-street-map",
            title="USA K-Means Accident Hotspots",
        )

        fig.update_layout(
            height=650,
            margin=dict(l=0, r=0, t=55, b=0),
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with method:

        st.subheader("🧠 USA K-Means Method")

        st.write(
            "The USA accident dataset contains Start_Lat and "
            "Start_Lng coordinates, so geographic K-Means clustering "
            "can be applied directly."
        )

        st.write(
            f"The dashboard uses a representative sample of "
            f"{sample_size:,} records to keep the interface responsive."
        )

        st.write(
            "Clusters are ranked by accident concentration. The "
            "highest-concentration clusters are classified as High, "
            "followed by Medium and Low."
        )

        st.info(
            "These hotspot levels are analytical classifications, "
            "not live emergency warnings."
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "🔥 SafeRoad AI — Accident Hotspot Detection | "
    "India state-level hotspot analysis + USA geographic K-Means analysis"
)
