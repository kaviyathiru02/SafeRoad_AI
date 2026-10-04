import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path


# ============================================================
# SAFE ROAD AI - INDIA ACCIDENT ANALYSIS
# ============================================================

st.set_page_config(
    page_title="India Accident Analysis | SafeRoad AI",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

INDIA_DATA = ROOT_DIR / "dataset" / "India_ML_Dataset.csv"
HOTSPOT_DATA = ROOT_DIR / "dataset" / "India_Accident_Hotspots.csv"


# ============================================================
# PAGE STYLE
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background:
            radial-gradient(
                circle at top right,
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

    h1, h2, h3 {
        color: #ffffff !important;
    }

    p, label {
        color: #cbd8e8 !important;
    }

    .page-title {
        color: #ffffff;
        font-size: 40px;
        font-weight: 800;
        margin-bottom: 4px;
    }

    .page-subtitle {
        color: #9fb5cb;
        font-size: 16px;
        margin-bottom: 22px;
    }

    .summary-card {
        background: linear-gradient(
            145deg,
            rgba(14, 34, 57, 0.98),
            rgba(8, 24, 42, 0.98)
        );
        border: 1px solid rgba(110, 180, 240, 0.22);
        border-radius: 16px;
        padding: 22px;
        min-height: 135px;
        box-shadow: 0 8px 26px rgba(0, 0, 0, 0.22);
    }

    .summary-label {
        font-size: 14px;
        font-weight: 650;
        color: #9fb3c8;
        margin-bottom: 10px;
    }

    .summary-value {
        font-size: 32px;
        font-weight: 800;
        color: #ffffff;
    }

    .summary-note {
        font-size: 12px;
        color: #7792ad;
        margin-top: 7px;
    }

    .risk-high {
        border-left: 4px solid #ef5350;
    }

    .risk-medium {
        border-left: 4px solid #ffab40;
    }

    .risk-low {
        border-left: 4px solid #35d39a;
    }

    .risk-total {
        border-left: 4px solid #54a8ff;
    }

    .risk-box {
        background: linear-gradient(
            135deg,
            rgba(16, 53, 64, 0.85),
            rgba(12, 33, 44, 0.95)
        );
        border-left: 4px solid #36d6a4;
        border-radius: 12px;
        padding: 17px 20px;
        color: #d9f7ef;
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
# DATA LOADING
# ============================================================

@st.cache_data
def load_india_dataset():

    if not INDIA_DATA.exists():
        return pd.DataFrame()

    try:
        data = pd.read_csv(
            INDIA_DATA,
            low_memory=False,
        )

        data.columns = [
            str(c).strip()
            for c in data.columns
        ]

        return data

    except Exception:
        return pd.DataFrame()


@st.cache_data
def load_hotspot_dataset():

    if not HOTSPOT_DATA.exists():
        return pd.DataFrame()

    try:
        data = pd.read_csv(
            HOTSPOT_DATA,
            low_memory=False,
        )

        data.columns = [
            str(c).strip()
            for c in data.columns
        ]

        return data

    except Exception:
        return pd.DataFrame()


df = load_india_dataset()
hotspot_df = load_hotspot_dataset()


# ============================================================
# REQUIRED DATA CHECK
# ============================================================

if df.empty:

    st.error("India dataset could not be loaded.")
    st.caption(str(INDIA_DATA))
    st.stop()


required_columns = [
    "Year",
    "State",
    "Accidents",
    "Deaths",
    "Injuries",
]

missing = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing:

    st.error(
        "India_ML_Dataset.csv is missing required columns: "
        + ", ".join(missing)
    )

    st.write(
        "Available columns:",
        df.columns.tolist()
    )

    st.stop()


# ============================================================
# CLEAN NUMERIC COLUMNS
# ============================================================

for column in [
    "Year",
    "Accidents",
    "Deaths",
    "Injuries",
]:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce",
    )

df["State"] = (
    df["State"]
    .astype(str)
    .str.strip()
)

df = df.dropna(
    subset=[
        "Year",
        "State",
    ]
).copy()

df["Year"] = df["Year"].astype(int)

# Unavailable metrics are preserved as NaN (not filled with 0) to maintain data integrity


# ============================================================
# YEAR FILTER
# ============================================================

years = sorted(
    df["Year"].unique().tolist()
)

if not years:

    st.error(
        "No valid year values were found in the India dataset."
    )

    st.stop()


selected_year = st.selectbox(
    "Select Year",
    years,
    index=len(years) - 1,
)


year_df = df[
    df["Year"] == selected_year
].copy()


# ============================================================
# HOTSPOT COUNTS
# ============================================================

high = 0
medium = 0
low = 0

if (
    not hotspot_df.empty
    and "Hotspot_Level" in hotspot_df.columns
):

    levels = (
        hotspot_df["Hotspot_Level"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    high = int(
        (levels == "high").sum()
    )

    medium = int(
        (levels == "medium").sum()
    )

    low = int(
        (levels == "low").sum()
    )


# ============================================================
# STATE COUNT
# ============================================================

states = int(
    year_df["State"].nunique()
)

if (
    not hotspot_df.empty
    and "State" in hotspot_df.columns
):

    total_states = int(
        hotspot_df["State"]
        .astype(str)
        .str.strip()
        .nunique()
    )

else:

    total_states = states


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="page-title">🇮🇳 India Accident Analysis</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="page-subtitle">'
    'Detailed analysis of road accidents across Indian states.'
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# HOTSPOT CLASSIFICATION NOTE
# ============================================================

st.caption(
    "Hotspot classification is based on the generated "
    "India hotspot dataset and is shown independently of "
    "the selected accident-analysis year."
)


# ============================================================
# SUMMARY CARDS
# ============================================================

c1, c2, c3, c4 = st.columns(4)


with c1:

    st.markdown(
        f"""
        <div class="summary-card risk-high">
            <div class="summary-label">🔴 HIGH RISK STATES</div>
            <div class="summary-value">{high}</div>
            <div class="summary-note">
                Final hotspot classification
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


with c2:

    st.markdown(
        f"""
        <div class="summary-card risk-medium">
            <div class="summary-label">🟠 MEDIUM RISK STATES</div>
            <div class="summary-value">{medium}</div>
            <div class="summary-note">
                Final hotspot classification
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


with c3:

    st.markdown(
        f"""
        <div class="summary-card risk-low">
            <div class="summary-label">🟢 LOW RISK STATES</div>
            <div class="summary-value">{low}</div>
            <div class="summary-note">
                Final hotspot classification
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


with c4:

    st.markdown(
        f"""
        <div class="summary-card risk-total">
            <div class="summary-label">🇮🇳 TOTAL STATES / UTs</div>
            <div class="summary-value">{total_states}</div>
            <div class="summary-note">
                State-level India dataset coverage
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# DASHBOARD TABS
# ============================================================

overview_tab, hotspots_tab, map_tab, trends_tab, state_tab = st.tabs(
    [
        "OVERVIEW",
        "HOTSPOTS",
        "MAP VIEW",
        "TRENDS",
        "STATE WISE DATA",
    ]
)


# ============================================================
# OVERVIEW TAB
# ============================================================

with overview_tab:

    left, right = st.columns(2)


    # --------------------------------------------------------
    # ACCIDENT TREND
    # --------------------------------------------------------

    with left:

        st.subheader("📈 Accidents Over Years (India)")

        trend = (
            df.dropna(subset=["Accidents"])
            .groupby(
                "Year",
                as_index=False
            )["Accidents"]
            .sum()
            .sort_values("Year")
        )

        if trend.empty:
            st.info("Accident-count data is not available in the current official dataset.")
        else:
            fig = px.line(
                trend,
                x="Year",
                y="Accidents",
                markers=True,
            )

            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=360,
                margin=dict(
                    l=20,
                    r=20,
                    t=25,
                    b=20,
                ),
                xaxis_title="Year",
                yaxis_title="Accidents",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )


    # --------------------------------------------------------
    # TOP STATES
    # --------------------------------------------------------

    with right:

        st.subheader(
            f"🏆 Top 5 States by Accidents ({selected_year})"
        )

        if not year_df["Accidents"].notna().any():
            st.info(
                f"Accident-count data is not available for {selected_year} in the current official dataset."
            )
        else:
            top_states = (
                year_df.dropna(subset=["Accidents"])
                .groupby(
                    "State",
                    as_index=False
                )["Accidents"]
                .sum()
                .sort_values(
                    "Accidents",
                    ascending=False,
                )
                .head(5)
                .sort_values(
                    "Accidents",
                    ascending=True,
                )
            )

            fig = px.bar(
                top_states,
                x="Accidents",
                y="State",
                orientation="h",
                text="Accidents",
            )

            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=360,
                margin=dict(
                    l=20,
                    r=20,
                    t=25,
                    b=20,
                ),
                xaxis_title="Accidents",
                yaxis_title="",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )


    # --------------------------------------------------------
    # YEAR SUMMARY
    # --------------------------------------------------------

    st.subheader(
        f"📊 Accident Statistics ({selected_year})"
    )

    accidents_display = (
        f"{int(year_df['Accidents'].sum()):,}"
        if year_df["Accidents"].notna().any()
        else "Data Unavailable"
    )

    deaths_display = (
        f"{int(year_df['Deaths'].sum()):,}"
        if year_df["Deaths"].notna().any()
        else "Data Unavailable"
    )

    injuries_display = (
        f"{int(year_df['Injuries'].sum()):,}"
        if year_df["Injuries"].notna().any()
        else "Data Unavailable"
    )

    m1, m2, m3 = st.columns(3)

    m1.metric(
        "Total Accidents",
        accidents_display,
    )

    m2.metric(
        "Total Deaths",
        deaths_display,
    )

    m3.metric(
        "Total Injuries",
        injuries_display,
    )


    # --------------------------------------------------------
    # HIGH-RISK STATES
    # --------------------------------------------------------

    if not hotspot_df.empty:

        high_df = hotspot_df.copy()

        if "Hotspot_Level" in high_df.columns:

            high_df["Hotspot_Level"] = (
                high_df["Hotspot_Level"]
                .astype(str)
                .str.strip()
            )

            high_df = high_df[
                high_df["Hotspot_Level"]
                .str.lower()
                == "high"
            ]

        if "Risk_Score" in high_df.columns:

            high_df["Risk_Score"] = pd.to_numeric(
                high_df["Risk_Score"],
                errors="coerce",
            )

            high_df = high_df.sort_values(
                "Risk_Score",
                ascending=False,
            )

        st.subheader("🔴 High Risk States")

        if (
            not high_df.empty
            and "State" in high_df.columns
        ):

            names = (
                high_df["State"]
                .astype(str)
                .tolist()
            )

            st.markdown(
                f"""
                <div class="risk-box">
                    <b>{len(names)} high-risk states identified</b>
                    <br><br>
                    {", ".join(names)}
                </div>
                """,
                unsafe_allow_html=True,
            )

        else:

            st.info(
                "No high-risk states are available."
            )


# ============================================================
# HOTSPOTS TAB
# ============================================================

with hotspots_tab:

    st.subheader("🔥 India Accident Hotspots")

    if hotspot_df.empty:

        st.warning(
            "India_Accident_Hotspots.csv was not found."
        )

    elif not {
        "State",
        "Risk_Score",
        "Hotspot_Level",
    }.issubset(hotspot_df.columns):

        st.warning(
            "Hotspot dataset does not contain State, "
            "Risk_Score and Hotspot_Level."
        )

        st.write(
            "Available columns:",
            hotspot_df.columns.tolist()
        )

    else:

        hotspot_view = hotspot_df[
            [
                "State",
                "Risk_Score",
                "Hotspot_Level",
            ]
        ].copy()

        hotspot_view["Risk_Score"] = pd.to_numeric(
            hotspot_view["Risk_Score"],
            errors="coerce",
        )

        hotspot_view = hotspot_view.sort_values(
            "Risk_Score",
            ascending=False,
        )

        st.dataframe(
            hotspot_view,
            use_container_width=True,
            hide_index=True,
        )

        st.subheader(
            "📊 Hotspot Distribution"
        )

        distribution = (
            hotspot_df["Hotspot_Level"]
            .astype(str)
            .str.strip()
            .value_counts()
            .rename_axis("Hotspot Level")
            .reset_index(name="States")
        )

        fig = px.bar(
            distribution,
            x="Hotspot Level",
            y="States",
            text="States",
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=350,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )


# ============================================================
# MAP TAB
# ============================================================

with map_tab:

    st.subheader("🗺️ India Hotspot Map")

    st.info(
        "Your current India hotspot dataset is state-level. "
        "The map uses approximate state-centre coordinates "
        "to visualize hotspot intensity rather than exact "
        "accident GPS points."
    )

    STATE_COORDINATES = {

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
        "Daman and Diu": (20.4143, 72.8328),
        "Dadra and Nagar Haveli": (20.1809, 73.0169),
        "Dadra and Nagar Haveli and Daman and Diu": (
            20.1809,
            73.0169,
        ),
        "Andaman and Nicobar Islands": (
            11.7401,
            92.6586,
        ),
        "Lakshadweep": (10.5667, 72.6417),
    }


    if hotspot_df.empty:

        st.warning(
            "Hotspot dataset is unavailable."
        )

    else:

        required_map_columns = {
            "State",
            "Risk_Score",
        }

        if not required_map_columns.issubset(
            hotspot_df.columns
        ):

            st.warning(
                "State and Risk_Score fields are required "
                "for the India hotspot map."
            )

        else:

            map_df = hotspot_df.copy()

            map_df["State"] = (
                map_df["State"]
                .astype(str)
                .str.strip()
            )

            map_df["Risk_Score"] = pd.to_numeric(
                map_df["Risk_Score"],
                errors="coerce",
            )

            map_df["Latitude"] = map_df["State"].map(
                lambda state:
                STATE_COORDINATES.get(
                    state,
                    (None, None)
                )[0]
            )

            map_df["Longitude"] = map_df["State"].map(
                lambda state:
                STATE_COORDINATES.get(
                    state,
                    (None, None)
                )[1]
            )

            map_df = map_df.dropna(
                subset=[
                    "Latitude",
                    "Longitude",
                    "Risk_Score",
                ]
            )

            if map_df.empty:

                st.warning(
                    "No state names could be matched "
                    "to map coordinates."
                )

            else:

                map_kwargs = {
                    "lat": "Latitude",
                    "lon": "Longitude",
                    "size": "Risk_Score",
                    "hover_name": "State",
                    "hover_data": {
                        "Risk_Score": ":.3f",
                        "Latitude": False,
                        "Longitude": False,
                    },
                    "zoom": 4.5,
                    "center": {
                        "lat": 22.5,
                        "lon": 79.0,
                    },
                    "mapbox_style": "open-street-map",
                    "title": "State-Level India Accident Hotspots",
                }

                if "Hotspot_Level" in map_df.columns:
                    map_kwargs["color"] = "Hotspot_Level"

                map_chart = px.scatter_mapbox(
                    map_df,
                    **map_kwargs,
                )

                map_chart.update_layout(
                    height=650,
                    margin=dict(
                        l=0,
                        r=0,
                        t=50,
                        b=0,
                    ),
                )

                st.plotly_chart(
                    map_chart,
                    use_container_width=True,
                )


# ============================================================
# TRENDS TAB
# ============================================================

with trends_tab:

    st.subheader("📈 India Accident Trends")

    metric = st.selectbox(
        "Select trend metric",
        [
            "Accidents",
            "Deaths",
            "Injuries",
        ],
    )

    metric_df = (
        df.dropna(subset=[metric])
        .groupby(
            "Year",
            as_index=False
        )[metric]
        .sum()
        .sort_values("Year")
    )

    if metric_df.empty:
        st.info(
            f"{metric} data is not available in the current official dataset."
        )
    else:
        fig = px.line(
            metric_df,
            x="Year",
            y=metric,
            markers=True,
            title=f"{metric} Over Years",
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=450,
            margin=dict(
                l=20,
                r=20,
                t=55,
                b=20,
            ),
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    st.subheader(
        f"📊 State-wise Accidents in {selected_year}"
    )

    if not year_df["Accidents"].notna().any():
        st.info(
            "Accident-count data is not available for this year in the current official dataset."
        )
    else:
        state_year = (
            year_df.dropna(subset=["Accidents"])
            .groupby(
                "State",
                as_index=False
            )["Accidents"]
            .sum()
            .sort_values(
                "Accidents",
                ascending=False,
            )
        )

        fig2 = px.bar(
            state_year,
            x="Accidents",
            y="State",
            orientation="h",
            title=f"State-wise Accidents in {selected_year}",
        )

        fig2.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=650,
            margin=dict(
                l=20,
                r=20,
                t=55,
                b=20,
            ),
        )

        st.plotly_chart(
            fig2,
            use_container_width=True,
        )


# ============================================================
# STATE WISE DATA TAB
# ============================================================

with state_tab:

    st.subheader(
        f"📋 State-wise Accident Data ({selected_year})"
    )

    state_table = (
        year_df
        .groupby(
            "State",
            as_index=False
        )[
            [
                "Accidents",
                "Deaths",
                "Injuries",
            ]
        ]
        .sum(min_count=1)
        .sort_values(
            "Accidents",
            ascending=False,
            na_position="last",
        )
    )

    st.dataframe(
        state_table,
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "⬇️ Download India State-wise Data",
        data=state_table
        .to_csv(index=False)
        .encode("utf-8"),
        file_name=(
            f"India_State_Wise_{selected_year}.csv"
        ),
        mime="text/csv",
        use_container_width=True,
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        🇮🇳 SafeRoad AI &nbsp;|&nbsp;
        India Accident Analysis
        <br>
        Intelligent Accident Risk Prediction & Hotspot Detection Platform
    </div>
    """,
    unsafe_allow_html=True,
)
