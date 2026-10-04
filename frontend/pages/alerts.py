import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

from utils.data_loader import load_india, load_usa, find_col

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MORTH_WEATHER_FILE = PROJECT_ROOT / "dataset" / "Indian Datasets" / "India_Weather_Accidents_MoRTH.csv"


# ============================================================
# SAFE ROAD AI - ALERTS & NOTIFICATIONS
# ============================================================

st.set_page_config(
    page_title="SafeRoad AI Alerts",
    page_icon="🔔",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PAGE STYLE
# NOTE:
# The visible content uses native Streamlit components.
# This avoids the <div> code-box issue seen in the previous version.
# ============================================================

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
            radial-gradient(
                circle at 10% 85%,
                rgba(70, 45, 120, 0.18),
                transparent 28%
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
        padding-top: 1.6rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3, h4 {
        color: #ffffff !important;
    }

    p, li, label {
        color: #d2dfed !important;
    }

    [data-testid="stMetric"] {
        background: rgba(12, 32, 54, 0.95);
        border: 1px solid rgba(96, 176, 236, 0.18);
        border-radius: 16px;
        padding: 14px;
    }

    [data-testid="stMetricLabel"] {
        color: #a9bfd4 !important;
    }

    [data-testid="stMetricValue"] {
        color: #ffffff !important;
    }

    .alert-panel {
        background: rgba(12, 34, 57, 0.94);
        border-radius: 16px;
        padding: 20px;
        min-height: 150px;
        border: 1px solid rgba(96, 176, 236, 0.18);
    }

    .alert-panel h4 {
        color: #ffffff;
        font-size: 18px;
        margin: 0 0 10px 0;
    }

    .alert-number {
        color: #ffffff;
        font-size: 30px;
        font-weight: 800;
    }

    .alert-text {
        color: #b9ccdf;
        font-size: 13px;
        line-height: 1.55;
        margin-top: 7px;
    }

    .future-panel {
        background:
            linear-gradient(
                135deg,
                rgba(17, 57, 67, 0.96),
                rgba(8, 30, 42, 0.98)
            );
        border: 1px solid rgba(61, 211, 168, 0.20);
        border-radius: 16px;
        padding: 18px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.title("🔔 Alerts & Notifications")

st.caption(
    "Analytical safety alerts generated from historical accident data."
)


# ============================================================
# DATASET SELECTION
# ============================================================

country = st.selectbox(
    "Select Dataset",
    ["India", "USA"],
)

df = load_india() if country == "India" else load_usa()


# ============================================================
# VALIDATION
# ============================================================

if df.empty:
    st.error(f"{country} dataset could not be loaded.")
    st.stop()

df = df.copy()

df.columns = [
    str(column).strip()
    for column in df.columns
]


# ============================================================
# COLUMN DETECTION
# ============================================================

weather_col = find_col(
    df,
    [
        "Weather_Condition",
        "Weather",
        "weather",
    ],
)

severity_col = find_col(
    df,
    [
        "Severity",
        "Accident_Severity",
        "severity",
    ],
)

state_col = find_col(
    df,
    [
        "State",
        "State/UT",
        "State_Name",
        "state",
    ],
)


# ============================================================
# CALCULATE ALERT VALUES
# ============================================================

rain_count = 0
storm_count = 0
fog_count = 0
weather_source_label = "Historical Records"
weather_source_type = "sensor"

if country == "India" and MORTH_WEATHER_FILE.exists():
    try:
        morth_df = pd.read_csv(MORTH_WEATHER_FILE)
        def get_morth_accidents(substr):
            matched = morth_df[morth_df["Weather_Condition"].str.contains(substr, case=False, na=False)]
            return int(matched["Accidents"].sum()) if not matched.empty else 0

        rain_count = get_morth_accidents("rain")
        fog_count = get_morth_accidents("fog|mist")
        storm_count = get_morth_accidents("hail|sleet|storm")
        weather_source_label = "MoRTH National Accidents"
        weather_source_type = "morth"
    except Exception:
        pass
elif weather_col:

    weather = (
        df[weather_col]
        .fillna("")
        .astype(str)
        .str.lower()
    )

    rain_count = int(
        weather.str.contains(
            r"rain|drizzle",
            regex=True,
            na=False,
        ).sum()
    )

    storm_count = int(
        weather.str.contains(
            r"storm|thunder",
            regex=True,
            na=False,
        ).sum()
    )

    fog_count = int(
        weather.str.contains(
            r"fog|haze|mist",
            regex=True,
            na=False,
        ).sum()
    )
    weather_source_label = "Historical Records"
    weather_source_type = "sensor"


# ============================================================
# MONITORING OVERVIEW
# ============================================================

st.subheader("📊 Alert Monitoring Overview")

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric(
        "Dataset",
        country,
    )

with m2:
    st.metric(
        "Total Records",
        f"{len(df):,}",
    )

with m3:
    clean_state_df = df.copy()
    if state_col:
        clean_state_df = clean_state_df[~clean_state_df[state_col].astype(str).str.contains(r"Total|All India", case=False, na=False)]
    st.metric(
        "States / Regions",
        (
            f"{clean_state_df[state_col].nunique():,}"
            if state_col
            else "N/A"
        ),
    )

with m4:
    if country == "India":
        alert_sources = 3
    else:
        alert_sources = sum(
            value is not None
            for value in [
                weather_col,
                severity_col,
                state_col,
            ]
        )
    st.metric(
        "Alert Sources",
        alert_sources,
    )


# ============================================================
# ANALYTICAL SAFETY ALERTS
# ============================================================

st.subheader("🚨 Analytical Safety Alerts")

a1, a2, a3 = st.columns(3)

with a1:

    st.markdown("### 🌧️ Rain / Drizzle Records")

    st.metric(
        weather_source_label,
        f"{rain_count:,}",
    )

    st.caption(
        "Annual recorded crashes during rainy conditions (MoRTH Table 3.8)."
        if weather_source_type == "morth"
        else "Records associated with rain or drizzle conditions."
    )


with a2:

    st.markdown("### ⛈️ Storm / Hail Records")

    st.metric(
        weather_source_label,
        f"{storm_count:,}",
    )

    st.caption(
        "Annual recorded crashes during hail, sleet or storm (MoRTH Table 3.8)."
        if weather_source_type == "morth"
        else "Records associated with storm or thunder-related conditions."
    )


with a3:

    st.markdown("### 🌫️ Fog / Mist Records")

    st.metric(
        weather_source_label,
        f"{fog_count:,}",
    )

    st.caption(
        "Annual recorded crashes during foggy or misty conditions (MoRTH Table 3.8)."
        if weather_source_type == "morth"
        else "Records associated with fog, haze or mist conditions."
    )


# ============================================================
# SAFETY INTERPRETATION
# ============================================================

st.subheader("🧠 Safety Alert Interpretation")

weather_related = rain_count + storm_count + fog_count

if weather_source_type == "morth":
    st.warning(
        f"🌧️ **Adverse Weather Pattern (Official MoRTH All-India Benchmark):** {weather_related:,} annual accidents "
        f"occur during adverse weather across India (Rain: {rain_count:,}, Fog/Mist: {fog_count:,}, Hail/Sleet: {storm_count:,}). "
        "While sunny/clear conditions account for 74.2% of crashes, low-visibility fog in northern winters and wet highway surfaces "
        "during monsoons represent critical seasonal hazard alerts."
    )
elif not weather_col:

    st.info(
        "Weather information is not available in the selected dataset."
    )

elif weather_related > 0:

    st.warning(
        f"Weather-related pattern detected: {weather_related:,} "
        "records contain rain, storm, fog, haze or mist-related "
        "weather descriptions. These records can support historical "
        "weather-risk analysis."
    )

else:

    st.info(
        "No rain, storm, fog, haze or mist pattern was detected "
        "using the available weather descriptions."
    )


# ============================================================
# SEVERITY DISTRIBUTION
# ============================================================

st.subheader("⚠️ Accident Severity Distribution")

if country == "India" and ("Fatality_Rate" in df.columns or ("Deaths" in df.columns and "Accidents" in df.columns)):
    clean_calc_df = df.copy()
    if state_col:
        clean_calc_df = clean_calc_df[~clean_calc_df[state_col].astype(str).str.contains(r"Total|All India", case=False, na=False)]

    if "Fatality_Rate" in clean_calc_df.columns:
        sev_metric = clean_calc_df.groupby("State")["Fatality_Rate"].mean()
    else:
        sev_metric = (clean_calc_df.groupby("State")["Deaths"].sum() / clean_calc_df.groupby("State")["Accidents"].sum()) * 100

    sev_metric = sev_metric.dropna()

    def severity_bracket(val):
        if val >= 50:
            return "Critical Severity (> 50 Deaths/100 Crashes)"
        elif val >= 35:
            return "High Severity (35 - 50 Deaths/100 Crashes)"
        elif val >= 20:
            return "Moderate Severity (20 - 35 Deaths/100 Crashes)"
        else:
            return "Low Fatality (< 20 Deaths/100 Crashes)"

    bracket_series = sev_metric.apply(severity_bracket)
    bracket_order = [
        "Critical Severity (> 50 Deaths/100 Crashes)",
        "High Severity (35 - 50 Deaths/100 Crashes)",
        "Moderate Severity (20 - 35 Deaths/100 Crashes)",
        "Low Fatality (< 20 Deaths/100 Crashes)",
    ]
    counts_dict = bracket_series.value_counts().to_dict()
    bracket_df = pd.DataFrame([
        {"Severity Level": b, "States": counts_dict.get(b, 0)}
        for b in bracket_order
        if counts_dict.get(b, 0) > 0
    ])

    fig = px.bar(
        bracket_df,
        x="Severity Level",
        y="States",
        text="States",
        color="Severity Level",
        color_discrete_map={
            "Critical Severity (> 50 Deaths/100 Crashes)": "#ff4b4b",
            "High Severity (35 - 50 Deaths/100 Crashes)": "#ffa15a",
            "Moderate Severity (20 - 35 Deaths/100 Crashes)": "#636efa",
            "Low Fatality (< 20 Deaths/100 Crashes)": "#00cc96",
        },
        title="India Regional Accident Severity Index (Persons Killed per 100 Accidents — MoRTH Standard)",
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=420,
        margin=dict(l=20, r=20, t=55, b=20),
        showlegend=False,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.caption(
        "📌 **MoRTH Standard Metric:** In official Indian road safety governance, *Accident Severity* is quantified as "
        "the number of persons killed per 100 accidents. High-speed highway corridors in states like Punjab, Bihar, and Jharkhand "
        "exhibit critical fatality severity despite moderate total accident volumes."
    )

elif severity_col:

    severity_values = pd.to_numeric(
        df[severity_col],
        errors="coerce",
    ).dropna()

    if not severity_values.empty:

        severity_counts = (
            severity_values
            .value_counts()
            .sort_index()
            .rename("Records")
            .reset_index()
        )

        severity_counts.columns = [
            "Severity",
            "Records",
        ]

        fig = px.bar(
            severity_counts,
            x="Severity",
            y="Records",
            text="Records",
            title=f"{country} Accident Severity Distribution",
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=420,
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

    else:

        st.info(
            "Severity values are not usable as numeric data."
        )

else:

    st.info(
        "Severity information is not available in the selected dataset."
    )


# ============================================================
# HIGHEST ACCIDENT-RECORD STATES
# ============================================================

st.subheader("📍 Highest Accident-Record States / Regions")

if state_col:
    clean_states_df = df.copy()
    clean_states_df = clean_states_df[~clean_states_df[state_col].astype(str).str.contains(r"Total|All India", case=False, na=False)]

    if country == "India" and "Accidents" in clean_states_df.columns:
        state_counts = (
            clean_states_df.groupby(state_col, as_index=False)["Accidents"]
            .sum()
            .rename(columns={state_col: "State / Region", "Accidents": "Total Recorded Accidents"})
            .sort_values("Total Recorded Accidents", ascending=False)
            .head(10)
        )
        metric_col_name = "Total Recorded Accidents"
        chart_title = "Top 10 Indian States by Recorded Accident Volume"
    else:
        state_counts = (
            clean_states_df[state_col]
            .dropna()
            .astype(str)
            .str.strip()
            .replace("", pd.NA)
            .dropna()
            .value_counts()
            .head(10)
            .rename("Records")
            .reset_index()
        )

        state_counts.columns = [
            "State / Region",
            "Records",
        ]
        metric_col_name = "Records"
        chart_title = "Top 10 Regions by Record Count"

    left, right = st.columns([1.05, 1])

    with left:

        st.dataframe(
            state_counts,
            use_container_width=True,
            hide_index=True,
        )

    with right:

        fig = px.bar(
            state_counts.sort_values(
                metric_col_name,
                ascending=True,
            ),
            x=metric_col_name,
            y="State / Region",
            orientation="h",
            title=chart_title,
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=430,
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

else:

    st.info(
        "State or region information is not available."
    )


# ============================================================
# ALERT CATEGORIES
# ============================================================

st.subheader("🔔 Alert Categories")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown("### 🌧️ Weather Alerts")
    st.write(
        "Identify historical accident patterns associated "
        "with adverse weather conditions."
    )

with c2:
    st.markdown("### ⚠️ Severity Alerts")
    st.write(
        "Monitor the distribution of accident severity levels "
        "within the selected dataset."
    )

with c3:
    st.markdown("### 📍 Regional Alerts")
    st.write(
        "Identify regions with comparatively high accident-record "
        "concentration."
    )

with c4:
    st.markdown("### 🧠 Analytical Insights")
    st.write(
        "Convert historical accident patterns into useful "
        "road-safety information."
    )


# ============================================================
# IMPORTANT NOTICE
# ============================================================

st.subheader("ℹ️ Important Notice")

st.info(
    "These are analytical alerts generated from historical accident "
    "data. They are intended for analysis and decision support and "
    "are not live emergency notifications."
)


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "🔔 SafeRoad AI — Alerts & Notifications | "
    "Historical accident-data based analytical safety insights"
)
