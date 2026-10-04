import streamlit as st
import pandas as pd
from pathlib import Path
import plotly.express as px

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "dataset"
INDIA_DIR = DATASET_DIR / "Indian Datasets"
USA_DIR = DATASET_DIR / "US Datasets"

# ============================================================
# HOME PAGE STYLE
# IMPORTANT: This version intentionally avoids custom HTML
# divs. Streamlit native components are used so HTML cannot
# appear as visible code/text.
# ============================================================

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #07111f 0%, #0b1d32 50%, #102d4d 100%);
}

.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #061322 0%, #0b1d32 100%);
}

[data-testid="stSidebar"] * {
    color: #eaf2ff;
}

.hero-box {
    padding: 34px 38px;
    border-radius: 22px;
    background: linear-gradient(135deg, #123b65, #081a2e);
    border: 1px solid #245a87;
    margin-bottom: 28px;
}

.hero-title {
    font-size: 46px;
    font-weight: 800;
    color: white;
}

.hero-subtitle {
    font-size: 20px;
    color: #b9d8f5;
    margin-top: 8px;
}

.hero-text {
    font-size: 16px;
    line-height: 1.7;
    color: #dcecff;
    margin-top: 16px;
}

.section-title {
    color: white;
    font-size: 27px;
    font-weight: 750;
    margin-top: 24px;
    margin-bottom: 15px;
}

.card {
    background: rgba(15, 39, 66, 0.88);
    border: 1px solid #28567d;
    border-radius: 18px;
    padding: 22px;
    min-height: 175px;
}

.card h3 {
    color: white;
    margin-top: 0;
}

.card p {
    color: #c9dcef;
    line-height: 1.55;
}

.status {
    background: rgba(13, 45, 71, 0.9);
    border: 1px solid #2b648e;
    border-radius: 14px;
    padding: 15px 18px;
    color: #dcecff;
}

.pipeline {
    background: rgba(9, 29, 50, 0.92);
    border: 1px solid #28567d;
    border-radius: 18px;
    padding: 22px;
}

.pipeline-step {
    background: #123b65;
    border-radius: 12px;
    padding: 12px;
    text-align: center;
    color: white;
    margin: 7px 0;
    font-weight: 600;
}

.pipeline-arrow {
    text-align: center;
    color: #8ec8f5;
    font-size: 20px;
}

.footer-box {
    text-align: center;
    padding: 22px;
    border-top: 1px solid #28567d;
    color: #9fb8ce;
    margin-top: 30px;
}


    /* ---------- FORCE NATIVE STREAMLIT TEXT TO STAY VISIBLE ---------- */

    [data-testid="stMetric"] {
        background: rgba(15, 39, 66, 0.88) !important;
        border: 1px solid #28567d !important;
        border-radius: 18px !important;
        padding: 20px !important;
    }

    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] *,
    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] *,
    [data-testid="stMetricDelta"],
    [data-testid="stMetricDelta"] * {
        color: #f4f7fb !important;
    }

    [data-testid="stMetricLabel"] {
        font-size: 15px !important;
        font-weight: 600 !important;
    }

    [data-testid="stMetricValue"] {
        font-size: 34px !important;
        font-weight: 700 !important;
    }

    [data-testid="stAlert"] {
        color: #ffffff !important;
        border-radius: 14px !important;
    }

    [data-testid="stAlert"] p,
    [data-testid="stAlert"] div,
    [data-testid="stAlert"] span {
        color: #eaf7ff !important;
    }

    /* Plotly chart titles, labels and axes */
    .js-plotly-plot .plotly .gtitle {
        fill: #ffffff !important;
    }

    .js-plotly-plot .plotly .xtitle,
    .js-plotly-plot .plotly .ytitle {
        fill: #b9d8f5 !important;
    }

    .js-plotly-plot .plotly .xtick text,
    .js-plotly-plot .plotly .ytick text {
        fill: #b9d8f5 !important;
    }

</style>
""", unsafe_allow_html=True)

# ============================================================
# DATA LOADERS
# ============================================================

@st.cache_data
def load_india_data():
    candidates = [
        DATASET_DIR / "India_Final_Checked_Dataset.csv",
        DATASET_DIR / "India_Integrated_Road_Accidents.csv",
        DATASET_DIR / "India_ML_Dataset.csv",
        DATASET_DIR / "India_Accident_Hotspots.csv",
        INDIA_DIR / "cleaned_india_accidents.csv",
        INDIA_DIR / "india_accidents_eda.csv",
    ]

    for file in candidates:
        if file.exists():
            try:
                df = pd.read_csv(file)
                return df
            except Exception:
                pass

    return pd.DataFrame()


@st.cache_data
def load_usa_data():
    candidates = [
        USA_DIR / "cleaned_US_Accidents.csv",
        USA_DIR / "hotspot_US_Accidents.csv",
        USA_DIR / "preprocessed_US_Accidents.csv",
        USA_DIR / "US_Accidents_Dec21_updated.csv",
    ]

    for file in candidates:
        if file.exists():
            try:
                # Load only useful Home-page columns when available.
                df = pd.read_csv(file, usecols=lambda c: c in [
                    "Severity",
                    "Start_Time",
                    "Start_Lat",
                    "Start_Lng",
                    "State",
                    "City",
                    "Weather_Condition",
                ])
                return df
            except Exception:
                try:
                    return pd.read_csv(file)
                except Exception:
                    pass

    return pd.DataFrame()


india_df = load_india_data()
usa_df = load_usa_data()

# ============================================================
# HELPERS
# ============================================================

def find_column(df, names):
    for name in names:
        if name in df.columns:
            return name
    return None


india_state_col = find_column(
    india_df,
    ["State", "state", "STATE", "State/UT", "State_Name", "State Name"]
)

usa_state_col = find_column(
    usa_df,
    ["State", "state", "STATE", "State_Name"]
)

india_records = len(india_df)
usa_records = len(usa_df)

india_states = int(india_df[india_state_col].nunique()) if india_state_col else 0
usa_states = int(usa_df[usa_state_col].nunique()) if usa_state_col else 0

total_records = india_records + usa_records
total_regions = india_states + usa_states

# ============================================================
# HERO
# ============================================================

st.markdown("""
<div class="hero-box">
    <div class="hero-title">🚦 SafeRoad AI</div>
    <div class="hero-subtitle">
        Intelligent Accident Risk Prediction & Hotspot Detection Platform
    </div>
    <div class="hero-text">
        An AI-powered road safety platform that transforms historical
        accident data into meaningful risk information, accident hotspot
        insights and safety-oriented analysis for India and the United States.
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# OVERVIEW
# ============================================================

st.markdown('<div class="section-title">📊 Road Safety Overview</div>', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("🇮🇳 India Accident Records", f"{india_records:,}")

with c2:
    st.metric("🇺🇸 USA Accident Records", f"{usa_records:,}")

with c3:
    st.metric("🌎 States / Regions", f"{total_regions:,}")

with c4:
    st.metric("📁 Total Records", f"{total_records:,}")

# ============================================================
# DATASET STATUS
# ============================================================

st.markdown('<div class="section-title">🔗 Dataset Connection</div>', unsafe_allow_html=True)

s1, s2 = st.columns(2)

with s1:
    if india_df.empty:
        st.error("🇮🇳 India dataset is not connected.")
    else:
        st.success(f"🇮🇳 India dataset connected — {india_records:,} records")

with s2:
    if usa_df.empty:
        st.error("🇺🇸 USA dataset is not connected.")
    else:
        st.success(f"🇺🇸 USA dataset connected — {usa_records:,} records")

# ============================================================
# MAIN FEATURES
# ============================================================

st.markdown('<div class="section-title">🚀 Core Platform Features</div>', unsafe_allow_html=True)

f1, f2, f3 = st.columns(3)

with f1:
    st.markdown("""
    <div class="card">
        <h3>🔥 Accident Hotspot Detection</h3>
        <p>
        K-Means clustering groups accident locations and helps identify
        areas with higher accident concentration.
        </p>
    </div>
    """, unsafe_allow_html=True)

with f2:
    st.markdown("""
    <div class="card">
        <h3>⚠️ Accident Risk Prediction</h3>
        <p>
        Random Forest uses accident-related and environmental features
        to classify accident risk into meaningful risk levels.
        </p>
    </div>
    """, unsafe_allow_html=True)

with f3:
    st.markdown("""
    <div class="card">
        <h3>🗺️ Interactive Safety Visualization</h3>
        <p>
        Maps, charts and analytical views help users understand where
        and under which conditions accidents are concentrated.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# DATASET COMPARISON
# ============================================================

st.markdown('<div class="section-title">🌍 India vs USA Dataset Overview</div>', unsafe_allow_html=True)

comparison_df = pd.DataFrame({
    "Dataset": ["India", "USA"],
    "Accident Records": [india_records, usa_records],
    "States / Regions": [india_states, usa_states],
})

fig = px.bar(
    comparison_df,
    x="Dataset",
    y="Accident Records",
    text="Accident Records",
    title="Accident Records Available to SafeRoad AI",
)

fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=20, r=20, t=60, b=20),
    font=dict(color="#eaf2ff"),
    title_font=dict(color="#ffffff"),
)

st.plotly_chart(fig, use_container_width=True)

# ============================================================
# USA ACCIDENT TREND
# ============================================================

if not usa_df.empty and "Start_Time" in usa_df.columns:
    usa_temp = usa_df.copy()
    usa_temp["Start_Time"] = pd.to_datetime(
        usa_temp["Start_Time"],
        errors="coerce",
        dayfirst=True
    )

    usa_temp = usa_temp.dropna(subset=["Start_Time"])

    if not usa_temp.empty:
        yearly = (
            usa_temp.assign(Year=usa_temp["Start_Time"].dt.year)
            .groupby("Year")
            .size()
            .reset_index(name="Accidents")
        )

        st.markdown(
            '<div class="section-title">📈 USA Accident Trend</div>',
            unsafe_allow_html=True
        )

        trend_fig = px.line(
            yearly,
            x="Year",
            y="Accidents",
            markers=True,
            title="USA Accident Records by Year",
        )

        trend_fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#eaf2ff"),
            title_font=dict(color="#ffffff"),
            xaxis=dict(
                title_font=dict(color="#b9d8f5"),
                tickfont=dict(color="#b9d8f5"),
            ),
            yaxis=dict(
                title_font=dict(color="#b9d8f5"),
                tickfont=dict(color="#b9d8f5"),
            ),
        )

        st.plotly_chart(trend_fig, use_container_width=True)

# ============================================================
# MACHINE LEARNING
# ============================================================

st.markdown('<div class="section-title">🤖 Machine Learning Architecture</div>', unsafe_allow_html=True)

m1, m2 = st.columns(2)

with m1:
    st.markdown("""
    <div class="card">
        <h3>🌲 Random Forest Classifier</h3>
        <p>
        Used for accident risk prediction. Multiple decision trees
        learn patterns from accident-related features and produce
        a risk classification.
        </p>
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown("""
    <div class="card">
        <h3>🔵 K-Means Clustering</h3>
        <p>
        Used for hotspot detection. Accident coordinates are grouped
        into clusters so concentrated accident regions can be visualized.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# PROCESS PIPELINE
# ============================================================

st.markdown('<div class="section-title">⚙️ SafeRoad AI Processing Pipeline</div>', unsafe_allow_html=True)

st.markdown("""
<div class="pipeline">
    <div class="pipeline-step">📁 Accident Dataset</div>
    <div class="pipeline-arrow">↓</div>
    <div class="pipeline-step">🧹 Data Cleaning</div>
    <div class="pipeline-arrow">↓</div>
    <div class="pipeline-step">⚙️ Data Preprocessing</div>
    <div class="pipeline-arrow">↓</div>
    <div class="pipeline-step">📊 Exploratory Data Analysis</div>
    <div class="pipeline-arrow">↓</div>
    <div class="pipeline-step">🤖 Machine Learning</div>
    <div class="pipeline-arrow">↓</div>
    <div class="pipeline-step">🔥 Hotspot Detection + ⚠️ Risk Prediction</div>
    <div class="pipeline-arrow">↓</div>
    <div class="pipeline-step">🗺️ Interactive Safety Dashboard</div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# PROJECT VALUE
# ============================================================

st.markdown('<div class="section-title">🎯 What Makes SafeRoad AI Different?</div>', unsafe_allow_html=True)

st.info(
    "Instead of only displaying historical accident statistics, "
    "SafeRoad AI converts accident data into actionable analysis: "
    "hotspot identification, risk classification and safety-oriented "
    "visual insights."
)

# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer-box">
    🚦 SafeRoad AI<br>
    Intelligent Accident Risk Prediction & Hotspot Detection Platform<br><br>
    AI & Data Science — Credit-Based Project
</div>
""", unsafe_allow_html=True)