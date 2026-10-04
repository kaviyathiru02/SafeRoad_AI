import streamlit as st
import textwrap

# ============================================================
# SAFE ROAD AI - ABOUT PAGE
# ============================================================

st.set_page_config(
    page_title="About SafeRoad AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# GLOBAL PAGE STYLE
# ============================================================

st.markdown(
textwrap.dedent("""
<style>
.stApp {
    background:
        radial-gradient(circle at 85% 5%, rgba(30, 80, 125, 0.30), transparent 32%),
        radial-gradient(circle at 8% 85%, rgba(70, 45, 140, 0.20), transparent 28%),
        linear-gradient(135deg, #06111f 0%, #0a1b30 50%, #102d4b 100%);
    color: #eef6ff;
}

.block-container {
    max-width: 1450px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}

h1, h2, h3, h4 {
    color: #ffffff !important;
}

p, li {
    color: #cfdeee !important;
}

.about-hero {
    padding: 38px 42px;
    min-height: 285px;
    border-radius: 24px;
    background: linear-gradient(125deg, rgba(18, 57, 95, 0.98), rgba(7, 25, 44, 0.98));
    border: 1px solid rgba(82, 175, 240, 0.28);
    box-shadow: 0 18px 45px rgba(0,0,0,0.30);
}

.badge {
    display: inline-block;
    padding: 7px 13px;
    border-radius: 999px;
    background: rgba(61, 211, 168, 0.12);
    border: 1px solid rgba(61, 211, 168, 0.28);
    color: #6ee7c1;
    font-size: 12px;
    font-weight: 750;
    letter-spacing: 0.5px;
}

.hero-title {
    margin-top: 16px;
    font-size: 46px;
    font-weight: 850;
    color: white;
}

.hero-title span {
    color: #42d6aa;
}

.hero-subtitle {
    margin-top: 8px;
    font-size: 20px;
    font-weight: 650;
    color: #9fd7ff;
}

.hero-text {
    margin-top: 16px;
    max-width: 900px;
    font-size: 16px;
    line-height: 1.8;
    color: #d7e7f7;
}

.section-title {
    margin-top: 30px;
    margin-bottom: 17px;
    font-size: 28px;
    font-weight: 800;
    color: white;
}

.card {
    background: linear-gradient(145deg, rgba(14, 38, 64, 0.98), rgba(7, 24, 41, 0.98));
    border: 1px solid rgba(96, 175, 235, 0.18);
    border-radius: 18px;
    padding: 23px;
    min-height: 205px;
    box-shadow: 0 9px 28px rgba(0,0,0,0.20);
}

.card h3 {
    color: white;
    font-size: 20px;
    margin-bottom: 10px;
}

.card p {
    color: #c9d9e9;
    line-height: 1.65;
    font-size: 14px;
}

.algorithm-card {
    background: linear-gradient(145deg, rgba(13, 36, 60, 0.98), rgba(7, 23, 40, 0.98));
    border: 1px solid rgba(96, 175, 235, 0.18);
    border-radius: 17px;
    padding: 21px;
    min-height: 220px;
}

.algorithm-icon {
    font-size: 32px;
}

.algorithm-card h3 {
    color: white;
    font-size: 18px;
    margin: 8px 0;
}

.algorithm-card p {
    color: #c6d7e8;
    line-height: 1.6;
    font-size: 14px;
}

.workflow {
    background: linear-gradient(145deg, rgba(11, 34, 57, 0.98), rgba(6, 22, 39, 0.98));
    border: 1px solid rgba(96, 175, 235, 0.18);
    border-radius: 18px;
    padding: 23px;
}

.workflow-step {
    background: #123a61;
    border: 1px solid rgba(109, 189, 240, 0.18);
    border-radius: 11px;
    padding: 13px;
    text-align: center;
    color: white;
    font-weight: 700;
    margin: 7px 0;
}

.workflow-arrow {
    text-align: center;
    color: #69d7ff;
    font-size: 21px;
}

.future-card {
    background: linear-gradient(135deg, rgba(17, 57, 67, 0.95), rgba(8, 30, 42, 0.98));
    border: 1px solid rgba(61, 211, 168, 0.20);
    border-radius: 18px;
    padding: 22px;
}

.future-item {
    background: rgba(37, 81, 90, 0.32);
    border-radius: 10px;
    padding: 10px 13px;
    margin: 7px 0;
    color: #d8eee8;
}

.footer-box {
    text-align: center;
    margin-top: 35px;
    padding-top: 22px;
    border-top: 1px solid rgba(255,255,255,0.08);
    color: #8198af;
    font-size: 13px;
}
</style>
"""),
unsafe_allow_html=True,
)

# ============================================================
# HERO
# ============================================================

st.html(
    """
    <div class="about-hero">

        <div class="badge">
            AI-POWERED ROAD SAFETY PLATFORM
        </div>

        <h1 class="hero-title">
            🛡️ About SafeRoad <span>AI</span>
        </h1>

        <div class="hero-subtitle">
            Intelligent Accident Risk Prediction & Hotspot Detection
        </div>

        <p class="hero-text">
            SafeRoad AI combines accident-data analysis, machine learning
            and geographical visualization to understand accident patterns,
            identify high-risk areas and transform historical accident
            information into meaningful road-safety insights.
        </p>

    </div>
    """
)

# ============================================================
# WHAT IS SAFEROAD AI?
# ============================================================

st.markdown(
    '<div class="section-title">🚦 What is SafeRoad AI?</div>',
    unsafe_allow_html=True,
)

col1, col2 = st.columns(2)

with col1:
    st.markdown(
    textwrap.dedent("""
    <div class="card">
        <h3>🎯 Project Purpose</h3>
        <p>
            SafeRoad AI analyzes historical road accident information
            to help users understand accident-prone regions and risky
            road or environmental conditions.
        </p>
        <p>
            The platform goes beyond displaying statistics by converting
            accident data into risk information, hotspot insights and
            safety-oriented recommendations.
        </p>
    </div>
    """),
    unsafe_allow_html=True,
    )

with col2:
    st.markdown(
    textwrap.dedent("""
    <div class="card">
        <h3>💡 Core Idea</h3>
        <p>
            The system brings data preparation, exploratory analysis,
            machine learning, hotspot detection and visualization into
            one integrated road-safety platform.
        </p>
        <p>
            It supports analysis for both India and the United States.
        </p>
    </div>
    """),
    unsafe_allow_html=True,
    )

# ============================================================
# ALGORITHMS
# ============================================================

st.markdown(
    '<div class="section-title">🤖 Algorithms & Methods Used</div>',
    unsafe_allow_html=True,
)

algorithm_data = [
    ("🌲", "Random Forest",
     "Used for accident risk prediction by learning relationships between accident-related and environmental features."),
    ("🔥", "K-Means Clustering",
     "Used for hotspot detection by grouping accident locations or regional accident patterns into clusters."),
    ("🧹", "Data Preprocessing",
     "Includes data cleaning, missing-value handling, feature preparation and transformation before modeling."),
    ("⚙️", "Feature Engineering",
     "Creates useful derived indicators and model inputs to represent accident risk and accident concentration."),
]

cols = st.columns(4)

for col, (icon, title, desc) in zip(cols, algorithm_data):
    with col:
        st.markdown(
        textwrap.dedent(f"""
        <div class="algorithm-card">
            <div class="algorithm-icon">{icon}</div>
            <h3>{title}</h3>
            <p>{desc}</p>
        </div>
        """),
        unsafe_allow_html=True,
        )

# ============================================================
# DATASETS
# ============================================================

st.markdown(
    '<div class="section-title">📂 Datasets Used</div>',
    unsafe_allow_html=True,
)

dataset_data = [
    ("🇮🇳", "India Road Accident Dataset",
     "Used for Indian accident statistics, state-wise analysis, trends and hotspot/risk analysis."),
    ("🇺🇸", "US Road Accident Dataset",
     "Contains accident severity, time, location, weather and road-related attributes."),
    ("🌦️", "Weather Information",
     "Weather conditions help analyze the environmental factors associated with accidents."),
    ("🚗", "Traffic & Road Features",
     "Road and traffic attributes support accident pattern analysis and risk assessment."),
]

cols = st.columns(4)

for col, (icon, title, desc) in zip(cols, dataset_data):
    with col:
        st.markdown(
        textwrap.dedent(f"""
        <div class="card">
            <h3>{icon} {title}</h3>
            <p>{desc}</p>
        </div>
        """),
        unsafe_allow_html=True,
        )

# ============================================================
# WORKFLOW
# ============================================================

st.markdown(
    '<div class="section-title">🔄 SafeRoad AI Workflow</div>',
    unsafe_allow_html=True,
)

st.markdown(
textwrap.dedent("""
<div class="workflow">
    <div class="workflow-step">📥 Data Collection</div>
    <div class="workflow-arrow">↓</div>
    <div class="workflow-step">🧹 Data Cleaning</div>
    <div class="workflow-arrow">↓</div>
    <div class="workflow-step">⚙️ Data Preprocessing & Feature Engineering</div>
    <div class="workflow-arrow">↓</div>
    <div class="workflow-step">📊 Exploratory Data Analysis</div>
    <div class="workflow-arrow">↓</div>
    <div class="workflow-step">🌲 Random Forest Risk Prediction</div>
    <div class="workflow-arrow">↓</div>
    <div class="workflow-step">🔥 K-Means Hotspot Detection</div>
    <div class="workflow-arrow">↓</div>
    <div class="workflow-step">🗺️ Interactive Visualization</div>
    <div class="workflow-arrow">↓</div>
    <div class="workflow-step">🛡️ Safety Insights & Recommendations</div>
</div>
"""),
unsafe_allow_html=True,
)

# ============================================================
# DIFFERENCE
# ============================================================

st.markdown(
    '<div class="section-title">🎯 What Makes SafeRoad AI Different?</div>',
    unsafe_allow_html=True,
)

st.info(
    "Instead of only showing historical accident statistics, SafeRoad AI "
    "converts the analysis into risk information, hotspot identification "
    "and safety-oriented insights."
)

# ============================================================
# FUTURE SCOPE
# ============================================================

st.markdown(
    '<div class="section-title">🚀 Future Scope</div>',
    unsafe_allow_html=True,
)

future_items = [
    "Real-time accident risk prediction",
    "Live weather API integration",
    "Real-time traffic data integration",
    "Real-time alerts and notifications",
    "Safe route recommendation",
    "District-level risk prediction",
    "Mobile application for road users",
    "Real-time hotspot monitoring",
]

future_html = "\n".join(
    f'<div class="future-item">• {item}</div>'
    for item in future_items
)

st.markdown(
textwrap.dedent(f"""
<div class="future-card">
    {future_html}
</div>
"""),
unsafe_allow_html=True,
)

# ============================================================
# FOOTER
# ============================================================

st.markdown(
textwrap.dedent("""
<div class="footer-box">
    🛡️ SafeRoad AI<br>
    Intelligent Accident Risk Prediction & Hotspot Detection Platform
    <br><br>
    AI & Data Science — Credit-Based Project
</div>
"""),
unsafe_allow_html=True,
)
