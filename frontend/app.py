import streamlit as st
from pathlib import Path
import base64

# ============================================================
# SAFE ROAD AI - MAIN APPLICATION
# ============================================================

st.set_page_config(
    page_title="SafeRoad AI",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# LOAD GLOBAL BACKGROUND IMAGE
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
BACKGROUND_IMAGE = BASE_DIR / "assets" / "saferoad_global_bg.png"

with open(BACKGROUND_IMAGE, "rb") as image_file:
    background_base64 = base64.b64encode(image_file.read()).decode()


# ============================================================
# GLOBAL APPLICATION STYLE
# ============================================================

st.markdown(
    f"""
    <style>

    /* ========================================================
       MAIN APPLICATION BACKGROUND
       ======================================================== */

    .stApp {{
        background-image:
            linear-gradient(
                rgba(3, 15, 29, 0.78),
                rgba(3, 15, 29, 0.78)
            ),
            url("data:image/png;base64,{background_base64}");

        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;

        min-height: 100vh;
    }}


    /* ========================================================
       MAIN CONTENT
       ======================================================== */

    .block-container {{
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        max-width: 1450px;
    }}


    /* ========================================================
       SIDEBAR
       ======================================================== */

    section[data-testid="stSidebar"] {{
        background:
            linear-gradient(
                180deg,
                #061322 0%,
                #0a1c31 55%,
                #102b49 100%
            ) !important;

        border-right: 1px solid rgba(120, 180, 230, 0.18);
    }}


    section[data-testid="stSidebar"],
    section[data-testid="stSidebar"] * {{
        color: #e9f3ff !important;
    }}


    section[data-testid="stSidebar"] div[role="button"],
    section[data-testid="stSidebar"] a {{
        color: #dcecff !important;
    }}


    section[data-testid="stSidebar"]
    div[role="button"][aria-current="page"] {{
        background: rgba(77, 143, 207, 0.28) !important;
        color: #ffffff !important;
        border-radius: 10px;
    }}


    section[data-testid="stSidebar"]
    div[role="button"][aria-current="page"] * {{
        color: #ffffff !important;
    }}


    section[data-testid="stSidebar"] button {{
        color: #ffffff !important;
    }}


    /* ========================================================
       HEADINGS
       ======================================================== */

    h1,
    h2,
    h3,
    h4 {{
        color: #ffffff !important;
    }}


    p,
    li,
    label {{
        color: #d6e4f2 !important;
    }}


    /* ========================================================
       INPUTS
       ======================================================== */

    div[data-baseweb="select"] > div {{
        background-color: rgba(245, 248, 252, 0.96) !important;
        color: #132238 !important;
    }}


    div[data-baseweb="select"] * {{
        color: #132238 !important;
    }}


    /* ========================================================
       GENERAL METRICS
       ======================================================== */

    [data-testid="stMetric"] {{
        background: rgba(12, 32, 54, 0.92) !important;
        border: 1px solid rgba(95, 170, 235, 0.18) !important;
        border-radius: 14px !important;
        padding: 14px !important;
    }}


    [data-testid="stMetricLabel"] {{
        color: #a9bdd2 !important;
    }}


    [data-testid="stMetricValue"] {{
        color: #ffffff !important;
    }}


    [data-testid="stMetricDelta"] {{
        color: #bcd2e8 !important;
    }}


    /* ========================================================
       ALERTS / INFO BOXES
       ======================================================== */

    [data-testid="stAlert"] p {{
        color: inherit !important;
    }}


    /* ========================================================
       DATAFRAME
       ======================================================== */

    [data-testid="stDataFrame"] {{
        border-radius: 12px;
        overflow: hidden;
    }}


    /* ========================================================
       BUTTONS
       ======================================================== */

    .stButton > button {{
        border-radius: 10px;
        min-height: 44px;
    }}


    /* ========================================================
       FOOTER
       ======================================================== */

    footer {{
        visibility: hidden;
    }}

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# PAGE NAVIGATION
# ============================================================

pages = [

    st.Page(
        "pages/home.py",
        title="Home",
        icon="🏠",
        default=True,
    ),

    st.Page(
        "pages/india_analysis.py",
        title="India Analysis",
        icon="🇮🇳",
    ),

    st.Page(
        "pages/usa_analysis.py",
        title="USA Analysis",
        icon="🇺🇸",
    ),

    st.Page(
        "pages/comparative_analysis.py",
        title="Comparative Analysis",
        icon="⚖️",
    ),

    st.Page(
        "pages/risk_prediction.py",
        title="Risk Prediction",
        icon="⚠️",
    ),

    st.Page(
        "pages/hotspot_detection.py",
        title="Accident Hotspots",
        icon="🔥",
    ),

    st.Page(
        "pages/map_view.py",
        title="Map View",
        icon="🗺️",
    ),

    st.Page(
        "pages/alerts.py",
        title="Alerts & Notifications",
        icon="🔔",
    ),

    st.Page(
        "pages/reports.py",
        title="Reports",
        icon="📄",
    ),

    st.Page(
        "pages/about.py",
        title="About SafeRoad AI",
        icon="ℹ️",
    ),
]


# ============================================================
# RUN NAVIGATION
# ============================================================

st.navigation(pages).run()