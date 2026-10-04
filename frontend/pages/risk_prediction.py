import streamlit as st
import pandas as pd
import joblib
from datetime import time
from pathlib import Path

from utils.data_loader import load_india, load_usa, country_states


# ============================================================
# SAFE ROAD AI - ACCIDENT RISK PREDICTION
# ============================================================

st.set_page_config(
    page_title="Risk Prediction | SafeRoad AI",
    page_icon="⚠️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

DATASET_DIR = ROOT_DIR / "dataset"
MODEL_DIR = ROOT_DIR / "models"

INDIA_MODEL_FILE = MODEL_DIR / "india_risk_model.pkl"
USA_MODEL_FILE = MODEL_DIR / "usa_risk_model.pkl"


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
                rgba(29, 78, 121, 0.28),
                transparent 32%
            ),
            radial-gradient(
                circle at 8% 90%,
                rgba(70, 45, 130, 0.16),
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
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3, h4 {
        color: #ffffff !important;
    }

    p, li, label {
        color: #cfdeed !important;
    }

    [data-testid="stMetric"] {
        background: rgba(12, 32, 54, 0.96);
        border: 1px solid rgba(96, 176, 236, 0.18);
        border-radius: 15px;
        padding: 14px;
    }

    [data-testid="stMetricLabel"] {
        color: #a9bfd4 !important;
    }

    [data-testid="stMetricValue"] {
        color: #ffffff !important;
    }

    [data-testid="stAlert"] {
        border-radius: 14px;
    }

    .footer {
        margin-top: 35px;
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
# MODEL LOADER
# ============================================================

@st.cache_resource
def load_saved_model(model_path):

    if not model_path.exists():
        return None

    try:
        return joblib.load(model_path)
    except Exception:
        return None


india_model_data = load_saved_model(INDIA_MODEL_FILE)
usa_model_data = load_saved_model(USA_MODEL_FILE)

# Safe dictionary unpacking
india_model = (
    india_model_data["model"]
    if isinstance(india_model_data, dict) and "model" in india_model_data
    else None
)
india_features = (
    india_model_data["features"]
    if isinstance(india_model_data, dict) and "features" in india_model_data
    else []
)

usa_model = (
    usa_model_data["model"]
    if isinstance(usa_model_data, dict) and "model" in usa_model_data
    else None
)
usa_features = (
    usa_model_data["features"]
    if isinstance(usa_model_data, dict) and "features" in usa_model_data
    else []
)


# ============================================================
# DATA HELPERS
# ============================================================

@st.cache_data
def load_selected_data(country):

    if country == "India":
        return load_india()

    return load_usa()


@st.cache_data
def get_states_for_country(country):

    try:
        states = country_states(country)
    except Exception:
        states = []

    if not states:

        data = load_selected_data(country)

        if not data.empty and "State" in data.columns:

            states = (
                data["State"]
                .dropna()
                .astype(str)
                .str.strip()
                .unique()
                .tolist()
            )

    states = sorted(
        [
            state
            for state in states
            if str(state).strip()
        ]
    )

    return states


@st.cache_data
def get_india_state_years(state_name, required_features):
    """Retrieve available historical years and identify complete reporting years for an Indian state."""
    candidates = [
        ROOT_DIR / "outputs" / "features_india.csv",
        DATASET_DIR / "India_Final_Checked_Dataset.csv",
        DATASET_DIR / "India_ML_Dataset.csv"
    ]
    df = pd.DataFrame()
    for path in candidates:
        if path.exists():
            try:
                temp_df = pd.read_csv(path)
                if "State" in temp_df.columns and "Year" in temp_df.columns:
                    df = temp_df
                    break
            except Exception:
                continue

    if df.empty:
        return [], [], None

    clean_name = str(state_name).strip().lower()
    matched = df[df["State"].astype(str).str.strip().str.lower() == clean_name]
    if matched.empty:
        return [], [], None

    # Available distinct years sorted descending (most recent first)
    all_years = sorted([int(y) for y in matched["Year"].dropna().unique()], reverse=True)

    # Complete years where all 5 required features are non-null
    complete_matched = matched.dropna(subset=required_features)
    complete_years = sorted([int(y) for y in complete_matched["Year"].dropna().unique()], reverse=True)

    # Default to latest complete year if available; otherwise latest available year
    default_year = complete_years[0] if complete_years else (all_years[0] if all_years else None)

    return all_years, complete_years, default_year


@st.cache_data
def get_india_state_features(state_name, required_features, selected_year=None):
    """Retrieve legitimate historical accident features for an Indian state and optional year."""
    candidates = [
        ROOT_DIR / "outputs" / "features_india.csv",
        DATASET_DIR / "India_Final_Checked_Dataset.csv",
        DATASET_DIR / "India_ML_Dataset.csv"
    ]
    df = pd.DataFrame()
    for path in candidates:
        if path.exists():
            try:
                temp_df = pd.read_csv(path)
                if "State" in temp_df.columns and all(f in temp_df.columns for f in required_features):
                    df = temp_df
                    break
            except Exception:
                continue

    if df.empty:
        return None, None, "India feature dataset could not be loaded."

    clean_name = str(state_name).strip().lower()
    matched = df[df["State"].astype(str).str.strip().str.lower() == clean_name]
    if matched.empty:
        return None, None, f"No historical accident record found for state '{state_name}'."

    # If a specific year was requested
    if selected_year is not None and "Year" in matched.columns:
        year_matched = matched[matched["Year"] == int(selected_year)]
        if year_matched.empty:
            return None, int(selected_year), f"No record found for '{state_name}' in Year {selected_year}."

        target_row = year_matched.iloc[0]
        missing_feats = [f for f in required_features if pd.isna(target_row[f])]
        if missing_feats:
            return None, int(selected_year), f"features {', '.join(missing_feats)} are not recorded in official reports for Year {selected_year}."

        features_dict = {f: float(target_row[f]) for f in required_features}
        return features_dict, int(selected_year), None

    # Fallback: select latest complete record
    valid_rows = matched.dropna(subset=required_features)
    if valid_rows.empty:
        return None, None, f"Historical records for '{state_name}' contain incomplete metrics for {', '.join(required_features)}."

    if "Year" in valid_rows.columns:
        valid_rows = valid_rows.sort_values("Year", ascending=False)

    latest = valid_rows.iloc[0]
    year_used = int(latest["Year"]) if "Year" in latest and not pd.isna(latest["Year"]) else "Historical"

    features_dict = {f: float(latest[f]) for f in required_features}
    return features_dict, year_used, None


# ============================================================
# ANALYTICAL RISK ENGINE
# ============================================================

def predict_risk(
    weather,
    traffic,
    road_condition,
    road_type,
    selected_time,
):

    score = 0.20

    reasons = []

    # --------------------------------------------------------
    # WEATHER
    # --------------------------------------------------------

    if weather == "Heavy Rain":

        score += 0.30

        reasons.append(
            "Heavy rain conditions"
        )

    elif weather == "Storm":

        score += 0.28

        reasons.append(
            "Storm conditions"
        )

    elif weather in ["Light Rain", "Fog"]:

        score += 0.18

        reasons.append(
            f"{weather} conditions"
        )

    # --------------------------------------------------------
    # TRAFFIC
    # --------------------------------------------------------

    if traffic == "High":

        score += 0.25

        reasons.append(
            "High traffic"
        )

    elif traffic == "Medium":

        score += 0.12

        reasons.append(
            "Medium traffic"
        )

    # --------------------------------------------------------
    # ROAD CONDITION
    # --------------------------------------------------------

    if road_condition == "Wet / Slippery":

        score += 0.15

        reasons.append(
            "Wet / slippery road"
        )

    elif road_condition == "Damaged":

        score += 0.12

        reasons.append(
            "Damaged road"
        )

    elif road_condition == "Under Construction":

        score += 0.12

        reasons.append(
            "Road under construction"
        )

    # --------------------------------------------------------
    # ROAD TYPE
    # --------------------------------------------------------

    if road_type == "Expressway":

        score += 0.05

        reasons.append(
            "High-speed road type"
        )

    elif road_type == "National Highway":

        score += 0.03

        reasons.append(
            "National highway"
        )

    # --------------------------------------------------------
    # PEAK TIME
    # --------------------------------------------------------

    hour = selected_time.hour

    if hour in range(7, 10) or hour in range(17, 21):

        score += 0.08

        reasons.append(
            "Peak travel time"
        )

    # --------------------------------------------------------
    # LIMIT SCORE
    # --------------------------------------------------------

    score = min(
        max(score, 0.0),
        0.98
    )

    # --------------------------------------------------------
    # RISK LEVEL
    # --------------------------------------------------------

    if score >= 0.65:

        return (
            "HIGH RISK",
            score,
            reasons
        )

    if score >= 0.40:

        return (
            "MEDIUM RISK",
            score,
            reasons
        )

    return (
        "LOW RISK",
        score,
        reasons
    )


# ============================================================
# HEADER
# ============================================================

st.title(
    "⚠️ Accident Risk Prediction"
)

st.caption(
    "Enter road, traffic and environmental conditions "
    "to estimate the relative accident-risk level."
)


# ============================================================
# COUNTRY
# ============================================================

country = st.selectbox(
    "Country",
    [
        "India",
        "USA"
    ]
)


selected_df = load_selected_data(
    country
)

states = get_states_for_country(
    country
)


if not states:

    states = [
        "Other"
    ]


# ============================================================
# DEFAULT STATE
# ============================================================

default_state = "Tamil Nadu"

if (
    country == "India"
    and default_state in states
):

    state_index = states.index(
        default_state
    )

else:

    state_index = 0


# ============================================================
# INPUT SECTION
# ============================================================

st.subheader(
    "🚗 Enter Road & Environmental Conditions"
)

left, right = st.columns(
    2,
    gap="large"
)


# ============================================================
# LOCATION AND ROAD
# ============================================================

with left:

    st.markdown(
        "### 📍 Location & Road"
    )

    state = st.selectbox(
        "State / Region",
        states,
        index=state_index
    )

    if country == "India":
        avail_years, complete_years, default_year = get_india_state_years(state, india_features)
        if avail_years:
            year_idx = avail_years.index(default_year) if default_year in avail_years else 0
            selected_year = st.selectbox(
                "Historical Benchmark Year",
                avail_years,
                index=year_idx,
                help="Select the reporting year to retrieve genuine official state accident statistics for Random Forest ML inference."
            )
            complete_text = ", ".join(str(y) for y in sorted(complete_years)) if complete_years else "None"
            st.caption(
                f"📅 **Historical Data Availability:** {', '.join(str(y) for y in sorted(avail_years))}  \n"
                f"*(Complete 5-feature ML metrics available for: **{complete_text}**)*"
            )
        else:
            selected_year = None
    else:
        selected_year = None


    district = st.text_input(
        "Specific District / City",
        placeholder=(
            "Example: Chennai"
            if country == "India"
            else "Example: Los Angeles"
        )
    )

    road_type = st.selectbox(
        "Road Type",
        [
            "National Highway",
            "State Highway",
            "City Road",
            "Rural Road",
            "Expressway"
        ]
    )

    traffic = st.selectbox(
        "Traffic Condition",
        [
            "Low",
            "Medium",
            "High"
        ]
    )


# ============================================================
# ENVIRONMENTAL CONDITIONS
# ============================================================

with right:

    st.markdown(
        "### 🌦️ Environmental Conditions"
    )

    weather = st.selectbox(
        "Weather Condition",
        [
            "Clear",
            "Light Rain",
            "Heavy Rain",
            "Fog",
            "Storm",
            "Other"
        ]
    )

    road_condition = st.selectbox(
        "Road Condition",
        [
            "Dry",
            "Wet / Slippery",
            "Damaged",
            "Under Construction"
        ]
    )

    selected_time = st.time_input(
        "Time of Day",
        value=time(14, 30)
    )

    st.caption(
        "Morning and evening peak periods are treated "
        "as higher-exposure periods in the current "
        "analytical scoring system."
    )


# ============================================================
# PREDICT BUTTON
# ============================================================

st.divider()

predict_button = st.button(
    "🔮 Predict Risk",
    type="primary",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    # 1. Analytical Risk Assessment (Always calculated as transparent situational baseline)
    risk, score, reasons = predict_risk(
        weather=weather,
        traffic=traffic,
        road_condition=road_condition,
        road_type=road_type,
        selected_time=selected_time
    )


    # ========================================================
    # SECTION 1: MACHINE LEARNING PREDICTION (RANDOM FOREST)
    # ========================================================

    st.subheader(
        "🤖 Random Forest Machine Learning Prediction"
    )

    if country == "India":

        if india_model is not None and india_features:

            state_feats, year_used, err_msg = get_india_state_features(
                state,
                india_features,
                selected_year=selected_year
            )

            if state_feats is not None:

                try:
                    # Construct single-row input strictly in model feature order
                    input_data = pd.DataFrame(
                        [[state_feats[f] for f in india_features]],
                        columns=india_features,
                        dtype=float
                    )

                    ml_prediction = india_model.predict(input_data)[0]

                    # Probabilities if available
                    if hasattr(india_model, "predict_proba"):
                        ml_probabilities = india_model.predict_proba(input_data)[0]
                        classes = list(india_model.classes_)
                        prob_map = {
                            c: round(float(p) * 100, 1)
                            for c, p in zip(classes, ml_probabilities)
                        }
                    else:
                        ml_probabilities = None
                        prob_map = {}

                    # Display ML result
                    pred_str = str(ml_prediction).upper()

                    basis_str = f"{state} — {year_used} Historical Record"

                    if pred_str == "HIGH":
                        st.error(
                            f"🚨 **Random Forest Prediction: HIGH RISK**\n\n"
                            f"State-level macro accident risk for **{state}** is classified as "
                            f"**HIGH** by the trained Random Forest model based on verified "
                            f"historical records ({basis_str})."
                        )
                    elif pred_str == "MEDIUM":
                        st.warning(
                            f"⚠️ **Random Forest Prediction: MEDIUM RISK**\n\n"
                            f"State-level macro accident risk for **{state}** is classified as "
                            f"**MEDIUM** by the trained Random Forest model based on verified "
                            f"historical records ({basis_str})."
                        )
                    else:
                        st.success(
                            f"✅ **Random Forest Prediction: LOW RISK**\n\n"
                            f"State-level macro accident risk for **{state}** is classified as "
                            f"**LOW** by the trained Random Forest model based on verified "
                            f"historical records ({basis_str})."
                        )

                    # ML Metric Summary Columns
                    ml_col1, ml_col2, ml_col3, ml_col4 = st.columns(4)

                    with ml_col1:
                        st.metric(
                            "ML Predicted Risk",
                            pred_str
                        )

                    with ml_col2:
                        conf_val = (
                            f"{max(ml_probabilities) * 100:.1f}%"
                            if ml_probabilities is not None
                            else "N/A"
                        )
                        st.metric(
                            "Model Confidence",
                            conf_val
                        )

                    with ml_col3:
                        st.metric(
                            "Model Architecture",
                            "Random Forest (India)"
                        )

                    with ml_col4:
                        st.metric(
                            "Data Basis",
                            basis_str
                        )

                    # Probability distribution breakdown
                    if prob_map:
                        st.markdown(
                            "**Random Forest Class Probabilities:**"
                        )
                        pcols = st.columns(len(prob_map))
                        for i, (cls_name, prob_pct) in enumerate(prob_map.items()):
                            with pcols[i]:
                                st.metric(
                                    f"P({cls_name})",
                                    f"{prob_pct}%"
                                )

                    # Feature transparency expander
                    with st.expander(
                        f"📋 View Legitimate State Features Used for {state} ({basis_str})"
                    ):
                        feat_df = pd.DataFrame([
                            {
                                "Feature Name": f,
                                "Historical Value": state_feats[f],
                                "Data Source": f"Verified India Dataset ({year_used})"
                            }
                            for f in india_features
                        ])
                        st.dataframe(
                            feat_df,
                            use_container_width=True,
                            hide_index=True
                        )
                        st.caption(
                            "📌 **Model Context:** This 5-feature Random Forest evaluates state-level cross-sectional "
                            "accident metrics for the selected year. (The separate Time-Based model under "
                            "`India_Time_Based_Final_Evaluation.csv` evaluates 2020 lagged indicators to forecast 2021 future risk)."
                        )

                except Exception as e:
                    st.error(
                        f"❌ Random Forest prediction execution failed: {e}"
                    )

            else:
                st.warning(
                    "⚠️ **Complete historical data is unavailable for this State/Year combination.**"
                )
                yr_display = f"Year {year_used}" if year_used else (f"Year {selected_year}" if selected_year else "the selected year")
                st.info(
                    f"The India Random Forest model strictly requires 5 state-level historical accident "
                    f"metrics ({', '.join(india_features)}). For **{state} ({yr_display})**, {err_msg}\n\n"
                    f"**Scientific Integrity Notice:** To prevent spurious or fabricated predictions, missing statistical values "
                    f"are never estimated or synthesized. Please select a historical reporting year with complete data, "
                    f"or refer to the Situational Analytical Risk Score below."
                )


        else:
            st.error(
                "⚠️ India Random Forest model file is not available."
            )

    elif country == "USA":

        st.warning(
            "⚠️ **ML Model Input Mismatch: Required Features Unavailable in Manual Form**"
        )
        st.write(
            f"The trained USA Random Forest model (`models/usa_risk_model.pkl`) "
            f"requires **{len(usa_features)} high-resolution pre-crash sensor, infrastructure, and telemetry features** "
            f"(strictly leakage-free — `Severity` is excluded):"
        )
        st.code(
            ", ".join(usa_features) if usa_features else "Features unavailable"
        )
        st.info(
            "ℹ️ **Scientific Integrity Notice:**\n\n"
            "The current manual form does not provide all telemetry features required by the USA Random Forest model "
            "(such as exact `Distance(mi)`, `Temperature(F)`, `Humidity(%)`, `Wind_Speed(mph)`, and physical infrastructure flags). "
            "**No synthetic values are generated.**\n\n"
            "The application continues using the transparent **📊 Situational Analytical Risk Score** below "
            "to evaluate the manual road, traffic, and environmental conditions."
        )


    st.divider()


    # ========================================================
    # SECTION 2: SITUATIONAL ANALYTICAL RISK SCORE
    # ========================================================

    st.subheader(
        "📊 Situational Analytical Risk Score"
    )

    st.caption(
        "Real-time situational assessment based on current road type, traffic, "
        "weather, surface condition, and time of day."
    )


    if risk == "HIGH RISK":

        st.error(
            "🚨 **HIGH SITUATIONAL RISK**\n\n"
            "The selected conditions indicate an elevated analytical risk level."
        )

    elif risk == "MEDIUM RISK":

        st.warning(
            "⚠️ **MEDIUM SITUATIONAL RISK**\n\n"
            "The selected conditions indicate a moderate analytical risk level."
        )

    else:

        st.success(
            "✅ **LOW SITUATIONAL RISK**\n\n"
            "The selected conditions indicate a comparatively low analytical risk level."
        )


    # Metric Summary
    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.metric(
            "Analytical Score",
            f"{score:.2f}"
        )

    with m2:
        st.metric(
            "Risk Percentage",
            f"{score * 100:.1f}%"
        )

    with m3:
        st.metric(
            "Country",
            country
        )

    with m4:
        st.metric(
            "State / Region",
            state
        )


    # Risk Level Progress
    st.progress(
        score,
        text=f"Current analytical risk score: {score:.2f}"
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.success(
            "🟢 LOW RISK\n\n"
            "0.00 – 0.39"
        )

    with c2:
        st.warning(
            "🟠 MEDIUM RISK\n\n"
            "0.40 – 0.64"
        )

    with c3:
        st.error(
            "🔴 HIGH RISK\n\n"
            "0.65 – 1.00"
        )


    # Input Summary
    st.subheader(
        "📋 Input Summary"
    )

    summary = pd.DataFrame(
        [
            {
                "Country": country,
                "State / Region": state,
                "District / City": (
                    district.strip()
                    if district.strip()
                    else "Not specified"
                ),
                "Road Type": road_type,
                "Weather": weather,
                "Traffic": traffic,
                "Road Condition": road_condition,
                "Time": selected_time.strftime(
                    "%H:%M"
                )
            }
        ]
    )

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True
    )


    # Contributing Conditions
    st.subheader(
        "🔎 Contributing Situational Conditions"
    )

    if reasons:
        for reason in reasons:
            st.write(
                f"• {reason}"
            )
    else:
        st.write(
            "• No additional elevated-risk conditions detected."
        )


    # Why this risk?
    st.subheader(
        "💡 Why this risk level?"
    )

    if risk == "HIGH RISK":
        st.warning(
            "Several selected conditions increase the analytical risk score. "
            "Pay particular attention to weather, traffic, and road surface conditions."
        )
    elif risk == "MEDIUM RISK":
        st.info(
            "The selected conditions contain some elevated-risk factors, but the "
            "overall analytical score remains below the high-risk threshold."
        )
    else:
        st.success(
            "The selected conditions do not produce a high analytical risk score "
            "under the current scoring rules."
        )


    # Safety Recommendations
    st.subheader(
        "🛡️ Safety Recommendation"
    )

    if risk == "HIGH RISK":
        recommendations = [
            "Reduce vehicle speed.",
            "Maintain a larger following distance.",
            "Avoid sudden braking or lane changes.",
            "Use headlights during poor visibility.",
            "Drive cautiously through the selected area."
        ]
    elif risk == "MEDIUM RISK":
        recommendations = [
            "Maintain a controlled speed.",
            "Keep sufficient distance from other vehicles.",
            "Monitor weather and road conditions.",
            "Avoid unnecessary sudden manoeuvres."
        ]
    else:
        recommendations = [
            "Follow the posted speed limit.",
            "Maintain a safe following distance.",
            "Follow traffic regulations.",
            "Continue monitoring road and weather conditions."
        ]

    for recommendation in recommendations:
        st.write(
            f"✅ {recommendation}"
        )


    # ========================================================
    # SECTION 3: MODEL DIAGNOSTICS & STATUS
    # ========================================================

    st.divider()

    st.subheader(
        "⚙️ Machine Learning Models Status & Architecture"
    )

    india_loaded = india_model is not None
    usa_loaded = usa_model is not None

    mc1, mc2 = st.columns(2)

    with mc1:
        if india_loaded:
            target_classes = ", ".join(map(str, getattr(india_model, "classes_", [])))
            st.success(
                f"🇮🇳 **India Random Forest Model — Loaded**  \n"
                f"• Model: `{type(india_model).__name__}`  \n"
                f"• Target: `Risk_Level` ({target_classes})  \n"
                f"• Features: {len(india_features)}"
            )
        else:
            st.error(
                "🇮🇳 India Random Forest Model — Not Loaded"
            )

    with mc2:
        if usa_loaded:
            target_classes = ", ".join(map(str, getattr(usa_model, "classes_", [])))
            st.success(
                f"🇺🇸 **USA Random Forest Model — Loaded**  \n"
                f"• Model: `{type(usa_model).__name__}`  \n"
                f"• Target: `High_Severity` ({target_classes})  \n"
                f"• Features: {len(usa_features)}"
            )
        else:
            st.error(
                "🇺🇸 USA Random Forest Model — Not Loaded"
            )

    with st.expander(
        "🔍 View Saved Model Feature Signatures"
    ):
        if india_loaded:
            st.markdown(
                f"**India Model Features ({len(india_features)}):** `{india_features}`"
            )
        if usa_loaded:
            st.markdown(
                f"**USA Model Features ({len(usa_features)}):** `{usa_features}`"
            )

    st.caption(
        "SafeRoad AI maintains a strict distinction between historical machine "
        "learning predictions and situational analytical risk scoring to ensure "
        "rigorous, non-fabricated road safety evaluations."
    )


# ============================================================
# BEFORE PREDICTION
# ============================================================

else:

    st.info(
        "Enter the conditions above and select "
        "**Predict Risk** to generate the risk assessment."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    "---"
)

st.markdown(
    """
    <div class="footer">
        ⚠️ SafeRoad AI — Accident Risk Prediction
        <br>
        Intelligent Accident Risk Prediction & Hotspot Detection Platform
    </div>
    """,
    unsafe_allow_html=True
)
