import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

from utils.data_loader import load_india, load_usa, find_col

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "dataset"
MORTH_WEATHER_FILE = DATASET_DIR / "Indian Datasets" / "India_Weather_Accidents_MoRTH.csv"


# ============================================================
# SAFE ROAD AI - COMPARATIVE ANALYSIS
# ============================================================

st.set_page_config(
    page_title="SafeRoad AI - Comparative Analysis",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# GLOBAL STYLE
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background:
            radial-gradient(
                circle at 86% 4%,
                rgba(30, 82, 128, 0.28),
                transparent 32%
            ),
            linear-gradient(
                135deg,
                #06111f 0%,
                #0a1d32 52%,
                #102d4b 100%
            );
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

    p, li, label {
        color: #cfdeee !important;
    }

    [data-testid="stMetric"] {
        background: rgba(12, 32, 54, 0.95);
        border: 1px solid rgba(100, 176, 236, 0.18);
        border-radius: 16px;
        padding: 14px;
    }

    [data-testid="stMetricLabel"] {
        color: #a9bfd4 !important;
    }

    [data-testid="stMetricValue"] {
        color: #ffffff !important;
    }

    .section-note {
        background: rgba(13, 42, 66, 0.75);
        border: 1px solid rgba(96, 176, 236, 0.16);
        border-radius: 14px;
        padding: 15px 18px;
        color: #cfe2f5;
        margin-bottom: 18px;
    }

    .comparison-card {
        background: linear-gradient(
            145deg,
            rgba(14, 38, 64, 0.97),
            rgba(7, 24, 42, 0.98)
        );
        border: 1px solid rgba(96, 176, 236, 0.18);
        border-radius: 17px;
        padding: 20px;
        min-height: 160px;
        box-shadow: 0 9px 26px rgba(0,0,0,0.20);
    }

    .comparison-card-title {
        color: #ffffff;
        font-size: 18px;
        font-weight: 750;
        margin-bottom: 8px;
    }

    .comparison-card-value {
        color: #ffffff;
        font-size: 30px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .comparison-card-note {
        color: #a9bdd1;
        font-size: 13px;
        line-height: 1.45;
    }

    .insight-box {
        background: linear-gradient(
            135deg,
            rgba(16, 57, 67, 0.92),
            rgba(8, 30, 42, 0.98)
        );
        border: 1px solid rgba(61, 211, 168, 0.18);
        border-radius: 15px;
        padding: 18px;
        color: #d7efe8;
    }

    .footer {
        text-align: center;
        margin-top: 35px;
        padding-top: 22px;
        border-top: 1px solid rgba(255,255,255,0.08);
        color: #8198af;
        font-size: 13px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def get_data():
    india = load_india()
    usa = load_usa()
    return india, usa


india_df, usa_df = get_data()


# ============================================================
# VALIDATE / CLEAN
# ============================================================

if not india_df.empty:
    india_df = india_df.copy()
    india_df.columns = [str(c).strip() for c in india_df.columns]

if not usa_df.empty:
    usa_df = usa_df.copy()
    usa_df.columns = [str(c).strip() for c in usa_df.columns]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def numeric_mean(df, column_name):
    if not column_name or column_name not in df.columns:
        return None

    values = pd.to_numeric(df[column_name], errors="coerce").dropna()

    if values.empty:
        return None

    return float(values.mean())


def find_year_column(df):
    return find_col(
        df,
        [
            "Year",
            "year",
            "Date",
            "Start_Time",
        ],
    )


def yearly_record_counts(df):
    if df.empty:
        return pd.DataFrame(columns=["Year", "Records"])

    year_col = find_col(
        df,
        [
            "Year",
            "year",
        ],
    )

    if year_col:
        years = pd.to_numeric(df[year_col], errors="coerce")
        result = years.dropna().astype(int).value_counts().sort_index()
        return result.rename("Records").reset_index(
            name=None
        ).rename(columns={"index": "Year"})

    date_col = find_col(
        df,
        [
            "Start_Time",
            "Date",
            "date",
        ],
    )

    if date_col:
        dates = pd.to_datetime(df[date_col], errors="coerce")
        valid = dates.dropna()

        if not valid.empty:
            result = (
                valid.dt.year
                .value_counts()
                .sort_index()
                .rename("Records")
                .reset_index()
            )
            result.columns = ["Year", "Records"]
            return result

    return pd.DataFrame(columns=["Year", "Records"])


def state_count(df):
    state_col = find_col(
        df,
        [
            "State",
            "State/UT",
            "State_Name",
            "state",
        ],
    )

    if not state_col:
        return None

    series = (
        df[state_col]
        .dropna()
        .astype(str)
        .str.strip()
    )

    series = series[series != ""]

    return int(series.nunique())


def severity_distribution(df):
    col = find_col(
        df,
        [
            "Severity",
            "Accident_Severity",
            "severity",
        ],
    )

    if not col:
        return pd.DataFrame(columns=["Severity", "Records"])

    values = pd.to_numeric(df[col], errors="coerce").dropna()

    if values.empty:
        return pd.DataFrame(columns=["Severity", "Records"])

    result = (
        values.value_counts()
        .sort_index()
        .rename("Records")
        .reset_index()
    )

    result.columns = ["Severity", "Records"]
    return result


def weather_top(df, limit=7):
    col = find_col(
        df,
        [
            "Weather_Condition",
            "Weather",
            "weather",
        ],
    )

    if not col:
        return pd.DataFrame(columns=["Weather", "Records"])

    result = (
        df[col]
        .dropna()
        .astype(str)
        .str.strip()
    )

    result = result[result != ""]

    result = (
        result.value_counts()
        .head(limit)
        .rename("Records")
        .reset_index()
    )

    result.columns = ["Weather", "Records"]
    return result


# ============================================================
# COLUMNS
# ============================================================

india_severity = find_col(
    india_df,
    ["Severity", "Accident_Severity", "severity"],
)

usa_severity = find_col(
    usa_df,
    ["Severity", "Accident_Severity", "severity"],
)

india_states = state_count(india_df)
usa_states = state_count(usa_df)

# Official MoRTH Accident Severity Index for India: (Deaths / Accidents) * 100 or mean(Fatality_Rate)
clean_india_sev = india_df.copy()
if "State" in clean_india_sev.columns:
    clean_india_sev = clean_india_sev[~clean_india_sev["State"].astype(str).str.contains(r"Total|All India", case=False, na=False)]

if "Fatality_Rate" in clean_india_sev.columns:
    india_avg_severity_val = float(clean_india_sev["Fatality_Rate"].dropna().mean())
    india_severity_display = f"{india_avg_severity_val:.1f}% (Fatality Rate)"
elif "Deaths" in clean_india_sev.columns and "Accidents" in clean_india_sev.columns:
    tot_d = pd.to_numeric(clean_india_sev["Deaths"], errors="coerce").fillna(0).sum()
    tot_a = pd.to_numeric(clean_india_sev["Accidents"], errors="coerce").fillna(0).sum()
    india_avg_severity_val = (tot_d / tot_a * 100) if tot_a > 0 else None
    india_severity_display = f"{india_avg_severity_val:.1f}% (Fatality Rate)" if india_avg_severity_val else "N/A"
else:
    india_avg_severity_val = numeric_mean(india_df, india_severity)
    india_severity_display = f"{india_avg_severity_val:.2f}" if india_avg_severity_val is not None else "N/A"

usa_avg_severity = numeric_mean(usa_df, usa_severity)
usa_severity_display = f"{usa_avg_severity:.2f} (Scale 1–4)" if usa_avg_severity is not None else "N/A"


# ============================================================
# HEADER
# ============================================================

st.title("⚖️ Comparative Analysis")

st.caption(
    "India vs USA accident datasets — comparative statistics, "
    "severity patterns, regional coverage and accident trends."
)


# ============================================================
# DATASET STATUS
# ============================================================

if india_df.empty:
    st.error("India dataset could not be loaded.")

if usa_df.empty:
    st.error("USA dataset could not be loaded.")

if india_df.empty and usa_df.empty:
    st.stop()


st.markdown(
    """
    <div class="section-note">
        <b>Comparison basis:</b> the values below are calculated directly
        from the currently connected India and USA datasets.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATASET SUMMARY
# ============================================================

st.subheader("📊 Dataset Summary")

summary = pd.DataFrame(
    {
        "Country": ["India", "USA"],
        "Dataset Records": [
            len(india_df),
            len(usa_df),
        ],
        "States / Regions": [
            india_states if india_states is not None else "N/A",
            usa_states if usa_states is not None else "N/A",
        ],
        "Average Severity": [
            india_severity_display,
            usa_severity_display,
        ],
    }
)

st.dataframe(
    summary,
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# TOP METRICS
# ============================================================

st.subheader("🌍 At-a-Glance Comparison")

c1, c2, c3, c4 = st.columns(4)

total_records = len(india_df) + len(usa_df)

with c1:
    st.metric(
        "🇮🇳 India Records",
        f"{len(india_df):,}",
    )

with c2:
    st.metric(
        "🇺🇸 USA Records",
        f"{len(usa_df):,}",
    )

with c3:
    st.metric(
        "📁 Combined Records",
        f"{total_records:,}",
    )

with c4:

    if india_states is not None and usa_states is not None:
        st.metric(
            "🌎 Combined State / Region Coverage",
            f"{india_states + usa_states:,}",
        )
    else:
        st.metric(
            "🌎 State / Region Coverage",
            "N/A",
        )


# ============================================================
# RECORD COMPARISON CHART
# ============================================================

st.subheader("📈 Accident Records Available in Each Dataset")

records_chart_df = pd.DataFrame(
    {
        "Country": ["India", "USA"],
        "Dataset Records": [
            len(india_df),
            len(usa_df),
        ],
    }
)

fig = px.bar(
    records_chart_df,
    x="Country",
    y="Dataset Records",
    text="Dataset Records",
    title="Accident Records by Dataset",
)

fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    height=440,
    margin=dict(l=20, r=20, t=60, b=20),
    yaxis_title="Dataset Records",
    xaxis_title="Country",
)

st.plotly_chart(
    fig,
    use_container_width=True,
)


# ============================================================
# TWO-COLUMN COMPARISON
# ============================================================

st.subheader("🔎 Analytical Comparison")

left, right = st.columns(2)


with left:

    st.markdown("### 🇮🇳 India")

    india_cards = [
        (
            "Dataset Records",
            f"{len(india_df):,}",
        ),
        (
            "States / Regions",
            (
                f"{india_states:,}"
                if india_states is not None
                else "N/A"
            ),
        ),
        (
            "Average Severity",
            india_severity_display,
        ),
    ]

    for label, value in india_cards:
        st.metric(label, value)


with right:

    st.markdown("### 🇺🇸 USA")

    usa_cards = [
        (
            "Dataset Records",
            f"{len(usa_df):,}",
        ),
        (
            "States / Regions",
            (
                f"{usa_states:,}"
                if usa_states is not None
                else "N/A"
            ),
        ),
        (
            "Average Severity",
            usa_severity_display,
        ),
    ]

    for label, value in usa_cards:
        st.metric(label, value)


# ============================================================
# SEVERITY COMPARISON
# ============================================================

st.subheader("⚠️ Severity Comparison")

sev_left, sev_right = st.columns(2)

with sev_left:
    st.markdown("### 🇮🇳 India — MoRTH Severity Index")
    if "Fatality_Rate" in clean_india_sev.columns or ("Deaths" in clean_india_sev.columns and "Accidents" in clean_india_sev.columns):
        if "Fatality_Rate" in clean_india_sev.columns:
            sev_metric_in = clean_india_sev.groupby("State")["Fatality_Rate"].mean()
        else:
            sev_metric_in = (clean_india_sev.groupby("State")["Deaths"].sum() / clean_india_sev.groupby("State")["Accidents"].sum()) * 100

        sev_metric_in = sev_metric_in.dropna()

        def severity_bracket(val):
            if val >= 50:
                return "Critical (> 50 Deaths/100 Crashes)"
            elif val >= 35:
                return "High (35 - 50 Deaths/100 Crashes)"
            elif val >= 20:
                return "Moderate (20 - 35 Deaths/100 Crashes)"
            else:
                return "Low (< 20 Deaths/100 Crashes)"

        bracket_series = sev_metric_in.apply(severity_bracket)
        bracket_order = [
            "Critical (> 50 Deaths/100 Crashes)",
            "High (35 - 50 Deaths/100 Crashes)",
            "Moderate (20 - 35 Deaths/100 Crashes)",
            "Low (< 20 Deaths/100 Crashes)",
        ]
        counts_dict = bracket_series.value_counts().to_dict()
        bracket_df = pd.DataFrame([
            {"Severity Level": b, "States": counts_dict.get(b, 0)}
            for b in bracket_order
            if counts_dict.get(b, 0) > 0
        ])

        fig_in = px.bar(
            bracket_df,
            x="Severity Level",
            y="States",
            text="States",
            color="Severity Level",
            color_discrete_map={
                "Critical (> 50 Deaths/100 Crashes)": "#ff4b4b",
                "High (35 - 50 Deaths/100 Crashes)": "#ffa15a",
                "Moderate (20 - 35 Deaths/100 Crashes)": "#636efa",
                "Low (< 20 Deaths/100 Crashes)": "#00cc96",
            },
            title="India: States by Fatality Severity Index",
        )
        fig_in.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=380,
            showlegend=False,
            margin=dict(l=20, r=20, t=55, b=20),
        )
        st.plotly_chart(fig_in, use_container_width=True)
        st.caption("📌 **MoRTH Metric:** Fatality Rate = (Deaths / Accidents) × 100 (national benchmark).")
    else:
        st.info("India severity metric not available.")

with sev_right:
    st.markdown("### 🇺🇸 USA — Incident Impact Severity")
    severity_usa = severity_distribution(usa_df)
    if not severity_usa.empty:
        fig_us = px.bar(
            severity_usa,
            x="Severity",
            y="Records",
            text="Records",
            color="Severity",
            title="USA: Incidents by Severity Rating (Scale 1–4)",
        )
        fig_us.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=380,
            showlegend=False,
            margin=dict(l=20, r=20, t=55, b=20),
        )
        st.plotly_chart(fig_us, use_container_width=True)
        st.caption("📌 **US Metric:** Sensor-recorded impact scale from 1 (minimal) to 4 (critical closure).")
    else:
        st.info("USA severity metric not available.")


# ============================================================
# ACCIDENT TREND COMPARISON
# ============================================================

st.subheader("📅 Accident Trend Comparison")

india_trend = yearly_record_counts(india_df)
usa_trend = yearly_record_counts(usa_df)

trend_frames = []

if not india_trend.empty:
    india_trend = india_trend.copy()
    india_trend["Country"] = "India"
    trend_frames.append(india_trend)

if not usa_trend.empty:
    usa_trend = usa_trend.copy()
    usa_trend["Country"] = "USA"
    trend_frames.append(usa_trend)

if trend_frames:

    combined_trend = pd.concat(
        trend_frames,
        ignore_index=True,
    )

    fig = px.line(
        combined_trend,
        x="Year",
        y="Records",
        color="Country",
        markers=True,
        title="Accident Records Over Time",
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=450,
        margin=dict(l=20, r=20, t=60, b=20),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

else:

    st.info(
        "Year/date information is not available for trend comparison."
    )


# ============================================================
# WEATHER COMPARISON
# ============================================================

st.subheader("🌦️ Common Weather Conditions")

MORTH_WEATHER_FILE = PROJECT_ROOT / "dataset" / "Indian Datasets" / "India_Weather_Accidents_MoRTH.csv"

weather_india = weather_top(india_df)
weather_usa = weather_top(usa_df)

weather_left, weather_right = st.columns(2)

with weather_left:

    st.markdown("### 🇮🇳 India — National Weather Distribution (MoRTH Benchmark)")

    if not weather_india.empty:

        st.dataframe(
            weather_india,
            use_container_width=True,
            hide_index=True,
        )

    elif MORTH_WEATHER_FILE.exists():

        morth_weather = pd.read_csv(MORTH_WEATHER_FILE)

        display_df = morth_weather[[
            "Weather_Condition",
            "Accidents",
            "Percentage_Share"
        ]].rename(
            columns={
                "Weather_Condition": "Weather Condition",
                "Percentage_Share": "Share (%)"
            }
        )

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "📌 **Official Source:** Ministry of Road Transport and Highways (MoRTH), Government of India — "
            "*Road Accidents in India* (Table 3.8: Road Accidents Classified by Weather Condition).  \n"
            "*(Note: This represents the official All-India national benchmark. "
            "Individual state-level open datasets record annual administrative totals rather than micro-meteorology).*"
        )

    else:

        st.info(
            "ℹ️ India weather-level accident data is not available in the current state-level dataset."
        )


with weather_right:

    st.markdown("### 🇺🇸 USA — Incident Telemetry Distribution")

    if weather_usa.empty:

        st.info(
            "Weather information is not available."
        )

    else:

        st.dataframe(
            weather_usa,
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "📌 **Source:** US Accidents Dataset (sensor-recorded incident-level weather telemetry)."
        )


# ============================================================
# KEY INSIGHT
# ============================================================

st.subheader("💡 Key Comparative Insight")

larger_country = (
    "USA"
    if len(usa_df) > len(india_df)
    else "India"
)

ratio_text = "N/A"

if len(india_df) > 0 and len(usa_df) > 0:

    if len(usa_df) >= len(india_df):
        ratio = len(usa_df) / len(india_df)
        ratio_text = f"approximately {ratio:.1f}×"
    else:
        ratio = len(india_df) / len(usa_df)
        ratio_text = f"approximately {ratio:.1f}×"


st.markdown(
    f"""
    <div class="insight-box">
        <b>Dataset scale:</b> The {larger_country} dataset currently
        contains the larger number of connected accident records
        ({ratio_text} relative to the other dataset).
        <br><br>
        <b>Important:</b> dataset size should not by itself be interpreted
        as a direct comparison of road safety between the two countries,
        because the datasets may differ in coverage, time period,
        collection method and feature definitions.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DOWNLOAD COMPARISON
# ============================================================

st.subheader("⬇️ Download Comparative Summary")

download_df = summary.copy()

st.download_button(
    label="📄 Download Comparison CSV",
    data=download_df.to_csv(index=False).encode("utf-8"),
    file_name="SafeRoad_AI_Comparative_Summary.csv",
    mime="text/csv",
    use_container_width=True,
)


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "⚖️ SafeRoad AI — Comparative Analysis | "
    "India vs USA accident-data comparison"
)