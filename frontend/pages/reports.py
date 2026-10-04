import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
from utils.data_loader import load_india, load_usa, find_col

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MORTH_WEATHER_FILE = PROJECT_ROOT / "dataset" / "Indian Datasets" / "India_Weather_Accidents_MoRTH.csv"

# ============================================================
# SAFE ROAD AI - REPORTS
# Professional dataset reporting and summary page
# ============================================================

st.set_page_config(
    page_title="SafeRoad AI Reports",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
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
                rgba(29, 78, 121, 0.25),
                transparent 30%
            ),
            linear-gradient(
                135deg,
                #06111f 0%,
                #0a1d32 50%,
                #102d4b 100%
            );
        color: #eef6ff;
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
        color: #cfdeed !important;
    }

    [data-testid="stMetric"] {
        background: rgba(12, 32, 54, 0.94);
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

    .report-card {
        background: rgba(12, 34, 57, 0.92);
        border: 1px solid rgba(96, 176, 236, 0.17);
        border-radius: 16px;
        padding: 20px;
        min-height: 135px;
    }

    .report-card-title {
        color: #ffffff;
        font-size: 17px;
        font-weight: 750;
        margin-bottom: 7px;
    }

    .report-card-text {
        color: #bfd1e2;
        font-size: 14px;
        line-height: 1.55;
    }

    .report-note {
        background: rgba(17, 61, 73, 0.70);
        border-left: 4px solid #42d6aa;
        border-radius: 11px;
        padding: 15px 18px;
        color: #d6f3ea;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.title("📄 SafeRoad AI Reports")

st.caption(
    "Generate a professional summary of the currently connected accident dataset."
)


# ============================================================
# DATASET SELECTION
# ============================================================

country = st.selectbox(
    "Select Report Dataset",
    ["India", "USA"],
)


df = load_india() if country == "India" else load_usa()


# ============================================================
# DATASET VALIDATION
# ============================================================

if df.empty:

    st.error(
        f"{country} dataset could not be loaded."
    )

    st.info(
        "Please verify that the dataset path and connection are correct "
        "before generating the report."
    )

    st.stop()


# Make a working copy
df = df.copy()

# Clean column names
df.columns = [
    str(column).strip()
    for column in df.columns
]


# ============================================================
# COLUMN DETECTION
# ============================================================

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

weather_col = find_col(
    df,
    [
        "Weather_Condition",
        "Weather",
        "weather",
    ],
)

date_col = find_col(
    df,
    [
        "Start_Time",
        "Date",
        "date",
        "Year",
    ],
)

accidents_col = find_col(
    df,
    [
        "Accidents",
        "Accident_Count",
        "Total_Accidents",
    ],
)

deaths_col = find_col(
    df,
    [
        "Deaths",
        "Death",
        "Total_Deaths",
    ],
)

injuries_col = find_col(
    df,
    [
        "Injuries",
        "Injury",
        "Total_Injuries",
    ],
)


# ============================================================
# REPORT OVERVIEW METRICS
# ============================================================

st.subheader("📊 Report Overview")

c1, c2, c3, c4 = st.columns(4)


with c1:

    st.metric(
        "Total Records",
        f"{len(df):,}",
    )


with c2:

    st.metric(
        "Dataset Columns",
        f"{len(df.columns):,}",
    )


with c3:

    if state_col:
        clean_state_df = df.copy()
        clean_state_df = clean_state_df[~clean_state_df[state_col].astype(str).str.contains(r"Total|All India", case=False, na=False)]

        state_count = (
            clean_state_df[state_col]
            .dropna()
            .astype(str)
            .str.strip()
            .nunique()
        )

        st.metric(
            "States / Regions",
            f"{state_count:,}",
        )

    else:

        st.metric(
            "States / Regions",
            "N/A",
        )


with c4:

    if country == "India":
        clean_sev_df = df.copy()
        if state_col:
            clean_sev_df = clean_sev_df[~clean_sev_df[state_col].astype(str).str.contains(r"Total|All India", case=False, na=False)]

        if "Fatality_Rate" in clean_sev_df.columns:
            avg_sev = clean_sev_df["Fatality_Rate"].dropna().mean()
            st.metric(
                "Avg Severity (Fatality Rate)",
                f"{avg_sev:.1f}%",
            )
        elif deaths_col and accidents_col:
            tot_d = pd.to_numeric(clean_sev_df[deaths_col], errors="coerce").fillna(0).sum()
            tot_a = pd.to_numeric(clean_sev_df[accidents_col], errors="coerce").fillna(0).sum()
            avg_sev = (tot_d / tot_a * 100) if tot_a > 0 else 0
            st.metric(
                "Avg Severity (Fatality Rate)",
                f"{avg_sev:.1f}%",
            )
        else:
            st.metric("Average Severity", "N/A")

    elif severity_col:

        severity_values = pd.to_numeric(
            df[severity_col],
            errors="coerce",
        ).dropna()

        if not severity_values.empty:

            st.metric(
                "Average Severity",
                f"{severity_values.mean():.2f}",
            )

        else:

            st.metric(
                "Average Severity",
                "N/A",
            )

    else:

        st.metric(
            "Average Severity",
            "N/A",
        )


# ============================================================
# DATASET CONNECTION STATUS
# ============================================================

st.subheader("🔗 Dataset Connection")

st.success(
    f"{country} dataset connected successfully."
)


st.markdown(
    f"""
    **Connected dataset:** `{country}`  
    **Rows available:** `{len(df):,}`  
    **Columns available:** `{len(df.columns):,}`
    """
)


# ============================================================
# KEY FINDINGS
# ============================================================

st.subheader("📌 Key Findings")

finding_col1, finding_col2 = st.columns(2)


# ------------------------------------------------------------
# TOP STATES
# ------------------------------------------------------------

with finding_col1:

    st.markdown("### 🏆 Top States / Regions")

    if state_col:
        clean_state_findings = df.copy()
        clean_state_findings = clean_state_findings[~clean_state_findings[state_col].astype(str).str.contains(r"Total|All India", case=False, na=False)]

        if country == "India" and accidents_col:
            state_table = (
                clean_state_findings.groupby(state_col, as_index=False)[accidents_col]
                .sum()
                .rename(columns={state_col: "State / Region", accidents_col: "Total Accidents"})
                .sort_values("Total Accidents", ascending=False)
                .head(10)
            )

            st.dataframe(
                state_table,
                use_container_width=True,
                hide_index=True,
            )

        else:

            state_counts = (
                clean_state_findings[state_col]
                .dropna()
                .astype(str)
                .str.strip()
                .replace("", pd.NA)
                .dropna()
                .value_counts()
                .head(10)
            )

            state_table = (
                state_counts
                .rename("Records")
                .reset_index()
            )

            state_table.columns = [
                "State / Region",
                "Records",
            ]

            st.dataframe(
                state_table,
                use_container_width=True,
                hide_index=True,
            )

    else:

        st.info(
            "State/region information is not available in this dataset."
        )


# ------------------------------------------------------------
# WEATHER
# ------------------------------------------------------------

with finding_col2:

    st.markdown("### 🌦️ Most Common Weather Conditions")

    if country == "India" and MORTH_WEATHER_FILE.exists():
        try:
            morth_w = pd.read_csv(MORTH_WEATHER_FILE)
            weather_table = morth_w[["Weather_Condition", "Accidents", "Percentage_Share"]].rename(
                columns={
                    "Weather_Condition": "Weather Condition",
                    "Percentage_Share": "Share (%)"
                }
            )
            st.dataframe(
                weather_table,
                use_container_width=True,
                hide_index=True,
            )
            st.caption("📌 **Official MoRTH Benchmark:** Table 3.8 — Road Accidents Classified by Weather Condition (National Aggregate).")
        except Exception:
            st.info("Weather information could not be loaded.")

    elif weather_col:

        weather_counts = (
            df[weather_col]
            .dropna()
            .astype(str)
            .str.strip()
            .replace("", pd.NA)
            .dropna()
            .value_counts()
            .head(10)
        )

        weather_table = (
            weather_counts
            .rename("Records")
            .reset_index()
        )

        weather_table.columns = [
            "Weather Condition",
            "Records",
        ]

        st.dataframe(
            weather_table,
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.info(
            "Weather information is not available in this dataset."
        )


# ============================================================
# ACCIDENT SEVERITY ANALYSIS
# ============================================================

st.subheader("⚠️ Accident Severity Analysis")

if country == "India" and ("Fatality_Rate" in df.columns or (deaths_col and accidents_col)):
    clean_sev_chart = df.copy()
    if state_col:
        clean_sev_chart = clean_sev_chart[~clean_sev_chart[state_col].astype(str).str.contains(r"Total|All India", case=False, na=False)]

    if "Fatality_Rate" in clean_sev_chart.columns:
        sev_metric = clean_sev_chart.groupby("State")["Fatality_Rate"].mean()
    else:
        sev_metric = (clean_sev_chart.groupby("State")[deaths_col].sum() / clean_sev_chart.groupby("State")[accidents_col].sum()) * 100

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

    chart = px.bar(
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
        title="India Accident Severity Index Distribution (Persons Killed per 100 Accidents — MoRTH Standard)",
    )

    chart.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=400,
        margin=dict(l=20, r=20, t=55, b=20),
        showlegend=False,
    )

    st.plotly_chart(
        chart,
        use_container_width=True,
    )

    st.caption(
        "📌 **MoRTH Definition:** Accident Severity is measured as persons killed per 100 accidents. "
        "States in northern and eastern corridors (such as Punjab, Bihar, and Jharkhand) record critical fatality rates exceeding 50%."
    )

elif severity_col:

    severity_series = pd.to_numeric(
        df[severity_col],
        errors="coerce",
    ).dropna()

    if not severity_series.empty:

        severity_counts = (
            severity_series
            .value_counts()
            .sort_index()
            .rename("Records")
            .reset_index()
        )

        severity_counts.columns = [
            "Severity",
            "Records",
        ]

        chart = px.bar(
            severity_counts,
            x="Severity",
            y="Records",
            text="Records",
            title=f"{country} Accident Severity Distribution",
        )

        chart.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=400,
            margin=dict(l=20, r=20, t=55, b=20),
        )

        st.plotly_chart(
            chart,
            use_container_width=True,
        )

    else:

        st.info(
            "Severity values could not be interpreted as numeric data."
        )

else:

    st.info(
        "Severity information is not available in this dataset."
    )


# ============================================================
# TREND ANALYSIS
# ============================================================

st.subheader("📈 Accident Trend")

if date_col and date_col != "Year":

    trend_df = df[[date_col]].copy()

    trend_df[date_col] = pd.to_datetime(
        trend_df[date_col],
        errors="coerce",
    )

    trend_df = trend_df.dropna(
        subset=[date_col]
    )

    if not trend_df.empty:

        trend_df["Year"] = (
            trend_df[date_col]
            .dt.year
        )

        yearly = (
            trend_df
            .groupby("Year")
            .size()
            .reset_index(name="Records")
        )

        yearly = yearly.sort_values("Year")

        chart = px.line(
            yearly,
            x="Year",
            y="Records",
            markers=True,
            title=f"{country} Accident Records Over Time",
        )

        chart.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=400,
            margin=dict(l=20, r=20, t=55, b=20),
        )

        st.plotly_chart(
            chart,
            use_container_width=True,
        )

    else:

        st.info(
            "Date information is not available for trend analysis."
        )

elif date_col == "Year":

    if country == "India" and accidents_col:
        clean_trend = df.copy()
        if state_col:
            clean_trend = clean_trend[~clean_trend[state_col].astype(str).str.contains(r"Total|All India", case=False, na=False)]

        yearly = (
            clean_trend.groupby("Year")[accidents_col]
            .sum()
            .reset_index(name="Accidents")
            .sort_values("Year")
        )

        chart = px.line(
            yearly,
            x="Year",
            y="Accidents",
            markers=True,
            title="India Total Road Accidents Over Time (Annual Aggregates)",
        )

        chart.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=400,
        )

        st.plotly_chart(
            chart,
            use_container_width=True,
        )

    else:

        year_values = pd.to_numeric(
            df[date_col],
            errors="coerce",
        ).dropna()

        if not year_values.empty:

            yearly = (
                year_values
                .value_counts()
                .sort_index()
                .rename("Records")
                .reset_index()
            )

            yearly.columns = [
                "Year",
                "Records",
            ]

            chart = px.line(
                yearly,
                x="Year",
                y="Records",
                markers=True,
                title=f"{country} Accident Records Over Time",
            )

            chart.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=400,
            )

            st.plotly_chart(
                chart,
                use_container_width=True,
            )

        else:

            st.info(
                "Year information is not available."
            )

else:

    st.info(
        "Date/year information is not available in this dataset."
    )


# ============================================================
# DIRECT ACCIDENT TOTALS
# ============================================================

if any([accidents_col, deaths_col, injuries_col]):

    st.subheader("📋 Accident Totals")

    total_cols = st.columns(3)


    with total_cols[0]:

        if accidents_col:

            clean_acc_df = df.copy()
            if country == "India" and state_col:
                clean_acc_df = clean_acc_df[~clean_acc_df[state_col].astype(str).str.contains(r"Total|All India", case=False, na=False)]

            value = pd.to_numeric(
                clean_acc_df[accidents_col],
                errors="coerce",
            ).fillna(0).sum()

            st.metric(
                "Total Accidents",
                f"{value:,.0f}",
            )

        else:

            st.metric(
                "Total Accidents",
                "N/A",
            )


    with total_cols[1]:

        if deaths_col:

            clean_dth_df = df.copy()
            if country == "India" and state_col:
                clean_dth_df = clean_dth_df[~clean_dth_df[state_col].astype(str).str.contains(r"Total|All India", case=False, na=False)]

            value = pd.to_numeric(
                clean_dth_df[deaths_col],
                errors="coerce",
            ).fillna(0).sum()

            st.metric(
                "Total Deaths",
                f"{value:,.0f}",
            )

        else:

            st.metric(
                "Total Deaths",
                "N/A",
            )


    with total_cols[2]:

        if injuries_col:

            clean_inj_df = df.copy()
            if country == "India" and state_col:
                clean_inj_df = clean_inj_df[~clean_inj_df[state_col].astype(str).str.contains(r"Total|All India", case=False, na=False)]

            value = pd.to_numeric(
                clean_inj_df[injuries_col],
                errors="coerce",
            ).fillna(0).sum()

            st.metric(
                "Total Injuries",
                f"{value:,.0f}",
            )

        else:

            st.metric(
                "Total Injuries",
                "N/A",
            )


# ============================================================
# DATASET PREVIEW
# ============================================================

st.subheader("🔍 Dataset Preview")

preview_rows = st.slider(
    "Number of rows to preview",
    min_value=5,
    max_value=20,
    value=10,
)

st.dataframe(
    df.head(preview_rows),
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# DOWNLOAD REPORT
# ============================================================

st.subheader("⬇️ Download Report")


avg_severity_text = "N/A"

if country == "India":
    clean_sev_dl = df.copy()
    if state_col:
        clean_sev_dl = clean_sev_dl[~clean_sev_dl[state_col].astype(str).str.contains(r"Total|All India", case=False, na=False)]
    if "Fatality_Rate" in clean_sev_dl.columns:
        avg_sev_dl = clean_sev_dl["Fatality_Rate"].dropna().mean()
        avg_severity_text = f"{avg_sev_dl:.1f}% (MoRTH Fatality Rate)"
    elif deaths_col and accidents_col:
        tot_d = pd.to_numeric(clean_sev_dl[deaths_col], errors="coerce").fillna(0).sum()
        tot_a = pd.to_numeric(clean_sev_dl[accidents_col], errors="coerce").fillna(0).sum()
        avg_sev_dl = (tot_d / tot_a * 100) if tot_a > 0 else 0
        avg_severity_text = f"{avg_sev_dl:.1f}% (MoRTH Fatality Rate)"

elif severity_col:

    severity_values = pd.to_numeric(
        df[severity_col],
        errors="coerce",
    ).dropna()

    if not severity_values.empty:

        avg_severity_text = (
            f"{severity_values.mean():.2f}"
        )


state_summary_text = "N/A"

if state_col:
    clean_st_dl = df.copy()
    clean_st_dl = clean_st_dl[~clean_st_dl[state_col].astype(str).str.contains(r"Total|All India", case=False, na=False)]

    if country == "India" and accidents_col:
        top_states_dl = (
            clean_st_dl.groupby(state_col)[accidents_col]
            .sum()
            .sort_values(ascending=False)
            .head(5)
        )
        state_summary_text = "\n".join(
            f"- {state}: {int(count):,} accidents"
            for state, count in top_states_dl.items()
        )
    else:
        state_summary = (
            clean_st_dl[state_col]
            .dropna()
            .astype(str)
            .str.strip()
            .value_counts()
            .head(5)
        )

        if not state_summary.empty:

            state_summary_text = "\n".join(
                f"- {state}: {count:,} records"
                for state, count in state_summary.items()
            )


weather_summary_text = "N/A"

if country == "India" and MORTH_WEATHER_FILE.exists():
    try:
        morth_w_dl = pd.read_csv(MORTH_WEATHER_FILE)
        weather_summary_text = "\n".join(
            f"- {row['Weather_Condition']}: {int(row['Accidents']):,} ({row['Percentage_Share']}%)"
            for _, row in morth_w_dl.iterrows()
        )
    except Exception:
        pass

elif weather_col:

    weather_summary = (
        df[weather_col]
        .dropna()
        .astype(str)
        .str.strip()
        .value_counts()
        .head(5)
    )

    if not weather_summary.empty:

        weather_summary_text = "\n".join(
            f"- {weather}: {count:,} records"
            for weather, count in weather_summary.items()
        )


report_text = f"""
============================================================
                    SAFEROAD AI REPORT
============================================================

Dataset: {country}

Dataset Overview
----------------
Total records       : {len(df):,}
Total columns       : {len(df.columns):,}
States / Regions    : {state_count if state_col else "N/A"}
Average severity    : {avg_severity_text}

Top States / Regions
--------------------
{state_summary_text}

Most Common Weather Conditions
------------------------------
{weather_summary_text}

Available Columns
-----------------
{", ".join(df.columns.tolist())}

============================================================
Generated by SafeRoad AI
Intelligent Accident Risk Prediction & Hotspot Detection
============================================================
""".strip()


st.download_button(
    label="📄 Download SafeRoad AI Report",
    data=report_text,
    file_name=f"SafeRoad_AI_{country}_Report.txt",
    mime="text/plain",
    use_container_width=True,
)


# ============================================================
# REPORT NOTE
# ============================================================

st.markdown(
    """
    <div class="report-note">
        <b>Report note:</b>
        This report summarizes the currently connected dataset.
        The reported values are generated from the dataset available
        to SafeRoad AI and are not manually entered placeholder values.
    </div>
    """,
    unsafe_allow_html=True,
)
