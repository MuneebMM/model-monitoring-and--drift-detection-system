"""Streamlit monitoring dashboard for the Bank Marketing drift detection system."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# ── Constants ──────────────────────────────────────────────────────────────────
API_URL = "http://localhost:8000"

PROJECT_ROOT = Path(__file__).parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"
DRIFT_SUMMARY_PATH = REPORTS_DIR / "drift_summary.csv"
REFERENCE_PATH = PROJECT_ROOT / "data" / "reference" / "reference_data.csv"
PRODUCTION_DIR = PROJECT_ROOT / "data" / "production"

NUMERICAL_FEATURES = ["age", "balance", "duration", "campaign"]
CATEGORICAL_OPTIONS = {
    "job": [
        "admin.", "blue-collar", "entrepreneur", "housemaid", "management",
        "retired", "self-employed", "services", "student", "technician",
        "unemployed",
    ],
    "marital": ["divorced", "married", "single"],
    "education": ["primary", "secondary", "tertiary"],
    "default": ["no", "yes"],
    "housing": ["no", "yes"],
    "loan": ["no", "yes"],
    "contact": ["cellular", "telephone"],
    "month": [
        "jan", "feb", "mar", "apr", "may", "jun",
        "jul", "aug", "sep", "oct", "nov", "dec",
    ],
    "poutcome": ["failure", "other", "success"],
}

PAGES = ["Overview", "Drift Analysis", "Batch Explorer", "Live Prediction"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Model Monitoring",
    page_icon="📊",
    layout="wide",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        border-left: 4px solid #4CAF50;
    }
    .metric-card.warning { border-left-color: #FF9800; }
    .metric-card.danger  { border-left-color: #F44336; }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        margin: 4px 0;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .drift-green  { color: #4CAF50; }
    .drift-yellow { color: #FF9800; }
    .drift-red    { color: #F44336; }
</style>
""", unsafe_allow_html=True)


# ── Data loading (cached) ─────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def load_drift_summary() -> Optional[pd.DataFrame]:
    """Load the drift summary CSV from reports/.

    Returns:
        DataFrame with per-batch drift metrics, or None if missing.
    """
    if not DRIFT_SUMMARY_PATH.exists():
        return None
    return pd.read_csv(DRIFT_SUMMARY_PATH)


@st.cache_data(ttl=60)
def load_reference_data() -> Optional[pd.DataFrame]:
    """Load reference (baseline) data for distribution comparison.

    Returns:
        Reference DataFrame or None if missing.
    """
    if not REFERENCE_PATH.exists():
        return None
    return pd.read_csv(REFERENCE_PATH)


@st.cache_data(ttl=60)
def load_batch_data(batch_name: str) -> Optional[pd.DataFrame]:
    """Load a production batch CSV.

    Args:
        batch_name: e.g. "batch_01".

    Returns:
        Batch DataFrame or None if missing.
    """
    path = PRODUCTION_DIR / f"{batch_name}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data(ttl=60)
def load_drift_report_json(batch_name: str) -> Optional[Dict[str, Any]]:
    """Load Evidently drift report JSON for a batch.

    Args:
        batch_name: e.g. "batch_01".

    Returns:
        Parsed JSON dict or None if missing.
    """
    path = REPORTS_DIR / f"{batch_name}_drift_report.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def extract_feature_drift_scores(report: Dict[str, Any]) -> Dict[str, float]:
    """Pull per-feature drift scores from an Evidently JSON report.

    Args:
        report: Parsed Evidently drift report.

    Returns:
        Dict mapping feature name to drift distance score.
    """
    scores: Dict[str, float] = {}
    for metric in report.get("metrics", []):
        config = metric.get("config", {})
        metric_type = config.get("type", "")
        if metric_type != "evidently:metric_v2:ValueDrift":
            continue
        col_name = config.get("column", "")
        value = metric.get("value")
        if col_name and col_name not in ("y", "prediction") and isinstance(value, (int, float)):
            scores[col_name] = float(value)
    return scores


def fetch_health() -> Optional[Dict[str, Any]]:
    """Call the /health API endpoint.

    Returns:
        Health response dict or None on failure.
    """
    try:
        resp = requests.get(f"{API_URL}/health", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException:
        return None


def drift_color(share: float) -> str:
    """Return a CSS class suffix based on drift severity.

    Args:
        share: Fraction of drifted features (0.0 – 1.0).

    Returns:
        "green", "yellow", or "red".
    """
    if share < 0.1:
        return "green"
    if share < 0.4:
        return "yellow"
    return "red"


def metric_card(label: str, value: str, style: str = "") -> str:
    """Render an HTML metric card.

    Args:
        label: Card subtitle.
        value: Main display value.
        style: Additional CSS class ("warning", "danger", or "").

    Returns:
        HTML string.
    """
    cls = f"metric-card {style}".strip()
    return f"""
    <div class="{cls}">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
    </div>
    """


# ── Sidebar ────────────────────────────────────────────────────────────────────

def render_sidebar() -> str:
    """Render sidebar with navigation and model info.

    Returns:
        Selected page name.
    """
    st.sidebar.title("📊 Model Monitoring Dashboard")
    st.sidebar.divider()

    page = st.sidebar.radio("Navigation", PAGES, label_visibility="collapsed")

    st.sidebar.divider()
    st.sidebar.subheader("Model Info")

    health = fetch_health()
    if health:
        st.sidebar.markdown(f"**Model:** RandomForest (baseline)")
        st.sidebar.markdown(f"**Reference rows:** {health['reference_data_rows']:,}")
        st.sidebar.markdown(f"**Latest batch:** {health.get('latest_batch', 'N/A')}")
        if health.get("latest_batch_accuracy") is not None:
            st.sidebar.markdown(f"**Latest accuracy:** {health['latest_batch_accuracy']:.2%}")
        if health.get("last_drift_check"):
            st.sidebar.markdown(f"**Last check:** {health['last_drift_check'][:19]}")
    else:
        st.sidebar.warning("API unreachable — data loaded from files only")

    return page


# ── Page: Overview ─────────────────────────────────────────────────────────────

def page_overview(summary: pd.DataFrame) -> None:
    """Render the Overview page with KPI cards, trend charts, and alerts.

    Args:
        summary: Drift summary DataFrame.
    """
    st.header("Overview")

    latest = summary.iloc[-1]

    # Alert banner
    if latest["dataset_drift_detected"]:
        st.error("⚠️ Data Drift Detected — Model Performance May Be Degraded")
    elif latest["drifted_feature_share"] > 0.1:
        st.warning("⚡ Mild feature drift detected — monitor closely")

    # KPI cards
    color = drift_color(latest["drifted_feature_share"])
    drift_label = "Yes" if latest["dataset_drift_detected"] else "No"
    drift_style = "danger" if latest["dataset_drift_detected"] else ("warning" if color == "yellow" else "")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(metric_card("Accuracy (Latest)", f"{latest['accuracy']:.2%}"), unsafe_allow_html=True)
    with c2:
        st.markdown(metric_card("F1 Score (Latest)", f"{latest['f1']:.2%}"), unsafe_allow_html=True)
    with c3:
        st.markdown(metric_card("Dataset Drift", f"<span class='drift-{color}'>{drift_label}</span>", drift_style), unsafe_allow_html=True)
    with c4:
        st.markdown(metric_card("Features Drifted", f"{latest['drifted_feature_share']:.1%}", drift_style), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Performance trend
    col_left, col_right = st.columns(2)

    with col_left:
        fig_perf = go.Figure()
        fig_perf.add_trace(go.Scatter(
            x=summary["batch_name"], y=summary["accuracy"],
            mode="lines+markers", name="Accuracy",
            line=dict(color="#2196F3", width=2),
        ))
        fig_perf.add_trace(go.Scatter(
            x=summary["batch_name"], y=summary["f1"],
            mode="lines+markers", name="F1 Score",
            line=dict(color="#FF9800", width=2),
        ))
        fig_perf.update_layout(
            title="Model Performance Across Batches",
            xaxis_title="Batch", yaxis_title="Score",
            yaxis=dict(range=[0, 1]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            height=400, template="plotly_white",
        )
        st.plotly_chart(fig_perf, use_container_width=True)

    with col_right:
        colors = [
            "#4CAF50" if s < 0.1 else "#FF9800" if s < 0.4 else "#F44336"
            for s in summary["drifted_feature_share"]
        ]
        fig_drift = go.Figure(go.Bar(
            x=summary["batch_name"],
            y=summary["drifted_feature_share"],
            marker_color=colors,
            text=[f"{v:.1%}" for v in summary["drifted_feature_share"]],
            textposition="outside",
        ))
        fig_drift.update_layout(
            title="Share of Drifted Features Per Batch",
            xaxis_title="Batch", yaxis_title="Drifted Share",
            yaxis=dict(range=[0, 1]),
            height=400, template="plotly_white",
        )
        st.plotly_chart(fig_drift, use_container_width=True)


# ── Page: Drift Analysis ──────────────────────────────────────────────────────

def page_drift_analysis(summary: pd.DataFrame) -> None:
    """Render the Drift Analysis page with heatmap and summary table.

    Args:
        summary: Drift summary DataFrame.
    """
    st.header("Drift Analysis")

    # Build heatmap data from per-batch JSON reports
    all_scores: Dict[str, Dict[str, float]] = {}
    for batch_name in summary["batch_name"]:
        report = load_drift_report_json(batch_name)
        if report:
            all_scores[batch_name] = extract_feature_drift_scores(report)

    if all_scores:
        features = sorted(
            {f for scores in all_scores.values() for f in scores},
            key=lambda f: max(scores.get(f, 0) for scores in all_scores.values()),
            reverse=True,
        )
        batches = list(all_scores.keys())

        z = [[all_scores[b].get(f, 0) for b in batches] for f in features]

        fig_heat = go.Figure(go.Heatmap(
            z=z, x=batches, y=features,
            colorscale="RdYlGn_r",
            text=[[f"{v:.3f}" for v in row] for row in z],
            texttemplate="%{text}",
            colorbar=dict(title="Drift Score"),
        ))
        fig_heat.update_layout(
            title="Feature Drift Scores Across Batches",
            xaxis_title="Batch", yaxis_title="Feature",
            height=500, template="plotly_white",
        )
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.warning("No drift report JSON files found in reports/")

    # Filters and summary table
    st.subheader("Drift Summary Table")

    drift_types = ["All"] + sorted(summary["drift_type"].unique().tolist())
    selected_type = st.selectbox("Filter by drift type", drift_types)

    display_df = summary.copy()
    if selected_type != "All":
        display_df = display_df[display_df["drift_type"] == selected_type]

    st.dataframe(
        display_df.style.format({
            "drifted_feature_share": "{:.2%}",
            "target_drift_p_value": "{:.6f}",
            "accuracy": "{:.4f}",
            "precision": "{:.4f}",
            "recall": "{:.4f}",
            "f1": "{:.4f}",
        }),
        use_container_width=True,
        hide_index=True,
    )


# ── Page: Batch Explorer ──────────────────────────────────────────────────────

def page_batch_explorer(summary: pd.DataFrame) -> None:
    """Render the Batch Explorer page with per-batch deep dive.

    Args:
        summary: Drift summary DataFrame.
    """
    st.header("Batch Explorer")

    batch_names = summary["batch_name"].tolist()
    selected = st.selectbox("Select batch", batch_names)

    row = summary[summary["batch_name"] == selected].iloc[0]

    # Metrics row
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Drift Type", row["drift_type"])
    c2.metric("Accuracy", f"{row['accuracy']:.4f}")
    c3.metric("F1", f"{row['f1']:.4f}")

    color = drift_color(row["drifted_feature_share"])
    c4.metric("Drifted Share", f"{row['drifted_feature_share']:.1%}")
    c5.metric("Target Drift", "Yes" if row["target_drift_detected"] else "No")

    # Drifted features list
    report = load_drift_report_json(selected)
    if report:
        scores = extract_feature_drift_scores(report)
        drifted = {f: s for f, s in scores.items() if s > 0.1}
        if drifted:
            st.subheader("Drifted Features (score > 0.1)")
            drift_df = pd.DataFrame(
                [{"Feature": f, "Drift Score": s} for f, s in sorted(drifted.items(), key=lambda x: -x[1])]
            )
            st.dataframe(drift_df, use_container_width=True, hide_index=True)
        else:
            st.success("No features exceeded the drift threshold for this batch.")

    # Evidently report link
    html_path = REPORTS_DIR / f"{selected}_drift_report.html"
    if html_path.exists():
        with st.expander("View Full Evidently Drift Report"):
            report_html = html_path.read_text(encoding="utf-8")
            st.components.v1.html(report_html, height=800, scrolling=True)

    # Distribution comparison
    st.subheader("Feature Distributions: Reference vs Batch")

    reference = load_reference_data()
    batch_data = load_batch_data(selected)

    if reference is not None and batch_data is not None:
        cols = st.columns(len(NUMERICAL_FEATURES))
        for i, feat in enumerate(NUMERICAL_FEATURES):
            with cols[i]:
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=reference[feat], name="Reference",
                    opacity=0.6, marker_color="#2196F3",
                    histnorm="probability density",
                ))
                fig.add_trace(go.Histogram(
                    x=batch_data[feat], name=selected,
                    opacity=0.6, marker_color="#F44336",
                    histnorm="probability density",
                ))
                fig.update_layout(
                    title=feat,
                    barmode="overlay",
                    height=300,
                    showlegend=(i == 0),
                    template="plotly_white",
                    margin=dict(l=40, r=20, t=40, b=40),
                )
                st.plotly_chart(fig, use_container_width=True)


# ── Page: Live Prediction ─────────────────────────────────────────────────────

def page_live_prediction() -> None:
    """Render the Live Prediction page with input form and gauge chart."""
    st.header("Live Prediction")
    st.markdown("Enter customer details to get a real-time prediction from the deployed model.")

    col_form, col_result = st.columns([2, 1])

    with col_form:
        with st.form("prediction_form"):
            r1c1, r1c2, r1c3, r1c4 = st.columns(4)
            age = r1c1.number_input("Age", min_value=18, max_value=120, value=35)
            job = r1c2.selectbox("Job", CATEGORICAL_OPTIONS["job"])
            marital = r1c3.selectbox("Marital Status", CATEGORICAL_OPTIONS["marital"])
            education = r1c4.selectbox("Education", CATEGORICAL_OPTIONS["education"])

            r2c1, r2c2, r2c3, r2c4 = st.columns(4)
            balance = r2c1.number_input("Balance (€)", min_value=-10000, max_value=200000, value=1500)
            housing = r2c2.selectbox("Housing Loan", CATEGORICAL_OPTIONS["housing"])
            loan = r2c3.selectbox("Personal Loan", CATEGORICAL_OPTIONS["loan"])
            default = r2c4.selectbox("Credit Default", CATEGORICAL_OPTIONS["default"])

            r3c1, r3c2, r3c3, r3c4 = st.columns(4)
            contact = r3c1.selectbox("Contact Type", CATEGORICAL_OPTIONS["contact"])
            day_of_week = r3c2.number_input("Day of Month", min_value=1, max_value=31, value=15)
            month = r3c3.selectbox("Month", CATEGORICAL_OPTIONS["month"])
            duration = r3c4.number_input("Call Duration (sec)", min_value=0, max_value=5000, value=300)

            r4c1, r4c2, r4c3, _ = st.columns(4)
            campaign = r4c1.number_input("Campaign Contacts", min_value=1, max_value=100, value=2)
            pdays = r4c2.number_input("Days Since Prev Contact", min_value=-1, max_value=999, value=-1)
            previous = r4c3.number_input("Previous Contacts", min_value=0, max_value=100, value=0)
            poutcome = st.selectbox("Previous Outcome", CATEGORICAL_OPTIONS["poutcome"])

            submitted = st.form_submit_button("Predict", use_container_width=True)

    with col_result:
        if submitted:
            payload = {
                "age": age, "job": job, "marital": marital, "education": education,
                "default": default, "balance": balance, "housing": housing, "loan": loan,
                "contact": contact, "day_of_week": day_of_week, "month": month,
                "duration": duration, "campaign": campaign, "pdays": pdays,
                "previous": previous, "poutcome": poutcome,
            }

            try:
                resp = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
                resp.raise_for_status()
                result = resp.json()

                prediction = result["prediction"]
                probability = result["probability"]

                # Result display
                if prediction == "yes":
                    st.success(f"**Prediction: YES** — Will subscribe")
                else:
                    st.info(f"**Prediction: NO** — Will not subscribe")

                # Gauge chart
                gauge_color = "#4CAF50" if prediction == "yes" else "#F44336"
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=probability * 100,
                    number={"suffix": "%"},
                    title={"text": "Subscription Probability"},
                    gauge=dict(
                        axis=dict(range=[0, 100]),
                        bar=dict(color=gauge_color),
                        steps=[
                            dict(range=[0, 30], color="#FFEBEE"),
                            dict(range=[30, 70], color="#FFF3E0"),
                            dict(range=[70, 100], color="#E8F5E9"),
                        ],
                        threshold=dict(
                            line=dict(color="black", width=2),
                            thickness=0.8,
                            value=50,
                        ),
                    ),
                ))
                fig_gauge.update_layout(height=300, margin=dict(t=60, b=20))
                st.plotly_chart(fig_gauge, use_container_width=True)

            except requests.ConnectionError:
                st.error("Cannot connect to API. Make sure the FastAPI server is running on port 8000.")
            except requests.HTTPError as exc:
                st.error(f"API error: {exc.response.text}")
        else:
            st.markdown(
                "<div style='text-align:center; color:#999; padding-top:80px;'>"
                "Fill out the form and click <b>Predict</b>"
                "</div>",
                unsafe_allow_html=True,
            )


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    """Entry point — render sidebar and dispatch to selected page."""
    page = render_sidebar()

    summary = load_drift_summary()

    if page == "Live Prediction":
        page_live_prediction()
        return

    if summary is None or summary.empty:
        st.error("No drift summary data found. Run `python src/monitor.py` first.")
        return

    if page == "Overview":
        page_overview(summary)
    elif page == "Drift Analysis":
        page_drift_analysis(summary)
    elif page == "Batch Explorer":
        page_batch_explorer(summary)


if __name__ == "__main__":
    main()
