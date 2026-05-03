from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import plotly.express as px
import streamlit as st

from pipeline import TwoStageAnomalyPipeline


st.set_page_config(
    page_title="Two-Stage Network Anomaly Detection",
    page_icon="🛡️",
    layout="wide",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=IBM+Plex+Mono:wght@500&display=swap');

    :root {
        --ink: #0d1b2a;
        --muted: #415a77;
        --bg-soft: #f4f8fb;
        --accent: #0f766e;
        --accent-2: #f97316;
    }

    html, body, [class*="css"] {
        font-family: 'Space Grotesk', sans-serif;
    }

    .hero {
        border-radius: 18px;
        padding: 1.25rem 1.4rem;
        margin-bottom: 0.8rem;
        background: linear-gradient(120deg, #e0f2fe 0%, #ecfeff 50%, #fff7ed 100%);
        border: 1px solid rgba(15, 118, 110, 0.25);
    }

    .hero h1 { color: var(--ink); margin: 0; font-size: 2rem; letter-spacing: 0.02em; }
    .hero p  { margin: 0.45rem 0 0 0; color: var(--muted); font-size: 1rem; }

    div[data-testid="metric-container"] {
        background: #ffffff;
        border: 1px solid #dce8ef;
        border-radius: 12px;
        padding: 0.7rem;
    }

    .caption-box {
        background: var(--bg-soft);
        border-left: 4px solid var(--accent);
        padding: 0.65rem 0.75rem;
        border-radius: 8px;
        color: #1e293b;
        margin: 0.4rem 0 1rem 0;
    }

    .mono { font-family: 'IBM Plex Mono', monospace; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
project_root      = Path(__file__).resolve().parents[1]
default_data_path = project_root / "PREPROCESSING" / "processed_cicids2017.csv"


# ── Cached loaders ────────────────────────────────────────────────────────────
@st.cache_resource
def get_pipeline() -> TwoStageAnomalyPipeline:
    return TwoStageAnomalyPipeline(project_root=project_root)


@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def parse_class_mapping(text: str) -> Optional[Dict[int, str]]:
    if not text.strip():
        return None
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("Class mapping must be a JSON object.")
    return {int(k): str(v) for k, v in parsed.items()}


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero">
      <h1>Two-Stage Network Threat Inference</h1>
      <p>Stage 1 detects anomalous traffic using Autoencoder + XGBoost.
         Stage 2 classifies detected anomalies into DDoS, PortScan, BruteForce, or General.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Run Settings")

    data_mode = st.radio(
        "Input Data",
        options=["Use default processed_cicids2017.csv", "Upload CSV"],
        index=0,
    )

    uploaded_file = None
    if data_mode == "Upload CSV":
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    row_cap = st.slider(
        "Rows to process",
        min_value=1_000, max_value=300_000, value=50_000, step=1_000,
    )

    st.selectbox("Stage-1 model", ["xgb_binary"],      index=0)
    st.selectbox("Stage-2 model", ["xgb_multiclass"],  index=0)

    class_map_text = st.text_area(
        "Optional class mapping JSON",
        value='{"1": "DDoS", "2": "PortScan", "3": "BruteForce", "4": "General"}',
        height=120,
        help="Keys 1-4 map to model outputs. Edit only if you want custom label names.",
    )

    run_clicked = st.button("Run Inference", type="primary", use_container_width=True)

# ── Caption ───────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="caption-box">
      Default dataset: <span class="mono">PREPROCESSING/processed_cicids2017.csv</span> —
      binary-labeled (0 = benign, 1 = anomaly). Stage-1 metrics are shown when a
      <span class="mono">Label</span> column is present.
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Run ───────────────────────────────────────────────────────────────────────
if run_clicked:
    with st.spinner("Loading models and running pipeline..."):

        pipeline = get_pipeline()

        # Load data
        if data_mode == "Upload CSV":
            if uploaded_file is None:
                st.error("Please upload a CSV file first.")
                st.stop()
            data_df     = pd.read_csv(uploaded_file)
            source_name = uploaded_file.name
        else:
            data_df     = load_csv(default_data_path)
            source_name = "PREPROCESSING/processed_cicids2017.csv"

        if len(data_df) > row_cap:
            data_df = data_df.head(row_cap).copy()

        # Parse class mapping
        try:
            class_mapping = parse_class_mapping(class_map_text)
        except Exception as exc:
            st.error(f"Invalid class mapping JSON: {exc}")
            st.stop()

        # Run pipeline
        result = pipeline.run(
            input_df=data_df,
            stage1_model_name="xgb_binary",
            stage2_model_name="xgb_multiclass",
            user_class_map=class_mapping,
        )

    # ── Warnings ──────────────────────────────────────────────────────────────
    for w in result.warnings:
        st.warning(w)

    # ── Summary metrics ───────────────────────────────────────────────────────
    st.subheader("Inference Summary")
    c1, c2, c3, c4, c5 = st.columns(5)

    total      = int(result.metrics.get("total_records", 0))
    anomalies  = int(result.metrics.get("anomalies_detected", 0))
    rate       = float(result.metrics.get("anomaly_rate", 0.0))
    elapsed    = float(result.metrics.get("elapsed_seconds", 0.0))

    c1.metric("Rows Processed",    f"{total:,}")
    c2.metric("Anomalies Detected", f"{anomalies:,}")
    c3.metric("Anomaly Rate",       f"{rate * 100:.2f}%")
    c4.metric("Time (s)",           f"{elapsed:.2f}s")
    c5.metric("Source",             source_name[:30])

    # ── Stage-1 evaluation (only if Label column was present) ─────────────────
    if "stage1_accuracy" in result.metrics:
        st.subheader("Stage-1 Evaluation")
        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Accuracy",  f"{result.metrics['stage1_accuracy']:.4f}")
        e2.metric("Precision", f"{result.metrics['stage1_precision']:.4f}")
        e3.metric("Recall",    f"{result.metrics['stage1_recall']:.4f}")
        e4.metric("F1 Score",  f"{result.metrics['stage1_f1']:.4f}")

    # ── Charts ────────────────────────────────────────────────────────────────
    st.subheader("Prediction Distributions")
    chart_col1, chart_col2 = st.columns(2)

    # Stage 1 — donut
    split_counts = (
        result.predictions["stage1_is_anomaly"]
        .value_counts()
        .rename(index={0: "Benign", 1: "Anomaly"})
    )
    fig1 = px.pie(
        names=split_counts.index,
        values=split_counts.values,
        title="Stage-1 Traffic Split",
        color=split_counts.index,
        color_discrete_map={"Benign": "#14b8a6", "Anomaly": "#f97316"},
        hole=0.45,
    )
    chart_col1.plotly_chart(fig1, use_container_width=True)

    # Stage 2 — bar chart (anomalies only)
    attack_counts = result.predictions.loc[
        result.predictions["stage1_is_anomaly"] == 1, "stage2_attack_class"
    ].value_counts()

    if not attack_counts.empty:
        fig2 = px.bar(
            x=attack_counts.index,
            y=attack_counts.values,
            title="Stage-2 Attack Type Distribution",
            labels={"x": "Attack Class", "y": "Count"},
            color=attack_counts.index,
            color_discrete_map={
                "DDoS"       : "#ef4444",
                "PortScan"   : "#f97316",
                "BruteForce" : "#a855f7",
                "General"    : "#64748b",
            },
        )
        fig2.update_layout(showlegend=False)
        chart_col2.plotly_chart(fig2, use_container_width=True)
    else:
        chart_col2.info("No anomalies detected — stage-2 chart is empty.")

    # ── Final label distribution bar ──────────────────────────────────────────
    st.subheader("Final Label Distribution")
    final_counts = result.predictions["final_prediction"].value_counts().reset_index()
    final_counts.columns = ["Label", "Count"]
    fig3 = px.bar(
        final_counts, x="Label", y="Count",
        title="All Predictions (Benign + Attack Types)",
        color="Label",
        color_discrete_map={
            "Benign"     : "#14b8a6",
            "DDoS"       : "#ef4444",
            "PortScan"   : "#f97316",
            "BruteForce" : "#a855f7",
            "General"    : "#64748b",
        },
    )
    fig3.update_layout(showlegend=False)
    st.plotly_chart(fig3, use_container_width=True)

    # ── Predictions table ─────────────────────────────────────────────────────
    st.subheader("Predictions Table")
    preferred_cols = [
        "Label",
        "stage1_anomaly_score",
        "stage1_is_anomaly",
        "stage2_attack_class_id",
        "stage2_attack_class",
        "final_prediction",
    ]
    display_cols = [c for c in preferred_cols if c in result.predictions.columns]
    st.dataframe(result.predictions[display_cols], use_container_width=True, height=430)

    # ── Download ──────────────────────────────────────────────────────────────
    csv_bytes = result.predictions.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇ Download full prediction CSV",
        data=csv_bytes,
        file_name="two_stage_predictions.csv",
        mime="text/csv",
    )

else:
    st.info("Choose settings from the sidebar and click **Run Inference**.")
