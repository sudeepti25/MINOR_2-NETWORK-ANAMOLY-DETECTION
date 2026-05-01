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
    page_icon="shield",
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

    .hero h1 {
        color: var(--ink);
        margin: 0;
        font-size: 2rem;
        letter-spacing: 0.02em;
    }

    .hero p {
        margin: 0.45rem 0 0 0;
        color: var(--muted);
        font-size: 1rem;
    }

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

    .mono {
        font-family: 'IBM Plex Mono', monospace;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

project_root = Path(__file__).resolve().parents[1]
default_data_path = project_root / "PREPROCESSING" / "processed_cicids2017.csv"


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
        raise ValueError("Class mapping must be a JSON object, for example: {\"1\": \"DDoS\"}.")
    normalized: Dict[int, str] = {}
    for k, v in parsed.items():
        normalized[int(k)] = str(v)
    return normalized


st.markdown(
    """
    <div class="hero">
      <h1>Two-Stage Network Threat Inference</h1>
      <p>Stage 1 detects anomaly traffic from encoded autoencoder features. Stage 2 classifies detected anomalies into attack classes.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Run Settings")

    data_mode = st.radio(
        "Input Data",
        options=["Use default preprocessed_cicids2017.csv", "Upload CSV"],
        index=0,
    )

    uploaded_file = None
    if data_mode == "Upload CSV":
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    row_cap = st.slider("Rows to process", min_value=1000, max_value=300000, value=50000, step=1000)

    stage1_model_name = st.selectbox("Stage-1 model (anomaly detection)", ["xgboost", "randomforest"], index=0)
    stage2_model_name = st.selectbox(
        "Stage-2 model (attack type classification)", ["xgboost_stage2", "randomforest_stage2"], index=0
    )

    class_map_text = st.text_area(
        "Optional class mapping JSON",
        value='{"1": "DDoS", "2": "PortScan", "3": "WebAttack", "4": "Infiltration"}',
        height=130,
        help="Use only if your stage-2 labels are numeric IDs and you want readable names.",
    )

    run_clicked = st.button("Run Inference", type="primary", use_container_width=True)

st.markdown(
    """
    <div class="caption-box">
      Default test dataset is <span class="mono">PREPROCESSING/processed_cicids2017.csv</span>.
      It is binary-labeled (benign vs anomaly), so stage-1 metrics are shown directly when <span class="mono">Label</span> exists.
    </div>
    """,
    unsafe_allow_html=True,
)

if run_clicked:
    with st.spinner("Loading models and running two-stage pipeline..."):
        pipeline = get_pipeline()

        if data_mode == "Upload CSV":
            if uploaded_file is None:
                st.error("Please upload a CSV file first.")
                st.stop()
            data_df = pd.read_csv(uploaded_file)
            source_name = uploaded_file.name
        else:
            data_df = load_csv(default_data_path)
            source_name = str(default_data_path.relative_to(project_root))

        if len(data_df) > row_cap:
            data_df = data_df.head(row_cap).copy()

        try:
            class_mapping = parse_class_mapping(class_map_text)
        except Exception as exc:
            st.error(f"Invalid class mapping JSON: {exc}")
            st.stop()

        result = pipeline.run(
            input_df=data_df,
            stage1_model_name=stage1_model_name,
            stage2_model_name=stage2_model_name,
            user_class_map=class_mapping,
        )

    if result.warnings:
        for warning in result.warnings:
            st.warning(warning)

    st.subheader("Inference Summary")
    c1, c2, c3, c4 = st.columns(4)

    total_records = int(result.metrics.get("total_records", 0))
    anomalies_detected = int(result.metrics.get("anomalies_detected", 0))
    anomaly_rate = float(result.metrics.get("anomaly_rate", 0.0))

    c1.metric("Rows Processed", f"{total_records:,}")
    c2.metric("Anomalies Detected", f"{anomalies_detected:,}")
    c3.metric("Anomaly Rate", f"{anomaly_rate * 100:.2f}%")
    c4.metric("Data Source", source_name)

    if "stage1_accuracy" in result.metrics:
        st.subheader("Stage-1 Evaluation (from Label column)")
        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Accuracy", f"{result.metrics['stage1_accuracy']:.4f}")
        e2.metric("Precision", f"{result.metrics['stage1_precision']:.4f}")
        e3.metric("Recall", f"{result.metrics['stage1_recall']:.4f}")
        e4.metric("F1", f"{result.metrics['stage1_f1']:.4f}")

    st.subheader("Prediction Distributions")
    chart_col_1, chart_col_2 = st.columns(2)

    split_counts = result.predictions["stage1_is_anomaly"].value_counts().rename(index={0: "Benign", 1: "Anomaly"})
    fig_stage1 = px.pie(
        names=split_counts.index,
        values=split_counts.values,
        title="Stage-1 Traffic Split",
        color=split_counts.index,
        color_discrete_map={"Benign": "#14b8a6", "Anomaly": "#f97316"},
        hole=0.45,
    )
    chart_col_1.plotly_chart(fig_stage1, use_container_width=True)

    attack_counts = result.predictions.loc[result.predictions["stage1_is_anomaly"] == 1, "stage2_attack_class"].value_counts()
    if not attack_counts.empty:
        fig_stage2 = px.bar(
            x=attack_counts.index,
            y=attack_counts.values,
            title="Stage-2 Attack Type Distribution (Detected Anomalies)",
            labels={"x": "Attack Class", "y": "Count"},
            color=attack_counts.values,
            color_continuous_scale="Tealgrn",
        )
        chart_col_2.plotly_chart(fig_stage2, use_container_width=True)
    else:
        chart_col_2.info("No anomalies were detected in this run, so stage-2 chart is empty.")

    st.subheader("Predictions Table")
    preferred_cols = [
        "Label",
        "stage1_anomaly_score",
        "stage1_is_anomaly",
        "stage2_attack_class_id",
        "stage2_attack_class",
        "final_prediction",
    ]
    display_cols = [col for col in preferred_cols if col in result.predictions.columns]
    st.dataframe(result.predictions[display_cols], use_container_width=True, height=430)

    csv_bytes = result.predictions.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download full prediction CSV",
        data=csv_bytes,
        file_name="two_stage_predictions.csv",
        mime="text/csv",
    )

else:
    st.info("Choose settings from the sidebar and click Run Inference.")
