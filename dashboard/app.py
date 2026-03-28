"""
LLMEval Dashboard — Streamlit leaderboard & analysis UI.

Run with:
    streamlit run dashboard/app.py
"""

import glob
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LLMEval Dashboard",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #e0e0e0;
        text-align: center;
    }
    .winner-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
    }
    .stMetric label { font-size: 13px !important; }
</style>
""", unsafe_allow_html=True)


# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_results(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def find_result_files(results_dir: str = "results") -> list[str]:
    files = sorted(glob.glob(f"{results_dir}/*.json"), reverse=True)
    return [f for f in files if "comparison" not in f]


def get_score_df(results: list[dict]) -> pd.DataFrame:
    rows = []
    for model_data in results:
        row = {"Model": model_data["model_name"]}
        row.update(model_data.get("aggregate_scores", {}))
        row["Avg latency (ms)"] = round(model_data.get("avg_latency_ms", 0), 1)
        row["Total cost ($)"] = round(model_data.get("total_cost_usd", 0), 5)
        row["Errors"] = model_data.get("errors", 0)
        rows.append(row)
    return pd.DataFrame(rows)


def get_sample_df(results: list[dict]) -> pd.DataFrame:
    rows = []
    for model_data in results:
        for s in model_data.get("samples", []):
            row = {
                "Model": model_data["model_name"],
                "Sample ID": s["id"],
                "Input": s["input"][:80] + "..." if len(s["input"]) > 80 else s["input"],
                "Reference": s["reference"][:80],
                "Prediction": s["prediction"][:80] if s["prediction"] else "ERROR",
                "Latency (ms)": round(s.get("latency_ms", 0), 1),
                "Error": s.get("error", ""),
            }
            row.update(s.get("scores", {}))
            rows.append(row)
    return pd.DataFrame(rows)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.shields.io/badge/LLMEval-v0.1.0-purple", width=150)
    st.title("🧪 LLMEval")
    st.caption("LLM Evaluation & Benchmarking Framework")
    st.divider()

    result_files = find_result_files()
    if not result_files:
        st.warning("No result files found in `results/` folder.\nRun an eval first!")
        st.code("python -m llmeval.cli run --config configs/example.yaml")
        st.stop()

    selected_file = st.selectbox(
        "📂 Select result file",
        result_files,
        format_func=lambda x: Path(x).name,
    )

    st.divider()
    st.markdown("**Quick links**")
    st.markdown("📖 [README](https://github.com/YOUR_USERNAME/llmeval)")
    st.markdown("⚙️ [Configs](configs/)")


# ── Load data ─────────────────────────────────────────────────────────────────
results = load_results(selected_file)
score_df = get_score_df(results)
sample_df = get_sample_df(results)

metric_cols = [c for c in score_df.columns if c not in
               ["Model", "Avg latency (ms)", "Total cost ($)", "Errors"]]

run_name = results[0].get("run_name", "eval_run") if results else "eval_run"

# ── Header ────────────────────────────────────────────────────────────────────
st.title("📊 LLMEval Leaderboard")
st.caption(f"Run: **{run_name}** · {results[0]['total_samples']} samples · {len(results)} model(s)")
st.divider()

# ── Top metrics row ───────────────────────────────────────────────────────────
if metric_cols:
    overall = score_df.copy()
    overall["Overall"] = overall[metric_cols].mean(axis=1)
    winner = overall.loc[overall["Overall"].idxmax(), "Model"]

    cols = st.columns(len(results) + 1)
    with cols[0]:
        st.metric("🏆 Overall winner", winner)

    for i, row in overall.iterrows():
        with cols[i + 1]:
            st.metric(
                f"{'🥇' if row['Model'] == winner else '🥈'} {row['Model']}",
                f"{row['Overall']:.4f}",
                delta=f"${row['Total cost ($)']:.5f} cost",
            )

    st.divider()

# ── Score matrix table ────────────────────────────────────────────────────────
st.subheader("Score matrix")

display_df = score_df.copy()
if metric_cols:
    display_df["Overall"] = display_df[metric_cols].mean(axis=1).round(4)

st.dataframe(
    display_df.style
    .highlight_max(subset=metric_cols + (["Overall"] if metric_cols else []), color="#d4edda")
    .format({c: "{:.4f}" for c in metric_cols + (["Overall"] if metric_cols else [])}),
    use_container_width=True,
    height=200,
)

# ── Charts row ────────────────────────────────────────────────────────────────
if metric_cols:
    st.subheader("Score breakdown")
    col1, col2 = st.columns(2)

    with col1:
        # Radar chart
        fig = go.Figure()
        for _, row in score_df.iterrows():
            fig.add_trace(go.Scatterpolar(
                r=[row[m] for m in metric_cols],
                theta=metric_cols,
                fill="toself",
                name=row["Model"],
            ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title="Metric radar",
            height=380,
            margin=dict(t=50, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Bar chart
        melted = score_df[["Model"] + metric_cols].melt(
            id_vars="Model", var_name="Metric", value_name="Score"
        )
        fig2 = px.bar(
            melted, x="Metric", y="Score", color="Model",
            barmode="group", title="Score by metric",
            height=380, range_y=[0, 1],
        )
        fig2.update_layout(margin=dict(t=50, b=20))
        st.plotly_chart(fig2, use_container_width=True)

# ── Latency & cost ────────────────────────────────────────────────────────────
st.subheader("Latency & cost")
col3, col4 = st.columns(2)

with col3:
    fig3 = px.bar(
        score_df, x="Model", y="Avg latency (ms)",
        title="Average latency per model", color="Model", height=300,
    )
    st.plotly_chart(fig3, use_container_width=True)

with col4:
    fig4 = px.bar(
        score_df, x="Model", y="Total cost ($)",
        title="Total API cost per model", color="Model", height=300,
    )
    st.plotly_chart(fig4, use_container_width=True)

# ── Per-sample explorer ───────────────────────────────────────────────────────
st.divider()
st.subheader("🔍 Sample explorer")

model_filter = st.selectbox("Filter by model", ["All"] + [r["model_name"] for r in results])
filtered_df = sample_df if model_filter == "All" else sample_df[sample_df["Model"] == model_filter]

st.dataframe(filtered_df, use_container_width=True, height=350)

# ── Score distribution ────────────────────────────────────────────────────────
if metric_cols and not sample_df.empty:
    st.divider()
    st.subheader("📈 Score distribution")
    selected_metric = st.selectbox("Select metric", metric_cols)
    if selected_metric in sample_df.columns:
        fig5 = px.histogram(
            sample_df, x=selected_metric, color="Model",
            nbins=20, barmode="overlay", opacity=0.7,
            title=f"{selected_metric} distribution across samples",
            height=350,
        )
        st.plotly_chart(fig5, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("LLMEval · MIT License · Built with Streamlit + Plotly")
