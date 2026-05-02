from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import streamlit as st


ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT / "outputs" / "runs"
PLOTS_DIR = ROOT / "outputs" / "plots"


MODEL_CONFIG: Dict[str, Dict[str, object]] = {
    "Baseline CNN": {
        "report": RUNS_DIR / "standalone_cnn_report.json",
        "plots_dir": PLOTS_DIR / "cnn_regression",
        "plot_prefix": "cnn_regression",
        "summary": "Single-channel 1D CNN using dZ/dt only for direct PEP and AVC regression.",
        "strengths": "Simple benchmark, interpretable baseline, fast to train.",
        "methodology": "Processed dZ/dt segments are passed through stacked Conv1D blocks with direct two-target regression using MSE loss.",
        "badge": "",
    },
    "Improved CNN": {
        "report": RUNS_DIR / "cnn_improved_report.json",
        "plots_dir": PLOTS_DIR / "cnn_improved",
        "plot_prefix": "cnn_improved",
        "summary": "Deeper dual-channel CNN using ECG and dZ/dt for stronger feature extraction and more stable regression.",
        "strengths": "Improved representation learning, stronger regularization, better standalone PEP behavior.",
        "methodology": "Dual-channel heartbeat segments are processed by deeper convolutional blocks with batch normalization, adaptive pooling, dropout, and SmoothL1 loss.",
        "badge": "",
    },
    "Smooth-Clipped Dual CNN": {
        "report": RUNS_DIR / "cnn_dual_smooth_clipped_report.json",
        "plots_dir": PLOTS_DIR / "cnn_dual_smooth_clipped",
        "plot_prefix": "cnn_dual_smooth_clipped",
        "summary": "Dual-branch regression model trained with clipped targets to reduce sensitivity to noisy AVC labels.",
        "strengths": "More robust supervision, explicit ECG + dZ/dt fusion, improved noisy-label handling.",
        "methodology": "Separate signal branches learn complementary features, then a weighted SmoothL1 objective emphasizes AVC while training on clipped targets.",
        "badge": "",
    },
    "Advanced Dual CNN": {
        "report": RUNS_DIR / "cnn_dual_advanced_report.json",
        "plots_dir": PLOTS_DIR / "cnn_dual_advanced",
        "plot_prefix": "cnn_dual_advanced",
        "summary": "Physiological dual-branch model that predicts PEP and LVET, then reconstructs AVC as PEP + LVET.",
        "strengths": "Strong physiological interpretability, explicit reconstruction analysis, promising research contribution.",
        "methodology": "The model learns ECG and dZ/dt features jointly, predicts PEP and LVET through separate heads, and uses Log-Cosh loss before reconstructing AVC.",
        "badge": "Best Physiological Model",
    },
    "ResNet": {
        "report": RUNS_DIR / "resnet_report.json",
        "plots_dir": PLOTS_DIR / "resnet",
        "plot_prefix": "resnet",
        "summary": "Residual 1D CNN baseline for deeper stable feature learning on single-channel dZ/dt input.",
        "strengths": "Improved gradient flow, deeper optimization stability, strong benchmark for deeper CNN design.",
        "methodology": "Residual skip connections preserve signal flow across deeper convolution blocks while the shared training pipeline applies physiological consistency regularization.",
        "badge": "",
    },
    "TCN": {
        "report": RUNS_DIR / "tcn_report.json",
        "plots_dir": PLOTS_DIR / "tcn",
        "plot_prefix": "tcn",
        "summary": "Temporal Convolutional Network with dilated convolutions for longer-range heartbeat dependency modeling.",
        "strengths": "Best temporal modeling behavior, larger receptive field, strong research potential.",
        "methodology": "Dilated temporal convolutions expand the receptive field without recurrence, enabling subject-independent timing estimation over longer temporal contexts.",
        "badge": "Best Temporal Model",
    },
}


PLOT_TABS = [
    ("Loss Curve", "loss_curve"),
    ("Validation MAE", "val_mae_curve"),
    ("Predicted vs True", "predicted_vs_true"),
    ("Error Histogram", "error_histogram"),
]


def load_report(report_path: Path) -> Dict[str, object]:
    with report_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def get_metrics(report: Dict[str, object]) -> Dict[str, float]:
    test_metrics = report.get("test_metrics", {})
    return {
        "MAE": float(test_metrics.get("mean_mae_ms", 0.0)),
        "RMSE": float(test_metrics.get("mean_rmse_ms", 0.0)),
        "MedAE": float(test_metrics.get("mean_medae_ms", 0.0)),
        "+/-10 ms": float(test_metrics.get("mean_acc_10ms_%", 0.0)),
        "+/-20 ms": float(test_metrics.get("mean_acc_20ms_%", 0.0)),
        "PEP MAE": float(test_metrics.get("avo_mae_ms", 0.0)),
        "AVC MAE": float(test_metrics.get("avc_mae_ms", 0.0)),
        "Bias": float(test_metrics.get("mean_bias_ms", 0.0)),
    }


def resolve_plot_path(model_cfg: Dict[str, object], suffix: str) -> Path:
    return Path(model_cfg["plots_dir"]) / f"{model_cfg['plot_prefix']}_{suffix}.png"


def render_plot(plot_path: Path, caption: str) -> None:
    """
    Render a plot in a centered content column so large PNG exports do not
    expand across the full page width.
    """
    left_pad, center_col, right_pad = st.columns([0.6, 4.8, 0.6])
    with center_col:
        st.image(str(plot_path), width=980)
        st.caption(caption)


st.set_page_config(
    page_title="Physiological Parameter Estimation Dashboard",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1280px;
    }
    .badge {
        display: inline-block;
        padding: 0.35rem 0.7rem;
        border-radius: 999px;
        border: 1px solid rgba(128, 128, 128, 0.35);
        font-size: 0.85rem;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
        background: rgba(120, 120, 120, 0.08);
    }
    .info-card {
        border: 1px solid rgba(128, 128, 128, 0.25);
        border-radius: 14px;
        padding: 1rem 1rem 0.4rem 1rem;
        background: rgba(120, 120, 120, 0.04);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.title("Model Explorer")
    selected_model = st.selectbox("Model Selection", list(MODEL_CONFIG.keys()))
    st.markdown("### Dashboard Notes")
    st.caption("Research-oriented comparison of physiological deep learning models for PEP and AVC prediction.")
    st.markdown('<div class="badge">Best Temporal Model -> TCN</div>', unsafe_allow_html=True)
    st.markdown('<div class="badge">Best Physiological Model -> Advanced Dual CNN</div>', unsafe_allow_html=True)

model_cfg = MODEL_CONFIG[selected_model]
report_path = Path(model_cfg["report"])

st.title("Deep Learning-Based Physiological Parameter Estimation")
st.caption(
    "Prediction of PEP and AVC using ECG and dZ/dt signals through CNN, TCN, "
    "and physiological dual-branch models."
)

if not report_path.exists():
    st.error(f"Report not found: {report_path}")
    st.stop()

report = load_report(report_path)
metrics = get_metrics(report)

st.markdown("### Model Overview")
info_col, badge_col = st.columns([5, 2])
with info_col:
    st.info(
        f"**Architecture Summary:** {model_cfg['summary']}\n\n"
        f"**Strengths:** {model_cfg['strengths']}\n\n"
        f"**Methodology:** {model_cfg['methodology']}"
    )
with badge_col:
    if model_cfg["badge"]:
        st.markdown(f'<div class="badge">{model_cfg["badge"]}</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="info-card">
        <strong>Report</strong><br>
        <code>{report_path.name}</code>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("### Test Metrics")
metric_row_1 = st.columns(4)
metric_row_1[0].metric("MAE", f"{metrics['MAE']:.2f} ms")
metric_row_1[1].metric("RMSE", f"{metrics['RMSE']:.2f} ms")
metric_row_1[2].metric("MedAE", f"{metrics['MedAE']:.2f} ms")
metric_row_1[3].metric("Bias", f"{metrics['Bias']:.2f} ms")

metric_row_2 = st.columns(4)
metric_row_2[0].metric("Accuracy +/-10 ms", f"{metrics['+/-10 ms']:.1f}%")
metric_row_2[1].metric("Accuracy +/-20 ms", f"{metrics['+/-20 ms']:.1f}%")
metric_row_2[2].metric("PEP MAE", f"{metrics['PEP MAE']:.2f} ms")
metric_row_2[3].metric("AVC MAE", f"{metrics['AVC MAE']:.2f} ms")

st.markdown("### Graphs")
tab_labels = [label for label, _ in PLOT_TABS]
if selected_model == "Advanced Dual CNN":
    tab_labels.append("AVC Reconstruction")

tabs = st.tabs(tab_labels)

for index, (label, suffix) in enumerate(PLOT_TABS):
    plot_path = resolve_plot_path(model_cfg, suffix)
    with tabs[index]:
        if plot_path.exists():
            render_plot(plot_path, plot_path.name)
        else:
            st.warning(f"Plot not found: {plot_path}")

if selected_model == "Advanced Dual CNN":
    reconstruction_path = Path(model_cfg["plots_dir"]) / "cnn_dual_advanced_avc_reconstruction.png"
    with tabs[-1]:
        if reconstruction_path.exists():
            render_plot(reconstruction_path, reconstruction_path.name)
        else:
            st.warning(f"Plot not found: {reconstruction_path}")

with st.expander("Artifact Paths"):
    artifacts = report.get("artifacts", {})
    if artifacts:
        for key, value in artifacts.items():
            st.code(f"{key}: {value}")
    else:
        st.caption("No artifact metadata available in this report.")
