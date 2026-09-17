"""
CEdge-XAI: Master Streamlit Dashboard.
Interactive Explainable AI Interface for Real-Time IoT Anomaly Detection.
"""

import os
import pandas as pd
import streamlit as st

# Configure wide layout and page theme
st.set_page_config(
    page_title="CEdge-XAI | IoT Anomaly Detection",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.image("https://img.icons8.com/fluency/96/processor.png", width=64)
st.sidebar.title("CEdge-XAI Platform")
st.sidebar.markdown(
    "**Causal, Counterfactual & Edge-Deployable XAI for Real-Time IoT Anomaly Detection**"
)
st.sidebar.markdown("---")
st.sidebar.info(
    "💡 **Architecture**: Split-Brain Engine\n\n"
    "• **Edge Tier**: ONNX INT8 (< 2ms)\n\n"
    "• **Server Tier**: Tigramite PCMCI + Counterfactuals\n\n"
    "• **Federated**: FedAvg across 3 sites"
)

# Main Landing Page
st.title("⚡ CEdge-XAI: Thesis Demonstration Platform")
st.subheader(
    "Causal, Counterfactual, and Edge-Deployable Explainable AI for Real-Time IoT Anomaly Detection"
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Detection Latency", "0.185 ms", delta="-91% vs PyTorch")
col2.metric("Edge Model Footprint", "0.588 MB", delta="INT8 Quantized")
col3.metric("Primary Model F1", "0.9848", delta="Precision: 1.000")
col4.metric("Privacy Level", "100% On-Device", delta="FedAvg 3-Sites")

st.markdown("---")

col_left, col_right = st.columns([3, 2])

with col_left:
    st.markdown("### 🎯 Problem Statement & Thesis Motivation")
    st.markdown("""
        Modern Industrial Internet of Things (IIoT) facilities generate high-frequency multivariate telemetry across dozens of physical sensors.
        Traditional anomaly detectors exhibit two critical flaws:
        1. **Black-Box Opacity**: They flag anomalies without revealing physical root causes or actionable interventions.
        2. **Deployment Bottlenecks**: Deep neural networks are too computationally intensive for resource-constrained edge gateways.

        **CEdge-XAI** resolves these dual challenges by introducing:
        - **Split-Brain Architecture**: Ultra-fast INT8 ONNX anomaly detection at the edge sensor interface (< 0.2 ms), coupled with deep causal-counterfactual reasoning on an edge server.
        - **Physics-Aware Counterfactual Engine**: Computes exact 'what-if' interventions that identify root causes and generate plain-English engineering explanations.
        - **Tigramite PCMCI Causal Discovery**: Uncovers directed causal propagation networks across physical sensors ($p < 0.01$).
        - **Privacy-Preserving Federated Learning**: Trains global LSTM-Autoencoders across decentralized manufacturing sites with zero raw data transmission.
        """)

    st.markdown("### 📂 Dashboard Navigation")
    st.markdown("""
        Use the sidebar to explore each experimental thesis pillar:
        - **Page 1: Overview & Architecture**: System topology, 13-sensor physical hierarchy, and paper assets.
        - **Page 2: Real-Time Anomaly Detection**: 4-day continuous telemetry streaming, A1-A7 anomaly episodes.
        - **Page 3: Explainable AI: Counterfactuals**: Minimal required interventions, physics clamping vs gradient descent.
        - **Page 4: Causal Discovery**: Tigramite PCMCI directed DAG, upstream/downstream fault cascades.
        - **Page 5: Federated Learning**: Multi-factory FedAvg training monitor, convergence dynamics, and privacy metrics.
        - **Page 6: Edge Deployment & Benchmarking**: PyTorch vs ONNX FP32 vs ONNX INT8 latency, RAM, and hardware guides.
        """)

with col_right:
    st.markdown("### 🏛️ Directed Causal Topology")
    causal_img = "paper_assets/figures/causal_graph.png"
    if os.path.exists(causal_img):
        st.image(causal_img, use_container_width=True)

    st.markdown("### 📊 Benchmark Architecture Summary")
    comp_csv = "paper_assets/tables/model_comparison.csv"
    if os.path.exists(comp_csv):
        df_comp = pd.read_csv(comp_csv)
        st.dataframe(df_comp[["Model", "F1-Score", "Latency (ms)", "Size (MB)"]], hide_index=True)
