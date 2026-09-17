"""
Page 5: Federated Learning Monitor for CEdge-XAI.
Monitors 3 distributed edge factories (A, B, C) participating in FedAvg coordination.
Displays communication round convergence, local losses, and Centralized vs Federated efficacy.
"""

import os
import sys
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

st.set_page_config(page_title="Federated Learning | CEdge-XAI", layout="wide", page_icon="🌐")
st.title("🌐 Decentralized Federated Edge Learning (FedAvg)")

st.markdown(
    "To comply with stringent industrial confidentiality regulations, edge nodes train local models "
    "on-premise and exchange **only** neural network weights via **Flower FedAvg**, sharing zero raw telemetry."
)

# High-Level Metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Federated Edge Nodes", "3 Factories", delta="A, B, C")
col2.metric("Communication Rounds", "10 Rounds", delta="FedAvg")
col3.metric("Final Global F1", "0.8140", delta="Centralized: 0.9848")
col4.metric("Data Privacy Level", "100%", delta="Zero Telemetry Leaves Premise")

st.markdown("---")

tab1, tab2, tab3 = st.tabs(
    [
        "📉 Convergence Dynamics",
        "🏭 Edge Factory Node Status",
        "⚖️ Federated vs Centralized Efficacy",
    ]
)

with tab1:
    st.subheader("FedAvg Convergence Across 10 Communication Rounds")

    csv_path = "paper_assets/tables/fl_convergence.csv"
    if os.path.exists(csv_path):
        df_conv = pd.read_csv(csv_path)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df_conv["Round"],
                y=df_conv["Loss_A"],
                mode="lines+markers",
                name="Factory A (Day 1)",
                line=dict(color="#2b5c8f", dash="dot"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df_conv["Round"],
                y=df_conv["Loss_B"],
                mode="lines+markers",
                name="Factory B (Day 2)",
                line=dict(color="#e07a5f", dash="dot"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df_conv["Round"],
                y=df_conv["Loss_C"],
                mode="lines+markers",
                name="Factory C (Day 3)",
                line=dict(color="#81b29a", dash="dot"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df_conv["Round"],
                y=df_conv["Global_Loss"],
                mode="lines+markers",
                name="Global Aggregated Loss",
                line=dict(color="#222222", width=3),
            )
        )

        fig.update_layout(
            title="Training Loss Trajectory Across Edge Factories",
            xaxis_title="Communication Round",
            yaxis_title="Reconstruction Error (MSE)",
            template="plotly_white",
            height=450,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df_conv, use_container_width=True)
    else:
        st.info("Run `python models/federated/fl_server.py` to generate convergence metrics.")

with tab2:
    st.subheader("Simulated Edge Factory Node Topology")
    factories = [
        {
            "Node ID": "Factory_A",
            "Location": "Detroit Plant (Day 1)",
            "Samples": 1920,
            "Local Epochs": 3,
            "Status": "Synced (Round 10)",
        },
        {
            "Node ID": "Factory_B",
            "Location": "Munich Plant (Day 2)",
            "Samples": 1920,
            "Local Epochs": 3,
            "Status": "Synced (Round 10)",
        },
        {
            "Node ID": "Factory_C",
            "Location": "Tokyo Plant (Day 3)",
            "Samples": 1920,
            "Local Epochs": 3,
            "Status": "Synced (Round 10)",
        },
    ]
    st.table(pd.DataFrame(factories))
    st.info(
        "💡 **Data Isolation Guarantee**: Each edge gateway processes 1,920 continuous minutes of local telemetry. "
        "Network transmission is strictly limited to 0.48 MB parameter payloads per round, preserving bandwidth."
    )

with tab3:
    st.subheader("Federated vs. Centralized Performance Tradeoff")
    fig_col1, fig_col2 = st.columns(2)
    with fig_col1:
        if os.path.exists("paper_assets/figures/fl_vs_centralized.png"):
            st.image(
                "paper_assets/figures/fl_vs_centralized.png",
                caption="Figure 7: Centralized vs Federated Learning Efficacy",
            )
    with fig_col2:
        if os.path.exists("paper_assets/figures/fl_convergence.png"):
            st.image(
                "paper_assets/figures/fl_convergence.png",
                caption="Figure 6: Multi-Site FedAvg Convergence Plot",
            )
