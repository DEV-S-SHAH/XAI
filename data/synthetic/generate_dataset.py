#!/usr/bin/env python3
"""
Synthetic IoT Dataset Generator for CEdge-XAI (10 Sensors, 3 Days).
Encodes 5-layer causal hierarchy and 5 industrial failure scenarios (A1 - A5).
Generates 4320 rows and publication-quality EDA overview plot.
"""

import argparse
import logging
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

FEATURE_COLUMNS: List[str] = [
    "ambient_temp",
    "ambient_humidity",
    "fan_speed",
    "lubrication_flow",
    "motor_temp",
    "spindle_speed",
    "vibration",
    "acoustic_emission",
    "power_draw",
    "tool_wear",
]


def load_dataset_config(config_path: str = "config/config.yaml") -> Dict:
    """Load dataset configuration from YAML file."""
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
            return cfg.get("data", {})
    return {"total_rows": 4320, "synthetic_csv": "data/synthetic/factory_iot_data.csv"}


def generate_factory_dataset(
    output_path: str = "data/synthetic/factory_iot_data.csv",
    figures_dir: str = "paper_assets/figures",
    seed: int = 42,
    total_rows: int = 4320,
) -> pd.DataFrame:
    """Generate 3-day continuous multivariate IoT telemetry dataset (4320 rows)."""
    np.random.seed(seed)
    timestamps = pd.date_range(start="2024-01-01 00:00:00", periods=total_rows, freq="1min")
    t_min = np.arange(total_rows)
    hours = timestamps.hour.values

    # Temporal & Operational Cycles
    diurnal_temp = 5.5 * np.sin(2 * np.pi * t_min / 1440 - np.pi / 2)
    work_hour_mask = (hours >= 8) & (hours < 18)
    night_shift_mask = (hours >= 22) | (hours < 6)

    base_load = np.full(total_rows, 55.0)
    base_load[work_hour_mask] = 80.0
    base_load[night_shift_mask] = 30.0
    load_pct = np.clip(base_load + np.random.normal(0, 4.0, size=total_rows), 20.0, 95.0)

    # Layer 1: Environmental Features
    ambient_temp = np.clip(
        26.5 + diurnal_temp + np.random.normal(0, 0.5, size=total_rows), 20.0, 35.0
    )
    ambient_humidity = np.clip(
        50.0 - 1.8 * (ambient_temp - 26.5) + np.random.normal(0, 1.5, size=total_rows),
        30.0,
        70.0,
    )

    # Layer 2: Actuators
    fan_speed = np.clip(
        1150.0 + 20.0 * (ambient_temp - 26.5) + np.random.normal(0, 25.0, size=total_rows),
        800.0,
        1500.0,
    )
    lubrication_flow = np.clip(3.5 + np.random.normal(0, 0.2, size=total_rows), 2.0, 5.0)

    # Layer 3: Primary Effects
    spindle_speed = np.clip(
        1200.0 + 18.0 * load_pct + np.random.normal(0, 35.0, size=total_rows),
        1000.0,
        3000.0,
    )

    # Initialize Anomaly Trackers
    anomaly = np.zeros(total_rows, dtype=int)
    anomaly_type = np.array(["Normal"] * total_rows, dtype=object)

    # A1: Fan Failure (Day 1: rows 510 to 545, 35 mins)
    a1_slice = slice(510, 545)
    fan_speed[a1_slice] = 0.0
    anomaly[a1_slice] = 1
    anomaly_type[a1_slice] = "A1_Fan_Failure"

    # A2: Lubrication Leak (Day 1: rows 840 to 875, 35 mins)
    a2_slice = slice(840, 875)
    lubrication_flow[a2_slice] = 0.05
    anomaly[a2_slice] = 1
    anomaly_type[a2_slice] = "A2_Lubrication_Leak"

    # A3: Heatwave (Day 2: rows 1800 to 1840, 40 mins)
    a3_slice = slice(1800, 1840)
    ambient_temp[a3_slice] = 52.0 + np.random.normal(0, 0.4, size=40)
    anomaly[a3_slice] = 1
    anomaly_type[a3_slice] = "A3_Heatwave"

    # Motor Temp dependent on ambient_temp, fan_speed, load
    motor_cool = 0.02 * (fan_speed - 1150.0)
    motor_temp = (
        56.0
        + 0.75 * (ambient_temp - 26.5)
        + 0.22 * (load_pct - 55.0)
        - motor_cool
        + np.random.normal(0, 0.6, size=total_rows)
    )
    motor_temp[a1_slice] += 25.0
    motor_temp = np.clip(motor_temp, 40.0, 96.0)

    # A5: Spindle Overload (Day 3: rows 3420 to 3458, 38 mins)
    a5_slice = slice(3420, 3458)
    spindle_speed[a5_slice] = 4800.0 + np.random.normal(0, 40.0, size=38)
    anomaly[a5_slice] = 1
    anomaly_type[a5_slice] = "A5_Spindle_Overload"

    # Layer 4: Secondary Effects (Vibration, Acoustic)
    lub_friction = 3.6 / np.clip(lubrication_flow, 0.02, 5.0) - 0.9
    temp_vib = 0.03 * (motor_temp - 56.0)
    speed_vib = 0.0006 * (spindle_speed - 2000.0)
    vibration = np.clip(
        1.6 + lub_friction + temp_vib + speed_vib + np.random.normal(0, 0.1, size=total_rows),
        0.5,
        12.0,
    )

    # A4: Sensor Glitch (Day 2: rows 2400 to 2425, 25 mins)
    a4_slice = slice(2400, 2425)
    physical_vib = vibration.copy()
    vibration[a4_slice] = 99.0 + np.random.normal(0, 0.2, size=25)
    anomaly[a4_slice] = 1
    anomaly_type[a4_slice] = "A4_Sensor_Glitch"

    acoustic_emission = np.clip(
        53.0
        + 5.8 * physical_vib
        + 0.12 * (motor_temp - 56.0)
        + np.random.normal(0, 0.7, size=total_rows),
        48.0,
        95.0,
    )

    # Layer 5: Output (Power Draw, Tool Wear)
    power_draw = np.clip(
        7.2
        + 0.07 * (motor_temp - 56.0)
        + 1.1 * (physical_vib - 1.6)
        + 0.002 * (spindle_speed - 2000.0)
        + np.random.normal(0, 0.3, size=total_rows),
        4.5,
        25.0,
    )

    shift_cycle = t_min % 480
    base_wear_rate = 0.045 * (
        1.0
        + 0.35 * (physical_vib - 1.6)
        + 0.0003 * (spindle_speed - 2000.0)
        + 0.004 * (load_pct - 55.0)
    )
    tool_wear = (
        5.0
        + shift_cycle * np.clip(base_wear_rate, 0.01, 0.09)
        + np.random.normal(0, 0.4, size=total_rows)
    )
    tool_wear[a5_slice] += 16.0
    tool_wear = np.clip(tool_wear, 0.0, 50.0)

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "ambient_temp": np.round(ambient_temp, 2),
            "ambient_humidity": np.round(ambient_humidity, 2),
            "fan_speed": np.round(fan_speed, 1),
            "lubrication_flow": np.round(lubrication_flow, 2),
            "motor_temp": np.round(motor_temp, 2),
            "spindle_speed": np.round(spindle_speed, 1),
            "vibration": np.round(vibration, 2),
            "acoustic_emission": np.round(acoustic_emission, 2),
            "power_draw": np.round(power_draw, 2),
            "tool_wear": np.round(tool_wear, 2),
            "anomaly": anomaly,
            "anomaly_type": anomaly_type,
        }
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(
        f"Saved synthetic dataset to {output_path} ({len(df)} rows, {len(FEATURE_COLUMNS)} features)"
    )

    render_overview_plot(df, figures_dir)
    return df


def render_overview_plot(df: pd.DataFrame, figures_dir: str = "paper_assets/figures") -> None:
    """Render and save 300 DPI dataset overview plot."""
    os.makedirs(figures_dir, exist_ok=True)
    fig, axes = plt.subplots(5, 2, figsize=(14, 12), sharex=True)
    axes = axes.flatten()

    for i, feat in enumerate(FEATURE_COLUMNS):
        axes[i].plot(df["timestamp"], df[feat], color="#2980b9", lw=0.8, label="Normal")
        anom_mask = df["anomaly"] == 1
        axes[i].scatter(
            df.loc[anom_mask, "timestamp"],
            df.loc[anom_mask, feat],
            color="#e74c3c",
            s=3,
            label="Anomaly",
        )
        axes[i].set_title(feat.replace("_", " ").title(), fontweight="bold", fontsize=10)
        axes[i].set_ylabel("Value")
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    overview_path = os.path.join(figures_dir, "dataset_overview.png")
    plt.savefig(overview_path, dpi=300)
    plt.close()
    logger.info(f"Saved dataset overview figure to {overview_path}")


if __name__ == "__main__":
    cfg_data = load_dataset_config()
    default_csv = cfg_data.get("synthetic_csv", "data/synthetic/factory_iot_data.csv")
    total_samples = cfg_data.get("total_rows", 4320)

    parser = argparse.ArgumentParser(description="Generate 3-day synthetic IoT sensor dataset.")
    parser.add_argument("--output", type=str, default=default_csv)
    parser.add_argument("--figdir", type=str, default="paper_assets/figures")
    parser.add_argument("--rows", type=int, default=total_samples)
    args = parser.parse_args()

    generate_factory_dataset(output_path=args.output, figures_dir=args.figdir, total_rows=args.rows)
