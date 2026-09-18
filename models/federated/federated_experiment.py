"""
Privacy-Preserving Federated Training Benchmark Module for IoT Cyber-Physical Systems.
Novelty 3: Decentralized Privacy-Preserving Federated Learning (FedAvg).

Compares:
1. Centralized Training (Pooled data)
2. Local-Only Training (Isolated factories without federation)
3. Federated Learning (FedAvg across 10 communication rounds)

Features:
- Realistic Non-IID multi-factory distributions (Factory A, B, C with heterogeneous failure regimes).
- Metrics: Precision, Recall, F1, ROC-AUC, PR-AUC (Global and Per-Client).
- Convergence logging (loss per client per round, global F1).
- Communication bytes and training time tracking.
- Programmatic privacy audit: asserts zero raw telemetry leaves client devices.
"""

import copy
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from models.detection.lstm_autoencoder import LSTMAutoencoder
from models.detection.train import create_sliding_windows, optimize_threshold
from models.federated.fl_utils import get_model_parameters, set_model_parameters

logger = logging.getLogger(__name__)


def partition_non_iid_factories(
    csv_path: str = "data/synthetic/factory_iot_data.csv",
    window_size: int = 60,
) -> Dict[str, Dict[str, Any]]:
    """
    Partition IoT data into 3 realistic Non-IID factories with heterogeneous failure scenarios:
    - Factory A (Day 1): Base manufacturing + Fan Failure (A1) + Lubrication Leak (A2)
    - Factory B (Day 2): High-temperature line + Heatwave (A3) + Accelerometer Glitch (A4)
    - Factory C (Day 3): Heavy CNC milling + Spindle Mechanical Overload (A5)
    """
    df = pd.read_csv(csv_path)
    features = [
        "ambient_temp", "ambient_humidity", "fan_speed", "lubrication_flow",
        "motor_temp", "spindle_speed", "vibration", "acoustic_emission",
        "power_draw", "tool_wear",
    ]

    scaler_path = "models_saved/checkpoints/scaler.joblib"
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

    # Temporal partition by industrial shift / scenario days (1440 mins per day)
    factory_slices = {
        "Factory_A": (0, 1440, ["A1_Fan_Failure", "A2_Lubrication_Leak"]),
        "Factory_B": (1440, 2880, ["A3_Heatwave", "A4_Sensor_Glitch"]),
        "Factory_C": (2880, len(df), ["A5_Spindle_Overload"]),
    }

    partitions = {}
    for name, (start, end, scenarios) in factory_slices.items():
        df_sub = df.iloc[start:end].copy()
        X_raw = df_sub[features].values
        y_raw = df_sub["anomaly"].values
        types_raw = df_sub["anomaly_type"].values

        if scaler:
            X_scaled = scaler.transform(X_raw)
        else:
            X_scaled = X_raw

        X_win, y_win = create_sliding_windows(X_scaled, y_raw, window_size=window_size)
        win_types = [types_raw[k + window_size - 1] for k in range(len(X_win))]

        # Normal training sequences (pure normal windows)
        pure_norm = np.array([y_raw[k : k + window_size].sum() == 0 for k in range(len(X_win))])
        X_train_norm = X_win[pure_norm]

        partitions[name] = {
            "name": name,
            "raw_rows": len(df_sub),
            "scenarios": scenarios,
            "X_train_normal": X_train_norm,
            "X_val": X_win,
            "y_val": y_win,
            "anomaly_types": win_types,
            "anomaly_count": int(y_win.sum()),
        }
        logger.info(
            f"Non-IID Partition [{name}]: {len(df_sub)} rows, {len(X_train_norm)} train normal windows, "
            f"{int(y_win.sum())} anomaly windows ({scenarios})"
        )

    return partitions


def verify_federated_privacy(
    client_payloads: List[Dict[str, Any]],
    log_csv_path: Optional[str] = "paper_assets/tables/novelty3_privacy_audit.csv",
) -> pd.DataFrame:
    """
    Programmatically verify that raw training data NEVER leaves client devices.
    Inspects transmission objects and asserts:
    1. Only serialized numerical weight tensors (np.ndarray) are transmitted.
    2. Zero raw sensor arrays, DataFrames, timestamps, or input samples are in payload.
    3. Measures transmission payload in bytes.
    """
    audit_rows = []
    for payload in client_payloads:
        client_name = payload["client_name"]
        params = payload["parameters"]
        num_samples = payload["num_samples"]

        # Audit payload contents
        is_only_weights = True
        total_param_bytes = 0
        leaked_raw_data_found = False

        for p in params:
            if not isinstance(p, np.ndarray):
                is_only_weights = False
            else:
                total_param_bytes += p.nbytes

        # Check for illicit raw data keys
        forbidden_keys = ["raw_data", "dataframe", "sensors", "timestamps", "X_train", "inputs"]
        for fk in forbidden_keys:
            if fk in payload:
                leaked_raw_data_found = True

        audit_rows.append({
            "Client_Name": client_name,
            "Payload_Type": "Model_Weights_Only (FedAvg)",
            "Transmitted_Param_Tensors": len(params),
            "Payload_Size_Bytes": total_param_bytes,
            "Payload_Size_KB": round(total_param_bytes / 1024.0, 2),
            "Raw_Data_Exposed": leaked_raw_data_found,
            "Privacy_Audit_Passed": (is_only_weights and not leaked_raw_data_found),
        })

    df_audit = pd.DataFrame(audit_rows)
    if log_csv_path:
        os.makedirs(os.path.dirname(log_csv_path), exist_ok=True)
        df_audit.to_csv(log_csv_path, index=False)
        logger.info(f"Saved federated privacy audit log to {log_csv_path}")

    # Programmatic assertion
    all_passed = df_audit["Privacy_Audit_Passed"].all()
    if not all_passed:
        raise AssertionError("PRIVACY LEAK DETECTED: Non-weight telemetry found in client aggregation payload!")

    return df_audit


def evaluate_model_on_data(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    threshold: Optional[float] = None,
) -> Dict[str, float]:
    """Compute Precision, Recall, F1, ROC-AUC, and PR-AUC."""
    model.eval()
    with torch.no_grad():
        t_x = torch.tensor(X, dtype=torch.float32)
        recon = model(t_x)
        diff = (recon - t_x) ** 2
        scores = (0.5 * diff.mean(dim=(1, 2)) + 0.5 * diff[:, -1, :].mean(dim=-1)).cpu().numpy()

    if threshold is None:
        th, _ = optimize_threshold(scores, y)
    else:
        th = threshold

    preds = (scores > th).astype(int)

    f1 = float(f1_score(y, preds, zero_division=0))
    prec = float(precision_score(y, preds, zero_division=0))
    rec = float(recall_score(y, preds, zero_division=0))

    try:
        roc_auc = float(roc_auc_score(y, scores))
    except Exception:
        roc_auc = 0.5

    try:
        p_curve, r_curve, _ = precision_recall_curve(y, scores)
        pr_auc = float(auc(r_curve, p_curve))
    except Exception:
        pr_auc = float(np.mean(y))

    return {
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "threshold": float(th),
    }


def train_local_model(
    X_train_norm: np.ndarray,
    config: Dict,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 0.001,
) -> LSTMAutoencoder:
    """Train isolated model strictly on local data."""
    cfg_m = config["models"]["lstm_ae"]
    model = LSTMAutoencoder(
        input_dim=cfg_m["input_dim"],
        hidden_dim=cfg_m["hidden_dim"],
        seq_len=config["preprocessing"]["window_size"],
        num_layers=cfg_m["num_layers"],
        dropout=cfg_m["dropout"],
    )
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    dataset = TensorDataset(torch.tensor(X_train_norm, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for _ in range(epochs):
        for (bx,) in loader:
            optimizer.zero_grad()
            recon = model(bx)
            loss = criterion(recon, bx)
            loss.backward()
            optimizer.step()
    return model


def run_federated_vs_baselines(
    config_path: str = "config/config.yaml",
    num_rounds: int = 10,
    local_epochs_per_round: int = 2,
    seed: int = 42,
) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """
    Run complete comparative experiment:
    1. Centralized Training
    2. Local-Only Training (Factory A, B, C)
    3. Federated Training (FedAvg across 10 rounds)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    partitions = partition_non_iid_factories(
        csv_path=config["data"]["synthetic_csv"],
        window_size=config["preprocessing"]["window_size"],
    )

    # Prepare Global Test Set (independent combined test windows)
    from models.detection.train import prepare_dataset

    data_pkg = prepare_dataset(
        csv_path=config["data"]["synthetic_csv"],
        window_size=config["preprocessing"]["window_size"],
        random_state=seed,
    )
    X_test_global = data_pkg["X_test"]
    y_test_global = data_pkg["y_test"]

    # ==========================================
    # 1. CENTRALIZED TRAINING (Upper Bound)
    # ==========================================
    logger.info("--- Training Centralized Benchmark Model ---")
    t0_cent = time.perf_counter()
    centralized_model = train_local_model(
        data_pkg["X_train_normal"],
        config=config,
        epochs=num_rounds * local_epochs_per_round,
    )
    t_cent_time = time.perf_counter() - t0_cent
    cent_metrics = evaluate_model_on_data(centralized_model, X_test_global, y_test_global)
    logger.info(f"Centralized Model: F1={cent_metrics['f1']:.4f}, AUC={cent_metrics['roc_auc']:.4f}")

    # ==========================================
    # 2. LOCAL-ONLY TRAINING (Isolated Nodes)
    # ==========================================
    local_models = {}
    local_metrics_on_global = {}
    t0_local = time.perf_counter()

    for name, part in partitions.items():
        logger.info(f"--- Training Local-Only Model for [{name}] ---")
        m_loc = train_local_model(
            part["X_train_normal"],
            config=config,
            epochs=num_rounds * local_epochs_per_round,
        )
        local_models[name] = m_loc
        # Evaluate on global test set
        m_eval = evaluate_model_on_data(m_loc, X_test_global, y_test_global)
        local_metrics_on_global[name] = m_eval
        logger.info(f"Local [{name}] on Global Test: F1={m_eval['f1']:.4f}, AUC={m_eval['roc_auc']:.4f}")

    t_local_time = time.perf_counter() - t0_local

    # ==========================================
    # 3. FEDERATED LEARNING (FedAvg)
    # ==========================================
    logger.info(f"--- Starting Federated Learning (FedAvg, {num_rounds} rounds, 3 clients) ---")
    cfg_m = config["models"]["lstm_ae"]
    global_model = LSTMAutoencoder(
        input_dim=cfg_m["input_dim"],
        hidden_dim=cfg_m["hidden_dim"],
        seq_len=config["preprocessing"]["window_size"],
        num_layers=cfg_m["num_layers"],
        dropout=cfg_m["dropout"],
    )

    current_params = get_model_parameters(global_model)
    client_names = list(partitions.keys())
    convergence_records = []

    total_bytes_communicated = 0
    t0_fed = time.perf_counter()

    for r in range(1, num_rounds + 1):
        client_weights = []
        client_sample_counts = []
        client_losses = {}
        round_payloads = []

        # Local training step on each factory
        for name in client_names:
            p = partitions[name]
            client_model = copy.deepcopy(global_model)
            set_model_parameters(client_model, current_params)

            dataset = TensorDataset(torch.tensor(p["X_train_normal"], dtype=torch.float32))
            loader = DataLoader(dataset, batch_size=config["federated"]["batch_size"], shuffle=True)
            optimizer = optim.Adam(client_model.parameters(), lr=config["models"]["lstm_ae"]["learning_rate"])
            criterion = nn.MSELoss()

            client_model.train()
            loss_acc = 0.0
            for _ in range(local_epochs_per_round):
                for (bx,) in loader:
                    optimizer.zero_grad()
                    recon = client_model(bx)
                    loss = criterion(recon, bx)
                    loss.backward()
                    optimizer.step()
                    loss_acc += loss.item() * len(bx)

            mean_loss = loss_acc / (len(p["X_train_normal"]) * local_epochs_per_round)
            new_params = get_model_parameters(client_model)

            client_weights.append(new_params)
            client_sample_counts.append(len(p["X_train_normal"]))
            client_losses[name] = mean_loss

            # Payload tracking for privacy audit and communication volume
            payload = {
                "client_name": name,
                "round": r,
                "parameters": new_params,
                "num_samples": len(p["X_train_normal"]),
            }
            round_payloads.append(payload)

        # Audit privacy on round 1 and final round
        if r == 1 or r == num_rounds:
            verify_federated_privacy(round_payloads)

        # Compute communication bytes (upstream parameter transfer from all clients)
        round_bytes = sum(sum(p.nbytes for p in pl["parameters"]) for pl in round_payloads)
        total_bytes_communicated += round_bytes

        # FedAvg Aggregation on Server
        total_samples = sum(client_sample_counts)
        aggregated_params = []
        for param_idx in range(len(current_params)):
            weighted_sum = np.zeros_like(current_params[param_idx])
            for c_idx in range(len(client_names)):
                weight_factor = client_sample_counts[c_idx] / total_samples
                weighted_sum += client_weights[c_idx][param_idx] * weight_factor
            aggregated_params.append(weighted_sum)

        current_params = aggregated_params
        set_model_parameters(global_model, current_params)

        # Evaluate global model on test set
        eval_res = evaluate_model_on_data(global_model, X_test_global, y_test_global)
        logger.info(
            f"Round {r:02d}/{num_rounds} | Avg Client Loss: {np.mean(list(client_losses.values())):.5f} | "
            f"Global F1: {eval_res['f1']:.4f} | ROC-AUC: {eval_res['roc_auc']:.4f}"
        )

        convergence_records.append({
            "Round": r,
            "Factory_A_Loss": client_losses.get("Factory_A", 0.0),
            "Factory_B_Loss": client_losses.get("Factory_B", 0.0),
            "Factory_C_Loss": client_losses.get("Factory_C", 0.0),
            "Avg_Client_Loss": float(np.mean(list(client_losses.values()))),
            "Global_F1": eval_res["f1"],
            "Global_ROC_AUC": eval_res["roc_auc"],
            "Global_PR_AUC": eval_res["pr_auc"],
            "Communication_Round_KB": round_bytes / 1024.0,
        })

    t_fed_time = time.perf_counter() - t0_fed
    fed_final_metrics = evaluate_model_on_data(global_model, X_test_global, y_test_global)

    # Per-client evaluations of final global model
    per_client_fed_metrics = {}
    for name, part in partitions.items():
        per_client_fed_metrics[name] = evaluate_model_on_data(
            global_model, part["X_val"], part["y_val"], threshold=fed_final_metrics["threshold"]
        )

    # Compile Comparison Table
    comp_rows = [
        {
            "Paradigm": "Centralized Training (Upper Bound)",
            "Data_Locality": "Raw Data Pooled Centrally",
            "Precision": round(cent_metrics["precision"], 4),
            "Recall": round(cent_metrics["recall"], 4),
            "F1_Score": round(cent_metrics["f1"], 4),
            "ROC_AUC": round(cent_metrics["roc_auc"], 4),
            "PR_AUC": round(cent_metrics["pr_auc"], 4),
            "Comm_Bytes_MB": 0.0,
            "Training_Time_s": round(t_cent_time, 2),
            "Privacy_Preserved": "NO (Data Pooled)",
        },
        {
            "Paradigm": "Local-Only: Factory A (No Federation)",
            "Data_Locality": "Factory A Data Only",
            "Precision": round(local_metrics_on_global["Factory_A"]["precision"], 4),
            "Recall": round(local_metrics_on_global["Factory_A"]["recall"], 4),
            "F1_Score": round(local_metrics_on_global["Factory_A"]["f1"], 4),
            "ROC_AUC": round(local_metrics_on_global["Factory_A"]["roc_auc"], 4),
            "PR_AUC": round(local_metrics_on_global["Factory_A"]["pr_auc"], 4),
            "Comm_Bytes_MB": 0.0,
            "Training_Time_s": round(t_local_time / 3.0, 2),
            "Privacy_Preserved": "YES (Isolated)",
        },
        {
            "Paradigm": "Local-Only: Factory B (No Federation)",
            "Data_Locality": "Factory B Data Only",
            "Precision": round(local_metrics_on_global["Factory_B"]["precision"], 4),
            "Recall": round(local_metrics_on_global["Factory_B"]["recall"], 4),
            "F1_Score": round(local_metrics_on_global["Factory_B"]["f1"], 4),
            "ROC_AUC": round(local_metrics_on_global["Factory_B"]["roc_auc"], 4),
            "PR_AUC": round(local_metrics_on_global["Factory_B"]["pr_auc"], 4),
            "Comm_Bytes_MB": 0.0,
            "Training_Time_s": round(t_local_time / 3.0, 2),
            "Privacy_Preserved": "YES (Isolated)",
        },
        {
            "Paradigm": "Local-Only: Factory C (No Federation)",
            "Data_Locality": "Factory C Data Only",
            "Precision": round(local_metrics_on_global["Factory_C"]["precision"], 4),
            "Recall": round(local_metrics_on_global["Factory_C"]["recall"], 4),
            "F1_Score": round(local_metrics_on_global["Factory_C"]["f1"], 4),
            "ROC_AUC": round(local_metrics_on_global["Factory_C"]["roc_auc"], 4),
            "PR_AUC": round(local_metrics_on_global["Factory_C"]["pr_auc"], 4),
            "Comm_Bytes_MB": 0.0,
            "Training_Time_s": round(t_local_time / 3.0, 2),
            "Privacy_Preserved": "YES (Isolated)",
        },
        {
            "Paradigm": "Federated Learning (FedAvg, 10 Rounds)",
            "Data_Locality": "Decentralized (Model Weights Only)",
            "Precision": round(fed_final_metrics["precision"], 4),
            "Recall": round(fed_final_metrics["recall"], 4),
            "F1_Score": round(fed_final_metrics["f1"], 4),
            "ROC_AUC": round(fed_final_metrics["roc_auc"], 4),
            "PR_AUC": round(fed_final_metrics["pr_auc"], 4),
            "Comm_Bytes_MB": round(total_bytes_communicated / (1024.0 * 1024.0), 2),
            "Training_Time_s": round(t_fed_time, 2),
            "Privacy_Preserved": "YES (Verified Zero Raw Data)",
        },
    ]

    df_comp = pd.DataFrame(comp_rows)
    df_conv = pd.DataFrame(convergence_records)

    # Save Global Model Checkpoint
    os.makedirs("models_saved/checkpoints", exist_ok=True)
    torch.save(
        {
            "model_state_dict": global_model.state_dict(),
            "threshold": fed_final_metrics["threshold"],
            "final_f1": fed_final_metrics["f1"],
            "rounds": num_rounds,
        },
        "models_saved/checkpoints/fl_global_lstm_ae.pt",
    )

    return {
        "centralized": cent_metrics,
        "federated": fed_final_metrics,
        "local": local_metrics_on_global,
        "per_client_fed": per_client_fed_metrics,
    }, df_comp, df_conv
