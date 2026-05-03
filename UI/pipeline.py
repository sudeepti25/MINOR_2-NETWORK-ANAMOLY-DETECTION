"""
pipeline.py
===========
Two-stage anomaly detection pipeline — based on inference_pipeline.ipynb.

Stage 1 : Autoencoder (PyTorch) + XGBoost binary  →  Benign / Anomaly
Stage 2 : XGBoost multi-class                      →  DDoS / PortScan / BruteForce / General

Key design decisions (matching training):
- feat_cols loaded from model2_feature_cols.pkl — used for BOTH models
- Scaler applied only to those feat_cols (however many were saved)
- Stage 2 receives cleaned, unscaled features
- Autoencoder loaded as raw state_dict, dims inferred from weight shapes
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, TensorDataset


# ── Autoencoder — architecture inferred from saved weights ────────────────────
class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, bottleneck: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, bottleneck),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
        )

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        return self.decoder(z), z


# ── Result container ──────────────────────────────────────────────────────────
@dataclass
class PipelineResult:
    predictions: pd.DataFrame
    metrics: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


# ── CIC-IDS-2017 string label → ground truth class ───────────────────────────
CICIDS_LABEL_MAP: Dict[str, str] = {
    "BENIGN"                          : "Benign",
    "DDoS"                            : "DDoS",
    "DoS Hulk"                        : "DDoS",
    "DoS GoldenEye"                   : "DDoS",
    "DoS Slowhttptest"                : "DDoS",
    "DoS slowloris"                   : "DDoS",
    "PortScan"                        : "PortScan",
    "FTP-Patator"                     : "BruteForce",
    "SSH-Patator"                     : "BruteForce",
    "Web Attack \u2013 Brute Force"   : "BruteForce",
    "Bot"                             : "General",
    "Web Attack \u2013 XSS"           : "General",
    "Web Attack \u2013 Sql Injection" : "General",
    "Infiltration"                    : "General",
    "Heartbleed"                      : "General",
}

M2_LABELS: Dict[int, str] = {
    0: "DDoS",
    1: "PortScan",
    2: "BruteForce",
    3: "General",
}


# ── Pipeline ──────────────────────────────────────────────────────────────────
class TwoStageAnomalyPipeline:

    AE_FILE     = "autoencoder.pt"
    M1_FILE     = "model1_binary.ubj"
    SCALER_FILE = "scaler.pkl"
    M2_FILE     = "model2_xgboost.json"
    FEAT_FILE   = "model2_feature_cols.pkl"

    # Exact 47 features the scaler + autoencoder were trained on
    M1_FEATURE_COLS = [
        'Destination Port', 'Flow Duration', 'Total Fwd Packets',
        'Total Length of Fwd Packets', 'Fwd Packet Length Max',
        'Fwd Packet Length Min', 'Fwd Packet Length Mean',
        'Bwd Packet Length Max', 'Bwd Packet Length Min',
        'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean',
        'Flow IAT Std', 'Flow IAT Min', 'Fwd IAT Min',
        'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max',
        'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags',
        'Fwd Header Length', 'Bwd Header Length', 'Bwd Packets/s',
        'Min Packet Length', 'FIN Flag Count', 'RST Flag Count',
        'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count',
        'Down/Up Ratio', 'Fwd Avg Bytes/Bulk', 'Fwd Avg Packets/Bulk',
        'Fwd Avg Bulk Rate', 'Bwd Avg Bytes/Bulk', 'Bwd Avg Packets/Bulk',
        'Bwd Avg Bulk Rate', 'Init_Win_bytes_forward', 'Init_Win_bytes_backward',
        'act_data_pkt_fwd', 'min_seg_size_forward',
        'Active Mean', 'Active Std', 'Active Max', 'Idle Std'
    ]

    def __init__(self, project_root: Path) -> None:
        self.project_root = Path(project_root)
        self.models_dir   = self.project_root / "MODELS"
        self.device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        d = self.models_dir

        # Feature cols — used for BOTH models (same as training)
        self.feat_cols: List[str] = joblib.load(d / self.FEAT_FILE)

        # Scaler — fitted on however many features were used in Model 1 training
        self.scaler = joblib.load(d / self.SCALER_FILE)

        # Autoencoder — raw state_dict, infer dims from weight shapes
        state_dict     = torch.load(d / self.AE_FILE, map_location=self.device)
        input_dim      = state_dict["encoder.0.weight"].shape[1]
        bottleneck     = state_dict["encoder.4.weight"].shape[0]
        self.ae        = Autoencoder(input_dim, bottleneck).to(self.device)
        self.ae.load_state_dict(state_dict)
        self.ae.eval()

        # Model 1 XGBoost binary
        self.model1 = xgb.XGBClassifier()
        self.model1.load_model(str(d / self.M1_FILE))

        # Model 2 XGBoost multi-class
        self.model2 = xgb.XGBClassifier()
        self.model2.load_model(str(d / self.M2_FILE))

    # ── Preprocessing ─────────────────────────────────────────────────────────
    def _preprocess_m1(self, df: pd.DataFrame, warnings: List[str]) -> np.ndarray:
        """Align to 47 M1 features, clean, scale — matches Model 1 training exactly."""
        df = df.copy()
        df.columns = df.columns.str.strip()

        missing = [c for c in self.M1_FEATURE_COLS if c not in df.columns]
        if missing:
            warnings.append(
                f"{len(missing)} Model-1 feature(s) missing, filled with 0: "
                f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
            )
            for c in missing:
                df[c] = 0.0

        X = df[self.M1_FEATURE_COLS].copy()
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X.fillna(X.median(numeric_only=True), inplace=True)
        X = X.clip(-1e6, 1e6)
        return self.scaler.transform(X)   # scaled numpy array for AE + M1 XGBoost

    def _preprocess_m2(self, df: pd.DataFrame, warnings: List[str]) -> np.ndarray:
        """Align to 78 M2 features, clean, unscaled — matches Model 2 training exactly."""
        df = df.copy()
        df.columns = df.columns.str.strip()

        missing = [c for c in self.feat_cols if c not in df.columns]
        if missing:
            warnings.append(
                f"{len(missing)} Model-2 feature(s) missing, filled with 0: "
                f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
            )
            for c in missing:
                df[c] = 0.0

        X = df[self.feat_cols].copy()
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X.fillna(X.median(numeric_only=True), inplace=True)
        X = X.clip(-1e6, 1e6)
        return X.values   # unscaled numpy array for M2 XGBoost

    # ── AE feature extraction ─────────────────────────────────────────────────
    def _extract_ae_features(self, X_scaled: np.ndarray, batch_size: int = 2048):
        """Returns per-feature squared errors + bottleneck vectors."""
        self.ae.eval()
        tensor = torch.tensor(X_scaled, dtype=torch.float32)
        loader = DataLoader(TensorDataset(tensor), batch_size=batch_size, shuffle=False)
        errors_list, z_list = [], []
        with torch.no_grad():
            for (xb,) in loader:
                xb = xb.to(self.device)
                recon, z = self.ae(xb)
                errors_list.append(((xb - recon) ** 2).cpu().numpy())
                z_list.append(z.cpu().numpy())
        return np.vstack(errors_list), np.vstack(z_list)

    @staticmethod
    def _build_m1_input(errors: np.ndarray, z: np.ndarray) -> np.ndarray:
        # Training used: np.concatenate([errors, bottlenecks], axis=1) only
        return np.concatenate([errors, z], axis=1)

    # ── Public run ────────────────────────────────────────────────────────────
    def run(
        self,
        input_df: pd.DataFrame,
        stage1_model_name: str = "xgb_binary",      # for app.py API compat
        stage2_model_name: str = "xgb_multiclass",  # for app.py API compat
        user_class_map: Optional[Dict[int, str]] = None,
    ) -> PipelineResult:

        warnings: List[str] = []
        t0 = time.time()

        # Resolve stage-2 label names
        # app.py sends keys 1-4, model outputs 0-3 → shift down
        if user_class_map:
            if all(k >= 1 for k in user_class_map):
                m2_labels = {k - 1: v for k, v in user_class_map.items()}
            else:
                m2_labels = user_class_map
        else:
            m2_labels = M2_LABELS

        # Strip column whitespace + handle CIC-IDS label col
        df = input_df.copy()
        df.columns = df.columns.str.strip()
        if " Label" in df.columns:
            df.rename(columns={" Label": "Label"}, inplace=True)

        has_label  = "Label" in df.columns
        raw_labels = df["Label"].copy() if has_label else None
        df_feats   = df.drop(columns=["Label"], errors="ignore")
        N = len(df_feats)

        # ── Preprocess ────────────────────────────────────────────────────────
        X_scaled = self._preprocess_m1(df_feats, warnings)   # 47 features, scaled
        X_m2_all = self._preprocess_m2(df_feats, warnings)   # 78 features, unscaled

        # ── Stage 1 ───────────────────────────────────────────────────────────
        errors, z = self._extract_ae_features(X_scaled)
        Xg_m1     = self._build_m1_input(errors, z)
        m1_pred   = self.model1.predict(Xg_m1)             # 0=benign, 1=anomaly
        m1_proba  = self.model1.predict_proba(Xg_m1)[:, 1]

        # ── Stage 2 (anomalies only) ──────────────────────────────────────────
        anomaly_mask = m1_pred == 1
        n_anomalies  = int(anomaly_mask.sum())

        m2_class_id = np.full(N, -1, dtype=int)
        m2_class    = np.array(["Benign"] * N, dtype=object)

        if n_anomalies > 0:
            X_m2          = X_m2_all[anomaly_mask]             # 78 features, unscaled
            preds_m2      = self.model2.predict(X_m2)          # 0-3
            m2_class_id[anomaly_mask] = preds_m2
            m2_class[anomaly_mask]    = [
                m2_labels.get(int(p), f"Class_{p}") for p in preds_m2
            ]
        else:
            warnings.append("No anomalies detected — stage-2 was skipped.")

        final_prediction = np.where(anomaly_mask, m2_class, "Benign")

        # ── Build result DataFrame ────────────────────────────────────────────
        preds_df = pd.DataFrame({
            "stage1_anomaly_score"  : np.round(m1_proba, 4),
            "stage1_is_anomaly"     : m1_pred.astype(int),
            "stage2_attack_class_id": m2_class_id,
            "stage2_attack_class"   : m2_class,
            "final_prediction"      : final_prediction,
        })

        if has_label:
            preds_df.insert(0, "Label", raw_labels.values)

        # ── Metrics ───────────────────────────────────────────────────────────
        elapsed = time.time() - t0
        metrics: Dict[str, float] = {
            "total_records"     : float(N),
            "anomalies_detected": float(n_anomalies),
            "anomaly_rate"      : float(n_anomalies / N) if N > 0 else 0.0,
            "elapsed_seconds"   : round(elapsed, 2),
        }

        if has_label:
            y_true = raw_labels.copy()
            if y_true.dtype == object:
                mapped = y_true.map(lambda x: CICIDS_LABEL_MAP.get(str(x).strip(), "General"))
                y_bin  = (mapped != "Benign").astype(int)
            else:
                y_bin = (y_true != 0).astype(int)
            try:
                metrics["stage1_accuracy"]  = float(accuracy_score(y_bin, m1_pred))
                metrics["stage1_precision"] = float(precision_score(y_bin, m1_pred, zero_division=0))
                metrics["stage1_recall"]    = float(recall_score(y_bin, m1_pred, zero_division=0))
                metrics["stage1_f1"]        = float(f1_score(y_bin, m1_pred, zero_division=0))
            except Exception as e:
                warnings.append(f"Could not compute stage-1 metrics: {e}")

        return PipelineResult(predictions=preds_df, metrics=metrics, warnings=warnings)
