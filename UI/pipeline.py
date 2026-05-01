from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


@dataclass
class PipelineArtifacts:
    autoencoder: Any
    encoder: Any
    scaler: Any
    feature_columns: List[str]
    stage1_models: Dict[str, Any]
    stage2_models: Dict[str, Any]


@dataclass
class PipelineResult:
    predictions: pd.DataFrame
    metrics: Dict[str, float]
    warnings: List[str]


class TwoStageAnomalyPipeline:
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.models_dir = self.project_root / "MODELS"
        self.artifacts = self._load_artifacts()

    def _load_artifacts(self) -> PipelineArtifacts:
        keras_module = importlib.import_module("tensorflow.keras")
        keras_models = importlib.import_module("tensorflow.keras.models")
        Model = getattr(keras_module, "Model")
        load_model = getattr(keras_models, "load_model")

        autoencoder = load_model(self.models_dir / "autoencoder.h5", compile=False)

        # Encoded feature extraction was trained from this latent layer in notebook flow.
        latent_idx = 3 if len(autoencoder.layers) > 3 else max(1, len(autoencoder.layers) // 2)
        encoder = Model(inputs=autoencoder.input, outputs=autoencoder.layers[latent_idx].output)

        scaler = joblib.load(self.models_dir / "scaler.pkl")
        feature_columns = list(joblib.load(self.models_dir / "feature_columns.pkl"))

        stage1_models = {
            "xgboost": joblib.load(self.models_dir / "xgboost_model.pkl"),
            "randomforest": joblib.load(self.models_dir / "rf_model.pkl"),
        }

        stage2_models = {
            "xgboost_stage2": joblib.load(self.models_dir / "xgboost_stage2.pkl"),
            # "randomforest_stage2": joblib.load(self.models_dir / "randomforest_stage2.pkl"),
        }

        return PipelineArtifacts(
            autoencoder=autoencoder,
            encoder=encoder,
            scaler=scaler,
            feature_columns=feature_columns,
            stage1_models=stage1_models,
            stage2_models=stage2_models,
        )

    @staticmethod
    def _is_model_fitted(model: Any) -> bool:
        fitted_markers = ["n_features_in_", "classes_", "estimators_", "_Booster", "booster_"]
        return any(hasattr(model, marker) for marker in fitted_markers)

    @staticmethod
    def _to_numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
        num_df = df.apply(pd.to_numeric, errors="coerce")
        return num_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def _prepare_scaled_features(self, raw_features: pd.DataFrame) -> pd.DataFrame:
        scaler = self.artifacts.scaler
        expected_cols = list(getattr(scaler, "feature_names_in_", raw_features.columns))

        prepared = raw_features.copy()
        for col in expected_cols:
            if col not in prepared.columns:
                prepared[col] = 0.0

        prepared = prepared[expected_cols]
        prepared = prepared.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        scaled = scaler.transform(prepared)
        return pd.DataFrame(scaled, columns=expected_cols, index=prepared.index)

    @staticmethod
    def _align_for_model(features: pd.DataFrame, model: Any) -> pd.DataFrame:
        if hasattr(model, "feature_names_in_"):
            expected = list(model.feature_names_in_)
            aligned = features.copy()
            for col in expected:
                if col not in aligned.columns:
                    aligned[col] = 0.0
            return aligned[expected]

        if hasattr(model, "n_features_in_"):
            n = int(model.n_features_in_)
            if features.shape[1] == n:
                return features
            if features.shape[1] > n:
                return features.iloc[:, :n]

            aligned = features.copy()
            for i in range(features.shape[1], n):
                aligned[f"missing_{i}"] = 0.0
            return aligned

        return features

    @staticmethod
    def _extract_binary_scores(model: Any, x: pd.DataFrame) -> np.ndarray:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(x)
            if proba.ndim == 2:
                classes = list(getattr(model, "classes_", np.arange(proba.shape[1])))
                if 1 in classes:
                    pos_idx = classes.index(1)
                else:
                    pos_idx = proba.shape[1] - 1
                return proba[:, pos_idx]
            return np.asarray(proba).reshape(-1)

        preds = np.asarray(model.predict(x)).reshape(-1)
        return preds.astype(float)

    @staticmethod
    def _coerce_binary(preds: np.ndarray) -> np.ndarray:
        arr = np.asarray(preds).reshape(-1)
        if np.issubdtype(arr.dtype, np.number):
            return (arr.astype(float) > 0).astype(int)
        return np.array([1 if str(v).lower() in {"1", "true", "anomaly"} else 0 for v in arr], dtype=int)

    @staticmethod
    def _default_attack_label(class_id: Any) -> str:
        return f"Attack Class {class_id}"

    @staticmethod
    def _parse_user_class_map(user_class_map: Optional[Dict[Any, Any]]) -> Dict[int, str]:
        if not user_class_map:
            return {}
        parsed: Dict[int, str] = {}
        for k, v in user_class_map.items():
            try:
                parsed[int(k)] = str(v)
            except Exception:
                continue
        return parsed

    def run(
        self,
        input_df: pd.DataFrame,
        stage1_model_name: str = "xgboost",
        stage2_model_name: str = "xgboost_stage2",
        user_class_map: Optional[Dict[Any, Any]] = None,
    ) -> PipelineResult:
        warnings: List[str] = []

        df = input_df.copy()
        has_label = "Label" in df.columns

        raw_x = df.drop(columns=["Label"], errors="ignore")
        raw_x = self._to_numeric_frame(raw_x)

        scaled_x = self._prepare_scaled_features(raw_x)

        encoded = self.artifacts.encoder.predict(scaled_x.values, verbose=0)
        encoded_df = pd.DataFrame(encoded, columns=self.artifacts.feature_columns, index=scaled_x.index)

        stage1_model = self.artifacts.stage1_models[stage1_model_name]
        stage1_input = self._align_for_model(encoded_df, stage1_model)

        stage1_scores = self._extract_binary_scores(stage1_model, stage1_input)
        stage1_pred_raw = stage1_model.predict(stage1_input)
        stage1_flags = self._coerce_binary(stage1_pred_raw)

        result = df.copy()
        result["stage1_model"] = stage1_model_name
        result["stage1_anomaly_score"] = stage1_scores
        result["stage1_is_anomaly"] = stage1_flags

        result["stage2_model"] = stage2_model_name
        result["stage2_attack_class_id"] = pd.Series([np.nan] * len(result), index=result.index, dtype="float")
        result["stage2_attack_class"] = "Benign"
        result["stage2_confidence"] = pd.Series([np.nan] * len(result), index=result.index, dtype="float")

        anomaly_mask = result["stage1_is_anomaly"] == 1
        anomaly_count = int(anomaly_mask.sum())

        stage2_model = self.artifacts.stage2_models[stage2_model_name]
        class_map = self._parse_user_class_map(user_class_map)

        if anomaly_count > 0:
            if not self._is_model_fitted(stage2_model):
                warnings.append(
                    "Stage-2 model is currently not fitted in MODELS. Replace with a trained stage-2 PKL to enable attack-type classification."
                )
                result.loc[anomaly_mask, "stage2_attack_class"] = "Anomaly (Type unavailable)"
            else:
                if hasattr(stage2_model, "n_features_in_") and int(stage2_model.n_features_in_) == encoded_df.shape[1]:
                    stage2_source = encoded_df
                else:
                    stage2_source = scaled_x

                stage2_input = self._align_for_model(stage2_source, stage2_model)
                stage2_input = stage2_input.loc[anomaly_mask]

                try:
                    stage2_pred = np.asarray(stage2_model.predict(stage2_input)).reshape(-1)
                    result.loc[anomaly_mask, "stage2_attack_class_id"] = stage2_pred

                    if hasattr(stage2_model, "predict_proba"):
                        stage2_proba = stage2_model.predict_proba(stage2_input)
                        result.loc[anomaly_mask, "stage2_confidence"] = np.max(stage2_proba, axis=1)

                    mapped_labels = []
                    for cid in stage2_pred:
                        try:
                            cid_int = int(cid)
                        except Exception:
                            mapped_labels.append(str(cid))
                            continue
                        mapped_labels.append(class_map.get(cid_int, self._default_attack_label(cid_int)))

                    result.loc[anomaly_mask, "stage2_attack_class"] = mapped_labels
                except Exception as exc:
                    warnings.append(f"Stage-2 prediction failed: {exc}")
                    result.loc[anomaly_mask, "stage2_attack_class"] = "Anomaly (Type unavailable)"

        result["final_prediction"] = np.where(
            result["stage1_is_anomaly"] == 1,
            result["stage2_attack_class"],
            "Benign",
        )

        metrics: Dict[str, float] = {
            "total_records": float(len(result)),
            "anomalies_detected": float(anomaly_count),
            "anomaly_rate": float((anomaly_count / len(result)) if len(result) else 0.0),
        }

        if has_label:
            y_true_bin = (pd.to_numeric(result["Label"], errors="coerce").fillna(0).astype(int) != 0).astype(int)
            y_pred_bin = result["stage1_is_anomaly"].astype(int)

            metrics.update(
                {
                    "stage1_accuracy": float(accuracy_score(y_true_bin, y_pred_bin)),
                    "stage1_precision": float(precision_score(y_true_bin, y_pred_bin, zero_division=0)),
                    "stage1_recall": float(recall_score(y_true_bin, y_pred_bin, zero_division=0)),
                    "stage1_f1": float(f1_score(y_true_bin, y_pred_bin, zero_division=0)),
                }
            )

        return PipelineResult(predictions=result, metrics=metrics, warnings=warnings)
