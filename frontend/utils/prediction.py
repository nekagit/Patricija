from __future__ import annotations

import os
import json
import warnings
from typing import Dict, Tuple, Optional, Any, List
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import shap
# LIME import removed as requested

from utils.logger import get_logger, log_performance, log_errors
from utils.cache import cache_prediction, get_model_cache, hash_input_data

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logger = get_logger(__name__)

class PredictionError(Exception):
    pass

class ModelLoader:
    
    def __init__(self, models_dir: str = None):
        if models_dir is None:
            models_dir = "models"
        self.models_dir = Path(models_dir)
        self.cache = get_model_cache()
        self._loaded_artifacts = {}
    
    @log_performance()
    def load_model_artifacts(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        cache_key = f"artifacts_{model_name or 'default'}"
        
        cached = self.cache.get_model(cache_key)
        if cached is not None:
            logger.debug(f"Using cached artifacts for {model_name or 'default'}")
            return cached
        
        artifacts = self._load_artifacts_from_disk(model_name)
        
        self.cache.put_model(cache_key, artifacts)
        
        return artifacts
    
    def _load_artifacts_from_disk(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        artifacts = {}
        
        model_path, pretty_name = self._resolve_model_path(model_name)
        if not model_path:
            raise PredictionError("No model found")
        
        artifacts["model_path"] = str(model_path)
        artifacts["model_pretty"] = pretty_name
        
        try:
            artifacts["model"] = joblib.load(model_path)
            logger.info(f"Loaded model: {pretty_name}")
        except Exception as e:
            raise PredictionError(f"Failed to load model: {e}")
        
        try:
            scaler_path = self.models_dir / "scaler.pkl"
            artifacts["scaler"] = joblib.load(scaler_path)
        except Exception as e:
            raise PredictionError(f"Failed to load scaler: {e}")
        
        try:
            encoders_path = self.models_dir / "label_encoders.pkl"
            artifacts["encoders"] = joblib.load(encoders_path)
        except Exception:
            artifacts["encoders"] = {}
            logger.warning("No label encoders found, proceeding without them")
        
        try:
            feature_names_path = self.models_dir / "feature_names.json"
            with open(feature_names_path, "r", encoding="utf-8") as f:
                artifacts["feature_names"] = json.load(f)
        except Exception as e:
            raise PredictionError(f"Failed to load feature names: {e}")
        
        # LIME background data removed as requested
        artifacts["lime_bg"] = None
        
        return artifacts
    
    def _resolve_model_path(self, model_name: Optional[str] = None) -> Tuple[Optional[Path], Optional[str]]:
        if model_name:
            slug = self._slug(model_name)
            model_path = self.models_dir / f"{slug}_model.pkl"
            if model_path.exists():
                return model_path, model_name
            else:
                logger.warning(f"Model file not found: {model_path}")
        
        # Fallback: find any available model
        for model_file in sorted(self.models_dir.glob("*_model.pkl")):
            pretty_name = model_file.stem.replace("_model", "").replace("_", " ").title()
            return model_file, pretty_name
        
        return None, None
    
    def _slug(self, s: str) -> str:
        return s.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")

class DataPreprocessor:
    
    @staticmethod
    @log_performance()
    def preprocess_input(
        applicant_data: Dict[str, Any],
        feature_columns: List[str],
        label_encoders: Dict[str, Any]
    ) -> pd.DataFrame:
        row = {}
        target_keys = {k for k in label_encoders if str(k).startswith("__TARGET__::")}
        
        for col in feature_columns:
            val_raw = applicant_data.get(col)
            
            if col in label_encoders and col not in target_keys:
                row[col] = DataPreprocessor._encode_categorical(
                    val_raw, label_encoders[col]
                )
            else:
                row[col] = DataPreprocessor._encode_numerical(val_raw)
        
        return pd.DataFrame([row], columns=feature_columns)
    
    @staticmethod
    def _encode_categorical(val_raw: Any, encoder: Any) -> int:
        try:
            val_str = str(val_raw) if val_raw is not None else str(getattr(encoder, "classes_", [""])[0])
            classes = list(getattr(encoder, "classes_", []))
            
            if classes and val_str not in classes:
                val_str = classes[0]
            
            return encoder.transform([val_str])[0]
        except Exception:
            return 0
    
    @staticmethod
    def _encode_numerical(val_raw: Any) -> float:
        try:
            v = pd.to_numeric(val_raw, errors="coerce")
            return 0.0 if pd.isna(v) else float(v)
        except Exception:
            return 0.0

class PredictionEngine:
    
    def __init__(self):
        self.model_loader = ModelLoader()
        self.preprocessor = DataPreprocessor()
        self.cache = get_model_cache()
    
    @cache_prediction()
    @log_performance()
    @log_errors()
    def predict(self, applicant_data: Dict[str, Any], model_name: Optional[str] = None) -> Dict[str, Any]:
        logger.info(f"Starting prediction for model: {model_name or 'default'}")
        
        artifacts = self.model_loader.load_model_artifacts(model_name)
        
        input_df = self.preprocessor.preprocess_input(
            applicant_data,
            artifacts["feature_names"],
            artifacts["encoders"]
        )
        
        input_scaled = artifacts["scaler"].transform(input_df)
        input_scaled_df = pd.DataFrame(input_scaled, columns=artifacts["feature_names"])
        
        prediction_result = self._make_prediction(
            artifacts["model"], input_scaled_df, artifacts["feature_names"]
        )
        
        explanations = self._generate_explanations(
            artifacts["model"], input_scaled_df, artifacts["feature_names"], artifacts["lime_bg"]
        )
        
        result = {
            **prediction_result,
            **explanations,
            "feature_names": artifacts["feature_names"],
            "model_name": artifacts["model_pretty"],
            "input_hash": hash_input_data(applicant_data)
        }
        
        logger.info(f"Prediction completed successfully: {result['prediction']}")
        return result
    
    def _make_prediction(self, model: Any, input_scaled_df: pd.DataFrame, feature_names: List[str]) -> Dict[str, Any]:
        predict_fn = self._get_predict_proba_fn(model, feature_names)
        proba = predict_fn(input_scaled_df.values)[0]
        
        pred_idx = np.argmax(proba)
        classes = self._get_class_names(model)
        
        good_idx, bad_idx = self._get_class_indices(model)
        prediction_text = "Bad" if pred_idx == bad_idx else "Good"
        
        return {
            "prediction": prediction_text,
            "probability_good": float(proba[good_idx]),
            "probability_bad": float(proba[bad_idx]),
            "confidence": float(max(proba)),
            "class_names": classes,
            "good_class_index": int(good_idx),
            "bad_class_index": int(bad_idx)
        }
    
    def _generate_explanations(
        self, model: Any, input_scaled_df: pd.DataFrame, 
        feature_names: List[str], lime_bg: Optional[np.ndarray]
    ) -> Dict[str, Any]:
        explanations = {}
        
        try:
            shap_explanation = self._build_shap_explanation(model, input_scaled_df, lime_bg)
            explanations["shap_explanation_object"] = shap_explanation
        except Exception as e:
            logger.warning(f"SHAP explanation failed: {e}")
            explanations["shap_explanation_object"] = None
        
        # LIME explanation generation removed as requested
        explanations["lime_explanation"] = None
        
        return explanations
    
    def _get_predict_proba_fn(self, model: Any, feature_names: List[str]):
        def predict_fn(X: np.ndarray) -> np.ndarray:
            Xdf = pd.DataFrame(X, columns=feature_names)
            
            if hasattr(model, "predict_proba"):
                P = np.asarray(model.predict_proba(Xdf))
                if P.ndim == 1:
                    return np.c_[1 - P, P]
                if P.shape[1] == 1:
                    return np.c_[1 - P[:, 0], P[:, 0]]
                return P[:, :2]
            
            s = model.decision_function(Xdf).astype(float).ravel()
            s_min, s_max = np.min(s), np.max(s)
            p1 = (s - s_min) / (s_max - s_min + 1e-9) if s_max > s_min else np.zeros_like(s)
            return np.c_[1 - p1, p1]
        
        return predict_fn
    
    def _get_class_names(self, model: Any) -> List[str]:
        return [str(c) for c in getattr(model, "classes_", ['0', '1'])]
    
    def _get_class_indices(self, model: Any, good_class_name: str = "Good", bad_class_name: str = "Bad") -> Tuple[int, int]:
        classes = self._get_class_names(model)
        
        try:
            good_idx = classes.index(good_class_name)
            bad_idx = classes.index(bad_class_name)
            return good_idx, bad_idx
        except ValueError:
            try:
                good_idx = classes.index('0')
                bad_idx = classes.index('1')
                return good_idx, bad_idx
            except ValueError:
                return 0, 1
    
    def _build_shap_explanation(self, model: Any, input_scaled_df: pd.DataFrame, bg_scaled: Optional[np.ndarray]):
        try:
            explainer = shap.TreeExplainer(model)
            return explainer(input_scaled_df)
        except Exception:
            try:
                background = shap.sample(bg_scaled, 100) if bg_scaled is not None and len(bg_scaled) > 100 else bg_scaled
                if background is None:
                    return None
                explainer = shap.KernelExplainer(model.predict_proba, background)
                return explainer(input_scaled_df)
            except Exception:
                return None
    
    def _build_lime_explanation(
        self, model: Any, feature_names: List[str], 
        input_scaled_row: np.ndarray, lime_bg: Optional[np.ndarray]
    ) -> Tuple[Optional[Any], Optional[str]]:
        # LIME explanation functionality removed as requested
        return None, "LIME explanations are not available"

prediction_engine = PredictionEngine()

def run_predictions(applicant_data: Dict[str, Any], model_name: Optional[str] = None) -> Dict[str, Any]:
    try:
        return prediction_engine.predict(applicant_data, model_name)
    except PredictionError as e:
        logger.error(f"Prediction error: {e}")
        return {"error": str(e)}
    except Exception as e:
        logger.error(f"Unexpected error during prediction: {e}")
        return {"error": f"Unexpected error: {e}"}
