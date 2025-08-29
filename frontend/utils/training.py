

from __future__ import annotations
import os
import json
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Callable
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve, auc

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


@dataclass
class TrainConfig:
    target: str = "loan_status"
    test_size: float = 0.2
    scaler: str = "StandardScaler"
    is_german_credit: bool = False
    estimator: str = "Random Forest"
    est_params: Optional[dict] = None
    training_mode: str = "new"
    existing_model_name: Optional[str] = None
    incremental_lr: Optional[float] = None


def _get_scaler(name: str):
    scalers = {
        "StandardScaler": StandardScaler(),
        "RobustScaler": RobustScaler(),
        "MinMaxScaler": MinMaxScaler()
    }
    return scalers.get(name, StandardScaler())


def load_existing_model(model_name: str) -> Tuple[object, object, Dict[str, object], list]:
    models_dir = Path("models")
    
    model_slug = model_name.lower().replace(" ", "_")
    model_path = models_dir / f"{model_slug}_model.pkl"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = joblib.load(model_path)
    scaler = joblib.load(models_dir / "scaler.pkl")
    encoders = joblib.load(models_dir / "label_encoders.pkl")
    
    with open(models_dir / "feature_names.json", "r") as f:
        feature_names = json.load(f)
    
    return model, scaler, encoders, feature_names


def _encode_target_variable(y_raw: pd.Series, target_name: str) -> Tuple[np.ndarray, LabelEncoder]:
    target_le = LabelEncoder()
    
    y_encoded = target_le.fit_transform(y_raw.astype(str))
    
    unique_classes = target_le.classes_
    if len(unique_classes) != 2:
        raise ValueError(f"Target must have exactly 2 classes. Found: {unique_classes}")
    
    print(f"Target encoding: {dict(zip(unique_classes, [0, 1]))}")
    return y_encoded, target_le


def _encode_features(X_raw: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    encoders = {}
    X_encoded = X_raw.copy()
    
    for col in X_encoded.columns:
        if X_encoded[col].dtype == 'object' or pd.api.types.is_categorical_dtype(X_encoded[col].dtype):
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
            encoders[col] = le
            print(f"Encoded {col}: {dict(zip(le.classes_, range(len(le.classes_))))}")
    
    return X_encoded, encoders


def _encode_features_with_existing_encoders(X_raw: pd.DataFrame, existing_encoders: Dict[str, object]) -> pd.DataFrame:
    X_encoded = X_raw.copy()
    
    for col in X_encoded.columns:
        if col in existing_encoders:
            le = existing_encoders[col]
            unique_vals = X_encoded[col].unique()
            for val in unique_vals:
                if val not in le.classes_:
                    X_encoded[col] = X_encoded[col].replace(val, le.classes_[0])
            X_encoded[col] = le.transform(X_encoded[col].astype(str))
    
    return X_encoded


def train_and_select(
    df: pd.DataFrame,
    config: TrainConfig,
    progress_cb: Optional[Callable[[float, str], None]] = None,
) -> Tuple[object, str, pd.Index, object, Dict[str, object], np.ndarray, np.ndarray, dict, np.ndarray]:
    p = progress_cb or (lambda *_: None)
    p(0.05, "Preprocessing data...")

    if config.target not in df.columns:
        raise ValueError(f"Target column '{config.target}' not found in DataFrame.")

    y_raw = df[config.target]
    X_raw = df.drop(columns=[config.target])
    
    print(f"Dataset shape: {df.shape}")
    print(f"Target distribution: {y_raw.value_counts()}")
    print(f"Features: {list(X_raw.columns)}")

    if config.training_mode == "incremental" and config.existing_model_name:
        p(0.1, f"Loading existing model: {config.existing_model_name}")
        existing_model, existing_scaler, existing_encoders, existing_features = load_existing_model(config.existing_model_name)
        
        target_encoder = existing_encoders.get(f"__TARGET__::{config.target}")
        if target_encoder is None:
            raise ValueError(f"Target encoder not found for {config.target}")
        
        y = target_encoder.transform(y_raw.astype(str))
        
        X_encoded = _encode_features_with_existing_encoders(X_raw, existing_encoders)
        
        X_encoded = X_encoded[existing_features]
        
        scaler = existing_scaler
        encoders = existing_encoders
        
        p(0.2, "Splitting data for incremental learning...")
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y, test_size=config.test_size, random_state=RANDOM_STATE, stratify=y
        )
        
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        p(0.4, f"Continuing training of {config.existing_model_name}...")
        
        if hasattr(existing_model, 'partial_fit'):
            existing_model.partial_fit(X_train_scaled, y_train)
            model = existing_model
        elif hasattr(existing_model, 'warm_start') and hasattr(existing_model, 'n_estimators'):
            original_n_estimators = existing_model.n_estimators
            existing_model.n_estimators += 50
            existing_model.fit(X_train_scaled, y_train)
            model = existing_model
        else:
            print("Model doesn't support incremental learning, retraining with new data...")
            model = existing_model.__class__(**existing_model.get_params())
            model.fit(X_train_scaled, y_train)
        
        model_name = f"{config.existing_model_name} (Updated)"
        
    else:
        y, target_encoder = _encode_target_variable(y_raw, config.target)
        
        X_encoded, feature_encoders = _encode_features(X_raw)
        
        encoders = {**feature_encoders, f"__TARGET__::{config.target}": target_encoder}

        p(0.2, "Splitting and scaling data...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y, test_size=config.test_size, random_state=RANDOM_STATE, stratify=y
        )

        scaler = _get_scaler(config.scaler)
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        p(0.4, f"Training {config.estimator}...")
        
        if config.estimator == "Random Forest":
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        elif config.estimator == "Logistic Regression":
            model = LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=RANDOM_STATE
            )
        elif config.estimator == "Gradient Boosting":
            model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=RANDOM_STATE
            )
        elif config.estimator == "XGBoost":
            model = xgb.XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=RANDOM_STATE,
                eval_metric='logloss',
                use_label_encoder=False
            )
        else:
            raise ValueError(f"Unknown estimator: {config.estimator}")

        model.fit(X_train_scaled, y_train)
        model_name = config.estimator

    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    
    print(f"Prediction probabilities range: {y_prob.min():.3f} to {y_prob.max():.3f}")
    print(f"Prediction distribution: {np.bincount(y_pred)}")

    p(0.8, "Evaluating model...")
    
    accuracy = accuracy_score(y_test, y_pred)
    try:
        roc_auc = roc_auc_score(y_test, y_prob)
    except ValueError:
        roc_auc = float('nan')
    
    pr, rc, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(rc, pr)
    
    metrics = {
        "Accuracy": float(accuracy),
        "ROC AUC": roc_auc,
        "PR AUC": pr_auc,
        "y_test": y_test,
        "y_prob": y_prob,
        "table": pd.DataFrame([{
            "Model": model_name,
            "Accuracy": f"{accuracy:.4f}",
            "ROC AUC": f"{roc_auc:.4f}",
            "PR AUC": f"{pr_auc:.4f}"
        }])
    }
    
    print(f"Model performance: Accuracy={accuracy:.3f}, ROC AUC={roc_auc:.3f}")
    
    p(1.0, "Training complete!")
    return model, model_name, X_encoded.columns, scaler, encoders, X_test_scaled, y_test, metrics, None  # LIME background removed


def save_artifacts(
    out_dir: str, 
    model_name: str, 
    model: object, 
    scaler: object, 
    encoders: dict, 
    features: list, 
    lime_background: np.ndarray | None = None
):
    os.makedirs(out_dir, exist_ok=True)
    slug = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
    
    joblib.dump(model, os.path.join(out_dir, f"{slug}_model.pkl"))
    joblib.dump(scaler, os.path.join(out_dir, "scaler.pkl"))
    joblib.dump(encoders, os.path.join(out_dir, "label_encoders.pkl"))
    
    with open(os.path.join(out_dir, "feature_names.json"), "w", encoding="utf-8") as f:
        json.dump(list(map(str, features)), f, indent=2)
        
    # LIME background saving removed as requested