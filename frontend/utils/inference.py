import os, json, warnings, joblib
import numpy as np
import pandas as pd
import shap
import lime.lime_tabular as ltt

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

SAFE_LIME_NUM_SAMPLES = 1500
SAFE_LIME_MAX_BG = 3000

def _load_paths(model_name: str):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.abspath(os.path.join(script_dir, ".."))
    models_dir = os.path.join(base_path, "models")
    return {
        "MODELS_DIR":   models_dir,
        "MODEL_PATH":   os.path.join(models_dir, f"{model_name}_model.pkl"),
        "ENCODERS_PATH":os.path.join(models_dir, "label_encoders.pkl"),
        "SCALER_PATH":  os.path.join(models_dir, "standard_scaler.pkl"),
        "COLUMNS_PATH": os.path.join(models_dir, "feature_names.json"),
        "BACKGROUND":   os.path.join(models_dir, "lime_background.npy"),
        "DATA_PATH":    os.path.join(base_path, "data", "german.data"),
    }

def _safe_load_artifacts(paths):
    try:
        model = joblib.load(paths["MODEL_PATH"])
        label_encoders = joblib.load(paths["ENCODERS_PATH"])
        scaler = joblib.load(paths["SCALER_PATH"])
        with open(paths["COLUMNS_PATH"], "r", encoding="utf-8") as f:
            feature_columns = json.load(f)
        return model, label_encoders, scaler, feature_columns, None
    except Exception as e:
        return None, None, None, None, f"Fehler beim Laden der Modelldateien: {e}"

def _align_input_row(applicant_data: dict, feature_columns: list, label_encoders: dict) -> pd.DataFrame:
    row = {}
    for col in feature_columns:
        if col in label_encoders:
            enc = label_encoders[col]
            val = str(applicant_data.get(col, enc.classes_[0]))
            if val not in enc.classes_:
                val = enc.classes_[0]
            row[col] = enc.transform([val])[0]
        else:
            v = pd.to_numeric(applicant_data.get(col, 0), errors="coerce")
            row[col] = 0 if pd.isna(v) else float(v)
    return pd.DataFrame([row], columns=feature_columns)

def _prepare_lime_background(paths, feature_columns, label_encoders, scaler):
    try:
        if os.path.exists(paths["BACKGROUND"]):
            X_bg = np.load(paths["BACKGROUND"])
            if X_bg.ndim == 2 and X_bg.shape[1] == len(feature_columns):
                if X_bg.shape[0] > SAFE_LIME_MAX_BG:
                    idx = np.random.RandomState(42).choice(X_bg.shape[0], SAFE_LIME_MAX_BG, replace=False)
                    X_bg = X_bg[idx]
                return X_bg.astype(np.float64, copy=False)
    except Exception:
        pass

    try:
        if not os.path.exists(paths["DATA_PATH"]):
            return None
        raw = pd.read_csv(paths["DATA_PATH"], sep=r"\s+", header=None, dtype=str)
        n_feat = len(feature_columns)
        if raw.shape[1] < n_feat:
            return None
        X_raw = raw.iloc[:, :n_feat].copy()

        X_enc = pd.DataFrame(index=X_raw.index, columns=feature_columns)
        for i, col in enumerate(feature_columns):
            s = X_raw.iloc[:, i]
            if col in label_encoders:
                enc = label_encoders[col]
                s = s.where(s.isin(enc.classes_), enc.classes_[0])
                X_enc[col] = enc.transform(s.astype(str))
            else:
                X_enc[col] = pd.to_numeric(s, errors="coerce").fillna(0)

        X_enc = X_enc.astype(float)
        X_bg = scaler.transform(X_enc)
        if X_bg.shape[0] > SAFE_LIME_MAX_BG:
            idx = np.random.RandomState(42).choice(X_bg.shape[0], SAFE_LIME_MAX_BG, replace=False)
            X_bg = X_bg[idx]
        return X_bg.astype(np.float64, copy=False)
    except Exception:
        return None

def _class_indices_for_good_bad(model):
    classes = getattr(model, "classes_", None)
    if classes is None or len(classes) != 2:
        return 0, 1, np.array([0, 1])
    if "Good" in classes and "Bad" in classes:
        good_idx = np.where(classes == "Good")[0][0]
        bad_idx = np.where(classes == "Bad")[0][0]
        return good_idx, bad_idx, classes
    if "0" in classes and "1" in classes:
        good_idx = np.where(classes == "0")[0][0]
        bad_idx = np.where(classes == "1")[0][0]
        return good_idx, bad_idx, classes
    return 0, 1, classes

def _get_predict_proba_fn(model, feature_columns):
    def predict_fn(X):
        Xdf = pd.DataFrame(X, columns=feature_columns)
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

def run_inference(applicant_data: dict, model_name: str = "random_forest"):
    paths = _load_paths(model_name)
    model, label_encoders, scaler, feature_columns, error = _safe_load_artifacts(paths)
    
    if error:
        return {"error": error}
    
    X_input = _align_input_row(applicant_data, feature_columns, label_encoders)
    X_scaled = scaler.transform(X_input)
    
    good_idx, bad_idx, classes = _class_indices_for_good_bad(model)
    
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_scaled)[0]
    else:
        proba = _get_predict_proba_fn(model, feature_columns)(X_scaled)[0]
    
    pred_idx = np.argmax(proba)
    prediction = "Good" if pred_idx == good_idx else "Bad"
    
    result = {
        "prediction": prediction,
        "probability_good": float(proba[good_idx]),
        "probability_bad": float(proba[bad_idx]),
        "confidence": float(max(proba))
    }
    
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_scaled)
        result["shap_explanation_object"] = shap_values
    except Exception:
        result["shap_explanation_object"] = None
    
    try:
        X_bg = _prepare_lime_background(paths, feature_columns, label_encoders, scaler)
        if X_bg is not None:
            explainer = ltt.LimeTabularExplainer(
                training_data=X_bg,
                feature_names=feature_columns,
                class_names=classes,
                mode="classification",
                discretize_continuous=False,
                random_state=42,
            )
            predict_fn = _get_predict_proba_fn(model, feature_columns)
            explanation = explainer.explain_instance(
                data_row=X_scaled[0],
                predict_fn=predict_fn,
                num_features=min(12, len(feature_columns)),
            )
            result["lime_explanation"] = explanation
        else:
            result["lime_explanation"] = None
    except Exception:
        result["lime_explanation"] = None
    
    return result
