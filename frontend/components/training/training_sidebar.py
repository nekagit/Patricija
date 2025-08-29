from __future__ import annotations
import streamlit as st
from typing import Any, Dict
import os
from pathlib import Path

def get_available_models() -> list[str]:
    models_dir = Path("models")
    if not models_dir.exists():
        return []
    
    model_files = list(models_dir.glob("*_model.pkl"))
    model_names = [f.stem.replace("_model", "").replace("_", " ").title() for f in model_files]
    return model_names

def render_sidebar(context: str = "default") -> Dict[str, Any]:
    with st.sidebar:
        st.header("⚙️ Training Einstellungen")

        training_mode = st.radio(
            "Training Modus",
            ["Neues Modell trainieren", "Bestehendes Modell weiter trainieren"],
            help="Wähle zwischen einem neuen Modell oder dem Weiterlernen eines bestehenden Modells.",
            key=f"training_mode_radio_{context}"
        )

        existing_models = get_available_models()
        selected_existing_model = None
        
        if training_mode == "Bestehendes Modell weiter trainieren":
            if not existing_models:
                st.warning("Keine trainierten Modelle gefunden. Trainiere zuerst ein neues Modell.")
                training_mode = "Neues Modell trainieren"
            else:
                selected_existing_model = st.selectbox(
                    "Bestehendes Modell auswählen",
                    existing_models,
                    help="Wähle ein bereits trainiertes Modell aus.",
                    key=f"existing_model_select_{context}"
                )
                
                if selected_existing_model:
                    st.info(f"Gewähltes Modell: {selected_existing_model}")
                    
                    incremental_lr = st.slider(
                        "Lernrate für Weiterlernen",
                        0.001, 0.1, 0.01,
                        help="Kleinere Lernrate für sanftes Weiterlernen",
                        key=f"incremental_lr_slider_{context}"
                    )

        if training_mode == "Neues Modell trainieren":
            estimator = st.selectbox(
                "Modell",
                ["Random Forest", "Logistic Regression", "Gradient Boosting"],
                index=0,
                help="Wähle den Klassifikator.",
                key=f"estimator_select_{context}"
            )
        else:
            estimator = "Incremental"

        test_size = st.slider("Testset-Größe", 0.1, 0.4, 0.2, help="Anteil der Testmenge.", key=f"test_size_slider_{context}")
        seed = st.number_input("Zufalls-Seed", min_value=0, max_value=999_999, value=42, step=1, key=f"seed_input_{context}")

        try:
            from utils.dataset_compatibility import get_target_variable_name
            default_target = get_target_variable_name()
        except ImportError:
            default_target = "loan_status"

        target_col = st.text_input("Zielspalte", value=default_target, key=f"target_col_input_{context}")
        
        scaler = st.selectbox("Skalierer", ["StandardScaler", "RobustScaler", "MinMaxScaler"], index=0, key=f"scaler_select_{context}")

        model_params: Dict[str, Any] = {}
        rf_params: Dict[str, Any] | None = None

        if training_mode == "Neues Modell trainieren":
            if estimator == "Random Forest":
                n_estimators = st.slider("Anzahl Bäume", 50, 200, 100, step=10, key=f"rf_n_estimators_{context}")
                max_depth = st.slider("Max. Tiefe", 5, 20, 10, step=1, key=f"rf_max_depth_{context}")

                rf_params = {
                    "n_estimators": int(n_estimators),
                    "max_depth": int(max_depth),
                    "random_state": int(seed),
                    "n_jobs": -1,
                }

            elif estimator == "Logistic Regression":
                C = st.slider("C (Regularisierung)", 0.1, 10.0, 1.0, key=f"lr_C_{context}")
                model_params = {
                    "C": float(C),
                    "max_iter": 1000,
                    "solver": "lbfgs",
                    "random_state": int(seed),
                }

            elif estimator == "Gradient Boosting":
                n_estimators = st.slider("Anzahl Bäume", 50, 200, 100, step=10, key=f"gb_n_estimators_{context}")
                learning_rate = st.slider("Lernrate", 0.01, 0.3, 0.1, key=f"gb_learning_rate_{context}")
                model_params = {
                    "n_estimators": int(n_estimators),
                    "learning_rate": float(learning_rate),
                    "max_depth": 5,
                    "random_state": int(seed),
                }

        # Simplified configuration - removed advanced options
        auto_detect = True              
        german_credit_flag = False      
        out_dir = "models"           
        show_cm = True                  
        show_roc = True
        show_pr = True

        return {
            "training_mode": training_mode,
            "selected_existing_model": selected_existing_model,
            "incremental_lr": incremental_lr if training_mode == "Bestehendes Modell weiter trainieren" else None,
            "auto_detect": auto_detect,
            "german_credit_flag": german_credit_flag,
            "target_col": target_col,
            "scaler": scaler,
            "test_size": float(test_size),
            "out_dir": out_dir,
            "estimator": estimator,
            "rf_params": rf_params,
            "model_params": model_params,
            "seed": int(seed),
            "show_cm": show_cm,
            "show_roc": show_roc,
            "show_pr": show_pr,
        }
