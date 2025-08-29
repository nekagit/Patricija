from __future__ import annotations
import logging
import queue
from pathlib import Path
import streamlit as st
import pandas as pd

from utils.visualizations import theme_css, render_eval_plots, render_single_result_table, render_feature_importance
from utils.training import TrainConfig
from utils.dataset_compatibility import clean_column_names, get_available_target_variables

from .training_sidebar import render_sidebar
from .training_upload import upload_and_preview
from .training_worker import launch_training_and_saving


def training_page():
    st.markdown(theme_css(), unsafe_allow_html=True)
    
    st.markdown("""
        <div class="main-title text-center text-secondary mb-4 animate-fade-in">
            <h1>⚙️ Modell-Training</h1>
            <p>Intelligente Modellentwicklung und -optimierung</p>
        </div>
    """, unsafe_allow_html=True)

    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Konfiguration", 
        "📊 Datenvorschau", 
        "🚀 Training", 
        "📈 Ergebnisse"
    ])
    
    with tab1:
        config_tab()
    
    with tab2:
        data_tab()
    
    with tab3:
        training_tab()
    
    with tab4:
        results_tab()

def config_tab():
    """Configuration tab with training settings."""
    st.markdown("### ⚙️ Trainingskonfiguration")
    
    cfg = render_sidebar("config")
    
    # Store configuration in session state for consistency across tabs
    st.session_state.training_config = cfg
    
    # Display configuration summary
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📋 Trainingsmodus")
        if cfg["training_mode"] == "Bestehendes Modell weiter trainieren":
            st.success(f"🔄 **Inkrementelles Lernen**: Modell '{cfg['selected_existing_model']}' wird mit neuen Daten weiter trainiert.")
        else:
            st.info("🆕 **Neues Modell**: Ein komplett neues Modell wird von Grund auf trainiert.")
    
    with col2:
        st.markdown("#### 🎯 Modellparameter")
        st.write(f"**Zielvariable:** {cfg['target_col']}")
        st.write(f"**Test-Größe:** {cfg['test_size']:.1%}")
        st.write(f"**Skalierung:** {cfg['scaler']}")
        if cfg["training_mode"] == "Bestehendes Modell weiter trainieren":
            st.write(f"**Lernrate:** {cfg['incremental_lr']}")
        else:
            # Show model-specific parameters
            if cfg['estimator'] == "Random Forest" and cfg.get('rf_params'):
                st.write(f"**Bäume:** {cfg['rf_params'].get('n_estimators', 'N/A')}")
                st.write(f"**Max. Tiefe:** {cfg['rf_params'].get('max_depth', 'N/A')}")
            elif cfg['estimator'] == "Gradient Boosting" and cfg.get('model_params'):
                st.write(f"**Bäume:** {cfg['model_params'].get('n_estimators', 'N/A')}")
                st.write(f"**Lernrate:** {cfg['model_params'].get('learning_rate', 'N/A')}")
            elif cfg['estimator'] == "Logistic Regression" and cfg.get('model_params'):
                st.write(f"**C (Regularisierung):** {cfg['model_params'].get('C', 'N/A')}")
    
    return cfg

def data_tab():
    """Data preview tab."""
    st.markdown("### 📊 Datenvorschau")
    
    # Get configuration for data loading
    cfg = render_sidebar("data")
    
    df, is_german_flag = upload_and_preview(cfg["auto_detect"], cfg["german_credit_flag"], "data_tab")

    if df is None:
        st.warning("⚠️ Bitte laden Sie zuerst Daten in der Konfiguration hoch.")
        return None

    # Clean column names and check target variable
    df_cleaned = clean_column_names(df.copy())
    available_columns = list(df_cleaned.columns)
    
    if cfg["target_col"] not in available_columns:
        st.error(f"Zielspalte '{cfg['target_col']}' nicht gefunden.")
        
        # Show available columns
        st.markdown("#### 📋 Verfügbare Spalten")
        st.dataframe(pd.DataFrame({'Spalten': available_columns}), use_container_width=True)
        
        # Show potential target variables
        potential_targets = get_available_target_variables(df_cleaned)
        if potential_targets:
            st.markdown("#### 🎯 Potentielle Zielvariablen")
            st.write(potential_targets)
            
            # Allow user to select target variable
            selected_target = st.selectbox(
                "Zielvariable auswählen:",
                potential_targets,
                index=0 if potential_targets else None
            )
            
            if st.button("Zielvariable bestätigen"):
                cfg["target_col"] = selected_target
                st.success(f"✅ Zielvariable auf '{selected_target}' gesetzt")
                st.rerun()
        else:
            st.error("❌ Keine geeignete Zielvariable gefunden")
            return None
    else:
        st.success(f"✅ Zielvariable '{cfg['target_col']}' gefunden")
    
    # Data quality metrics
    st.markdown("#### 📈 Datenqualität")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Zeilen", f"{len(df):,}")
    
    with col2:
        st.metric("Spalten", f"{len(df.columns)}")
    
    with col3:
        missing_count = df.isnull().sum().sum()
        st.metric("Fehlende Werte", f"{missing_count:,}")
    
    with col4:
        if cfg["target_col"] in df.columns:
            target_counts = df[cfg["target_col"]].value_counts()
            balance = min(target_counts) / max(target_counts) * 100
            st.metric("Klassenbalance", f"{balance:.1f}%")
    
    return df_cleaned

def training_tab():
    """Training execution tab."""
    st.markdown("### 🚀 Training starten")
    
    # Get configuration from session state or sidebar
    if hasattr(st.session_state, 'training_config'):
        cfg = st.session_state.training_config
    else:
        cfg = render_sidebar("training")
        st.session_state.training_config = cfg
    
    # Check if data is available
    df, is_german_flag = upload_and_preview(cfg["auto_detect"], cfg["german_credit_flag"], "training_tab")
    if df is None:
        st.warning("⚠️ Bitte laden Sie zuerst Daten hoch.")
        return
    
    df_cleaned = clean_column_names(df.copy())
    if cfg["target_col"] not in df_cleaned.columns:
        st.error("❌ Zielvariable nicht gefunden. Bitte überprüfen Sie die Konfiguration.")
        return
    
    button_text = "🚀 Automatisches Training starten"
    
    # Display current settings from sidebar configuration
    st.markdown("#### ⚙️ Aktuelle Einstellungen")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Modell:** {cfg['estimator']}")
        st.write(f"**Zielvariable:** {cfg['target_col']}")
        st.write(f"**Test-Größe:** {cfg['test_size']:.1%}")
        st.write(f"**Zufalls-Seed:** {cfg['seed']}")
    
    with col2:
        st.write(f"**Skalierer:** {cfg['scaler']}")
        st.write(f"**Datensatz-Größe:** {len(df):,} Zeilen")
        st.write(f"**Spalten:** {len(df.columns)}")
        if cfg['training_mode'] == "Bestehendes Modell weiter trainieren":
            st.write(f"**Lernrate:** {cfg['incremental_lr']}")
        else:
            # Show model-specific parameters
            if cfg['estimator'] == "Random Forest" and cfg.get('rf_params'):
                st.write(f"**Bäume:** {cfg['rf_params'].get('n_estimators', 'N/A')}")
                st.write(f"**Max. Tiefe:** {cfg['rf_params'].get('max_depth', 'N/A')}")
            elif cfg['estimator'] == "Gradient Boosting" and cfg.get('model_params'):
                st.write(f"**Bäume:** {cfg['model_params'].get('n_estimators', 'N/A')}")
                st.write(f"**Lernrate:** {cfg['model_params'].get('learning_rate', 'N/A')}")
            elif cfg['estimator'] == "Logistic Regression" and cfg.get('model_params'):
                st.write(f"**C (Regularisierung):** {cfg['model_params'].get('C', 'N/A')}")
    
    st.markdown("---")
    st.markdown("#### 🎯 Training starten")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if not st.button(button_text, type="primary", use_container_width=True):
            st.info("💡 Klicken Sie auf den Button, um das automatische Training zu starten. Alle Einstellungen werden automatisch optimiert.")
            return

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", force=True)

    if cfg["training_mode"] == "Bestehendes Modell weiter trainieren":
        config = TrainConfig(
            target=cfg["target_col"],
            test_size=cfg["test_size"],
            scaler=cfg["scaler"],
            is_german_credit=is_german_flag,
            estimator="Incremental",
            est_params=None,
            training_mode="incremental",
            existing_model_name=cfg["selected_existing_model"],
            incremental_lr=cfg["incremental_lr"]
        )
    else:
        # Use the correct estimator and parameters from sidebar
        estimator_name = cfg["estimator"]
        est_params = None
        
        if estimator_name == "Random Forest" and cfg.get("rf_params"):
            est_params = cfg["rf_params"]
        elif estimator_name in ["Logistic Regression", "Gradient Boosting"] and cfg.get("model_params"):
            est_params = cfg["model_params"]
        
        config = TrainConfig(
            target=cfg["target_col"],
            test_size=cfg["test_size"],
            scaler=cfg["scaler"],
            is_german_credit=is_german_flag,
            estimator=estimator_name,
            est_params=est_params,
            training_mode="new"
        )

    # Training progress
    prog = st.progress(0.0, text="Initialisiere …")
    status = st.empty()

    th, result_container, out_path, q = launch_training_and_saving(df, config, cfg["out_dir"])

    def drive_progress_and_get_result(thread, queue_obj, progress_bar, status_text, result_dict):
        while thread.is_alive():
            try:
                msg = queue_obj.get(timeout=0.1)
                if msg["type"] == "progress":
                    progress_bar.progress(msg["progress"], text=msg["message"])
                    status_text.text(msg["message"])
                elif msg["type"] == "result":
                    result_dict.update(msg["result"])
                    break
            except queue.Empty:
                continue
        thread.join()
        progress_bar.empty()
        status_text.empty()

    drive_progress_and_get_result(th, q, prog, status, result_container)

    if not result_container.get("ok"):
        st.error(f"❌ Training fehlgeschlagen: {result_container.get('err', 'Unbekannter Fehler.')}")
        if result_container.get("trace"):
            with st.expander("Fehlerdetails anzeigen"):
                st.code(result_container["trace"], language="python")
        return

    if cfg["training_mode"] == "Bestehendes Modell weiter trainieren":
        st.success(f"✔️ Modell '{result_container['model_name']}' erfolgreich weiter trainiert!")
    else:
        st.success(f"✔️ Neues Modell '{result_container['model_name']}' erfolgreich trainiert!")
    
    # Store results in session state for results tab
    st.session_state.training_results = result_container
    st.session_state.training_config = cfg

def results_tab():
    """Results display tab."""
    st.markdown("### 📈 Trainingsergebnisse")
    
    if not hasattr(st.session_state, 'training_results') or not st.session_state.training_results:
        st.info("ℹ️ Keine Trainingsergebnisse verfügbar. Führen Sie zuerst ein Training durch.")
        return
    
    result_container = st.session_state.training_results
    
    # Display metrics if available
    if result_container.get("metrics"):
        st.markdown("#### 🏁 Modellleistung")
        render_single_result_table(result_container["metrics"])
    
    # Create sub-tabs for different result types
    if result_container.get("metrics") and "y_test" in result_container["metrics"] and "y_prob" in result_container["metrics"]:
        y_test = result_container["metrics"]["y_test"]
        y_prob = result_container["metrics"]["y_prob"]
        
        eval_tab1, eval_tab2 = st.tabs([
            "📈 ROC-Kurve", 
            "📉 Precision-Recall"
        ])
        
        with eval_tab1:
            render_eval_plots(y_test, y_prob, show_cm=False, show_roc=True, show_pr=False)
        
        with eval_tab2:
            render_eval_plots(y_test, y_prob, show_cm=False, show_roc=False, show_pr=True)
    
    # Display feature importance if available
    if result_container.get("model") and hasattr(result_container["model"], 'feature_importances_'):
        st.markdown("#### 🎯 Merkmalswichtigkeit")
        render_feature_importance(
            feature_names=result_container.get("features", []),
            importances=result_container["model"].feature_importances_
        )
    
    # Model information
    if result_container.get("model_name"):
        st.markdown("#### ℹ️ Modellinformationen")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Modellname:** {result_container['model_name']}")
            st.write(f"**Trainingsmodus:** {result_container.get('training_mode', 'Unbekannt')}")
        
        with col2:
            st.write(f"**Zielvariable:** {st.session_state.training_config.get('target_col', 'Unbekannt')}")
            st.write(f"**Test-Größe:** {st.session_state.training_config.get('test_size', 0):.1%}")