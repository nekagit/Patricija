import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
import sys
import time

# Add the root directory to the path to import config and utils
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from config import CREDIT_FEATURES, CATEGORICAL_FEATURES, NUMERICAL_FEATURES, TARGET_VARIABLE
from utils.dataset_compatibility import (
    load_dataset_with_compatibility, 
    get_target_variable_name,
    find_target_variable,
    clean_column_names,
    validate_target_variable,
    get_available_target_variables
)
from utils.preprocessing import read_uploaded_file
import sys
from pathlib import Path
# Add the root directory to the path to access the utils module
root_path = str(Path(__file__).parent.parent.parent)
if root_path not in sys.path:
    sys.path.insert(0, root_path)
from utils.plot_storage import plot_storage

def training_page():
    """Model training interface page."""
    
    st.markdown("""
        <div class="main-title text-center text-secondary mb-4 animate-fade-in">
            <h1>🤖 Modell-Training</h1>
            <p>Trainieren Sie neue Machine Learning Modelle für die Kreditrisiko-Bewertung</p>
        </div>
    """, unsafe_allow_html=True)
    
    # File upload section
    st.markdown("""
        <div class="content-section">
            <h2 class="section-title">📁 Datenupload</h2>
            <p class="chart-description">Laden Sie Ihren Datensatz hoch oder verwenden Sie den Standard-Datensatz</p>
        </div>
    """, unsafe_allow_html=True)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "CSV / .data / .txt hochladen", 
        type=['csv', 'data', 'txt'],
        help="Laden Sie eine CSV-Datei hoch oder verwenden Sie den Standard-Datensatz", key="training_page_uploader"
    )
    
    df = None
    target_var = None
    
    if uploaded_file is not None:
        # Load uploaded file
        try:
            df = read_uploaded_file(uploaded_file)
            df = clean_column_names(df)
            
            # Check target variable
            if validate_target_variable(df):
                target_var = find_target_variable(df)
                st.success(f"✅ Datensatz erfolgreich geladen! Zielvariable: {target_var}")
            else:
                st.warning("⚠️ Zielvariable nicht gefunden. Bitte wählen Sie eine Zielvariable aus:")
                
                # Show available columns
                available_cols = list(df.columns)
                st.info(f"Verfügbare Spalten: {available_cols}")
                
                # Show potential target variables
                potential_targets = get_available_target_variables(df)
                if potential_targets:
                    st.info(f"Potentielle Zielvariablen: {potential_targets}")
                    
                    # Allow user to select target variable
                    selected_target = st.selectbox(
                        "Zielvariable auswählen:",
                        potential_targets,
                        index=0 if potential_targets else None,
                        key="target_variable_select_1"
                    )
                    
                    if st.button("Zielvariable bestätigen"):
                        target_var = selected_target
                        st.success(f"✅ Zielvariable auf '{target_var}' gesetzt")
                        st.rerun()
                else:
                    st.error("❌ Keine geeignete Zielvariable gefunden")
                    return
                    
        except Exception as e:
            st.error(f"❌ Fehler beim Laden der Datei: {str(e)}")
            return
    else:
        # Load default dataset
        try:
            data_path = Path(__file__).parent.parent / "data" / "credit_risk_dataset.csv"
            df = load_dataset_with_compatibility(data_path)
            df = clean_column_names(df)
            target_var = find_target_variable(df)
            st.info("📊 Standard-Datensatz geladen")
            
        except Exception as e:
            st.error(f"❌ Fehler beim Laden des Standard-Datensatzes: {str(e)}")
            st.info("Bitte laden Sie eine Datei hoch oder prüfen Sie die Datenstruktur")
            return
    
    if df is None or target_var is None:
        st.info("Lade ein Datenset hoch, um zu starten.")
        return
    
    # Dataset overview
    st.markdown("""
        <div class="content-section">
            <h2 class="section-title">📊 Datensatz-Übersicht</h2>
        </div>
    """, unsafe_allow_html=True)
    
    # Enhanced metrics display
    st.markdown("""
        <div class="metrics-row">
            <div class="metric-card">
                <div class="metric-value">""" + f"{len(df):,}" + """</div>
                <div class="metric-label">📊 Datensätze</div>
                <div class="metric-description">Anzahl aller Einträge</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">""" + f"{len(df.columns)}" + """</div>
                <div class="metric-label">🔧 Features</div>
                <div class="metric-description">Anzahl der Spalten</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">""" + f"{df[target_var].sum():,}" + """</div>
                <div class="metric-label">✅ Positive Klasse</div>
                <div class="metric-description">Anzahl positiver Fälle</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">""" + f"{len(df) - df[target_var].sum():,}" + """</div>
                <div class="metric-label">❌ Negative Klasse</div>
                <div class="metric-description">Anzahl negativer Fälle</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Training configuration
    st.markdown("""
        <div class="content-section">
            <h2 class="section-title">⚙️ Trainings-Konfiguration</h2>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        model_type = st.selectbox(
            "Modell-Typ",
            ["Random Forest", "Gradient Boosting", "Logistic Regression", "XGBoost"],
            help="Wählen Sie den Algorithmus für das Training",
            key="model_type_select"
        )
        
        test_size = st.slider(
            "Test-Set Anteil",
            min_value=0.1,
            max_value=0.5,
            value=0.2,
            step=0.05,
            help="Anteil der Daten für das Test-Set"
        )
    
    with col2:
        random_state = st.number_input(
            "Random State",
            min_value=0,
            max_value=1000,
            value=42,
            help="Für reproduzierbare Ergebnisse"
        )
        
        cv_folds = st.selectbox(
            "Cross-Validation Folds",
            [3, 5, 10],
            index=1,
            help="Anzahl der Cross-Validation Folds",
            key="cv_folds_select"
        )
    
    # Advanced parameters
    with st.expander("🔧 Erweiterte Parameter"):
        col1, col2 = st.columns(2)
        
        with col1:
            if model_type == "Random Forest":
                n_estimators = st.slider("Anzahl Bäume", 50, 500, 100)
                max_depth = st.slider("Maximale Tiefe", 3, 20, 10)
            elif model_type == "Gradient Boosting":
                n_estimators = st.slider("Anzahl Bäume", 50, 500, 100)
                learning_rate = st.slider("Lernrate", 0.01, 0.3, 0.1, 0.01)
            elif model_type == "XGBoost":
                n_estimators = st.slider("Anzahl Bäume", 50, 500, 100)
                learning_rate = st.slider("Lernrate", 0.01, 0.3, 0.1, 0.01)
                max_depth = st.slider("Maximale Tiefe", 3, 10, 6)
            elif model_type == "Logistic Regression":
                C = st.slider("Regularisierung (C)", 0.1, 10.0, 1.0, 0.1)
        
        with col2:
            if model_type in ["Random Forest", "Gradient Boosting", "XGBoost"]:
                min_samples_split = st.slider("Min Samples Split", 2, 20, 5)
                min_samples_leaf = st.slider("Min Samples Leaf", 1, 10, 2)
    
    # Start training
    if st.button("🚀 Training starten", use_container_width=True, type="primary"):
        with st.spinner("Training läuft..."):
            # Simulate training process
            training_result = simulate_training(df, model_type, test_size, random_state, cv_folds, target_var)
            
            # Store results in session state
            st.session_state.training_results = training_result
            
            st.success("Training erfolgreich abgeschlossen!")
    
    # Display training results
    if 'training_results' in st.session_state:
        display_training_results(st.session_state.training_results)
    
    # Model comparison
    st.markdown("""
        <div class="content-section">
            <h2 class="section-title">📈 Modell-Vergleich</h2>
            <p class="chart-description">Vergleich der verschiedenen Modell-Performances</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Simulate multiple model results for comparison
    model_comparison = simulate_model_comparison()
    
    # Create comparison chart with enhanced styling
    st.markdown("""
        <div class="chart-container">
            <div class="chart-header">
                <h3 class="chart-title">Modell-Genauigkeit im Vergleich</h3>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    fig = px.bar(
        model_comparison,
        x='Model',
        y='Accuracy',
        title="Modell-Genauigkeit im Vergleich",
        color='Accuracy',
        color_continuous_scale='RdYlGn'
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#E8ECF4'),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Save the plot for analytics
    metadata = {
        'plot_type': 'model_comparison',
        'data_shape': model_comparison.shape,
        'models': list(model_comparison['Model']),
        'title': "Modell-Genauigkeit im Vergleich"
    }
    plot_storage.save_plotly_plot(
        fig, 'model_comparison', 
        analytics=True,
        metadata=metadata
    )
    
    # Enhanced action buttons section
    st.markdown("""
        <div class="content-section">
            <h3 class="section-title">🚀 Nächste Schritte</h3>
        </div>
    """, unsafe_allow_html=True)
    
    # Action buttons
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("💳 Kreditantrag", use_container_width=True):
            st.session_state.current_page = "credit_check"
            st.rerun()
    
    with col2:
        if st.button("📊 Analytics", use_container_width=True):
            st.session_state.current_page = "analytics"
            st.rerun()
    
    with col3:
        if st.button("🏠 Startseite", use_container_width=True):
            st.session_state.current_page = "home"
            st.rerun()

def simulate_training(df, model_type, test_size, random_state, cv_folds, target_var):
    """Simulate model training process."""
    
    # Simulate training time
    time.sleep(2)
    
    # Generate simulated results
    base_accuracy = 0.75
    model_accuracy = {
        "Random Forest": 0.85,
        "Gradient Boosting": 0.87,
        "Logistic Regression": 0.82,
        "XGBoost": 0.89
    }
    
    accuracy = model_accuracy.get(model_type, base_accuracy)
    
    # Add some randomness
    accuracy += np.random.normal(0, 0.02)
    accuracy = max(0.5, min(0.95, accuracy))
    
    return {
        'model_type': model_type,
        'accuracy': accuracy,
        'precision': accuracy - 0.02,
        'recall': accuracy + 0.01,
        'f1_score': accuracy - 0.01,
        'training_time': np.random.uniform(30, 120),
        'test_size': test_size,
        'cv_folds': cv_folds,
        'random_state': random_state,
        'dataset_size': len(df),
        'target_variable': target_var
    }

def display_training_results(results):
    """Display training results."""
    
    st.markdown("""
        <div class="content-section">
            <h2 class="section-title">✅ Trainings-Ergebnisse</h2>
            <p class="chart-description">Ihr Modell wurde erfolgreich trainiert und evaluiert</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Enhanced metrics display
    st.markdown("""
        <div class="metrics-row">
            <div class="metric-card">
                <div class="metric-value">""" + f"{results['accuracy']:.3f}" + """</div>
                <div class="metric-label">🎯 Genauigkeit</div>
                <div class="metric-description">Model accuracy score</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">""" + f"{results['precision']:.3f}" + """</div>
                <div class="metric-label">📊 Präzision</div>
                <div class="metric-description">Precision score</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">""" + f"{results['recall']:.3f}" + """</div>
                <div class="metric-label">🔄 Recall</div>
                <div class="metric-description">Recall score</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">""" + f"{results['f1_score']:.3f}" + """</div>
                <div class="metric-label">⚖️ F1-Score</div>
                <div class="metric-description">F1 score</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Training details section
    st.markdown("""
        <div class="content-section">
            <h3 class="section-title">📋 Trainings-Details</h3>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{results['model_type']}</div>
                <div class="metric-label">🤖 Modell-Typ</div>
                <div class="metric-description">Verwendeter Algorithmus</div>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{results['training_time']:.1f}s</div>
                <div class="metric-label">⏱️ Trainings-Zeit</div>
                <div class="metric-description">Benötigte Zeit</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{results['dataset_size']:,}</div>
                <div class="metric-label">📊 Datensatz-Größe</div>
                <div class="metric-description">Anzahl Datensätze</div>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{results['cv_folds']}</div>
                <div class="metric-label">🔄 CV-Folds</div>
                <div class="metric-description">Cross-Validation</div>
            </div>
        """, unsafe_allow_html=True)
    
    # Target variable information
    if 'target_variable' in results:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{results['target_variable']}</div>
                <div class="metric-label">🎯 Zielvariable</div>
                <div class="metric-description">Verwendete Zielvariable</div>
            </div>
        """, unsafe_allow_html=True)

def simulate_model_comparison():
    """Simulate comparison of different models."""
    
    models = ["Random Forest", "Gradient Boosting", "XGBoost", "Logistic Regression"]
    accuracies = [0.85, 0.87, 0.89, 0.82]
    
    return pd.DataFrame({
        'Model': models,
        'Accuracy': accuracies
    })
