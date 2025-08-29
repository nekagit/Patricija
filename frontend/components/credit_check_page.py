import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import uuid
from datetime import datetime

# Import from frontend config
from config import VALIDATION_RULES, CATEGORICAL_VALUES
from utils.prediction import run_predictions
from utils.api_client import get_api_client


def credit_check_page():
    """Credit application page with form and prediction."""
    
    st.markdown("""
        <div class="main-title text-center text-secondary mb-4 animate-fade-in">
            <h1>💳 Kreditantrag</h1>
            <p>Füllen Sie das Formular aus, um Ihre Kreditwürdigkeit zu bewerten</p>
        </div>
    """, unsafe_allow_html=True)
    
    if 'form_data' not in st.session_state:
        st.session_state.form_data = {}
    
    # Check backend connection status
    api_client = get_api_client()
    backend_connected = api_client.check_backend_connection()
    
    if not backend_connected:
        st.markdown("""
            <div class="alert alert-warning" role="alert">
                <strong>⚠️ Backend nicht verbunden</strong><br>
                Die Anwendung läuft im Demo-Modus. Ergebnisse werden nicht im Backend gespeichert.
            </div>
        """, unsafe_allow_html=True)
    
    with st.form("credit_application_form"):
        
        st.markdown("""
            <div class="form-section animate-fade-in-up delay-100">
                <div class="form-section-header">
                    <div class="form-section-icon">👤</div>
                    <div class="form-section-title">Persönliche Informationen</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            person_age = st.number_input(
                "Alter",
                min_value=VALIDATION_RULES["person_age"]["min"],
                max_value=VALIDATION_RULES["person_age"]["max"],
                value=25,
                help="Ihr Alter in Jahren"
            )
            
            person_income = st.number_input(
                "Jahreseinkommen (€)",
                min_value=VALIDATION_RULES["person_income"]["min"],
                max_value=VALIDATION_RULES["person_income"]["max"],
                value=50000,
                step=1000,
                help="Ihr jährliches Einkommen in Euro"
            )
            
            person_home_ownership = st.selectbox(
                "Wohnsituation",
                options=CATEGORICAL_VALUES["person_home_ownership"],
                help="Ihre aktuelle Wohnsituation",
                key="person_home_ownership_select"
            )
        
        with col2:
            person_emp_length = st.number_input(
                "Beschäftigungsdauer (Jahre)",
                min_value=float(VALIDATION_RULES["person_emp_length"]["min"]),
                max_value=float(VALIDATION_RULES["person_emp_length"]["max"]),
                value=5.0,
                step=0.5,
                help="Dauer Ihrer Beschäftigung in Jahren"
            )
            
            cb_person_cred_hist_length = st.number_input(
                "Kredithistorie (Jahre)",
                min_value=VALIDATION_RULES["cb_person_cred_hist_length"]["min"],
                max_value=VALIDATION_RULES["cb_person_cred_hist_length"]["max"],
                value=3,
                help="Länge Ihrer Kredithistorie in Jahren"
            )
            
            cb_person_default_on_file = st.selectbox(
                "Frühere Zahlungsausfälle",
                options=CATEGORICAL_VALUES["cb_person_default_on_file"],
                help="Hatten Sie bereits Zahlungsausfälle?",
                key="cb_person_default_on_file_select"
            )
        
        st.markdown("""
            <div class="form-section animate-fade-in-up delay-200">
                <div class="form-section-header">
                    <div class="form-section-icon">💰</div>
                    <div class="form-section-title">Kreditinformationen</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            loan_intent = st.selectbox(
                "Kreditzweck",
                options=CATEGORICAL_VALUES["loan_intent"],
                help="Zweck des Kredits",
                key="loan_intent_select"
            )
            
            loan_grade = st.selectbox(
                "Kreditnote",
                options=CATEGORICAL_VALUES["loan_grade"],
                help="Ihre Kreditnote (A = beste, F = schlechteste)",
                key="loan_grade_select"
            )
            
            loan_amnt = st.number_input(
                "Kreditbetrag (€)",
                min_value=VALIDATION_RULES["loan_amnt"]["min"],
                max_value=VALIDATION_RULES["loan_amnt"]["max"],
                value=25000,
                step=1000,
                help="Gewünschter Kreditbetrag in Euro"
            )
        
        with col2:
            loan_int_rate = st.number_input(
                "Zinssatz (%)",
                min_value=float(VALIDATION_RULES["loan_int_rate"]["min"]),
                max_value=float(VALIDATION_RULES["loan_int_rate"]["max"]),
                value=7.5,
                step=0.1,
                help="Zinssatz in Prozent"
            )
            
            # Calculate loan percent income
            loan_percent_income = loan_amnt / person_income if person_income > 0 else 0
            st.metric(
                "Kreditanteil am Einkommen",
                f"{loan_percent_income:.1%}",
                help="Verhältnis von Kreditbetrag zu Jahreseinkommen"
            )
        
        st.markdown("""
            <div class="form-section animate-fade-in-up delay-300">
                <div class="form-section-header">
                    <div class="form-section-icon">📋</div>
                    <div class="form-section-title">Antrag einreichen</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col2:
            submitted = st.form_submit_button(
                "Antrag einreichen & Analyse starten",
                use_container_width=True,
                type="primary"
            )
        
        if submitted:
            # Validate form data
            if person_age < 18:
                st.error("Sie müssen mindestens 18 Jahre alt sein.")
                return
            
            if person_income <= 0:
                st.error("Das Jahreseinkommen muss größer als 0 sein.")
                return
            
            if loan_amnt <= 0:
                st.error("Der Kreditbetrag muss größer als 0 sein.")
                return
            
            # Prepare application data
            application_data = {
                "person_age": person_age,
                "person_income": float(person_income),
                "person_home_ownership": person_home_ownership,
                "person_emp_length": float(person_emp_length),
                "cb_person_cred_hist_length": cb_person_cred_hist_length,
                "cb_person_default_on_file": cb_person_default_on_file,
                "loan_intent": loan_intent,
                "loan_grade": loan_grade,
                "loan_amnt": float(loan_amnt),
                "loan_int_rate": float(loan_int_rate),
                "loan_percent_income": float(loan_percent_income)
            }
            
            # Store form data in session state
            st.session_state.form_data = application_data
            
            # Run prediction
            with st.spinner("KI-Analyse läuft..."):
                try:
                    # Get prediction from existing system
                    prediction_result = run_predictions(application_data)
                    
                    if "error" in prediction_result:
                        st.error(f"Fehler bei der Vorhersage: {prediction_result['error']}")
                        return
                    
                    # Store prediction result
                    st.session_state.prediction_result = prediction_result
                    st.session_state.applicant_data_readable = application_data
                    
                    # Add to session history
                    if 'history' not in st.session_state:
                        st.session_state.history = []
                    
                    history_entry = {
                        'prediction': prediction_result.get('prediction', 'Unknown'),
                        'probability': prediction_result.get('probability_good', 0.0) if prediction_result.get('prediction') == 'Good' else prediction_result.get('probability_bad', 0.0),
                        'data': application_data,
                        'timestamp': datetime.now().isoformat(),
                        'model_name': prediction_result.get('model_name', 'Unknown')
                    }
                    st.session_state.history.append(history_entry)
                    
                    # Always try to save to backend if connected
                    if backend_connected:
                        try:
                            # Create application in backend
                            application_response = api_client.create_application(application_data)
                            
                            if "error" not in application_response:
                                application_id = application_response.get("id")
                                # Store application ID in session state for plot storage
                                st.session_state.application_id = application_id
                                
                                # Create prediction in backend
                                prediction_data = {
                                    "application_id": application_id,
                                    "model_name": prediction_result.get("model_name", "Random Forest"),
                                    "model_version": "1.0.0",
                                    "prediction": prediction_result.get("prediction", "Good"),
                                    "probability_good": prediction_result.get("probability_good", 0.0),
                                    "probability_bad": prediction_result.get("probability_bad", 0.0),
                                    "confidence_score": prediction_result.get("confidence_score", 0.0),
                                    "risk_category": prediction_result.get("risk_category", "medium"),
                                    "input_features": application_data,
                                    "feature_importance": prediction_result.get("feature_importance"),
                                    "shap_values": prediction_result.get("shap_values"),
                                    "lime_explanation": prediction_result.get("lime_explanation"),
                                    "feature_contributions": prediction_result.get("feature_contributions"),
                                    "processing_time_ms": prediction_result.get("processing_time_ms", 0)
                                }
                                
                                prediction_response = api_client.create_prediction(prediction_data)
                                
                                if "error" not in prediction_response:
                                    st.success("✅ Ergebnisse erfolgreich im Backend gespeichert!")
                                else:
                                    st.warning("⚠️ Vorhersage erstellt, aber Backend-Speicherung fehlgeschlagen.")
                            else:
                                st.warning("⚠️ Vorhersage erstellt, aber Backend-Speicherung fehlgeschlagen.")
                        except Exception as backend_error:
                            st.warning(f"⚠️ Backend-Verbindung fehlgeschlagen: {str(backend_error)}")
                    else:
                        st.info("📊 Demo-Modus: Ergebnisse werden nicht im Backend gespeichert.")
                    
                    st.success("Analyse erfolgreich abgeschlossen!")
                    
                    # Don't navigate, just rerun to show results below
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Fehler bei der KI-Analyse: {str(e)}")
                    return
    
    # Display results directly below the form if available
    if 'prediction_result' in st.session_state and st.session_state.prediction_result:
        st.markdown("---")
        st.markdown("""
            <div class="results-section animate-fade-in-up delay-500">
                <h2 class="main-title text-center">📊 Ergebnis Ihrer Kreditanalyse</h2>
            </div>
        """, unsafe_allow_html=True)
        
        result = st.session_state.prediction_result
        if 'error' in result:
            st.error(f"Bei der Analyse ist ein Fehler aufgetreten: {result['error']}")
        else:
            prediction_text = result.get('prediction', 'Unbekannt')
            
            if prediction_text == "Good":
                st.success("🎉 Kredit genehmigt (Gute Bonität)", icon="✅")
                prob_good = result.get('probability_good', 0)
                st.progress(prob_good, text=f"Konfidenz der KI: {prob_good:.1%}")
            else:
                st.error("❌ Kredit abgelehnt (Hohes Risiko)", icon="🔥")
                prob_bad = result.get('probability_bad', 0)
                st.progress(prob_bad, text=f"Konfidenz der KI: {prob_bad:.1%}")
            
            st.markdown("---")
            
            # Import and render result tabs
            from .result.result_details_whatif import render_details_whatif_tab
            from .result.result_shap import render_shap_tab
            from .result.result_history_model import render_history_model_tab
            
            tab_details, tab_shap, tab_history = st.tabs([
                "📊 Ergebnis & Was-wäre-wenn",
                "🎯 Erklärung (SHAP)",
                "📜 Verlauf & Modelldetails"
            ])

            # Get application ID for plot storage
            application_id = None
            if backend_connected and 'prediction_result' in st.session_state:
                # Try to get application ID from backend response
                if hasattr(st.session_state, 'application_id'):
                    application_id = st.session_state.application_id
                else:
                    # Generate a temporary ID for demo mode
                    application_id = f"demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            with tab_details:
                render_details_whatif_tab(result, st.session_state.applicant_data_readable, application_id)

            with tab_shap:
                render_shap_tab(result, application_id)

            with tab_history:
                render_history_model_tab(result, application_id)
        
        # Add button to start new analysis
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔄 Neue Analyse starten", type="secondary", use_container_width=True):
                # Clear session state
                if 'prediction_result' in st.session_state:
                    del st.session_state.prediction_result
                if 'applicant_data_readable' in st.session_state:
                    del st.session_state.applicant_data_readable
                st.rerun()
    
    st.markdown("""
        <div class="glass-card mt-4 animate-fade-in-up delay-500">
            <h4>💡 Tipps für eine bessere Kreditwürdigkeit:</h4>
            <ul>
                <li>Halten Sie Ihre Kredithistorie sauber</li>
                <li>Vermeiden Sie hohe Kreditbeträge im Verhältnis zum Einkommen</li>
                <li>Stabile Beschäftigung verbessert Ihre Chancen</li>
                <li>Niedrigere Zinssätze deuten auf bessere Kreditwürdigkeit hin</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)