import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import uuid
from datetime import datetime

# Import from frontend config
from config import VALIDATION_RULES
from utils.credit_config import OPTIONS
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
    
    # Initialize model selection in session state
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = "Gradient Boosting"
    
    # Model selection outside the form
    st.markdown("""
        <div class="form-section animate-fade-in-up delay-50">
            <div class="form-section-header">
                <div class="form-section-icon">🤖</div>
                <div class="form-section-title">Modell-Auswahl</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("🔄 Gradient Boosting", use_container_width=True, 
                    type="primary" if st.session_state.selected_model == "Gradient Boosting" else "secondary"):
            st.session_state.selected_model = "Gradient Boosting"
            st.rerun()
    
    with col2:
        if st.button("🌲 Random Forest", use_container_width=True,
                    type="primary" if st.session_state.selected_model == "Random Forest" else "secondary"):
            st.session_state.selected_model = "Random Forest"
            st.rerun()
    
    with col3:
        st.markdown(f"""
            <div style="text-align: center; padding: 15px; background-color: rgba(255,255,255,0.1); border-radius: 5px; margin-top: 5px;">
                <small>📊 Aktuelles Modell:<br><strong>{st.session_state.selected_model}</strong></small>
            </div>
        """, unsafe_allow_html=True)
    
    # Active model info display
    model_info = {
        "Gradient Boosting": "Optimiert für komplexe Muster in Kreditdaten. Verwendet sequentielle Entscheidungsbäume mit Gradientenverstärkung.",
        "Random Forest": "Ensemble-Methode mit hoher Robustheit. Kombiniert mehrere Entscheidungsbäume für zuverlässige Vorhersagen."
    }
    
    st.info(f"""
    **🤖 Aktives Modell: {st.session_state.selected_model}**
    
    {model_info.get(st.session_state.selected_model, "Machine Learning Modell für Kreditrisiko-Bewertung.")}
    
    Klicken Sie auf die Buttons oben, um das Modell zu wechseln.
    """)
    
    with st.form("credit_application_form"):
        
        # Submit button at the top
        st.markdown("""
            <div class="form-section animate-fade-in-up delay-75">
                <div class="form-section-header">
                    <div class="form-section-icon">📋</div>
                    <div class="form-section-title">Antrag einreichen</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col2:
            submitted = st.form_submit_button(
                "🚀 Antrag einreichen & Analyse starten",
                use_container_width=True,
                type="primary"
            )
        
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
                "Alter (Jahre)",
                min_value=18,
                max_value=100,
                value=30,
                help="Ihr Alter in Jahren"
            )
            
            person_income = st.number_input(
                "Jahreseinkommen (€)",
                min_value=0,
                max_value=1000000,
                value=50000,
                step=1000,
                help="Ihr jährliches Einkommen in Euro"
            )
            
            person_home_ownership = st.selectbox(
                "Wohnsituation",
                options=["RENT", "OWN", "MORTGAGE", "OTHER"],
                format_func=lambda x: {
                    "RENT": "Miete",
                    "OWN": "Eigentum",
                    "MORTGAGE": "Hypothek",
                    "OTHER": "Sonstiges"
                }[x],
                help="Ihre aktuelle Wohnsituation"
            )
            
            person_emp_length = st.number_input(
                "Beschäftigungsdauer (Jahre)",
                min_value=0.0,
                max_value=50.0,
                value=5.0,
                step=0.5,
                help="Seit wie vielen Jahren sind Sie beschäftigt"
            )
        
        with col2:
            cb_person_cred_hist_length = st.number_input(
                "Kredithistorie Länge (Jahre)",
                min_value=0,
                max_value=50,
                value=3,
                help="Länge Ihrer Kredithistorie in Jahren"
            )
            
            cb_person_default_on_file = st.selectbox(
                "Standard in Datei",
                options=["Y", "N"],
                format_func=lambda x: "Ja" if x == "Y" else "Nein",
                help="Haben Sie bereits einen Zahlungsausfall in Ihrer Kredithistorie"
            )
            
            loan_intent = st.selectbox(
                "Kreditzweck",
                options=["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"],
                format_func=lambda x: {
                    "PERSONAL": "Persönlich",
                    "EDUCATION": "Ausbildung",
                    "MEDICAL": "Medizinisch",
                    "VENTURE": "Unternehmen",
                    "HOMEIMPROVEMENT": "Hausverbesserung",
                    "DEBTCONSOLIDATION": "Schuldenkonsolidierung"
                }[x],
                help="Zweck des Kredits"
            )
            
            loan_grade = st.selectbox(
                "Kreditnote",
                options=["A", "B", "C", "D", "E", "F"],
                help="Kreditnote (A = Beste, F = Schlechteste)"
            )
        
        st.markdown("""
            <div class="form-section animate-fade-in-up delay-200">
                <div class="form-section-header">
                    <div class="form-section-icon">💰</div>
                    <div class="form-section-title">Kreditdetails</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            loan_amnt = st.number_input(
                "Kreditbetrag (€)",
                min_value=100,
                max_value=1000000,
                value=20000,
                step=1000,
                help="Gewünschter Kreditbetrag in Euro"
            )
            
            loan_int_rate = st.number_input(
                "Kreditzinssatz (%)",
                min_value=0.0,
                max_value=100.0,
                value=12.0,
                step=0.1,
                help="Jährlicher Zinssatz in Prozent"
            )
        
        with col2:
            loan_percent_income = st.number_input(
                "Kreditanteil am Einkommen",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.01,
                help="Anteil des Kredits am jährlichen Einkommen (0.0 - 1.0)"
            )
        
        if submitted:
            # Validate form data
            if person_age < 18:
                st.error("Sie müssen mindestens 18 Jahre alt sein.")
                return
            
            if loan_amnt <= 0:
                st.error("Der Kreditbetrag muss größer als 0 sein.")
                return
            
            if loan_percent_income <= 0 or loan_percent_income > 1:
                st.error("Der Kreditanteil am Einkommen muss zwischen 0 und 1 liegen.")
                return
            
            # Prepare application data with new field names
            application_data = {
                "person_age": person_age,
                "person_income": person_income,
                "person_home_ownership": person_home_ownership,
                "person_emp_length": person_emp_length,
                "cb_person_cred_hist_length": cb_person_cred_hist_length,
                "cb_person_default_on_file": cb_person_default_on_file,
                "loan_intent": loan_intent,
                "loan_grade": loan_grade,
                "loan_amnt": loan_amnt,
                "loan_int_rate": loan_int_rate,
                "loan_percent_income": loan_percent_income
            }
            
            # Store in session state
            st.session_state.form_data = application_data
            
            # Generate unique application ID
            application_id = str(uuid.uuid4())
            application_data["application_id"] = application_id
            application_data["timestamp"] = datetime.now().isoformat()
            
            # Run prediction
            with st.spinner("Analysiere Kreditantrag..."):
                try:
                    prediction_result = run_predictions(application_data, st.session_state.selected_model)
                    
                    if prediction_result:
                        # Store results in session state
                        st.session_state.prediction_result = prediction_result
                        st.session_state.application_data = application_data
                        
                        # Convert application data to readable format
                        st.session_state.applicant_data_readable = {
                            "Alter": f"{person_age} Jahre",
                            "Jahreseinkommen": f"{person_income:,.0f} €",
                            "Wohnsituation": {
                                "RENT": "Miete",
                                "OWN": "Eigentum",
                                "MORTGAGE": "Hypothek",
                                "OTHER": "Sonstiges"
                            }[person_home_ownership],
                            "Beschäftigungsdauer": f"{person_emp_length} Jahre",
                            "Kredithistorie Länge": f"{cb_person_cred_hist_length} Jahre",
                            "Standard in Datei": "Ja" if cb_person_default_on_file == "Y" else "Nein",
                            "Kreditzweck": {
                                "PERSONAL": "Persönlich",
                                "EDUCATION": "Ausbildung",
                                "MEDICAL": "Medizinisch",
                                "VENTURE": "Unternehmen",
                                "HOMEIMPROVEMENT": "Hausverbesserung",
                                "DEBTCONSOLIDATION": "Schuldenkonsolidierung"
                            }[loan_intent],
                            "Kreditnote": loan_grade,
                            "Kreditbetrag": f"{loan_amnt:,.0f} €",
                            "Kreditzinssatz": f"{loan_int_rate}%",
                            "Kreditanteil am Einkommen": f"{loan_percent_income:.1%}"
                        }
                        
                        # Save to backend if connected
                        if backend_connected:
                            try:
                                # Save application to backend
                                save_response = api_client.save_application(application_data)
                                if "error" not in save_response:
                                    st.success("✅ Ergebnisse erfolgreich im Backend gespeichert!")
                                else:
                                    st.warning("⚠️ Konnte Ergebnisse nicht im Backend speichern.")
                            except Exception as e:
                                st.warning(f"⚠️ Backend-Speicherung fehlgeschlagen: {str(e)}")
                        else:
                            st.info("📊 Demo-Modus: Ergebnisse werden nicht im Backend gespeichert.")
                        
                        # Redirect to results page
                        st.session_state.page = "Ergebnis"
                        st.rerun()
                    else:
                        st.error("❌ Fehler bei der Vorhersage. Bitte versuchen Sie es erneut.")
                        
                except Exception as e:
                    st.error(f"❌ Fehler bei der Analyse: {str(e)}")
                    st.error("Bitte überprüfen Sie Ihre Eingaben und versuchen Sie es erneut.")
    
    # Display form instructions
    st.markdown("""
        <div class="form-instructions">
            <h3>📝 Anleitung</h3>
            <ul>
                <li>Füllen Sie alle Felder sorgfältig aus</li>
                <li>Geben Sie nur echte und korrekte Informationen an</li>
                <li>Die Analyse basiert auf den bereitgestellten Daten</li>
                <li>Ergebnisse werden sofort nach dem Absenden angezeigt</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)