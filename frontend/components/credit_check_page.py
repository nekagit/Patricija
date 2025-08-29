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
            age_years = st.number_input(
                "Alter (Jahre)",
                min_value=VALIDATION_RULES["age_years"]["min"],
                max_value=VALIDATION_RULES["age_years"]["max"],
                value=30,
                help="Ihr Alter in Jahren"
            )
            
            personal_status_sex = st.selectbox(
                "Persönlicher Status",
                options=list(OPTIONS["personal_status_sex"].keys()),
                format_func=lambda x: OPTIONS["personal_status_sex"][x],
                help="Ihr persönlicher Status",
                key="personal_status_sex_select"
            )
            
            num_dependents = st.number_input(
                "Anzahl Abhängige",
                min_value=VALIDATION_RULES["num_dependents"]["min"],
                max_value=VALIDATION_RULES["num_dependents"]["max"],
                value=1,
                help="Anzahl der Personen, die von Ihnen abhängig sind"
            )
            
            residence_since = st.number_input(
                "Wohnsitz seit (Jahre)",
                min_value=VALIDATION_RULES["residence_since"]["min"],
                max_value=VALIDATION_RULES["residence_since"]["max"],
                value=2,
                help="Seit wie vielen Jahren wohnen Sie am aktuellen Ort"
            )
        
        with col2:
            job = st.selectbox(
                "Beruf",
                options=list(OPTIONS["job"].keys()),
                format_func=lambda x: OPTIONS["job"][x],
                help="Ihr aktueller Beruf",
                key="job_select"
            )
            
            housing = st.selectbox(
                "Wohnsituation",
                options=list(OPTIONS["housing"].keys()),
                format_func=lambda x: OPTIONS["housing"][x],
                help="Ihre aktuelle Wohnsituation",
                key="housing_select"
            )
            
            telephone = st.selectbox(
                "Telefon",
                options=list(OPTIONS["telephone"].keys()),
                format_func=lambda x: OPTIONS["telephone"][x],
                help="Haben Sie ein Telefon",
                key="telephone_select"
            )
            
            foreign_worker = st.selectbox(
                "Ausländischer Arbeitnehmer",
                options=list(OPTIONS["foreign_worker"].keys()),
                format_func=lambda x: OPTIONS["foreign_worker"][x],
                help="Sind Sie ein ausländischer Arbeitnehmer",
                key="foreign_worker_select"
            )
        
        st.markdown("""
            <div class="form-section animate-fade-in-up delay-200">
                <div class="form-section-header">
                    <div class="form-section-icon">🏦</div>
                    <div class="form-section-title">Bank- und Kreditinformationen</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            checking_account_status = st.selectbox(
                "Girokonto Status",
                options=list(OPTIONS["checking_account_status"].keys()),
                format_func=lambda x: OPTIONS["checking_account_status"][x],
                help="Status Ihres Girokontos",
                key="checking_account_status_select"
            )
            
            savings_account = st.selectbox(
                "Sparkonto",
                options=list(OPTIONS["savings_account"].keys()),
                format_func=lambda x: OPTIONS["savings_account"][x],
                help="Status Ihres Sparkontos",
                key="savings_account_select"
            )
            
            credit_history = st.selectbox(
                "Kredithistorie",
                options=list(OPTIONS["credit_history"].keys()),
                format_func=lambda x: OPTIONS["credit_history"][x],
                help="Ihre Kredithistorie",
                key="credit_history_select"
            )
            
            num_existing_credits = st.number_input(
                "Bestehende Kredite",
                min_value=VALIDATION_RULES["num_existing_credits"]["min"],
                max_value=VALIDATION_RULES["num_existing_credits"]["max"],
                value=1,
                help="Anzahl Ihrer bestehenden Kredite"
            )
        
        with col2:
            purpose = st.selectbox(
                "Kreditzweck",
                options=list(OPTIONS["purpose"].keys()),
                format_func=lambda x: OPTIONS["purpose"][x],
                help="Zweck des Kredits",
                key="purpose_select"
            )
            
            property = st.selectbox(
                "Eigentum",
                options=list(OPTIONS["property"].keys()),
                format_func=lambda x: OPTIONS["property"][x],
                help="Art Ihres Eigentums",
                key="property_select"
            )
            
            other_installment_plans = st.selectbox(
                "Andere Ratenpläne",
                options=list(OPTIONS["other_installment_plans"].keys()),
                format_func=lambda x: OPTIONS["other_installment_plans"][x],
                help="Andere Ratenpläne",
                key="other_installment_plans_select"
            )
            
            other_debtors_guarantors = st.selectbox(
                "Andere Schuldner/Bürgen",
                options=list(OPTIONS["other_debtors_guarantors"].keys()),
                format_func=lambda x: OPTIONS["other_debtors_guarantors"][x],
                help="Andere Schuldner oder Bürgen",
                key="other_debtors_guarantors_select"
            )
        
        st.markdown("""
            <div class="form-section animate-fade-in-up delay-300">
                <div class="form-section-header">
                    <div class="form-section-icon">💼</div>
                    <div class="form-section-title">Beschäftigungsinformationen</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            employment_since = st.selectbox(
                "Beschäftigung seit",
                options=list(OPTIONS["employment_since"].keys()),
                format_func=lambda x: OPTIONS["employment_since"][x],
                help="Seit wann sind Sie beschäftigt",
                key="employment_since_select"
            )
        
        with col2:
            # Empty column for layout
            pass
        
        st.markdown("""
            <div class="form-section animate-fade-in-up delay-400">
                <div class="form-section-header">
                    <div class="form-section-icon">💰</div>
                    <div class="form-section-title">Kreditdetails</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            duration_months = st.number_input(
                "Kreditdauer (Monate)",
                min_value=VALIDATION_RULES["duration_months"]["min"],
                max_value=VALIDATION_RULES["duration_months"]["max"],
                value=12,
                help="Dauer des Kredits in Monaten"
            )
            
            credit_amount = st.number_input(
                "Kreditbetrag (€)",
                min_value=VALIDATION_RULES["credit_amount"]["min"],
                max_value=VALIDATION_RULES["credit_amount"]["max"],
                value=2000,
                step=100,
                help="Gewünschter Kreditbetrag in Euro"
            )
        
        with col2:
            installment_rate_percent = st.number_input(
                "Ratenanteil (%)",
                min_value=VALIDATION_RULES["installment_rate_percent"]["min"],
                max_value=VALIDATION_RULES["installment_rate_percent"]["max"],
                value=3,
                help="Anteil der Rate am verfügbaren Einkommen in Prozent"
            )
        

        
        if submitted:
            # Validate form data
            if age_years < 18:
                st.error("Sie müssen mindestens 18 Jahre alt sein.")
                return
            
            if credit_amount <= 0:
                st.error("Der Kreditbetrag muss größer als 0 sein.")
                return
            
            # Prepare application data
            application_data = {
                "checking_account_status": checking_account_status,
                "duration_months": duration_months,
                "credit_history": credit_history,
                "purpose": purpose,
                "credit_amount": credit_amount,
                "savings_account": savings_account,
                "employment_since": employment_since,
                "installment_rate_percent": installment_rate_percent,
                "personal_status_sex": personal_status_sex,
                "other_debtors_guarantors": other_debtors_guarantors,
                "residence_since": residence_since,
                "property": property,
                "age_years": age_years,
                "other_installment_plans": other_installment_plans,
                "housing": housing,
                "num_existing_credits": num_existing_credits,
                "job": job,
                "num_dependents": num_dependents,
                "telephone": telephone,
                "foreign_worker": foreign_worker
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
                        from utils.credit_config import to_readable
                        st.session_state.applicant_data_readable = to_readable(application_data)
                        
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