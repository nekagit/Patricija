import streamlit as st
import pandas as pd
from datetime import datetime
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from utils.api_client import get_api_client

def render_history_model_tab(result, application_id: str = None):
    st.subheader("📜 Verlauf aller Kreditanfragen")
    st.markdown("Hier sehen Sie die Historie aller Kreditanfragen aus der Datenbank.")
    
    # Get API client
    api_client = get_api_client()
    backend_connected = api_client.check_backend_connection()
    
    if not backend_connected:
        st.warning("⚠️ Backend nicht verbunden. Zeige nur lokale Sitzungsdaten.")
        render_session_history()
    else:
        # Fetch history from backend
        try:
            history_response = api_client.get_application_history(skip=0, limit=50)
            
            if "error" in history_response:
                st.error(f"Fehler beim Laden der Historie: {history_response['error']}")
                render_session_history()
            else:
                history_data = history_response.get("items", [])
                
                if not history_data:
                    st.info("📭 Keine Kreditanfragen in der Datenbank gefunden.")
                    render_session_history()
                else:
                    render_backend_history(history_data)
                    
        except Exception as e:
            st.error(f"Fehler beim Laden der Historie: {str(e)}")
            render_session_history()
    
    st.markdown("---")
    render_model_details(result)

def render_backend_history(history_data):
    """Render history data from backend."""
    st.success(f"✅ {len(history_data)} Kreditanfragen aus der Datenbank geladen")
    
    # Add filters
    col1, col2, col3 = st.columns(3)
    with col1:
        status_filter = st.selectbox(
            "Status filtern:",
            ["Alle", "approved", "rejected", "pending"],
            key="history_status_filter"
        )
    
    with col2:
        prediction_filter = st.selectbox(
            "Vorhersage filtern:",
            ["Alle", "Good", "Bad"],
            key="history_prediction_filter"
        )
    
    with col3:
        sort_by = st.selectbox(
            "Sortieren nach:",
            ["Neueste zuerst", "Älteste zuerst", "Einkommen", "Kreditbetrag"],
            key="history_sort"
        )
    
    # Filter data
    filtered_data = history_data
    if status_filter != "Alle":
        filtered_data = [app for app in filtered_data if app.get("status") == status_filter]
    
    if prediction_filter != "Alle":
        filtered_data = [app for app in filtered_data 
                        if any(pred.get("prediction") == prediction_filter 
                               for pred in app.get("predictions", []))]
    
    # Sort data
    if sort_by == "Neueste zuerst":
        filtered_data.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    elif sort_by == "Älteste zuerst":
        filtered_data.sort(key=lambda x: x.get("created_at", ""))
    elif sort_by == "Einkommen":
        filtered_data.sort(key=lambda x: x.get("application_data", {}).get("person_income", 0), reverse=True)
    elif sort_by == "Kreditbetrag":
        filtered_data.sort(key=lambda x: x.get("application_data", {}).get("loan_amnt", 0), reverse=True)
    
    st.info(f"📊 {len(filtered_data)} Anfragen entsprechen den Filtern")
    
    # Display history
    for i, app in enumerate(filtered_data):
        with st.expander(
            f"Anfrage #{app.get('application_number', f'{i+1}')} - "
            f"Status: {app.get('status', 'Unbekannt')} - "
            f"Erstellt: {format_date(app.get('created_at'))}"
        ):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("**Antragsdaten:**")
                app_data = app.get("application_data", {})
                display_application_data(app_data)
            
            with col2:
                st.markdown("**Vorhersagen:**")
                predictions = app.get("predictions", [])
                if predictions:
                    for pred in predictions:
                        display_prediction_summary(pred)
                else:
                    st.info("Keine Vorhersagen verfügbar")

def render_session_history():
    """Render session-based history when backend is not available."""
    st.subheader("Lokale Sitzungsdaten")
    
    if 'history' not in st.session_state or not st.session_state['history']:
        st.info("Bisher wurden in dieser Sitzung keine Anfragen gestellt.")
    else:
        for i, record in enumerate(reversed(st.session_state['history'])):
            with st.expander(f"Anfrage #{len(st.session_state['history']) - i} - Ergebnis: {record['prediction']}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        label=f"Wahrscheinlichkeit für '{record['prediction']}'",
                        value=f"{record['probability']:.1%}"
                    )
                with col2:
                    st.write("Ihre damaligen Angaben:")
                    st.dataframe(pd.DataFrame.from_dict(record['data'], orient='index', columns=["Eingabe"]), height=250)

def display_application_data(app_data):
    """Display application data in a formatted way."""
    # Personal information
    st.markdown("**Persönliche Daten:**")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Alter", f"{app_data.get('person_age', 'N/A')} Jahre")
        st.metric("Einkommen", f"€{app_data.get('person_income', 0):,.0f}")
        st.metric("Beschäftigungsdauer", f"{app_data.get('person_emp_length', 0):.1f} Jahre")
    with col2:
        st.metric("Kredithistorie", f"{app_data.get('cb_person_cred_hist_length', 0)} Jahre")
        st.metric("Wohnsituation", app_data.get('person_home_ownership', 'N/A'))
        st.metric("Vergangenheit", "Ja" if app_data.get('cb_person_default_on_file') == 'Y' else "Nein")
    
    # Loan information
    st.markdown("**Kreditdaten:**")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Kreditbetrag", f"€{app_data.get('loan_amnt', 0):,.0f}")
        st.metric("Zinssatz", f"{app_data.get('loan_int_rate', 0):.1f}%")
        st.metric("Zweck", app_data.get('loan_intent', 'N/A'))
    with col2:
        st.metric("Bonität", app_data.get('loan_grade', 'N/A'))
        st.metric("Einkommensanteil", f"{app_data.get('loan_percent_income', 0):.1%}")

def display_prediction_summary(pred):
    """Display prediction summary."""
    with st.container():
        st.markdown(f"**{pred.get('model_name', 'Unbekanntes Modell')} v{pred.get('model_version', '1.0')}**")
        
        prediction = pred.get('prediction', 'Unbekannt')
        if prediction == "Good":
            st.success(f"✅ Genehmigt ({pred.get('probability_good', 0):.1%})")
        else:
            st.error(f"❌ Abgelehnt ({pred.get('probability_bad', 0):.1%})")
        
        st.metric("Risikokategorie", pred.get('risk_category', 'Unbekannt'))
        if pred.get('processing_time_ms'):
            st.metric("Verarbeitungszeit", f"{pred.get('processing_time_ms')}ms")
        
        # Show feature importance if available
        if pred.get('feature_importance'):
            with st.expander("Feature Importance"):
                feature_importance = pred.get('feature_importance', {})
                if isinstance(feature_importance, dict):
                    # Convert to DataFrame for better display
                    df = pd.DataFrame(list(feature_importance.items()), columns=['Feature', 'Importance'])
                    df = df.sort_values('Importance', ascending=False)
                    st.bar_chart(df.set_index('Feature'))

def format_date(date_string):
    """Format date string for display."""
    if not date_string:
        return "Unbekannt"
    try:
        dt = datetime.fromisoformat(date_string.replace('Z', '+00:00'))
        return dt.strftime("%d.%m.%Y %H:%M")
    except:
        return date_string

def render_model_details(result):
    """Render model details section."""
    st.subheader("🤖 Informationen zum verwendeten KI-Modell")
    
    model_used = result.get('model_name', 'Random Forest') 
    
    st.info(f"**Verwendetes Modell:** {model_used.replace('_', ' ').title()}", icon="🤖")
    
    # Model performance metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Modellversion", result.get('model_version', '1.0.0'))
    with col2:
        st.metric("Verarbeitungszeit", f"{result.get('processing_time_ms', 0)}ms")
    with col3:
        st.metric("Cache Hit", "Ja" if result.get('cache_hit', False) else "Nein")
    
    st.markdown("""
    **Modellbeschreibung:**
    
    Je nach Auswahl auf der Anfrageseite können hier unterschiedliche Modelle zum Einsatz kommen. 
    Jedes Modell hat seine eigenen Stärken und lernt leicht unterschiedliche Muster in den Daten.
    
    **Verfügbare Modelle:**
    - **Random Forest**: Robuster Ensemble-Ansatz, gut für komplexe Daten
    - **Gradient Boosting**: Sequentielle Verbesserung, hohe Genauigkeit
    - **Neural Network**: Deep Learning Ansatz für komplexe Muster
    - **Support Vector Machine**: Effektiv für hochdimensionale Daten
    """)
    
    # Show model training info if available
    if 'model_training_info' in result:
        st.markdown("**Letztes Training:**")
        training_info = result['model_training_info']
        st.write(f"- Datum: {training_info.get('training_date', 'Unbekannt')}")
        st.write(f"- Genauigkeit: {training_info.get('accuracy', 0):.2%}")
        st.write(f"- ROC-AUC: {training_info.get('roc_auc', 0):.3f}")