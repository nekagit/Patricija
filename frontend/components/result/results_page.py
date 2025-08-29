import streamlit as st
import pandas as pd

from .result_details_whatif import render_details_whatif_tab
from .result_shap import render_shap_tab
from .result_lime import render_lime_tab
from .result_history_model import render_history_model_tab

def results_page():
    if 'prediction_result' not in st.session_state or 'applicant_data_readable' not in st.session_state:
        st.warning("Bitte füllen Sie zuerst das Formular auf der Seite 'Kreditanfrage' aus, um ein Ergebnis zu sehen.")
        if st.button("Zurück zur Kreditanfrage"):
            st.session_state.page = "Kreditanfrage"
            st.rerun()
        return

    result = st.session_state.get('prediction_result', {})
    if 'error' in result:
        st.error(f"Bei der Analyse ist ein Fehler aufgetreten: {result['error']}")
        return
    applicant_data_readable = st.session_state.get('applicant_data_readable', {})

    st.title("Ergebnis Ihrer Kreditanalyse")
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
    
    # Alle Inhalte direkt anzeigen ohne Tabs
    st.subheader("📊 Ergebnis & Was-wäre-wenn")
    render_details_whatif_tab(result, applicant_data_readable)
    
    st.markdown("---")
    
    st.subheader("🎯 Erklärung (SHAP)")
    render_shap_tab(result)
    
    st.markdown("---")
    
    st.subheader("🔬 Alternative Erklärung (LIME)")
    render_lime_tab(result)
    
    st.markdown("---")
    
    st.subheader("📜 Verlauf & Modelldetails")
    render_history_model_tab(result)