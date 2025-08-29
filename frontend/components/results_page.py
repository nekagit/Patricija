import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# Add the parent directory to the path to import config
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import RISK_CATEGORIES, CREDIT_FEATURES

def results_page():
    """Results display page."""
    
    # Check if form data exists
    if 'form_data' not in st.session_state or not st.session_state.form_data:
        st.error("Keine Daten verfügbar. Bitte füllen Sie zuerst das Kreditantragsformular aus.")
        if st.button("Zurück zum Formular"):
            st.session_state.current_page = "credit_check"
            st.rerun()
        return
    
    form_data = st.session_state.form_data
    
    # Simulate prediction (in a real app, this would call the ML model)
    prediction_result = simulate_prediction(form_data)
    
    st.markdown("""
        <div class="main-title text-center text-secondary mb-4 animate-fade-in">
            <h1>📊 Kreditwürdigkeitsanalyse</h1>
            <p>Ihre persönliche Kreditbewertung und Empfehlungen</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Main results section
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Risk score display
        risk_score = prediction_result['risk_score']
        risk_category = prediction_result['risk_category']
        risk_color = RISK_CATEGORIES[risk_category]['color']
        
        st.markdown(f"""
            <div class="result-card animate-fade-in-up delay-100">
                <div class="result-header">
                    <h2>Kreditwürdigkeits-Score</h2>
                </div>
                <div class="result-content">
                    <div style="text-align: center; margin: 20px 0;">
                        <div style="font-size: 48px; font-weight: bold; color: {risk_color};">
                            {risk_score:.1%}
                        </div>
                        <div style="font-size: 18px; color: {risk_color}; margin-top: 10px;">
                            {risk_category.upper()} RISIKO
                        </div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    # Risk assessment details
    st.markdown("""
        <div class="result-card animate-fade-in-up delay-200">
            <div class="result-header">
                <h3>🔍 Risikobewertung</h3>
            </div>
            <div class="result-content">
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Kreditwahrscheinlichkeit", f"{prediction_result['approval_probability']:.1%}")
        st.metric("Empfohlener Kreditbetrag", f"{prediction_result['recommended_amount']:,.0f} €")
    
    with col2:
        st.metric("Risikokategorie", prediction_result['risk_category'].title())
        st.metric("Vertrauensintervall", f"{prediction_result['confidence']:.1%}")
    
    st.markdown("</div></div>", unsafe_allow_html=True)
    
    # Key factors analysis
    st.markdown("""
        <div class="result-card animate-fade-in-up delay-300">
            <div class="result-header">
                <h3>📈 Wichtigste Faktoren</h3>
            </div>
            <div class="result-content">
    """, unsafe_allow_html=True)
    
    # Create feature importance chart
    feature_importance = prediction_result['feature_importance']
    
    # Create bar chart
    fig = px.bar(
        x=list(feature_importance.values()),
        y=list(feature_importance.keys()),
        orientation='h',
        title="Einfluss der Faktoren auf Ihre Kreditwürdigkeit",
        labels={'x': 'Wichtigkeit', 'y': 'Faktoren'},
        color=list(feature_importance.values()),
        color_continuous_scale='RdYlGn_r'
    )
    
    fig.update_layout(
        height=400,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("</div></div>", unsafe_allow_html=True)
    
    # Recommendations section
    st.markdown("""
        <div class="result-card animate-fade-in-up delay-400">
            <div class="result-header">
                <h3>💡 Empfehlungen</h3>
            </div>
            <div class="result-content">
    """, unsafe_allow_html=True)
    
    recommendations = prediction_result['recommendations']
    
    for i, recommendation in enumerate(recommendations, 1):
        st.markdown(f"""
            <div style="margin: 10px 0; padding: 10px; background: rgba(255,255,255,0.1); border-radius: 8px;">
                <strong>{i}.</strong> {recommendation}
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div></div>", unsafe_allow_html=True)
    
    # Action buttons
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("🔄 Neue Bewertung", use_container_width=True):
            st.session_state.current_page = "credit_check"
            st.rerun()
    
    with col2:
        if st.button("📊 Detaillierte Analyse", use_container_width=True):
            st.session_state.current_page = "analytics"
            st.rerun()
    
    with col3:
        if st.button("🏠 Zurück zur Startseite", use_container_width=True):
            st.session_state.current_page = "home"
            st.rerun()

def simulate_prediction(form_data):
    """Simulate ML prediction based on form data."""
    
    # Calculate a simple risk score based on the form data
    risk_factors = []
    
    # Age factor (younger = higher risk)
    age_factor = max(0, (25 - form_data['person_age']) / 25) if form_data['person_age'] < 25 else 0
    risk_factors.append(('Alter', age_factor))
    
    # Income factor (lower income = higher risk)
    income_factor = max(0, (50000 - form_data['person_income']) / 50000) if form_data['person_income'] < 50000 else 0
    risk_factors.append(('Einkommen', income_factor))
    
    # Employment length factor (shorter = higher risk)
    emp_factor = max(0, (5 - form_data['person_emp_length']) / 5) if form_data['person_emp_length'] < 5 else 0
    risk_factors.append(('Beschäftigungsdauer', emp_factor))
    
    # Loan amount to income ratio factor
    loan_ratio = form_data['loan_percent_income']
    ratio_factor = min(1, loan_ratio * 2)  # Higher ratio = higher risk
    risk_factors.append(('Kredit-Einkommen-Verhältnis', ratio_factor))
    
    # Interest rate factor (higher rate = higher risk)
    rate_factor = max(0, (form_data['loan_int_rate'] - 10) / 20) if form_data['loan_int_rate'] > 10 else 0
    risk_factors.append(('Zinssatz', rate_factor))
    
    # Credit history factor (shorter = higher risk)
    hist_factor = max(0, (3 - form_data['cb_person_cred_hist_length']) / 3) if form_data['cb_person_cred_hist_length'] < 3 else 0
    risk_factors.append(('Kredithistorie', hist_factor))
    
    # Previous default factor
    default_factor = 0.3 if form_data['cb_person_default_on_file'] == 'Y' else 0
    risk_factors.append(('Frühere Ausfälle', default_factor))
    
    # Home ownership factor
    home_risk = {
        'RENT': 0.1,
        'MORTGAGE': 0.05,
        'OWN': 0,
        'OTHER': 0.15
    }
    home_factor = home_risk.get(form_data['person_home_ownership'], 0.1)
    risk_factors.append(('Wohnsituation', home_factor))
    
    # Loan grade factor
    grade_risk = {
        'A': 0,
        'B': 0.05,
        'C': 0.1,
        'D': 0.2,
        'E': 0.3,
        'F': 0.4
    }
    grade_factor = grade_risk.get(form_data['loan_grade'], 0.1)
    risk_factors.append(('Kreditnote', grade_factor))
    
    # Calculate total risk score
    total_risk = sum(factor for _, factor in risk_factors)
    max_possible_risk = len(risk_factors) * 0.4  # Normalize to 0-1 range
    risk_score = min(1, total_risk / max_possible_risk)
    
    # Determine risk category
    if risk_score < 0.3:
        risk_category = 'low'
        approval_probability = 0.85
        recommended_amount = form_data['loan_amnt'] * 1.2
    elif risk_score < 0.6:
        risk_category = 'medium'
        approval_probability = 0.65
        recommended_amount = form_data['loan_amnt'] * 0.9
    else:
        risk_category = 'high'
        approval_probability = 0.35
        recommended_amount = form_data['loan_amnt'] * 0.6
    
    # Create feature importance dictionary
    feature_importance = dict(risk_factors)
    
    # Generate recommendations
    recommendations = []
    
    if age_factor > 0.1:
        recommendations.append("Erwägen Sie einen Co-Signer oder warten Sie auf mehr Lebenserfahrung.")
    
    if income_factor > 0.2:
        recommendations.append("Versuchen Sie, Ihr Einkommen zu steigern oder reduzieren Sie den Kreditbetrag.")
    
    if emp_factor > 0.2:
        recommendations.append("Stabile Beschäftigung verbessert Ihre Kreditwürdigkeit erheblich.")
    
    if ratio_factor > 0.5:
        recommendations.append("Reduzieren Sie den Kreditbetrag oder erhöhen Sie Ihr Einkommen.")
    
    if rate_factor > 0.2:
        recommendations.append("Verbessern Sie Ihre Kreditwürdigkeit für niedrigere Zinssätze.")
    
    if hist_factor > 0.2:
        recommendations.append("Bauen Sie eine längere Kredithistorie auf.")
    
    if default_factor > 0:
        recommendations.append("Vermeiden Sie zukünftige Zahlungsausfälle.")
    
    if len(recommendations) == 0:
        recommendations.append("Ihre Kreditwürdigkeit ist gut. Behalten Sie Ihre aktuellen Finanzpraktiken bei.")
    
    return {
        'risk_score': risk_score,
        'risk_category': risk_category,
        'approval_probability': approval_probability,
        'recommended_amount': recommended_amount,
        'confidence': 0.85,
        'feature_importance': feature_importance,
        'recommendations': recommendations
    }
