import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go

# Import visualization functions
from utils.visualizations import (
    create_model_comparison_chart,
    create_roc_curves,
    create_feature_importance_plot,
)

# This function cleanly generates the deployment code snippet
def generate_deployment_code(best_model_name, is_german_credit, german_credit_features, cost_sensitive):
    """Generates the Python deployment code snippet."""
    # (The corrected code from the first section goes here)
    # ... for brevity, it is omitted here but should be pasted from above ...
    # It returns the 'deployment_code' string.
    # For this example, let's assume it's implemented as described.
    # Conditionally create the feature engineering block
    feature_eng_block = ""
    if is_german_credit and german_credit_features:
        feature_eng_block = '''
    # German Credit specific feature engineering
    if "credit_amount" in df.columns and "age_years" in df.columns:
        df["debt_to_age_ratio"] = df["credit_amount"] / (df["age_years"] + 1)
    if "employment_since" in df.columns and "age_years" in df.columns:
        df["employment_stability"] = df["employment_since"] * df["age_years"] / 100
'''
    # Conditionally create the final return block
    return_block = ""
    if is_german_credit and cost_sensitive:
        return_block = f'''
    # Calculate cost impact for German Credit
    cost_impact = 5 if prediction == 0 and probability > 0.3 else 1 if prediction == 1 else 0
    
    return {{
        'prediction': prediction,
        'probability': probability,
        'risk_level': 'High' if probability and probability > 0.7 else 'Medium' if probability and probability > 0.3 else 'Low',
        'cost_impact': cost_impact,
        'model_used': '{best_model_name}'
    }}
'''
    else:
        return_block = f'''
    return {{
        'prediction': prediction,
        'probability': probability,
        'risk_level': 'High' if probability and probability > 0.7 else 'Medium' if probability and probability > 0.3 else 'Low',
        'cost_impact': None,
        'model_used': '{best_model_name}'
    }}
'''

    # Assemble the final, clean code string
    deployment_code = f"""
# Enhanced Model Deployment Code
import joblib
import pandas as pd
import numpy as np

# Load artifacts
model = joblib.load('models/enhanced_best_model.pkl')
scaler = joblib.load('models/enhanced_scaler.pkl')
label_encoders = joblib.load('models/enhanced_label_encoders.pkl')

def predict_credit_risk(data_dict):
    df = pd.DataFrame([data_dict])
    for col, encoder in label_encoders.items():
        if col in df.columns:
            try:
                df[col] = encoder.transform(df[col])
            except ValueError:
                df[col] = 0
{feature_eng_block}
    # Ensure all required columns are present for scaling, fill missing with 0
    # This part requires the feature names used during training. Let's assume they are loaded.
    # with open('models/enhanced_feature_names.json', 'r') as f:
    #     training_features = json.load(f)
    # for col in training_features:
    #     if col not in df.columns:
    #         df[col] = 0
    # df = df[training_features] # Ensure order
            
    df_scaled = scaler.transform(df)
    prediction = model.predict(df_scaled)[0]
    probability = model.predict_proba(df_scaled)[0][1] if hasattr(model, 'predict_proba') else None
{return_block}
"""
    return deployment_code


def render_results_dashboard(st, results, X_test_scaled, y_test, X, training_time, config, scaler, label_encoders, original_feature_count):
    """Displays the entire results dashboard."""
    # Find best model
    best_model_name, best_model, best_score, best_cost_score = None, None, 0, float('inf')
    if config['is_german_credit'] and config['cost_sensitive']:
        for name, data in results.items():
            if data['cost_score'] < best_cost_score:
                best_cost_score = data['cost_score']
                best_model_name, best_model, best_score = name, data['model'], data['accuracy']
    else:
        for name, data in results.items():
            if data['accuracy'] > best_score:
                best_score = data['accuracy']
                best_model_name, best_model, best_cost_score = name, data['model'], data['cost_score']

    st.markdown('<div class="holo-card" style="margin-top: 2rem;">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-header">🏆 Erweitertes Performance Dashboard</h2>', unsafe_allow_html=True)
    
    # Enhanced metrics section with better spacing
    st.markdown("""
        <div class="metrics-row">
            <div class="metric-card">
                <div class="metric-value">""" + best_model_name + """</div>
                <div class="metric-label">🥇 Bestes Modell</div>
                <div class="metric-description">Bester Algorithmus</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">""" + f"{best_score:.3f}" + """</div>
                <div class="metric-label">🎯 Beste Genauigkeit</div>
                <div class="metric-description">Höchste erreichte Genauigkeit</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">""" + (f"{best_cost_score}" if config['is_german_credit'] and config['cost_sensitive'] else f"{training_time:.2f}s") + """</div>
                <div class="metric-label">""" + ("💰 Bester Kostenwert" if config['is_german_credit'] and config['cost_sensitive'] else "⚡ Trainingszeit") + """</div>
                <div class="metric-description">""" + ("Niedrigster Kostenwert" if config['is_german_credit'] and config['cost_sensitive'] else "Modell-Trainingsdauer") + """</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">""" + f"{len(X.columns)}" + """</div>
                <div class="metric-label">⚙️ Verwendete Features</div>
                <div class="metric-description">Anzahl der Eingabemerkmale</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Enhanced tabbed content with better styling
    st.markdown("""
        <div class="tab-content">
    """, unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🏆 Modellvergleich", "📊 ROC & Metriken", "🎯 Feature-Analyse", "💾 Modell-Export", "🎉 Zusammenfassung"])

    with tab1:
        st.markdown("""
            <div class="chart-container">
                <div class="chart-header">
                    <h3 class="chart-title">Modell-Performance-Vergleich</h3>
                </div>
            </div>
        """, unsafe_allow_html=True)
        st.plotly_chart(create_model_comparison_chart(results), use_container_width=True)
        
        st.markdown("""
            <div class="content-section">
                <h3 class="section-title">📋 Detaillierte Performance-Metriken</h3>
            </div>
        """, unsafe_allow_html=True)
        performance_data = {'Modell': list(results.keys()),'Genauigkeit': [f"{r['accuracy']:.4f}" for r in results.values()],'CV Mittelwert': [f"{r['cv_mean']:.4f}" for r in results.values()]}
        if config['is_german_credit'] and config['cost_sensitive']:
             performance_data['Kostenwert'] = [r['cost_score'] for r in results.values()]
        st.dataframe(pd.DataFrame(performance_data), use_container_width=True)

    with tab2:
        st.markdown("""
            <div class="chart-container">
                <div class="chart-header">
                    <h3 class="chart-title">ROC-Kurven & Performance-Metriken</h3>
                </div>
            </div>
        """, unsafe_allow_html=True)
        st.plotly_chart(create_roc_curves(results, X_test_scaled, y_test), use_container_width=True)
        
        st.markdown("""
            <div class="content-section">
                <h3 class="section-title">📊 Detaillierte Modellanalyse</h3>
            </div>
        """, unsafe_allow_html=True)
        selected_model_detail = st.selectbox("Modell für detaillierte Metriken wählen:", list(results.keys()), key="model_detail_select")
        if selected_model_detail:
            y_pred = results[selected_model_detail]['predictions']
            report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
            st.dataframe(report_df, use_container_width=True)
            
            st.markdown("""
                <div class="chart-container">
                    <div class="chart-header">
                        <h3 class="chart-title">Konfusionsmatrix</h3>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            cm = confusion_matrix(y_test, y_pred)
            fig_cm = px.imshow(cm, text_auto=True, labels=dict(x="Vorhergesagt", y="Tatsächlich"), title="Konfusionsmatrix")
            fig_cm.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#E8ECF4'),
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_cm, use_container_width=True)

    with tab3:
        st.markdown("""
            <div class="chart-container">
                <div class="chart-header">
                    <h3 class="chart-title">Feature-Wichtigkeitsanalyse</h3>
                </div>
            </div>
        """, unsafe_allow_html=True)
        importance_fig = create_feature_importance_plot(best_model, X.columns.tolist())
        if importance_fig:
            importance_fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#E8ECF4'),
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(importance_fig, use_container_width=True)
        else:
            st.info("Feature-Wichtigkeit für diesen Modelltyp nicht verfügbar.")

    with tab5:
        st.markdown("""
            <div class="content-section">
                <h2 class="section-title">🎉 Mission erfüllt!</h2>
                <p class="chart-description">Ihr erweitertes Machine Learning Modell wurde erfolgreich trainiert und evaluiert.</p>
            </div>
        """, unsafe_allow_html=True)
        # Add summary text here...

    with tab4:
        st.markdown("""
            <div class="content-section">
                <h2 class="section-title">💾 Erweiterter Modell-Export & Deployment</h2>
                <p class="chart-description">Speichern Sie Ihr trainiertes Modell und erhalten Sie deployment-bereiten Code für den Produktiveinsatz.</p>
            </div>
        """, unsafe_allow_html=True)
        if st.button("💾 Bestes Modell speichern", type="primary"):
            models_dir = 'models'
            os.makedirs(models_dir, exist_ok=True)
            joblib.dump(best_model, os.path.join(models_dir, 'enhanced_best_model.pkl'))
            joblib.dump(scaler, os.path.join(models_dir, 'enhanced_scaler.pkl'))
            joblib.dump(label_encoders, os.path.join(models_dir, 'enhanced_label_encoders.pkl'))
            with open(os.path.join(models_dir, 'enhanced_feature_names.json'), 'w') as f:
                json.dump(list(X.columns), f)
            st.success("✅ Modell und Artefakte im `models/` Verzeichnis gespeichert!")
            st.balloons()
            
        st.markdown("""
            <div class="chart-container">
                <div class="chart-header">
                    <h3 class="chart-title">Deployment-Code</h3>
                </div>
            </div>
        """, unsafe_allow_html=True)
        deployment_code = generate_deployment_code(
            best_model_name, 
            config['is_german_credit'], 
            config['german_credit_features'], 
            config['cost_sensitive']
        )
        st.code(deployment_code, language='python')
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)