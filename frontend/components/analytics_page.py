import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
from pathlib import Path
import sys
import os
from datetime import datetime, timedelta

# Add the root directory to the path to import config and utils
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from config import NUMERICAL_FEATURES
from utils.dataset_compatibility import (
    load_dataset_with_compatibility, 
    get_target_variable_name,
    find_target_variable,
    clean_column_names
)
from utils.api_client import get_api_client
import sys
from pathlib import Path
# Add the root directory to the path to access the utils module
root_path = str(Path(__file__).parent.parent.parent)
if root_path not in sys.path:
    sys.path.insert(0, root_path)
from utils.plot_storage import plot_storage

def analytics_page():
    st.markdown("""
        <div class="main-title text-center text-secondary mb-4 animate-fade-in">
            <h1>📊 Intelligente Kreditanalyse</h1>
            <p>Erweiterte Datenanalyse und KI-gestützte Einblicke</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Always try to load data from backend first
    df = None
    backend_connected = False
    is_demo_data = False
    
    try:
        api_client = get_api_client()
        
        # Check if backend is connected
        if api_client.check_backend_connection():
            backend_connected = True
            
            # Try to get real data from backend
            applications_response = api_client.get_applications(limit=1000)
            
            if "error" not in applications_response and "items" in applications_response:
                applications = applications_response["items"]
                if applications and len(applications) > 0:
                    # Convert backend data to DataFrame
                    df = pd.DataFrame(applications)
                else:
                    # No real data in backend, try demo data
                    demo_response = api_client.get_demo_applications(limit=1000)
                    if "error" not in demo_response and "items" in demo_response:
                        df = pd.DataFrame(demo_response["items"])
                        is_demo_data = True
            else:
                # Backend error, try demo data
                demo_response = api_client.get_demo_applications(limit=1000)
                if "error" not in demo_response and "items" in demo_response:
                    df = pd.DataFrame(demo_response["items"])
                    is_demo_data = True
        else:
            # Backend not connected, try demo data
            demo_response = api_client.get_demo_applications(limit=1000)
            if "error" not in demo_response and "items" in demo_response:
                df = pd.DataFrame(demo_response["items"])
                is_demo_data = True
    
    except Exception as e:
        return
    
    # If still no data, try local files as last resort
    if df is None or len(df) == 0:
        possible_paths = [
            Path(__file__).parent.parent / "data" / "credit_risk_dataset.csv",
            Path(__file__).parent.parent / "data" / "sample_credit_data.csv",
            Path(__file__).parent.parent.parent / "frontend" / "data" / "credit_risk_dataset.csv",
            Path(__file__).parent.parent.parent / "frontend" / "data" / "sample_credit_data.csv"
        ]
        
        for data_path in possible_paths:
            if data_path.exists():
                try:
                    df = load_dataset_with_compatibility(data_path)
                    break
                except Exception as e:
                    continue
        
        if df is None or len(df) == 0:
            return
    
    # Show backend connection status
    if not backend_connected:
        st.markdown("""
            <div class="alert alert-warning" role="alert">
                <strong>⚠️ Backend nicht verbunden</strong><br>
                Die Anwendung läuft im Demo-Modus. Für vollständige Funktionalität starten Sie bitte das Backend.
            </div>
        """, unsafe_allow_html=True)
    
    if is_demo_data:
        st.markdown("""
            <div class="alert alert-info" role="alert">
                <strong>📊 Demo-Modus aktiv</strong><br>
                Sie sehen Demo-Daten für Demonstrationszwecke. Echte Daten werden angezeigt, sobald das Backend verbunden ist.
            </div>
        """, unsafe_allow_html=True)
    
    # Clean and process data
    df = clean_column_names(df)
    
    # Get target variable
    target_var = find_target_variable(df) or get_target_variable_name()
    
    if target_var not in df.columns:
        st.error(f"Zielvariable '{target_var}' nicht gefunden.")
        return
    
    # Create tabs for different analysis sections
    tab1, tab2, tab3 = st.tabs([
        "📈 Übersicht", 
        "🎯 Risikoanalyse", 
        "📋 Details"
    ])
    
    with tab1:
        overview_tab(df, target_var)
    
    with tab2:
        risk_analysis_tab(df, target_var)
    
    with tab3:
        details_tab(df, target_var)

def overview_tab(df, target_var):
    """Overview tab with key metrics and summary charts."""
    
    # Calculate key metrics
    total_records = len(df)
    positive_rate = (df[target_var].sum() / total_records) * 100
    
    # Safe calculations for key features
    avg_age = df['person_age'].mean() if 'person_age' in df.columns else 0
    avg_income = df['person_income'].mean() if 'person_income' in df.columns else 0
    avg_loan = df['loan_amnt'].mean() if 'loan_amnt' in df.columns else 0
    
    # Create metrics cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📊 Gesamt Datensätze",
            value=f"{total_records:,}",
            delta=None
        )
    
    with col2:
        st.metric(
            label="✅ Positive Rate",
            value=f"{positive_rate:.1f}%",
            delta=None
        )
    
    with col3:
        st.metric(
            label="👥 Ø Alter",
            value=f"{avg_age:.1f} Jahre",
            delta=None
        )
    
    with col4:
        st.metric(
            label="💰 Ø Einkommen",
            value=f"{avg_income:,.0f} €",
            delta=None
        )
    


def risk_analysis_tab(df, target_var):
    """Risk analysis tab with detailed risk factors."""
    
    st.markdown("### 🎯 Risikofaktor Analyse")
    
    # Risk factors analysis
    risk_factors = []
    
    if 'person_age' in df.columns:
        age_risk = df.groupby(target_var)['person_age'].mean()
        risk_factors.append(('Alter', age_risk))
    
    if 'person_income' in df.columns:
        income_risk = df.groupby(target_var)['person_income'].mean()
        risk_factors.append(('Einkommen', income_risk))
    
    if 'loan_amnt' in df.columns:
        loan_risk = df.groupby(target_var)['loan_amnt'].mean()
        risk_factors.append(('Kreditbetrag', loan_risk))
    
    # Display risk factors
    if risk_factors:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Durchschnittswerte nach Kreditqualität")
            risk_df = pd.DataFrame({
                'Faktor': [factor[0] for factor in risk_factors],
                'Gute Kredite': [factor[1].get(0, 0) for factor in risk_factors],
                'Risikokredite': [factor[1].get(1, 0) for factor in risk_factors]
            })
            st.dataframe(risk_df, use_container_width=True)
        
        with col2:
            # Risk factor visualization
            if len(risk_factors) > 0:
                factors = [factor[0] for factor in risk_factors]
                good_values = [factor[1].get(0, 0) for factor in risk_factors]
                bad_values = [factor[1].get(1, 0) for factor in risk_factors]
                
                fig_risk = go.Figure()
                fig_risk.add_trace(go.Bar(
                    name='Gute Kredite',
                    x=factors,
                    y=good_values,
                    marker_color='#2ECC71'
                ))
                fig_risk.add_trace(go.Bar(
                    name='Risikokredite',
                    x=factors,
                    y=bad_values,
                    marker_color='#E74C3C'
                ))
                fig_risk.update_layout(
                    title="Risikofaktoren Vergleich",
                    barmode='group',
                    height=400
                )
                st.plotly_chart(fig_risk, use_container_width=True)
    
    # Employment length analysis
    if 'person_emp_length' in df.columns:
        st.markdown("### 💼 Beschäftigungsdauer Analyse")
        
        col1, col2 = st.columns(2)
        
        with col1:
            emp_risk = df.groupby(target_var)['person_emp_length'].mean()
            st.metric("Ø Beschäftigungsdauer - Gute Kredite", f"{emp_risk.get(0, 0):.1f} Jahre")
            st.metric("Ø Beschäftigungsdauer - Risikokredite", f"{emp_risk.get(1, 0):.1f} Jahre")
        
        with col2:
            fig_emp = px.histogram(
                df,
                x='person_emp_length',
                color=target_var,
                title="Beschäftigungsdauer Verteilung",
                labels={'person_emp_length': 'Beschäftigungsdauer (Jahre)'},
                color_discrete_map={0: '#2ECC71', 1: '#E74C3C'}
            )
            st.plotly_chart(fig_emp, use_container_width=True)





def details_tab(df, target_var):
    """Details tab with comprehensive data analysis."""
    
    st.markdown("### 📋 Detaillierte Datenanalyse")
    
    # Data overview
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Datensatz Übersicht")
        st.write(f"**Anzahl Zeilen:** {len(df):,}")
        st.write(f"**Anzahl Spalten:** {len(df.columns)}")
        st.write(f"**Zielvariable:** {target_var}")
        
        # Missing values
        missing_data = df.isnull().sum()
        if missing_data.sum() > 0:
            st.markdown("#### ⚠️ Fehlende Werte")
            missing_df = pd.DataFrame({
                'Spalte': missing_data.index,
                'Fehlende Werte': missing_data.values
            })
            st.dataframe(missing_df[missing_df['Fehlende Werte'] > 0], use_container_width=True)
        else:
            st.success("✅ Keine fehlenden Werte im Datensatz")
    
    with col2:
        st.markdown("#### 📈 Zielvariable Details")
        if target_var in df.columns:
            target_stats = df[target_var].describe()
            st.dataframe(target_stats, use_container_width=True)
    
    # Feature statistics
    st.markdown("#### 📊 Feature Statistiken")
    
    # Get numerical features
    numerical_features = [col for col in NUMERICAL_FEATURES if col in df.columns]
    
    if numerical_features:
        stats_df = df[numerical_features].describe()
        st.dataframe(stats_df, use_container_width=True)
    
    # Categorical features
    categorical_features = [col for col in df.columns if col not in numerical_features and col != target_var]
    
    if categorical_features:
        st.markdown("#### 📝 Kategorische Features")
        for feature in categorical_features[:5]:  # Show first 5 categorical features
            if feature in df.columns:
                value_counts = df[feature].value_counts()
                st.write(f"**{feature}:**")
                st.dataframe(value_counts.head(10), use_container_width=True)
    
    # Data quality metrics
    st.markdown("#### 🎯 Datenqualität")
    
    col1, col2 = st.columns(2)
    
    with col1:
        completeness = (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        st.metric("Vollständigkeit", f"{completeness:.1f}%")
    
    with col2:
        if target_var in df.columns:
            balance = min(df[target_var].value_counts()) / max(df[target_var].value_counts()) * 100
            st.metric("Klassenbalance", f"{balance:.1f}%")

def clean_dataset(df):
    """Clean the dataset for analysis."""
    df_clean = df.copy()
    
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            df_clean[col] = df_clean[col].astype('category')
    
    return df_clean
