import streamlit as st
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from config import TARGET_VARIABLE, CREDIT_RISK_TARGET
except ImportError:
    TARGET_VARIABLE = "loan_status"
    CREDIT_RISK_TARGET = "loan_status"

from utils.dataset_compatibility import (
    validate_target_variable, 
    get_target_mapping, 
    load_dataset_with_compatibility,
    get_available_target_variables,
    find_target_variable,
    clean_column_names
)

def target_variable_sidebar():
    st.sidebar.markdown(" Zielvariable Konfiguration")
    
    try:
        # Load dataset with compatibility
        data_path = Path(__file__).parent.parent / "data" / "credit_risk_dataset.csv"
        
        # First try to load raw dataset to show debugging info
        df_raw = pd.read_csv(data_path)
        st.sidebar.info(f"**Debug: Raw columns:** {list(df_raw.columns)}")
        
        df = load_dataset_with_compatibility(data_path)
        
        # Clean column names for display
        df_display = clean_column_names(df.copy())
        available_columns = list(df_display.columns)
        
    except Exception as e:
        st.sidebar.error(f"Fehler beim Laden des Datensatzes: {str(e)}")
        available_columns = []
        df_display = None
    
    st.sidebar.info(f"**Aktuelle Zielvariable:** {TARGET_VARIABLE}")
    
    if available_columns and df_display is not None:
        # Check if target variable exists using improved detection
        if validate_target_variable(df_display):
            target_col = find_target_variable(df_display)
            st.sidebar.success(f"✅ Zielvariable '{target_col}' gefunden")
            
            target_dist = df_display[target_col].value_counts()
            st.sidebar.markdown("**Verteilung:**")
            for value, count in target_dist.items():
                percentage = (count / len(df_display)) * 100
                st.sidebar.text(f"  {value}: {count:,} ({percentage:.1f}%)")
        else:
            st.sidebar.error(f"❌ Zielspalte '{TARGET_VARIABLE}' nicht gefunden. Bitte in der Sidebar anpassen oder Daten prüfen.")
            
            # Show available columns
            st.sidebar.markdown("**Verfügbare Spalten:**")
            for col in available_columns:
                st.sidebar.text(f"  • {col}")
            
            # Show potential target variables
            potential_targets = get_available_target_variables(df_display)
            
            if potential_targets:
                st.sidebar.markdown("**Potentielle Zielvariablen:**")
                for target in potential_targets:
                    st.sidebar.text(f"  🔍 {target}")
                    
                # Allow user to select a different target variable
                st.sidebar.markdown("**Zielvariable ändern:**")
                selected_target = st.sidebar.selectbox(
                    "Neue Zielvariable wählen:",
                    potential_targets,
                    index=0 if potential_targets else None
                )
                
                if st.sidebar.button("Zielvariable aktualisieren"):
                    # Update the target variable in session state
                    st.session_state.target_variable = selected_target
                    st.sidebar.success(f"Zielvariable auf '{selected_target}' geändert")
                    st.rerun()
    
    mapping = get_target_mapping()
    st.sidebar.markdown("**Variablen-Mapping:**")
    st.sidebar.text(f"  Alte Variable: {mapping['old_target']}")
    st.sidebar.text(f"  Neue Variable: {mapping['new_target']}")
    st.sidebar.text(f"  Alias: {mapping['alias']}")
    
    st.sidebar.markdown("## ⚙️ Konfiguration")
    
    if available_columns and df_display is not None:
        st.sidebar.markdown(f"**Datensatz:**")
        st.sidebar.text(f"  Zeilen: {len(df_display):,}")
        st.sidebar.text(f"  Spalten: {len(available_columns)}")
        
        st.sidebar.markdown("**Datentypen:**")
        for col in available_columns[:5]:
            dtype = str(df_display[col].dtype)
            st.sidebar.text(f"  {col}: {dtype}")
        if len(available_columns) > 5:
            st.sidebar.text(f"  ... und {len(available_columns) - 5} weitere")
    
    with st.sidebar.expander("❓ Hilfe"):
        st.markdown("""
        **Zielvariable Problem?**
        
        Falls die Zielvariable nicht gefunden wird:
        1. Prüfen Sie den Dateinamen des Datensatzes
        2. Stellen Sie sicher, dass die Spalte existiert
        3. Überprüfen Sie die Schreibweise
        4. Wählen Sie eine andere Zielvariable aus der Liste
        
        **Unterstützte Formate:**
        - CSV-Dateien
        - Standard-Delimiters
        - UTF-8 Encoding
        
        **Automatische Erkennung:**
        - Spalten mit 'status', 'risk', 'default', 'target', 'label', 'class'
        - Case-insensitive Matching
        - Automatische Spaltenbereinigung
        """)
