import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from config import TARGET_VARIABLE, CREDIT_RISK_TARGET
except ImportError:
    TARGET_VARIABLE = "loan_status"
    CREDIT_RISK_TARGET = "loan_status"

def clean_column_names(df):
    """Clean column names by removing extra spaces and standardizing format."""
    df.columns = df.columns.str.strip()
    return df

def get_target_variable_name():
    return TARGET_VARIABLE

def find_target_variable(df):
    """Find the target variable in the dataset, handling variations in naming."""
    # First clean the column names
    df_cleaned = clean_column_names(df.copy())
    cleaned_columns = list(df_cleaned.columns)
    
    # Direct match
    if TARGET_VARIABLE in cleaned_columns:
        return TARGET_VARIABLE
    
    # Case-insensitive match
    for col in cleaned_columns:
        if col.lower() == TARGET_VARIABLE.lower():
            return col
    
    # Partial match for common target variable names
    target_keywords = ['loan_status', 'status', 'risk', 'default', 'target', 'label', 'class']
    for keyword in target_keywords:
        for col in cleaned_columns:
            if keyword.lower() in col.lower():
                return col
    
    return None

def map_target_variable(df):
    """Map target variable with better error handling."""
    # Clean column names first
    df_cleaned = clean_column_names(df.copy())
    
    target_col = find_target_variable(df_cleaned)
    if target_col:
        # Find the original column name (with spaces) to map correctly
        original_col = None
        for col in df.columns:
            if col.strip() == target_col:
                original_col = col
                break
        
        if original_col:
            df[CREDIT_RISK_TARGET] = df[original_col]
        else:
            df[CREDIT_RISK_TARGET] = df_cleaned[target_col]
        return df
    else:
        available_cols = list(df.columns)
        raise ValueError(f"Target variable '{TARGET_VARIABLE}' not found in dataset. Available columns: {available_cols}")

def validate_target_variable(df):
    """Validate target variable with better detection."""
    df = clean_column_names(df)
    return find_target_variable(df) is not None

def get_target_distribution(df):
    """Get target distribution with better error handling."""
    df = clean_column_names(df)
    
    target_col = find_target_variable(df)
    if not target_col:
        raise ValueError(f"Target variable '{TARGET_VARIABLE}' not found in dataset")
    
    return df[target_col].value_counts().to_dict()

def get_target_mapping():
    return {
        'old_target': 'credit_risk',
        'new_target': TARGET_VARIABLE,
        'alias': CREDIT_RISK_TARGET
    }

def load_dataset_with_compatibility(data_path):
    """Load dataset with improved compatibility and column cleaning."""
    try:
        df = pd.read_csv(data_path)
        
        # Clean numerical data - handle empty strings and spaces
        for col in df.columns:
            if col.strip() in ['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length']:
                # Replace empty strings and spaces with NaN
                df[col] = df[col].replace(['', ' ', 'nan', 'NaN'], pd.NA)
                # Convert to numeric, coercing errors to NaN
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Check if target variable exists before cleaning
        if not validate_target_variable(df):
            available_cols = list(df.columns)
            raise ValueError(f"Target variable '{TARGET_VARIABLE}' not found in dataset. Available columns: {available_cols}")
        
        # Map target variable (this will handle the cleaning internally)
        df = map_target_variable(df)
        
        # Finally clean all column names for consistency
        df = clean_column_names(df)
        
        return df
        
    except Exception as e:
        raise Exception(f"Error loading dataset: {str(e)}")

def get_feature_columns(df, exclude_target=True):
    """Get feature columns with better target variable handling."""
    df = clean_column_names(df)
    
    if exclude_target:
        target_col = find_target_variable(df)
        exclude_cols = [CREDIT_RISK_TARGET]
        if target_col:
            exclude_cols.append(target_col)
        return [col for col in df.columns if col not in exclude_cols]
    else:
        return list(df.columns)

def get_available_target_variables(df):
    """Get list of potential target variables in the dataset."""
    df = clean_column_names(df)
    
    target_keywords = ['status', 'risk', 'default', 'target', 'label', 'class', 'outcome']
    potential_targets = []
    
    for col in df.columns:
        if any(keyword in col.lower() for keyword in target_keywords):
            potential_targets.append(col)
    
    return potential_targets
