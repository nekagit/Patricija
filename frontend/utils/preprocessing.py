from __future__ import annotations
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# New credit risk dataset field mappings
CREDIT_RISK_DECODING_MAP = {
    'person_home_ownership': {
        'RENT': 'Miete',
        'OWN': 'Eigentum', 
        'MORTGAGE': 'Hypothek',
        'OTHER': 'Sonstiges'
    },
    'cb_person_default_on_file': {
        'Y': 'Ja',
        'N': 'Nein'
    },
    'loan_intent': {
        'PERSONAL': 'Persönlich',
        'EDUCATION': 'Ausbildung',
        'MEDICAL': 'Medizinisch',
        'VENTURE': 'Unternehmen',
        'HOMEIMPROVEMENT': 'Hausverbesserung',
        'DEBTCONSOLIDATION': 'Schuldenkonsolidierung'
    },
    'loan_grade': {
        'A': 'A (Beste)',
        'B': 'B (Gut)',
        'C': 'C (Mittel)',
        'D': 'D (Schlecht)',
        'E': 'E (Sehr schlecht)',
        'F': 'F (Schlechteste)'
    }
}

def decode_credit_risk_data(df: pd.DataFrame) -> pd.DataFrame:
    df_decoded = df.copy()
    for column, mapping in CREDIT_RISK_DECODING_MAP.items():
        if column in df_decoded.columns:
            df_decoded[column] = df_decoded[column].map(mapping).fillna(df_decoded[column])
    return df_decoded

def decode_german_credit_data(df: pd.DataFrame) -> pd.DataFrame:
    """Legacy function for German credit dataset decoding."""
    df_decoded = df.copy()
    
    # Old German credit dataset mappings
    GERMAN_CREDIT_DECODING_MAP = {
        'checking_account_status': {
            'A11': '< 0 DM', 'A12': '0 <= ... < 200 DM', 'A13': '>= 200 DM', 'A14': 'no checking account'
        },
        'credit_history': {
            'A30': 'no credits/all paid back', 'A31': 'all credits at this bank paid back', 'A32': 'existing credits paid back duly', 'A33': 'delay in paying off in the past', 'A34': 'critical account/other credits'
        },
        'purpose': {
            'A40': 'car (new)', 'A41': 'car (used)', 'A42': 'furniture/equipment', 'A43': 'radio/television', 'A44': 'domestic appliances', 'A45': 'repairs', 'A46': 'education', 'A47': 'vacation', 'A48': 'retraining', 'A49': 'business', 'A410': 'others'
        },
        'savings_account': {
            'A61': '< 100 DM', 'A62': '100 <= ... < 500 DM', 'A63': '500 <= ... < 1000 DM', 'A64': '>= 1000 DM', 'A65': 'unknown/no savings account'
        },
        'employment_since': {
            'A71': 'unemployed', 'A72': '< 1 year', 'A73': '1 <= ... < 4 years', 'A74': '4 <= ... < 7 years', 'A75': '>= 7 years'
        },
        'personal_status_sex': {
            'A91': 'male : divorced/separated', 'A92': 'female : divorced/separated/married', 'A93': 'male : single', 'A94': 'male : married/widowed', 'A95': 'female : single'
        },
        'other_debtors_guarantors': {
            'A101': 'none', 'A102': 'co-applicant', 'A103': 'guarantor'
        },
        'property': {
            'A121': 'real estate', 'A122': 'building society savings/life insurance', 'A123': 'car or other', 'A124': 'unknown/no property'
        },
        'other_installment_plans': {
            'A141': 'bank', 'A142': 'stores', 'A143': 'none'
        },
        'housing': {
            'A151': 'rent', 'A152': 'own', 'A153': 'for free'
        },
        'job': {
            'A171': 'unemployed/unskilled - non-resident', 'A172': 'unskilled - resident', 'A173': 'skilled employee/official', 'A174': 'management/self-employed/highly qualified'
        },
        'telephone': {
            'A191': 'none', 'A192': 'yes, registered under the customers name'
        },
        'foreign_worker': {
            'A201': 'yes', 'A202': 'no'
        }
    }
    
    for column, mapping in GERMAN_CREDIT_DECODING_MAP.items():
        if column in df_decoded.columns:
            df_decoded[column] = df_decoded[column].map(mapping).fillna(df_decoded[column])
    return df_decoded

def read_uploaded_file(uploaded) -> pd.DataFrame:
    name = uploaded.name.lower()
    if name.endswith((".data", ".txt")):
        return pd.read_csv(uploaded, sep=r'\s+', header=None)
    return pd.read_csv(uploaded)

def detect_credit_risk_dataset(df: pd.DataFrame) -> bool:
    """Detect if the dataset is the new credit risk dataset format."""
    expected_columns = [
        'person_age', 'person_income', 'person_home_ownership', 'person_emp_length',
        'cb_person_cred_hist_length', 'cb_person_default_on_file', 'loan_intent',
        'loan_grade', 'loan_amnt', 'loan_int_rate', 'loan_percent_income'
    ]
    
    # Check if all expected columns are present
    df_columns = [col.strip() for col in df.columns]
    return all(col in df_columns for col in expected_columns)

def detect_german_credit_dataset(df: pd.DataFrame) -> bool:
    """Detect if the dataset is the old German credit dataset format."""
    if df.shape[1] == 21:
        if any(v.startswith("A1") for v in df.iloc[:, 0].astype(str).unique()):
            return True
    return df.shape[1] == 20

def preprocess_credit_risk_data(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    dfp = df.copy()
    
    try:
        from utils.dataset_compatibility import get_target_variable_name
        target_var = get_target_variable_name()
    except ImportError:
        target_var = "loan_status"
    
    # Define categorical and numerical columns for the new dataset
    cat_cols = [c for c in CREDIT_RISK_DECODING_MAP.keys() if c in dfp.columns]
    num_cols = [c for c in dfp.columns if c not in cat_cols and c != target_var]
    
    return dfp, cat_cols, num_cols

def create_credit_risk_features(X: pd.DataFrame) -> pd.DataFrame:
    X_feat = X.copy()
    
    for col in X_feat.columns:
        if col in CREDIT_RISK_DECODING_MAP:
            X_feat[col] = X_feat[col].astype(str)
    
    return X_feat

def label_encode_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    encoders = {}
    df_encoded = df.copy()
    
    for column in df_encoded.columns:
        if df_encoded[column].dtype == 'object':
            le = LabelEncoder()
            df_encoded[column] = le.fit_transform(df_encoded[column].astype(str))
            encoders[column] = le
    
    return df_encoded, encoders

def preprocess_data(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Main preprocessing function that handles both old and new dataset formats."""
    if detect_credit_risk_dataset(df):
        return preprocess_credit_risk_data(df)
    else:
        # Fallback to old German credit dataset format for backward compatibility
        return preprocess_german_credit_data(df)

def preprocess_german_credit_data(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Legacy function for German credit dataset - kept for backward compatibility."""
    dfp = df.copy()
    
    try:
        from utils.dataset_compatibility import get_target_variable_name
        target_var = get_target_variable_name()
    except ImportError:
        target_var = "loan_status"
    
    # Old German credit dataset column names
    names = [
        'checking_account_status', 'duration_months', 'credit_history', 'purpose', 'credit_amount',
        'savings_account', 'employment_since', 'installment_rate_percent', 'personal_status_sex',
        'other_debtors_guarantors', 'residence_since', 'property', 'age_years', 'other_installment_plans',
        'housing', 'num_existing_credits', 'job', 'num_dependents', 'telephone', 'foreign_worker', target_var
    ]
    
    if dfp.shape[1] == 21 and all(isinstance(c, int) or str(c).isdigit() for c in dfp.columns):
        dfp.columns = names
    
    # Old German credit dataset mappings
    GERMAN_CREDIT_DECODING_MAP = {
        'checking_account_status': {
            'A11': '< 0 DM', 'A12': '0 <= ... < 200 DM', 'A13': '>= 200 DM', 'A14': 'no checking account'
        },
        'credit_history': {
            'A30': 'no credits/all paid back', 'A31': 'all credits at this bank paid back', 'A32': 'existing credits paid back duly', 'A33': 'delay in paying off in the past', 'A34': 'critical account/other credits'
        },
        'purpose': {
            'A40': 'car (new)', 'A41': 'car (used)', 'A42': 'furniture/equipment', 'A43': 'radio/television', 'A44': 'domestic appliances', 'A45': 'repairs', 'A46': 'education', 'A47': 'vacation', 'A48': 'retraining', 'A49': 'business', 'A410': 'others'
        },
        'savings_account': {
            'A61': '< 100 DM', 'A62': '100 <= ... < 500 DM', 'A63': '500 <= ... < 1000 DM', 'A64': '>= 1000 DM', 'A65': 'unknown/no savings account'
        },
        'employment_since': {
            'A71': 'unemployed', 'A72': '< 1 year', 'A73': '1 <= ... < 4 years', 'A74': '4 <= ... < 7 years', 'A75': '>= 7 years'
        },
        'personal_status_sex': {
            'A91': 'male : divorced/separated', 'A92': 'female : divorced/separated/married', 'A93': 'male : single', 'A94': 'male : married/widowed', 'A95': 'female : single'
        },
        'other_debtors_guarantors': {
            'A101': 'none', 'A102': 'co-applicant', 'A103': 'guarantor'
        },
        'property': {
            'A121': 'real estate', 'A122': 'building society savings/life insurance', 'A123': 'car or other', 'A124': 'unknown/no property'
        },
        'other_installment_plans': {
            'A141': 'bank', 'A142': 'stores', 'A143': 'none'
        },
        'housing': {
            'A151': 'rent', 'A152': 'own', 'A153': 'for free'
        },
        'job': {
            'A171': 'unemployed/unskilled - non-resident', 'A172': 'unskilled - resident', 'A173': 'skilled employee/official', 'A174': 'management/self-employed/highly qualified'
        },
        'telephone': {
            'A191': 'none', 'A192': 'yes, registered under the customers name'
        },
        'foreign_worker': {
            'A201': 'yes', 'A202': 'no'
        }
    }
    
    cat_cols = [c for c in GERMAN_CREDIT_DECODING_MAP.keys() if c in dfp.columns]
    num_cols = [c for c in dfp.columns if c not in cat_cols and c != target_var]
    
    return dfp, cat_cols, num_cols