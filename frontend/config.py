import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"
DATA_DIR = FRONTEND_DIR / "data"
MODELS_DIR = FRONTEND_DIR / "models"

APP_NAME = "XAI Kreditprüfung"
APP_VERSION = "1.0.0"
APP_DESCRIPTION = "Transparent Credit Assessment with Explainable AI"

STREAMLIT_PORT = int(os.getenv("STREAMLIT_PORT", "8501"))
STREAMLIT_HOST = os.getenv("STREAMLIT_HOST", "localhost")

DEFAULT_MODEL = "random_forest"
MODEL_CACHE_SIZE = 5

CREDIT_FEATURES = [
    "checking_account_status",
    "duration_months",
    "credit_history",
    "purpose",
    "credit_amount",
    "savings_account",
    "employment_since",
    "installment_rate_percent",
    "personal_status_sex",
    "other_debtors_guarantors",
    "residence_since",
    "property",
    "age_years",
    "other_installment_plans",
    "housing",
    "num_existing_credits",
    "job",
    "num_dependents",
    "telephone",
    "foreign_worker"
]

TARGET_VARIABLE = "loan_status"
CREDIT_RISK_TARGET = "loan_status"

CATEGORICAL_FEATURES = [
    "checking_account_status",
    "credit_history",
    "purpose",
    "savings_account",
    "employment_since",
    "personal_status_sex",
    "other_debtors_guarantors",
    "property",
    "other_installment_plans",
    "housing",
    "job",
    "telephone",
    "foreign_worker"
]

NUMERICAL_FEATURES = [
    "duration_months",
    "credit_amount",
    "installment_rate_percent",
    "residence_since",
    "age_years",
    "num_existing_credits",
    "num_dependents"
]

RISK_CATEGORIES = {
    "low": {"min_score": 0.7, "color": "#2ECC71"},
    "medium": {"min_score": 0.4, "color": "#F7C948"},
    "high": {"min_score": 0.0, "color": "#FF5C5C"}
}

VALIDATION_RULES = {
    "duration_months": {"min": 6, "max": 48},
    "credit_amount": {"min": 697, "max": 9055},
    "installment_rate_percent": {"min": 2, "max": 4},
    "residence_since": {"min": 2, "max": 4},
    "age_years": {"min": 22, "max": 67},
    "num_existing_credits": {"min": 1, "max": 2},
    "num_dependents": {"min": 1, "max": 2}
}

CATEGORICAL_VALUES = {
    "checking_account_status": ["A11", "A12", "A14"],
    "credit_history": ["A30", "A32", "A33", "A34"],
    "purpose": ["A40"],
    "savings_account": ["A61", "A65"],
    "employment_since": ["A71", "A73", "A74"],
    "personal_status_sex": ["A93"],
    "other_debtors_guarantors": ["A101", "A103"],
    "property": ["A121", "A122", "A124"],
    "other_installment_plans": ["A143"],
    "housing": ["A152"],
    "job": ["A172", "A173"],
    "telephone": ["A191", "A192"],
    "foreign_worker": ["A201"]
}

XAI_METHODS = ["shap", "lime"]
DEFAULT_XAI_METHOD = "shap"

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = PROJECT_ROOT / "logs" / "app.log"

def ensure_directories():
    directories = [
        PROJECT_ROOT / "logs",
        DATA_DIR,
        MODELS_DIR
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

ensure_directories()
