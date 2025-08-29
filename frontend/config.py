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
    "person_age",
    "person_income",
    "person_home_ownership",
    "person_emp_length",
    "cb_person_cred_hist_length",
    "cb_person_default_on_file",
    "loan_intent",
    "loan_grade",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income"
]

TARGET_VARIABLE = "loan_status"
CREDIT_RISK_TARGET = "loan_status"

CATEGORICAL_FEATURES = [
    "person_home_ownership",
    "cb_person_default_on_file",
    "loan_intent",
    "loan_grade"
]

NUMERICAL_FEATURES = [
    "person_age",
    "person_income",
    "person_emp_length",
    "cb_person_cred_hist_length",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income"
]

RISK_CATEGORIES = {
    "low": {"min_score": 0.7, "color": "#2ECC71"},
    "medium": {"min_score": 0.4, "color": "#F7C948"},
    "high": {"min_score": 0.0, "color": "#FF5C5C"}
}

VALIDATION_RULES = {
    "person_age": {"min": 18, "max": 100},
    "person_income": {"min": 0, "max": 1000000},
    "person_emp_length": {"min": 0.0, "max": 50.0},
    "cb_person_cred_hist_length": {"min": 0, "max": 50},
    "loan_amnt": {"min": 100, "max": 1000000},
    "loan_int_rate": {"min": 0.0, "max": 100.0},
    "loan_percent_income": {"min": 0.0, "max": 1.0}
}

CATEGORICAL_VALUES = {
    "person_home_ownership": ["RENT", "OWN", "MORTGAGE", "OTHER"],
    "cb_person_default_on_file": ["Y", "N"],
    "loan_intent": ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"],
    "loan_grade": ["A", "B", "C", "D", "E", "F"]
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
