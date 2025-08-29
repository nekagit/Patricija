from __future__ import annotations

FIELD_KEYS = [
    "person_age", "person_income", "person_home_ownership", "person_emp_length",
    "cb_person_cred_hist_length", "cb_person_default_on_file", "loan_intent",
    "loan_grade", "loan_amnt", "loan_int_rate", "loan_percent_income"
]

OPTIONS = {
    "person_home_ownership": {
        "RENT": "Miete",
        "OWN": "Eigentum",
        "MORTGAGE": "Hypothek",
        "OTHER": "Sonstiges"
    },
    "cb_person_default_on_file": {
        "Y": "Ja",
        "N": "Nein"
    },
    "loan_intent": {
        "PERSONAL": "Persönlich",
        "EDUCATION": "Ausbildung",
        "MEDICAL": "Medizinisch",
        "VENTURE": "Unternehmen",
        "HOMEIMPROVEMENT": "Hausverbesserung",
        "DEBTCONSOLIDATION": "Schuldenkonsolidierung"
    },
    "loan_grade": {
        "A": "A (Beste)",
        "B": "B (Gut)",
        "C": "C (Mittel)",
        "D": "D (Schlecht)",
        "E": "E (Sehr schlecht)",
        "F": "F (Schlechteste)"
    }
}

DEFAULTS = {
    "person_age": 30,
    "person_income": 50000,
    "person_home_ownership": "RENT",
    "person_emp_length": 5.0,
    "cb_person_cred_hist_length": 3,
    "cb_person_default_on_file": "N",
    "loan_intent": "PERSONAL",
    "loan_grade": "B",
    "loan_amnt": 20000,
    "loan_int_rate": 12.0,
    "loan_percent_income": 0.3
}

def to_readable(d: dict) -> dict:
    r = {}
    for k, v in d.items():
        if k in OPTIONS:
            r[k] = OPTIONS[k].get(str(v), str(v))
        else:
            r[k] = str(v)
    return r

def lab(options: dict, field: str, value: str) -> str:
    if field in options and value in options[field]:
        return options[field][value]
    return str(value)
