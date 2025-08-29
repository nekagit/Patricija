from __future__ import annotations

FIELD_KEYS = [
    "checking_account_status", "duration_months", "credit_history", "purpose",
    "credit_amount", "savings_account", "employment_since",
    "installment_rate_percent", "personal_status_sex", "other_debtors_guarantors",
    "residence_since", "property", "age_years", "other_installment_plans",
    "housing", "num_existing_credits", "job", "num_dependents",
    "telephone", "foreign_worker",
]

OPTIONS = {
    "checking_account_status": {
        "A11": "Kein Girokonto",
        "A12": "Guthaben < 0 DM",
        "A13": "0 ≤ Guthaben < 200 DM",
        "A14": "Guthaben ≥ 200 DM",
    },
    "credit_history": {
        "A30": "Keine Kredite / Alle zurückgezahlt",
        "A31": "Alle Kredite bei dieser Bank zurückgezahlt",
        "A32": "Bestehende Kredite pünktlich bezahlt",
        "A33": "Zahlungsschwierigkeiten in der Vergangenheit",
        "A34": "Kritisches Konto / Andere Kredite vorhanden",
    },
    "purpose": {
        "A40": "Neuwagen",
        "A41": "Gebrauchtwagen",
        "A42": "Möbel/Ausstattung",
        "A43": "Radio/Fernseher",
        "A44": "Haushaltsgeräte",
        "A45": "Reparaturen",
        "A46": "Ausbildung",
        "A49": "Geschäftlich",
        "A410": "Sonstiges",
    },
    "savings_account": {
        "A61": "< 100 DM",
        "A62": "100 – 500 DM",
        "A63": "500 – 1000 DM",
        "A64": "≥ 1000 DM",
        "A65": "Kein Sparkonto",
    },
    "employment_since": {
        "A71": "Arbeitslos",
        "A72": "< 1 Jahr",
        "A73": "1 – 4 Jahre",
        "A74": "4 – 7 Jahre",
        "A75": "≥ 7 Jahre",
    },
    "personal_status_sex": {
        "A91": "Männlich, geschieden/getrennt",
        "A92": "Weiblich, geschieden/getrennt/verheiratet",
        "A93": "Männlich, ledig",
        "A94": "Männlich, verheiratet/verwitwet",
    },
    "other_debtors_guarantors": {
        "A101": "Keine",
        "A102": "Mitantragsteller",
        "A103": "Bürge",
    },
    "property": {
        "A121": "Immobilien",
        "A122": "Bausparvertrag/Lebensversicherung",
        "A123": "Auto/Sonstiges",
        "A124": "Kein Eigentum",
    },
    "other_installment_plans": {
        "A141": "Bank",
        "A142": "Geschäfte",
        "A143": "Keine",
    },
    "housing": {
        "A151": "Miete",
        "A152": "Eigentum",
        "A153": "Kostenlos wohnen",
    },
    "job": {
        "A171": "Arbeitslos/Ungelernt",
        "A172": "Ungelernter Arbeitnehmer",
        "A173": "Facharbeiter/Angestellter",
        "A174": "Management/Selbstständig/Akademiker",
    },
}

DEFAULTS = {
    "duration_months": 24,
    "credit_amount": 2000,
    "installment_rate_percent": 2,
    "num_existing_credits": 1,
    "age_years": 35,
    "residence_since": 2,
    "num_dependents": 1,
    "telephone": "A192",
    "foreign_worker": "A202",
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
