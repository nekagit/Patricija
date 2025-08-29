from __future__ import annotations
import io
import copy
import json
import math
import numpy as np
import pandas as pd
import streamlit as st
from config import VALIDATION_RULES, CATEGORICAL_VALUES

def _pct(x: float) -> str:
    try:
        return f"{float(x):.1%}"
    except Exception:
        return "–"

def _safe_int(x, fallback=0):
    try:
        return int(x)
    except Exception:
        return fallback

def _safe_float(x, fallback=0.0):
    try:
        return float(x)
    except Exception:
        return fallback

def _baseline_probs(result: dict) -> tuple[float, float]:
    p_good, p_bad = 0.0, 0.0
    
    if "probability_good" in result and "probability_bad" in result:
        p_good = float(result["probability_good"])
        p_bad = float(result["probability_bad"])
    
    elif "probabilities" in result:
        probs = result["probabilities"]
        prediction = result.get("prediction", "")
        
        if isinstance(probs, (list, tuple)) and len(probs) >= 2:
            if str(prediction).lower() in ['good', '1', 'true']:
                if probs[1] > probs[0]:
                    p_good = float(probs[1])
                    p_bad = float(probs[0])
                else:
                    p_good = float(probs[0])
                    p_bad = float(probs[1])
            else:
                if probs[0] > probs[1]:
                    p_bad = float(probs[0])
                    p_good = float(probs[1])
                else:
                    p_bad = float(probs[1])
                    p_good = float(probs[0])
                    
        elif isinstance(probs, (list, tuple)) and len(probs) == 1:
            prob_value = float(probs[0])
            prediction = result.get("prediction", "")
            
            if str(prediction).lower() in ['good', '1', 'true']:
                p_good = prob_value
                p_bad = 1.0 - prob_value
            else:
                p_bad = prob_value
                p_good = 1.0 - prob_value
    
    elif "predict_proba" in result:
        proba = result["predict_proba"]
        prediction = result.get("prediction", "")
        
        if isinstance(proba, (list, tuple)) and len(proba) >= 2:
            if str(prediction).lower() in ['good', '1', 'true']:
                if proba[1] > proba[0]:
                    p_good = float(proba[1])
                    p_bad = float(proba[0])
                else:
                    p_good = float(proba[0])
                    p_bad = float(proba[1])
            else:
                if proba[0] > proba[1]:
                    p_bad = float(proba[0])
                    p_good = float(proba[1])
                else:
                    p_bad = float(proba[1])
                    p_good = float(proba[0])
    
    elif "prob_0" in result and "prob_1" in result:
        prediction = result.get("prediction", "")
        prob_0 = float(result["prob_0"])
        prob_1 = float(result["prob_1"])
        
        if str(prediction).lower() in ['good', '1', 'true']:
            if prob_1 > prob_0:
                p_good = prob_1
                p_bad = prob_0
            else:
                p_good = prob_0
                p_bad = prob_1
        else:
            if prob_0 > prob_1:
                p_bad = prob_0
                p_good = prob_1
            else:
                p_bad = prob_1
                p_good = prob_0
    
    elif "confidence" in result or "score" in result:
        conf = float(result.get("confidence", result.get("score", 0.5)))
        pred = result.get("prediction", "")
        if str(pred).lower() in ["good", "1", "true"]:
            p_good = conf
            p_bad = 1.0 - conf
        else:
            p_bad = conf
            p_good = 1.0 - conf
    
    if p_good == 0.0 and p_bad == 0.0:
        pred = result.get("prediction", "")
        if str(pred).lower() in ["good", "1", "true"]:
            p_good = 0.7
            p_bad = 0.3
        else:
            p_good = 0.3
            p_bad = 0.7
    
    total = p_good + p_bad
    if total > 0:
        p_good = p_good / total
        p_bad = p_bad / total
    else:
        p_good = 0.5
        p_bad = 0.5
    
    return p_good, p_bad

def _risk_text(prob_good: float) -> str:
    if prob_good >= 0.75:
        return "Niedriges Risiko (hohe Genehmigungswahrscheinlichkeit)"
    elif prob_good >= 0.50:
        return "Mittleres Risiko"
    elif prob_good >= 0.30:
        return "Erhöhtes Risiko"
    else:
        return "Hohes Risiko (geringe Genehmigungswahrscheinlichkeit)"

def _clone_and_update(base: dict, **updates) -> dict:
    d = dict(base or {})
    d.update({k: v for k, v in updates.items() if v is not None})
    return d

def _ensure_key(d: dict, key: str, default):
    if key not in d or d[key] in (None, ""):
        d[key] = default
    return d

def render_details_whatif_tab(result: dict, applicant_data_readable: dict, application_id: str = None):
    from utils.prediction import run_predictions
    
    st.subheader("📊 Ergebnis & Was-wäre-wenn")
    p_good, p_bad = _baseline_probs(result)
    pred = result.get("prediction", "Unbekannt")

    st.markdown(
        f"""
**Aktuelle Entscheidung:** **{('✅ Gute Bonität' if pred=='Good' else '❌ Hohes Risiko')}**  
**Einschätzung:** {_risk_text(p_good)}
"""
    )
    with st.expander("Wie wird das eingeordnet? (Erläuterung)",expanded=True):
        st.markdown(
            """
- **Das Modell** bewertet die Angaben und schätzt eine **Wahrscheinlichkeit** für *gute Bonität*.
- **Hohe Werte** deuten auf **Genehmigung** hin; niedrige auf **Ablehnung**.
- Die **Grenzen** (z. B. 0.3/0.5/0.75) sind Richtwerte – Institute können strengere/lockerere Schwellen nutzen.
- Nutzen Sie unten **Was-wäre-wenn** und **Szenarien**, um zu sehen, wie **konkrete Anpassungen** die Chance verändern.
"""
        )
    st.divider()
    
    original_raw = st.session_state.get("form_data", result.get("input_data_raw", {}))
    readable_display = st.session_state.get("applicant_data_readable", applicant_data_readable)

    st.markdown("#### Ihre ursprünglichen Angaben")
    if readable_display:
        df_in = pd.DataFrame.from_dict(readable_display, orient="index", columns=["Ihre Eingabe"])
        st.dataframe(df_in, use_container_width=True, height=360)
    else:
        st.info("Keine lesbaren Ursprungsangaben gefunden.")
