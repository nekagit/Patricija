from __future__ import annotations
import re
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st


def _prettify_rule(text: str) -> str:
    t = str(text).strip()
    m = re.match(r'^\s*([^\s<>=]+)\s*<\s*([A-Za-z0-9_]+)\s*<=\s*([^\s]+)\s*$', t)
    if m:
        a, feat, b = m.groups()
        return f"{feat.replace('_', ' ').title()} ∈ ({a}, {b}]"
    m = re.match(r'^\s*([A-Za-z0-9_]+)\s*([<>=]+)\s*([^\s]+)\s*$', t)
    if m:
        feat, op, val = m.groups()
        op = {'<=': '≤', '>=': '≥'}.get(op, op)
        return f"{feat.replace('_', ' ').title()} {op} {val}"
    m = re.match(r'^\s*([A-Za-z0-9_]+)\s+in\s+\{(.+)\}\s*$', t)
    if m:
        feat, vals = m.groups()
        return f"{feat.replace('_',' ').title()} ∈ {{{vals.replace('_',' ')}}}"
    return t.replace('_', ' ').title()


def _plot_barh(names, weights, title: str):
    colors = ["#2ECC71" if w > 0 else "#FF6B6B" for w in weights]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(range(len(weights)), weights, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Einfluss (Gewichtung)")
    ax.set_title(title)
    ax.axvline(0, color="grey", lw=0.8)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)


def _bullets_for(names, vals, predicted: str, max_items: int = 8) -> list[str]:
    lines = []
    for i, (n, v) in enumerate(zip(names[:max_items], vals[:max_items]), start=1):
        direction = ("unterstützt **Risiko**" if predicted == "Bad" else "unterstützt **Genehmigung**") if v > 0 \
                    else ("wirkt **gegen Risiko**" if predicted == "Bad" else "wirkt **gegen Genehmigung**")
        lines.append(f"- **{i}. {n}** → {direction} (Gewicht {v:+.4f})")
    return lines


def _render_from_lime(lime_exp, top_k: int, predicted: str) -> bool:
    try:
        pairs = lime_exp.as_list()
        if not pairs:
            return False
        rules, weights = zip(*pairs)
        weights = np.asarray(weights, dtype=float)
        idx = np.argsort(np.abs(weights))[::-1][:top_k]
        names = [_prettify_rule(rules[i]) for i in idx]
        vals = weights[idx]
    except Exception:
        return False

    st.write("**Was wird hier gezeigt?** Dieser Plot zeigt die **wichtigsten lokalen Faktoren** (Balkenlänge = Einflussstärke).")
    st.write("**Farben:** Grün = Beitrag in Richtung **vorhergesagter Klasse**, Rot = Beitrag **dagegen**.")
    st.write("**Hinweis:** Positive Gewichte verschieben die Entscheidung **in Richtung** der vorhergesagten Klasse.")

    _plot_barh(names, vals, f"Top-Faktoren (LIME) • Vorhersage: {predicted}")

    st.write("**Kurzinterpretation der Top-Faktoren:**")
    for line in _bullets_for(names, vals, predicted):
        st.write(line)

    st.write("**So lesen Sie den Plot:**")
    st.write("- Längere Balken bedeuten **stärkeren Einfluss** auf die Entscheidung.")
    st.write("- Balken **rechts von 0** (positiv) stützen die vorhergesagte Klasse direkt.")
    st.write("- Balken **links von 0** (negativ) wirken der Vorhersage entgegen.")

    st.write("**Tipps:**")
    st.write("- Konzentrieren Sie sich auf die **Top 3–5** Balken – sie erklären oft den Großteil der lokalen Entscheidung.")
    st.write("- Prüfen Sie, ob die gezeigten Regeln **geschäftlich sinnvoll** sind (Plausibilitätscheck).")
    st.write("- Kleine Unterschiede nahe 0 sind **nicht entscheidungsprägend**.")

    return True


def _render_from_shap(shap_obj, feature_names: list, top_k: int, predicted: str) -> bool:
    try:
        if not hasattr(shap_obj, 'values'):
            return False
        
        values = np.asarray(shap_obj.values)
        if values.ndim == 3:
            values = values[0, :, 1] if predicted == "Good" else values[0, :, 0]
        elif values.ndim == 2:
            values = values[0, :]
        else:
            values = values.flatten()
        
        if len(values) != len(feature_names):
            return False
        
        idx = np.argsort(np.abs(values))[::-1][:top_k]
        names = [feature_names[i] for i in idx]
        vals = values[idx]
        
    except Exception:
        return False

    st.write("**Alternative Erklärung (SHAP-basiert):**")
    st.write("Da LIME nicht verfügbar war, zeigen wir hier eine SHAP-basierte lokale Erklärung.")
    st.write("**Interpretation:** Ähnlich wie LIME, aber basierend auf SHAP-Werten.")

    _plot_barh(names, vals, f"Top-Faktoren (SHAP) • Vorhersage: {predicted}")

    st.write("**Kurzinterpretation der Top-Faktoren:**")
    for line in _bullets_for(names, vals, predicted):
        st.write(line)

    return True


def render_lime_tab(result: dict):
    st.subheader("🔬 Alternative Erklärung (LIME)")
    
    lime_exp = result.get("lime_explanation")
    shap_obj = result.get("shap_explanation_object")
    feature_names = result.get("feature_names", [])
    predicted = result.get("prediction", "Good")
    
    if lime_exp is None and shap_obj is None:
        st.warning("Keine LIME- oder SHAP-Erklärung verfügbar.")
        st.info("LIME benötigt spezielle Hintergrunddaten, die möglicherweise nicht verfügbar sind.")
        return
    
    top_k = st.slider("Anzahl Top-Faktoren", min_value=3, max_value=15, value=8)
    
    if lime_exp is not None:
        if not _render_from_lime(lime_exp, top_k, predicted):
            st.warning("LIME-Erklärung konnte nicht verarbeitet werden.")
            if shap_obj is not None and feature_names:
                st.info("Versuche SHAP-basierte Alternative...")
                _render_from_shap(shap_obj, feature_names, top_k, predicted)
    elif shap_obj is not None and feature_names:
        st.info("LIME nicht verfügbar, verwende SHAP-basierte Alternative.")
        _render_from_shap(shap_obj, feature_names, top_k, predicted)
    else:
        st.error("Keine Erklärungsmethode verfügbar.")
    
    st.markdown("---")
    st.markdown("**Über LIME:**")
    st.markdown("""
    LIME (Local Interpretable Model-agnostic Explanations) erklärt einzelne Vorhersagen durch:
    - **Lokale Approximation** des komplexen Modells um den betrachteten Punkt
    - **Interpretierbare Features** (z.B. "Alter > 30")
    - **Gewichtete Beiträge** zu der spezifischen Vorhersage
    
    **Vorteile:**
    - Einfach zu verstehen
    - Funktioniert mit jedem Modell
    - Lokale Perspektive
    
    **Nachteile:**
    - Nur lokale Erklärung (nicht global)
    - Kann instabil sein bei kleinen Änderungen
    """)
