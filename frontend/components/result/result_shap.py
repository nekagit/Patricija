from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import streamlit as st
import sys
from pathlib import Path
# Add the root directory to the path to access the utils module
root_path = str(Path(__file__).parent.parent.parent.parent)
if root_path not in sys.path:
    sys.path.insert(0, root_path)
from utils.plot_storage import plot_storage

def _explanation_for_class(
    expl: shap.Explanation,
    class_index: int,
    feature_names: list[str] | None,
) -> shap.Explanation:
    fn = feature_names or getattr(expl, "feature_names", None)

    vals = getattr(expl, "values", None)
    if vals is not None and getattr(vals, "ndim", 0) == 2:
        return shap.Explanation(
            values=expl.values,
            base_values=getattr(expl, "base_values", None),
            data=getattr(expl, "data", None),
            feature_names=fn,
        )

    vals = np.asarray(expl.values)
    if vals.ndim == 3:
        v = vals[:, :, class_index]
        base = getattr(expl, "base_values", None)
        if base is not None and np.ndim(base) == 2 and base.shape[1] > class_index:
            base = base[:, class_index]
        return shap.Explanation(
            values=v,
            base_values=base,
            data=getattr(expl, "data", None),
            feature_names=fn,
        )

    try:
        sl = expl[:, :, class_index]
        return shap.Explanation(
            values=np.asarray(sl.values),
            base_values=getattr(sl, "base_values", None),
            data=getattr(sl, "data", None),
            feature_names=fn,
        )
    except Exception:
        pass

    return shap.Explanation(
        values=np.atleast_2d(np.asarray(expl.values)),
        base_values=getattr(expl, "base_values", None),
        data=getattr(expl, "data", None),
        feature_names=fn,
    )


def _topk(names: list[str], vals: np.ndarray, k: int):
    idx = np.argsort(np.abs(vals))[::-1][: max(1, min(k, len(vals)))]
    return [names[i] for i in idx], vals[idx]


def _plot_radar(abs_vals: np.ndarray, names: list[str], application_id: str = None):
    if abs_vals.size == 0:
        st.info("Keine Daten für Radar-Chart verfügbar")
        return
    vmax = float(np.max(abs_vals)) or 1.0
    norm = abs_vals / vmax

    angles = np.linspace(0, 2 * np.pi, len(norm), endpoint=False).tolist()
    angles += angles[:1]
    values = np.concatenate([norm, norm[:1]])

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, values, alpha=0.35)
    ax.plot(angles, values, linewidth=2)
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(names, size=9)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    
    # Save the plot if application_id is provided
    if application_id:
        metadata = {
            'plot_type': 'shap_radar',
            'feature_count': len(names),
            'max_abs_value': float(vmax),
            'normalized': True
        }
        plot_storage.save_matplotlib_plot(
            fig, 'shap_radar', 
            application_id=application_id,
            metadata=metadata
        )


def _plot_signed_bar(names: list[str], vals: np.ndarray, title: str, application_id: str = None):
    if len(names) == 0:
        st.info("Keine Daten für Bar-Chart verfügbar")
        return

    fig, ax = plt.subplots(figsize=(8, max(4, len(names) * 0.3)))
    colors = ['red' if v < 0 else 'blue' for v in vals]
    bars = ax.barh(names, vals, color=colors, alpha=0.7)
    ax.set_xlabel('SHAP Value')
    ax.set_title(title)
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    
    # Save the plot if application_id is provided
    if application_id:
        metadata = {
            'plot_type': 'shap_signed_bar',
            'feature_count': len(names),
            'title': title,
            'positive_count': sum(1 for v in vals if v > 0),
            'negative_count': sum(1 for v in vals if v < 0)
        }
        plot_storage.save_matplotlib_plot(
            fig, 'shap_signed_bar', 
            application_id=application_id,
            metadata=metadata
        )


def render_shap_tab(result: dict, application_id: str = None):
    st.subheader("🎯 SHAP Erklärung")
    
    shap_obj = result.get("shap_explanation_object")
    if shap_obj is None:
        st.warning("Keine SHAP-Erklärung verfügbar. Das Modell unterstützt möglicherweise keine SHAP-Analyse.")
        return

    feature_names = result.get("feature_names", [])
    if not feature_names:
        st.warning("Feature-Namen nicht verfügbar.")
        return

    try:
        expl = _explanation_for_class(shap_obj, 1, feature_names)
        values = np.asarray(expl.values).flatten()
        
        if values.size == 0:
            st.warning("Keine SHAP-Werte verfügbar.")
            return

        st.markdown("#### 📊 Feature Impact")
        
        top_names, top_vals = _topk(feature_names, values, 10)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🔴 Negative Impact (Reduziert Kreditwürdigkeit)**")
            neg_names = [n for n, v in zip(top_names, top_vals) if v < 0]
            neg_vals = [v for v in top_vals if v < 0]
            if neg_names:
                _plot_signed_bar(neg_names, neg_vals, "Negative SHAP Values", application_id)
            else:
                st.info("Keine negativen SHAP-Werte")
        
        with col2:
            st.markdown("**🔵 Positive Impact (Erhöht Kreditwürdigkeit)**")
            pos_names = [n for n, v in zip(top_names, top_vals) if v > 0]
            pos_vals = [v for v in top_vals if v > 0]
            if pos_names:
                _plot_signed_bar(pos_names, pos_vals, "Positive SHAP Values", application_id)
            else:
                st.info("Keine positiven SHAP-Werte")

        st.markdown("#### 📈 Radar Chart")
        abs_vals = np.abs(values)
        _plot_radar(abs_vals, feature_names, application_id)

        st.markdown("#### 📋 Detaillierte SHAP-Werte")
        shap_df = pd.DataFrame({
            'Feature': feature_names,
            'SHAP Value': values,
            'Absolute Impact': abs_vals
        }).sort_values('Absolute Impact', ascending=False)
        
        st.dataframe(shap_df, use_container_width=True)

    except Exception as e:
        st.error(f"Fehler bei der SHAP-Analyse: {e}")
        st.info("Das Modell unterstützt möglicherweise keine SHAP-Analyse oder es gab einen Fehler bei der Berechnung.")