# components/training/training_results.py
from __future__ import annotations
from pathlib import Path
import json
import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score,
    roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
)
from sklearn.calibration import calibration_curve

from utils.visualizations import (
    render_results_table,
    render_eval_plots,      # keeps your existing trio (CM/ROC/PR)
    render_artifacts_list,
)

# -------------------------------
# Small helpers
# -------------------------------
def _metric_cards(y_true: np.ndarray, y_prob: np.ndarray, thr: float):
    y_pred = (y_prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    acc = (tp + tn) / max(tn + fp + fn + tp, 1)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except Exception:
        roc_auc = float("nan")
    try:
        pr_auc = average_precision_score(y_true, y_prob)
    except Exception:
        pr_auc = float("nan")

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Accuracy", f"{acc:.4f}")
    c2.metric("Precision", f"{prec:.4f}")
    c3.metric("Recall", f"{rec:.4f}")
    c4.metric("F1", f"{f1:.4f}")
    c5.metric("ROC AUC", f"{roc_auc:.4f}")
    c6.metric("PR AUC", f"{pr_auc:.4f}")

    with st.expander("Wie lese ich diese Werte?"):
        st.markdown(
            "- **Precision**: Anteil der als *positiv* vorhergesagten Fälle, die wirklich positiv sind.\n"
            "- **Recall**: Anteil der *tatsächlich positiven* Fälle, die gefunden wurden.\n"
            "- **F1**: Harmonie zwischen Precision & Recall (gut, wenn Klassen unausgewogen sind).\n"
            "- **ROC AUC**: Wie gut trennt das Modell Klassen über alle Schwellen hinweg.\n"
            "- **PR AUC**: Besser lesbar bei stark unausgewogenen Daten."
        )

def _confusion_heatmap(y_true: np.ndarray, y_prob: np.ndarray, thr: float):
    y_pred = (y_prob >= thr).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(cm, cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=[0, 1], yticks=[0, 1],
        xticklabels=["Negativ", "Positiv"], yticklabels=["Negativ", "Positiv"],
        xlabel="Vorhergesagt", ylabel="Tatsächlich"
    )
    thresh = cm.max() / 2 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    ax.set_title(f"Konfusionsmatrix (Threshold = {thr:.2f})")
    st.pyplot(fig, use_container_width=True)

def _probability_histograms(y_true: np.ndarray, y_prob: np.ndarray):
    # hist for each class
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.hist(y_prob[y_true == 0], bins=20, alpha=0.6, label="Klasse 0", density=True)
    ax.hist(y_prob[y_true == 1], bins=20, alpha=0.6, label="Klasse 1", density=True)
    ax.set_xlabel("Vorhergesagte Wahrscheinlichkeit (positiv)")
    ax.set_ylabel("Dichte")
    ax.set_title("Wahrscheinlichkeitsverteilung je Klasse")
    ax.legend()
    st.pyplot(fig, use_container_width=True)
    with st.expander("Was sehe ich hier?"):
        st.markdown(
            "Idealerweise liegen die Wahrscheinlichkeiten der Klasse **1** rechts (hoch), "
            "und der Klasse **0** links (niedrig). Je weniger Überlappung, desto besser trennt das Modell."
        )

def _calibration_plot(y_true: np.ndarray, y_prob: np.ndarray):
    try:
        frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")
        fig, ax = plt.subplots(figsize=(5.8, 4.2))
        ax.plot([0, 1], [0, 1], "--", label="Perfekt kalibriert")
        ax.plot(mean_pred, frac_pos, marker="o", linewidth=1.5, label="Modell")
        ax.set_xlabel("Durchschnittliche vorhergesagte Wahrscheinlichkeit")
        ax.set_ylabel("Tatsächlicher Positivenanteil")
        ax.set_title("Kalibrierung (Zuverlässigkeit der Wahrscheinlichkeiten)")
        ax.legend()
        st.pyplot(fig, use_container_width=True)
    except Exception:
        st.info("Kalibrierungsplot konnte nicht berechnet werden (zu wenige Punkte oder konstante Vorhersagen).")

def _threshold_sweep_curves(y_true: np.ndarray, y_prob: np.ndarray):
    # compute precision/recall/f1 over thresholds
    thresholds = np.linspace(0, 1, 101)
    precs, recs, f1s = [], [], []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        precs.append(precision_score(y_true, y_pred, zero_division=0))
        recs.append(recall_score(y_true, y_pred, zero_division=0))
        f1s.append(f1_score(y_true, y_pred, zero_division=0))

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(thresholds, precs, label="Precision")
    ax.plot(thresholds, recs, label="Recall")
    ax.plot(thresholds, f1s, label="F1")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Threshold-Explorer (Precision/Recall/F1)")
    ax.legend()
    st.pyplot(fig, use_container_width=True)
    with st.expander("Wie nutze ich das?"):
        st.markdown(
            "Wähle einen **Schwellwert**, der zu deinem Use-Case passt: "
            "z. B. hoher **Recall** (wenig False-Negatives) bei Risiko-Screening, "
            "oder hohe **Precision** (wenig False-Positives) bei knappen Ressourcen."
        )

def _lift_gains_plots(y_true: np.ndarray, y_prob: np.ndarray):
    # sort by probability desc
    order = np.argsort(-y_prob)
    y_sorted = y_true[order]
    n = len(y_true)
    positives = y_true.sum()
    if positives == 0:
        st.info("Lift/Gains nicht darstellbar (keine positiven Klassen).")
        return

    perc = np.arange(1, n + 1) / n
    cum_pos = np.cumsum(y_sorted)
    gains = cum_pos / positives  # cumulative recall
    lift = gains / perc

    # Gains
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(perc, gains, label="Gains-Kurve")
    ax.plot([0, 1], [0, 1], "--", label="Zufall")
    ax.set_xlabel("Anteil der adressierten Fälle")
    ax.set_ylabel("Anteil der gefundenen Positiven")
    ax.set_title("Cumulative Gains")
    ax.legend()
    st.pyplot(fig, use_container_width=True)

    # Lift
    fig2, ax2 = plt.subplots(figsize=(6.0, 4.0))
    ax2.plot(perc, lift, label="Lift")
    ax2.axhline(1.0, ls="--", color="grey")
    ax2.set_xlabel("Anteil der adressierten Fälle")
    ax2.set_ylabel("Lift (vs. Zufall)")
    ax2.set_title("Lift-Kurve")
    ax2.legend()
    st.pyplot(fig2, use_container_width=True)

    with st.expander("Interpretation"):
        st.markdown(
            "- **Gains**: Wie viel Prozent aller Positiven habe ich gefunden, wenn ich die Top-x % mit den höchsten Scores nehme?\n"
            "- **Lift**: Wie viel besser ist das als Zufall (Lift = 1.0)? Höher ist besser."
        )

def _ks_curve(y_true: np.ndarray, y_prob: np.ndarray):
    # Kolmogorov-Smirnov statistic for separation
    pos_scores = np.sort(y_prob[y_true == 1])
    neg_scores = np.sort(y_prob[y_true == 0])
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        st.info("KS-Kurve nicht darstellbar (eine Klasse fehlt).")
        return

    # ECDFs
    xs = np.linspace(0, 1, 200)
    cdf_pos = np.searchsorted(pos_scores, xs, side="right") / max(len(pos_scores), 1)
    cdf_neg = np.searchsorted(neg_scores, xs, side="right") / max(len(neg_scores), 1)
    ks_vals = np.abs(cdf_pos - cdf_neg)
    ks_stat = ks_vals.max()
    ks_x = xs[ks_vals.argmax()]

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(xs, cdf_pos, label="ECDF Positiv")
    ax.plot(xs, cdf_neg, label="ECDF Negativ")
    ax.vlines(ks_x, cdf_neg[ks_vals.argmax()], cdf_pos[ks_vals.argmax()], colors="red", linestyles="--", label=f"KS = {ks_stat:.3f}")
    ax.set_xlabel("Score (Wahrscheinlichkeit für Positiv)")
    ax.set_ylabel("ECDF")
    ax.set_title("KS-Kurve (Trennschärfe)")
    ax.legend()
    st.pyplot(fig, use_container_width=True)

    with st.expander("Was bedeutet KS?"):
        st.markdown(
            "Der **KS-Wert** misst den maximalen Abstand zwischen den beiden Verteilungen (Positiv/Negativ). "
            "Je größer, desto besser getrennt sind die Klassen."
        )

def _feature_importance(model, out_path: Path):
    # try to load feature names
    names = None
    feat_path = out_path / "feature_names.json"
    if feat_path.exists():
        try:
            with open(feat_path, "r", encoding="utf-8") as f:
                names = json.load(f)
        except Exception:
            names = None

    importances = None
    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        coef = getattr(model, "coef_", None)
        if coef is not None and np.ndim(coef) == 2 and coef.shape[0] == 1:
            importances = np.abs(coef[0])
    if importances is None:
        st.info("Feature-Wichtigkeit nicht verfügbar für dieses Modell.")
        return

    # Prepare names
    if names is None or len(names) != len(importances):
        names = [f"Feature {i}" for i in range(len(importances))]

    # Top-k
    k = min(20, len(importances))
    idx = np.argsort(importances)[::-1][:k]
    imp = importances[idx]
    labels = [str(names[i]) for i in idx]

    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    ax.barh(range(k), imp[::-1])
    ax.set_yticks(range(k))
    ax.set_yticklabels([labels[::-1][i] for i in range(k)])
    ax.set_xlabel("Wichtigkeit")
    ax.set_title("Top-Merkmalswichtigkeiten")
    ax.invert_yaxis()
    st.pyplot(fig, use_container_width=True)

    with st.expander("Hinweis"):
        st.markdown(
            "Bei **Baum-Modellen** ist das die durchschnittliche Reduktionswirkung (Impurity). "
            "Bei **linearen Modellen** wird |Koeffizient| gezeigt."
        )

def format_table_values(df):
    """Convert encoded values to human readable format"""
    display_df = df.copy()
    
    # Dictionary for common model name mappings
    model_mapping = {
        'A43': 'Random Forest',
        'B12': 'Logistic Regression', 
        'C89': 'XGBoost',
        'D56': 'SVM',
        'E78': 'Gradient Boosting',
        'F91': 'AdaBoost',
        'G23': 'Extra Trees',
        'H45': 'KNN',
        'I67': 'Naive Bayes',
        'J89': 'Decision Tree'
    }
    
    for col in display_df.columns:
        if display_df[col].dtype == 'object':
            # First try direct mapping
            display_df[col] = display_df[col].map(model_mapping).fillna(display_df[col])
            
            # Then try regex pattern for any remaining encoded values
            if display_df[col].dtype == 'object':
                display_df[col] = display_df[col].astype(str)
                display_df[col] = display_df[col].str.replace(r'^[A-Z]\d+$', r'Model_\g<0>', regex=True)
    
    return display_df

# -------------------------------
# Main renderer
# -------------------------------
def render_results_block(result: dict, show_cm: bool, show_roc: bool, show_pr: bool, out_path: Path):
    """
    Big 'Notion-like' gallery with explanations.
    """
    
    # Add error handling for missing keys
    try:
        # Check if result is a dictionary
        if not isinstance(result, dict):
            st.error(f"Expected dictionary but got {type(result)}")
            st.write("Received data:", result)
            return
        
        # Adapt to the actual structure your backend returns
        required_keys = ["best_name", "y_test", "y_prob"]
        missing_keys = [key for key in required_keys if key not in result]
        if missing_keys:
            st.error(f"Missing required keys: {missing_keys}")
            st.write("Available keys:", list(result.keys()))
            return
        
        # Extract data using the actual key structure
        best_name = result["best_name"]
        y_true = np.asarray(result["y_test"])
        y_prob = np.asarray(result["y_prob"])
        
        # Use table data for results (if available)
        results_table = result.get("table", None)
        metrics = result.get("metrics", None)

        # ── Section: Summary ──────────────────────────────────────────────────────
        st.subheader("🏆 Bestes Modell")
        st.write(best_name)
        st.caption("Dies ist das Modell mit der besten Primärmetrik aus dem Training.")

        st.markdown("### 📋 Cross-Validation & Testmetrik (Tabelle)")
        if results_table is not None:
            # Display the table data directly if available
            if isinstance(results_table, pd.DataFrame):
                display_table = format_table_values(results_table)
                st.dataframe(display_table, use_container_width=True)
            else:
                # Handle dict or other formats
                df = pd.DataFrame(results_table)
                display_table = format_table_values(df)
                st.dataframe(display_table, use_container_width=True)
        elif metrics is not None:
            # Display metrics if available
            df = pd.DataFrame([metrics])
            display_table = format_table_values(df)
            st.dataframe(display_table, use_container_width=True)
        else:
            st.info("No results table available")

        # ── Section: Threshold Explorer ───────────────────────────────────────────
        st.markdown("## 🎚️ Threshold-Explorer")
        st.markdown(
            "> **Warum?** Viele Entscheidungen hängen vom **Schwellwert** ab. "
            "Mit einem niedrigen Threshold fängst du mehr Positive (hoher Recall), riskierst aber mehr False-Positives."
        )
        thr = st.slider("Threshold für Positiv-Entscheidung", 0.0, 1.0, 0.50, 0.01)

        _metric_cards(y_true, y_prob, thr)
        _confusion_heatmap(y_true, y_prob, thr)

        # ── Section: Classic Curves (reuse your original trio) ───────────────────
        st.markdown("## 📈 Klassische Kurven (automatisch)")
        render_eval_plots(
            y_true=y_true,
            y_prob=y_prob,
            threshold_default=thr,
            show_cm=show_cm,
            show_roc=show_roc,
            show_pr=show_pr,
        )

        # ── Section: Distributions & Calibration ─────────────────────────────────
        st.markdown("## 🧪 Kalibrierung & Verteilungen")
        _probability_histograms(y_true, y_prob)
        _calibration_plot(y_true, y_prob)

        # ── Section: Threshold Sweep ─────────────────────────────────────────────
        st.markdown("## 🔍 Schwellen-Kurven (Precision/Recall/F1 über Threshold)")
        _threshold_sweep_curves(y_true, y_prob)

        # ── Section: Gains, Lift & KS ────────────────────────────────────────────
        st.markdown("## 📊 Ranking-Güte: Gains, Lift & KS")
        _lift_gains_plots(y_true, y_prob)
        _ks_curve(y_true, y_prob)

        # ── Section: Feature Importance (if available) ───────────────────────────
        st.markdown("## 🧠 Wichtigste Merkmale")
        # Try to load model from the output directory
        if "out_dir" in result:
            try:
                import pickle
                import joblib
                out_dir = Path(result["out_dir"])
                
                # Try different possible model file names and formats
                possible_files = [
                    f"{best_name.lower().replace(' ','_').replace('-','_')}_model.pkl",
                    f"{best_name.lower().replace(' ','_').replace('-','_')}_model.joblib",
                    "best_model.pkl",
                    "best_model.joblib",
                    "model.pkl",
                    "model.joblib"
                ]
                
                model = None
                for filename in possible_files:
                    model_file = out_dir / filename
                    if model_file.exists():
                        try:
                            # Try pickle first
                            with open(model_file, "rb") as f:
                                model = pickle.load(f)
                            st.success(f"Model loaded from {filename}")
                            break
                        except Exception as pickle_error:
                            try:
                                # Try joblib if pickle fails
                                model = joblib.load(model_file)
                                st.success(f"Model loaded from {filename} using joblib")
                                break
                            except Exception as joblib_error:
                                st.warning(f"Failed to load {filename}: Pickle error: {str(pickle_error)[:100]}...")
                                continue
                
                if model is not None:
                    _feature_importance(model, out_dir)
                else:
                    st.info("Could not load any model file for feature importance analysis")
                    st.write("Searched for files:", possible_files)
                    st.write("In directory:", out_dir)
                    # List actual files in the directory
                    if out_dir.exists():
                        actual_files = list(out_dir.glob("*.pkl")) + list(out_dir.glob("*.joblib"))
                        st.write("Available model files:", [f.name for f in actual_files])
                    
            except Exception as e:
                st.warning(f"Error during model loading: {str(e)[:100]}...")
        else:
            st.info("Output directory not available for feature importance analysis")

        # ── Section: Saved artifacts ─────────────────────────────────────────────
        st.markdown("## 💾 Gespeicherte Artefakte")
        if "out_dir" in result:
            out_dir = Path(result["out_dir"])
            files = [
                out_dir / f"{best_name.lower().replace(' ','_').replace('-','_')}_model.pkl",
                out_dir / "standard_scaler.pkl",
                out_dir / "label_encoders.pkl",
                out_dir / "feature_names.json",
            ]
            render_artifacts_list(files)
        else:
            st.info("Output directory not available")

        # ── Closing helper text ──────────────────────────────────────────────────
        st.markdown("---")
        st.markdown(
            "### 🧭 Nächste Schritte\n"
            "- **Threshold anpassen** und die Effekte auf Precision/Recall in deinem Prozess abwägen.\n"
            "- **Kalibrierung** prüfen: Wenn stark off, erwäge Platt-Scaling/Isotonic.\n"
            "- **Top-Merkmale** prüfen (Fachlichkeit!).\n"
            "- **Gains/Lift** nutzen, um Kampagnen/Manuellen Review zu priorisieren."
        )
        
    except KeyError as e:
        st.error(f"Missing key in result data: {e}")
        st.write("Full result structure:", result)
    except Exception as e:
        st.error(f"Error processing results: {str(e)}")
        st.write("Result data:", result)
        import traceback
        st.code(traceback.format_exc())