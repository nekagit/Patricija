from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, confusion_matrix,
    accuracy_score, f1_score, roc_auc_score, average_precision_score
)
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
# Add the root directory to the path to access the utils module
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from utils.plot_storage import plot_storage

def theme_css():
    return """
    <style>
      .st-emotion-cache-16txtl3 {font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Courier New", monospace;}
      .st-emotion-cache-1629p8f h1 {font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;}
    </style>
    """

def render_dataset_preview(df: pd.DataFrame, rows: int = 5, df_converted: pd.DataFrame | None = None, show_conversion: bool = False):
    st.subheader("👀 Vorschau")

    if show_conversion and df_converted is not None:
        tab1, tab2 = st.tabs(["Konvertiert (lesbar)", "Original (Rohdaten)"])
        with tab1:
            st.dataframe(df_converted.head(rows), use_container_width=True)
        with tab2:
            st.dataframe(df.head(rows), use_container_width=True)
    else:
        st.dataframe(df.head(rows), use_container_width=True)

def render_quality_dashboard(df: pd.DataFrame, analysis: dict):
    st.subheader("🧪 Datenqualität")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Zeilen", f"{analysis.get('samples', 0):,}")
    c2.metric("Features", f"{analysis.get('features', 0):,}")
    c3.metric("Missing", f"{analysis.get('missing_data', 0):,}")
    balance = analysis.get('class_balance', 0.0)
    c4.metric("Balance", f"{balance:.2f}")
    st.divider()

def render_results_table(results: dict, best_name: str) -> pd.DataFrame:
    rows = []
    for name, res in results.items():
        m = res.get('metrics', {})
        rows.append({
            "Model": name, "Primary": name == best_name,
            "CV Scoring": m.get("cv_scoring"), "CV Mean": m.get("cv_mean"),
            "CV Std": m.get("cv_std"), "ROC AUC": m.get("roc_auc"),
            "PR AUC": m.get("pr_auc"), "F1": m.get("f1"),
            "Accuracy": m.get("accuracy"),
        })
    df = pd.DataFrame(rows)
    return df.sort_values(by=['Primary', 'ROC AUC'], ascending=[False, False])

def render_single_result_table(metrics: dict):
    st.subheader("🏁 Ergebnisse")
    if 'table' in metrics and isinstance(metrics['table'], pd.DataFrame):
        st.dataframe(metrics['table'], use_container_width=True, hide_index=True)

def _plot_confusion(y_true: np.ndarray, y_pred: np.ndarray):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False, annot_kws={'size': 10})
    ax.set_xlabel('Vorhersage', fontsize=10)
    ax.set_ylabel('Tatsächlicher Wert', fontsize=10)
    ax.set_title('Konfusionsmatrix', fontsize=12, fontweight='bold')
    ax.set_xticklabels(['Gut', 'Schlecht'], fontsize=9)
    ax.set_yticklabels(['Gut', 'Schlecht'], fontsize=9)
    plt.tight_layout()
    return fig

def _plot_roc(y_true: np.ndarray, y_prob: np.ndarray):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC-Kurve (Fläche = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0]); ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Falsch-Positiv-Rate', fontsize=10)
    ax.set_ylabel('Richtig-Positiv-Rate', fontsize=10)
    ax.set_title('ROC-Kurve', fontsize=12, fontweight='bold')
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    return fig

def _plot_pr(y_true: np.ndarray, y_prob: np.ndarray):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(recall, precision, color='blue', lw=2, label=f'PR-Kurve (Fläche = {pr_auc:.2f})')
    ax.set_xlim([0.0, 1.0]); ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=10)
    ax.set_ylabel('Precision', fontsize=10)
    ax.set_title('Precision-Recall-Kurve', fontsize=12, fontweight='bold')
    ax.legend(loc="lower left", fontsize=9)
    plt.tight_layout()
    return fig

def render_eval_plots(y_true: np.ndarray, y_prob: np.ndarray, show_cm: bool = True, show_roc: bool = True, show_pr: bool = True,
                     application_id: str = None, analytics: bool = False, training: bool = False, model_name: str = None):
    y_pred = (y_prob >= 0.5).astype(int)
    
    if show_cm:
        st.subheader("📊 Konfusionsmatrix")
        fig = _plot_confusion(y_true, y_pred)
        st.pyplot(fig)
        
        # Save the plot
        metadata = {
            'plot_type': 'confusion_matrix',
            'y_true_shape': y_true.shape,
            'y_pred_shape': y_pred.shape,
            'threshold': 0.5
        }
        plot_storage.save_matplotlib_plot(
            fig, 'confusion_matrix', 
            application_id=application_id, 
            analytics=analytics, 
            training=training, 
            model_name=model_name,
            metadata=metadata
        )
        plt.close(fig)
    
    if show_roc:
        st.subheader("📈 ROC-Kurve")
        fig = _plot_roc(y_true, y_prob)
        st.pyplot(fig)
        
        # Save the plot
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        metadata = {
            'plot_type': 'roc_curve',
            'y_true_shape': y_true.shape,
            'roc_auc': roc_auc,
            'fpr_shape': fpr.shape,
            'tpr_shape': tpr.shape
        }
        plot_storage.save_matplotlib_plot(
            fig, 'roc_curve', 
            application_id=application_id, 
            analytics=analytics, 
            training=training, 
            model_name=model_name,
            metadata=metadata
        )
        plt.close(fig)
    
    if show_pr:
        st.subheader("📉 Precision-Recall-Kurve")
        fig = _plot_pr(y_true, y_prob)
        st.pyplot(fig)
        
        # Save the plot
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(recall, precision)
        metadata = {
            'plot_type': 'precision_recall_curve',
            'y_true_shape': y_true.shape,
            'pr_auc': pr_auc,
            'precision_shape': precision.shape,
            'recall_shape': recall.shape
        }
        plot_storage.save_matplotlib_plot(
            fig, 'precision_recall_curve', 
            application_id=application_id, 
            analytics=analytics, 
            training=training, 
            model_name=model_name,
            metadata=metadata
        )
        plt.close(fig)

def render_feature_importance(feature_names: list, importances: np.ndarray, top_n: int = 10,
                            application_id: str = None, analytics: bool = False, training: bool = False, model_name: str = None):
    st.subheader("🎯 Merkmalswichtigkeit")
    
    importance_df = pd.DataFrame({
        'Merkmal': feature_names,
        'Wichtigkeit': importances
    }).sort_values('Wichtigkeit', ascending=False).head(top_n)
    
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=importance_df, x='Wichtigkeit', y='Merkmal', ax=ax)
    ax.set_title(f'Top {top_n} Merkmalswichtigkeiten', fontsize=12, fontweight='bold')
    ax.set_xlabel('Wichtigkeit', fontsize=10)
    ax.set_ylabel('Merkmal', fontsize=10)
    plt.tight_layout()
    st.pyplot(fig)
    
    # Save the plot
    metadata = {
        'plot_type': 'feature_importance',
        'feature_count': len(feature_names),
        'top_n': top_n,
        'max_importance': float(importances.max()),
        'min_importance': float(importances.min()),
        'mean_importance': float(importances.mean())
    }
    plot_storage.save_matplotlib_plot(
        fig, 'feature_importance', 
        application_id=application_id, 
        analytics=analytics, 
        training=training, 
        model_name=model_name,
        metadata=metadata
    )
    plt.close(fig)
    
    st.dataframe(importance_df, use_container_width=True, hide_index=True)

def apply_chart_theme(fig, title=None, height=None):
    """
    Apply consistent dark theme styling to Plotly charts.
    
    Args:
        fig: Plotly figure object
        title: Chart title (optional)
        height: Chart height (optional)
    
    Returns:
        Updated figure with consistent styling
    """
    # Base layout updates
    fig.update_layout(
        # Background colors
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        
        # Font settings
        font=dict(
            family='Inter, system-ui, -apple-system, sans-serif',
            size=12,
            color='#E8ECF4'
        ),
        
        # Title styling
        title=dict(
            text=title,
            font=dict(
                family='Lexend, Inter, system-ui, sans-serif',
                size=18,
                color='#E8ECF4',
                weight=600
            ),
            x=0.5,
            xanchor='center',
            y=0.95,
            yanchor='top'
        ) if title else None,
        
        # Margins for better spacing
        margin=dict(l=20, r=20, t=40, b=20),
        
        # Height if specified
        height=height,
        
        # Legend styling
        showlegend=True,
        legend=dict(
            bgcolor='rgba(26, 29, 38, 0.8)',
            bordercolor='rgba(255, 255, 255, 0.12)',
            borderwidth=1,
            font=dict(color='#E8ECF4', size=11),
            x=1.02,
            xanchor='left',
            y=1,
            yanchor='top'
        ),
        
        # Hover template styling
        hovermode='closest',
        hoverlabel=dict(
            bgcolor='rgba(26, 29, 38, 0.9)',
            bordercolor='rgba(194, 73, 20, 0.3)',
            font=dict(color='#E8ECF4', size=11)
        )
    )
    
    # Update axes styling
    fig.update_xaxes(
        gridcolor='rgba(255, 255, 255, 0.08)',
        gridwidth=1,
        zerolinecolor='rgba(255, 255, 255, 0.12)',
        zerolinewidth=1,
        showline=True,
        linecolor='rgba(255, 255, 255, 0.12)',
        linewidth=1,
        tickfont=dict(color='#98A2B3', size=10),
        titlefont=dict(color='#E8ECF4', size=12, weight=600)
    )
    
    fig.update_yaxes(
        gridcolor='rgba(255, 255, 255, 0.08)',
        gridwidth=1,
        zerolinecolor='rgba(255, 255, 255, 0.12)',
        zerolinewidth=1,
        showline=True,
        linecolor='rgba(255, 255, 255, 0.12)',
        linewidth=1,
        tickfont=dict(color='#98A2B3', size=10),
        titlefont=dict(color='#E8ECF4', size=12, weight=600)
    )
    
    return fig

def create_styled_bar_chart(data, x_col, y_col, title, color_col=None, color_sequence=None):
    """
    Create a styled bar chart with consistent theme.
    
    Args:
        data: DataFrame
        x_col: Column name for x-axis
        y_col: Column name for y-axis
        title: Chart title
        color_col: Column name for color grouping (optional)
        color_sequence: Color sequence for bars (optional)
    
    Returns:
        Styled Plotly figure
    """
    if color_col:
        fig = px.bar(
            data, 
            x=x_col, 
            y=y_col, 
            color=color_col,
            title=title,
            color_discrete_sequence=color_sequence or ['#C24914', '#F4A261', '#E76F51', '#2ECC71']
        )
    else:
        fig = px.bar(
            data, 
            x=x_col, 
            y=y_col, 
            title=title,
            color_discrete_sequence=color_sequence or ['#C24914']
        )
    
    return apply_chart_theme(fig, title)

def create_styled_line_chart(data, x_col, y_col, title, color_col=None, color_sequence=None):
    """
    Create a styled line chart with consistent theme.
    
    Args:
        data: DataFrame
        x_col: Column name for x-axis
        y_col: Column name for y-axis
        title: Chart title
        color_col: Column name for color grouping (optional)
        color_sequence: Color sequence for lines (optional)
    
    Returns:
        Styled Plotly figure
    """
    if color_col:
        fig = px.line(
            data, 
            x=x_col, 
            y=y_col, 
            color=color_col,
            title=title,
            color_discrete_sequence=color_sequence or ['#C24914', '#F4A261', '#E76F51', '#2ECC71']
        )
    else:
        fig = px.line(
            data, 
            x=x_col, 
            y=y_col, 
            title=title,
            color_discrete_sequence=color_sequence or ['#C24914']
        )
    
    return apply_chart_theme(fig, title)

def create_styled_scatter_chart(data, x_col, y_col, title, color_col=None, size_col=None, color_sequence=None):
    """
    Create a styled scatter chart with consistent theme.
    
    Args:
        data: DataFrame
        x_col: Column name for x-axis
        y_col: Column name for y-axis
        title: Chart title
        color_col: Column name for color grouping (optional)
        size_col: Column name for point size (optional)
        color_sequence: Color sequence for points (optional)
    
    Returns:
        Styled Plotly figure
    """
    if color_col and size_col:
        fig = px.scatter(
            data, 
            x=x_col, 
            y=y_col, 
            color=color_col,
            size=size_col,
            title=title,
            color_discrete_sequence=color_sequence or ['#C24914', '#F4A261', '#E76F51', '#2ECC71']
        )
    elif color_col:
        fig = px.scatter(
            data, 
            x=x_col, 
            y=y_col, 
            color=color_col,
            title=title,
            color_discrete_sequence=color_sequence or ['#C24914', '#F4A261', '#E76F51', '#2ECC71']
        )
    else:
        fig = px.scatter(
            data, 
            x=x_col, 
            y=y_col, 
            title=title,
            color_discrete_sequence=color_sequence or ['#C24914']
        )
    
    return apply_chart_theme(fig, title)

def create_styled_histogram(data, x_col, title, nbins=20, color_sequence=None):
    """
    Create a styled histogram with consistent theme.
    
    Args:
        data: DataFrame
        x_col: Column name for x-axis
        title: Chart title
        nbins: Number of bins
        color_sequence: Color sequence for bars (optional)
    
    Returns:
        Styled Plotly figure
    """
    fig = px.histogram(
        data, 
        x=x_col, 
        nbins=nbins,
        title=title,
        color_discrete_sequence=color_sequence or ['#C24914']
    )
    
    return apply_chart_theme(fig, title)

def create_styled_pie_chart(data, names_col, values_col, title, color_sequence=None):
    """
    Create a styled pie chart with consistent theme.
    
    Args:
        data: DataFrame
        names_col: Column name for pie slice labels
        values_col: Column name for pie slice values
        title: Chart title
        color_sequence: Color sequence for slices (optional)
    
    Returns:
        Styled Plotly figure
    """
    fig = px.pie(
        data, 
        names=names_col, 
        values=values_col,
        title=title,
        color_discrete_sequence=color_sequence or ['#C24914', '#F4A261', '#E76F51', '#2ECC71', '#9B59B6', '#3498DB']
    )
    
    return apply_chart_theme(fig, title)

def create_styled_heatmap(data, title, color_sequence=None):
    """
    Create a styled heatmap with consistent theme.
    
    Args:
        data: Correlation matrix or 2D array
        title: Chart title
        color_sequence: Color sequence for heatmap (optional)
    
    Returns:
        Styled Plotly figure
    """
    fig = px.imshow(
        data,
        title=title,
        color_continuous_scale=color_sequence or 'RdYlBu_r',
        aspect='auto'
    )
    
    return apply_chart_theme(fig, title)

# Wrapper functions for automatic plot saving
def render_and_save_bar_chart(data, x_col, y_col, title, color_col=None, color_sequence=None,
                             application_id: str = None, analytics: bool = False, training: bool = False, model_name: str = None):
    """
    Create, render, and save a styled bar chart.
    """
    fig = create_styled_bar_chart(data, x_col, y_col, title, color_col, color_sequence)
    st.plotly_chart(fig, use_container_width=True)
    
    # Save the plot
    metadata = {
        'plot_type': 'bar_chart',
        'x_column': x_col,
        'y_column': y_col,
        'color_column': color_col,
        'data_shape': data.shape,
        'title': title
    }
    plot_storage.save_plotly_plot(
        fig, 'bar_chart', 
        application_id=application_id, 
        analytics=analytics, 
        training=training, 
        model_name=model_name,
        metadata=metadata
    )
    
    return fig

def render_and_save_line_chart(data, x_col, y_col, title, color_col=None, color_sequence=None,
                              application_id: str = None, analytics: bool = False, training: bool = False, model_name: str = None):
    """
    Create, render, and save a styled line chart.
    """
    fig = create_styled_line_chart(data, x_col, y_col, title, color_col, color_sequence)
    st.plotly_chart(fig, use_container_width=True)
    
    # Save the plot
    metadata = {
        'plot_type': 'line_chart',
        'x_column': x_col,
        'y_column': y_col,
        'color_column': color_col,
        'data_shape': data.shape,
        'title': title
    }
    plot_storage.save_plotly_plot(
        fig, 'line_chart', 
        application_id=application_id, 
        analytics=analytics, 
        training=training, 
        model_name=model_name,
        metadata=metadata
    )
    
    return fig

def render_and_save_scatter_chart(data, x_col, y_col, title, color_col=None, size_col=None, color_sequence=None,
                                 application_id: str = None, analytics: bool = False, training: bool = False, model_name: str = None):
    """
    Create, render, and save a styled scatter chart.
    """
    fig = create_styled_scatter_chart(data, x_col, y_col, title, color_col, size_col, color_sequence)
    st.plotly_chart(fig, use_container_width=True)
    
    # Save the plot
    metadata = {
        'plot_type': 'scatter_chart',
        'x_column': x_col,
        'y_column': y_col,
        'color_column': color_col,
        'size_column': size_col,
        'data_shape': data.shape,
        'title': title
    }
    plot_storage.save_plotly_plot(
        fig, 'scatter_chart', 
        application_id=application_id, 
        analytics=analytics, 
        training=training, 
        model_name=model_name,
        metadata=metadata
    )
    
    return fig

def render_and_save_histogram(data, x_col, title, nbins=20, color_sequence=None,
                             application_id: str = None, analytics: bool = False, training: bool = False, model_name: str = None):
    """
    Create, render, and save a styled histogram.
    """
    fig = create_styled_histogram(data, x_col, title, nbins, color_sequence)
    st.plotly_chart(fig, use_container_width=True)
    
    # Save the plot
    metadata = {
        'plot_type': 'histogram',
        'x_column': x_col,
        'nbins': nbins,
        'data_shape': data.shape,
        'title': title
    }
    plot_storage.save_plotly_plot(
        fig, 'histogram', 
        application_id=application_id, 
        analytics=analytics, 
        training=training, 
        model_name=model_name,
        metadata=metadata
    )
    
    return fig

def render_and_save_pie_chart(data, names_col, values_col, title, color_sequence=None,
                             application_id: str = None, analytics: bool = False, training: bool = False, model_name: str = None):
    """
    Create, render, and save a styled pie chart.
    """
    fig = create_styled_pie_chart(data, names_col, values_col, title, color_sequence)
    st.plotly_chart(fig, use_container_width=True)
    
    # Save the plot
    metadata = {
        'plot_type': 'pie_chart',
        'names_column': names_col,
        'values_column': values_col,
        'data_shape': data.shape,
        'title': title
    }
    plot_storage.save_plotly_plot(
        fig, 'pie_chart', 
        application_id=application_id, 
        analytics=analytics, 
        training=training, 
        model_name=model_name,
        metadata=metadata
    )
    
    return fig

def render_and_save_heatmap(data, title, color_sequence=None,
                           application_id: str = None, analytics: bool = False, training: bool = False, model_name: str = None):
    """
    Create, render, and save a styled heatmap.
    """
    fig = create_styled_heatmap(data, title, color_sequence)
    st.plotly_chart(fig, use_container_width=True)
    
    # Save the plot
    metadata = {
        'plot_type': 'heatmap',
        'data_shape': data.shape if hasattr(data, 'shape') else len(data),
        'title': title
    }
    plot_storage.save_plotly_plot(
        fig, 'heatmap', 
        application_id=application_id, 
        analytics=analytics, 
        training=training, 
        model_name=model_name,
        metadata=metadata
    )
    
    return fig