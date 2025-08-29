from __future__ import annotations
import streamlit as st
import pandas as pd

from utils.preprocessing import (
    read_uploaded_file,
    detect_german_credit_dataset,
    preprocess_german_credit_data,
    decode_german_credit_data
)
from utils.visualizations import render_dataset_preview


def upload_and_preview(auto_detect: bool, force_german_flag: bool, key: str = "default"):
    st.subheader("📁 Datenupload")
    uploaded = st.file_uploader("CSV / .data / .txt hochladen", type=['csv', 'data', 'txt'], key=f"file_uploader_{key}")
    if not uploaded:
        st.info("Lade ein Datenset hoch, um zu starten.")
        return None, False

    try:
        df_raw = read_uploaded_file(uploaded)
    except Exception as e:
        st.error(f"❌ Datei konnte nicht gelesen werden: {e}")
        return None, False

    is_german = detect_german_credit_dataset(df_raw) if auto_detect else force_german_flag
    
    df_for_training = df_raw.copy()
    df_for_preview = df_raw.copy()

    if is_german:
        st.success("🎯 German Credit Datensatz erkannt/aktiviert.")
        try:
            df_for_training, _, _ = preprocess_german_credit_data(df_for_training)
            df_for_preview, _, _ = preprocess_german_credit_data(df_for_preview)
            df_for_preview = decode_german_credit_data(df_for_preview)
        except Exception as e:
            st.warning(f"Preprocessing-Problem (wir machen weiter): {e}")

    render_dataset_preview(
        df=df_for_training, 
        df_converted=df_for_preview, 
        show_conversion=is_german
    )

    return df_for_training, is_german