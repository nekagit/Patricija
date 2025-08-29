import streamlit as st

def home_page():
    st.markdown("""
        <div class="hero-section animate-fade-in">
            <h1 class="hero-title text-gradient">Willkommen bei XAI Kreditprüfung</h1>
            <p class="hero-subtitle">Intelligente Kreditwürdigkeitsbewertung mit Explainable AI</p>
        </div>
    """, unsafe_allow_html=True)



