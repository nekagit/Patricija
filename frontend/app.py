import streamlit as st
import sys
import os
from pathlib import Path
from main import main

# Global theme & layout
st.set_page_config(page_title="XAI Kreditprüfung - Intelligente Kreditwürdigkeitsbewertung", layout="wide")

# Load CSS files
def load_css():
    css_files = [
        "css/main.css",
        "css/components.css", 
        "css/animations.css",
        "css/sidebar.css"
    ]
    
    css_content = ""
    for css_file in css_files:
        css_path = Path(__file__).parent / css_file
        if css_path.exists():
            with open(css_path, 'r', encoding='utf-8') as f:
                css_content += f.read() + "\n"
    
    return css_content

# Inject CSS
st.markdown(f"""
<style>
{load_css()}
</style>
""", unsafe_allow_html=True)



if __name__ == "__main__":
    if "page" not in st.session_state:
        st.session_state.page = "Startseite"
    main()
