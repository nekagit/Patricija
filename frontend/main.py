import streamlit as st
from components.home_page import home_page
from components.credit_check_page import credit_check_page
from components.result.results_page import results_page
from components.training.training_page import training_page
from components.analytics_page import analytics_page



def main():
    if 'page' not in st.session_state:
        st.session_state.page = "Startseite"

    with st.sidebar:
        st.markdown("""
            <div class="sidebar-content">
                <h2 class="text-heading mb-4">Navigation</h2>
            </div>
        """, unsafe_allow_html=True)
        
        if st.button("🏠 Startseite", on_click=lambda: st.session_state.update({"page": "Startseite"}), use_container_width=True, key="nav_home"):
            pass
        
        if st.button("📝 Kreditanfrage", on_click=lambda: st.session_state.update({"page": "Kreditanfrage"}), use_container_width=True, key="nav_credit"):
            pass

        st.markdown("""
            <div class="sidebar-content">
                <hr class="my-4">
                <h3 class="text-heading mb-3">Admin-Bereich</h3>
            </div>
        """, unsafe_allow_html=True)
        
        if st.button("⚙️ Modell trainieren", on_click=lambda: st.session_state.update({"page": "Training"}), use_container_width=True, key="nav_training"):
            pass
        
        if st.button("📊 Analyse-Dashboard", on_click=lambda: st.session_state.update({"page": "Analytics"}), use_container_width=True, key="nav_analytics"):
            pass

    if st.session_state.page == "Startseite":
        st.markdown('<h1 class="main-title">XAI Kreditprüfung</h1>', unsafe_allow_html=True)
        home_page()
    elif st.session_state.page == "Kreditanfrage":
        st.markdown('<h1 class="main-title">Kreditanfrage</h1>', unsafe_allow_html=True)
        credit_check_page()
    elif st.session_state.page == "Ergebnis":
        st.markdown('<h1 class="main-title">Ergebnis</h1>', unsafe_allow_html=True)
        results_page()
    elif st.session_state.page == "Training":
        st.markdown('<h1 class="main-title">Modell-Training</h1>', unsafe_allow_html=True)
        training_page()
    elif st.session_state.page == "Analytics":
        analytics_page()

if __name__ == "__main__":
    main()
