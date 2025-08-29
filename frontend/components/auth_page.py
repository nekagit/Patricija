import streamlit as st
from utils.api_client import login_user, get_api_client

def auth_page():
    """Authentication page with login and register forms."""
    
    st.markdown("""
        <div class="main-title text-center text-secondary mb-4 animate-fade-in">
            <h1>🔐 Anmeldung</h1>
            <p>Melden Sie sich an, um auf das Kreditprüfungssystem zuzugreifen</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for login and register
    tab1, tab2 = st.tabs(["Anmelden", "Registrieren"])
    
    with tab1:
        st.markdown("""
            <div class="form-section animate-fade-in-up delay-100">
                <div class="form-section-header">
                    <div class="form-section-icon">👤</div>
                    <div class="form-section-title">Bestehender Benutzer</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        with st.form("login_form"):
            username = st.text_input(
                "Benutzername",
                placeholder="Ihr Benutzername",
                help="Geben Sie Ihren Benutzernamen ein"
            )
            
            password = st.text_input(
                "Passwort",
                type="password",
                placeholder="Ihr Passwort",
                help="Geben Sie Ihr Passwort ein"
            )
            
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col2:
                login_submitted = st.form_submit_button(
                    "Anmelden",
                    use_container_width=True,
                    type="primary"
                )
            
            if login_submitted:
                if username and password:
                    if login_user(username, password):
                        st.success("Anmeldung erfolgreich!")
                        st.session_state.page = "Home"
                        st.rerun()
                else:
                    st.error("Bitte füllen Sie alle Felder aus.")
    
    with tab2:
        st.markdown("""
            <div class="form-section animate-fade-in-up delay-200">
                <div class="form-section-header">
                    <div class="form-section-icon">📝</div>
                    <div class="form-section-title">Neuer Benutzer</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        with st.form("register_form"):
            new_username = st.text_input(
                "Benutzername",
                placeholder="Wählen Sie einen Benutzernamen",
                help="Der Benutzername muss mindestens 3 Zeichen lang sein"
            )
            
            new_email = st.text_input(
                "E-Mail",
                placeholder="ihre.email@beispiel.com",
                help="Geben Sie eine gültige E-Mail-Adresse ein"
            )
            
            new_password = st.text_input(
                "Passwort",
                type="password",
                placeholder="Mindestens 8 Zeichen",
                help="Das Passwort muss mindestens 8 Zeichen lang sein"
            )
            
            confirm_password = st.text_input(
                "Passwort bestätigen",
                type="password",
                placeholder="Passwort wiederholen",
                help="Bestätigen Sie Ihr Passwort"
            )
            
            first_name = st.text_input(
                "Vorname",
                placeholder="Ihr Vorname",
                help="Optional"
            )
            
            last_name = st.text_input(
                "Nachname",
                placeholder="Ihr Nachname",
                help="Optional"
            )
            
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col2:
                register_submitted = st.form_submit_button(
                    "Registrieren",
                    use_container_width=True,
                    type="primary"
                )
            
            if register_submitted:
                if not all([new_username, new_email, new_password, confirm_password]):
                    st.error("Bitte füllen Sie alle Pflichtfelder aus.")
                elif new_password != confirm_password:
                    st.error("Die Passwörter stimmen nicht überein.")
                elif len(new_password) < 8:
                    st.error("Das Passwort muss mindestens 8 Zeichen lang sein.")
                elif len(new_username) < 3:
                    st.error("Der Benutzername muss mindestens 3 Zeichen lang sein.")
                else:
                    # Register user
                    api_client = get_api_client()
                    user_data = {
                        "username": new_username,
                        "email": new_email,
                        "password": new_password,
                        "first_name": first_name if first_name else None,
                        "last_name": last_name if last_name else None,
                        "role": "user"
                    }
                    
                    response = api_client.register(user_data)
                    
                    if "error" not in response:
                        st.success("Registrierung erfolgreich! Sie können sich jetzt anmelden.")
                        # Clear form
                        st.session_state.register_form = {}
                    else:
                        st.error(f"Registrierung fehlgeschlagen: {response.get('detail', 'Unbekannter Fehler')}")
    
    # Demo credentials info
    st.markdown("---")
    st.markdown("""
        <div class="info-box">
            <h4>Demo-Zugangsdaten</h4>
            <p><strong>Benutzer:</strong> demo / demo123</p>
            <p><strong>Analyst:</strong> analyst / analyst123</p>
            <p><strong>Admin:</strong> admin / admin123</p>
        </div>
    """, unsafe_allow_html=True)
