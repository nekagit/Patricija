import requests
import json
from typing import Dict, Any, Optional, List
import streamlit as st
from datetime import datetime

class APIClient:
    """API Client for communicating with the backend."""
    
    def __init__(self, base_url: str = "http://localhost:8000/api/v1"):
        self.base_url = base_url
        self.session = requests.Session()
        self.token = None
    
    def set_token(self, token: str):
        """Set authentication token."""
        self.token = token
        self.session.headers.update({"Authorization": f"Bearer {token}"})
    
    def clear_token(self):
        """Clear authentication token."""
        self.token = None
        if "Authorization" in self.session.headers:
            del self.session.headers["Authorization"]
    
    def check_backend_connection(self) -> bool:
        """Check if backend is connected and accessible."""
        try:
            response = self.session.get(f"{self.base_url.replace('/api/v1', '')}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def _make_request(self, method: str, endpoint: str, data: Optional[Dict] = None, params: Optional[Dict] = None) -> Dict[str, Any]:
        """Make HTTP request to API."""
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method.upper() == "GET":
                response = self.session.get(url, params=params, timeout=10)
            elif method.upper() == "POST":
                response = self.session.post(url, json=data, timeout=10)
            elif method.upper() == "PUT":
                response = self.session.put(url, json=data, timeout=10)
            elif method.upper() == "DELETE":
                response = self.session.delete(url, timeout=10)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
    
    # Authentication endpoints
    def register(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Register a new user."""
        return self._make_request("POST", "/auth/register", data=user_data)
    
    def login(self, username: str, password: str) -> Dict[str, Any]:
        """Login user and get access token."""
        response = self._make_request("POST", "/auth/login", data={
            "username": username,
            "password": password
        })
        
        if "access_token" in response:
            self.set_token(response["access_token"])
        
        return response
    
    def get_current_user(self) -> Dict[str, Any]:
        """Get current user information."""
        return self._make_request("GET", "/auth/me")
    
    # Credit Application endpoints
    def create_application(self, application_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new credit application."""
        return self._make_request("POST", "/applications/", data=application_data)
    
    def save_application(self, application_data: Dict[str, Any]) -> Dict[str, Any]:
        """Save a credit application (alias for create_application)."""
        return self.create_application(application_data)
    
    def get_applications(self, skip: int = 0, limit: int = 100, status: Optional[str] = None) -> Dict[str, Any]:
        """Get credit applications."""
        params = {"skip": skip, "limit": limit}
        if status:
            params["status"] = status
        return self._make_request("GET", "/applications/", params=params)
    
    def get_demo_applications(self, skip: int = 0, limit: int = 100) -> Dict[str, Any]:
        """Get demo applications when backend is not available."""
        return self._make_request("GET", "/applications/demo-data", params={"skip": skip, "limit": limit})
    
    def get_application(self, application_id: str) -> Dict[str, Any]:
        """Get a specific credit application."""
        return self._make_request("GET", f"/applications/{application_id}")
    
    def update_application(self, application_id: str, update_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update a credit application."""
        return self._make_request("PUT", f"/applications/{application_id}", data=update_data)
    
    def review_application(self, application_id: str, review_data: Dict[str, Any]) -> Dict[str, Any]:
        """Review a credit application."""
        return self._make_request("POST", f"/applications/{application_id}/review", data=review_data)
    
    def get_application_statistics(self) -> Dict[str, Any]:
        """Get application statistics."""
        return self._make_request("GET", "/applications/statistics/summary")
    
    def get_application_history(self, skip: int = 0, limit: int = 100) -> Dict[str, Any]:
        """Get credit application history with predictions."""
        params = {"skip": skip, "limit": limit}
        return self._make_request("GET", "/applications/history", params=params)
    
    def get_demo_history(self, skip: int = 0, limit: int = 100) -> Dict[str, Any]:
        """Get demo history data when backend has no real data."""
        params = {"skip": skip, "limit": limit}
        return self._make_request("GET", "/applications/demo-history", params=params)
    
    # Prediction endpoints
    def create_prediction(self, prediction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new prediction."""
        return self._make_request("POST", "/predictions/", data=prediction_data)
    
    def get_predictions(self, skip: int = 0, limit: int = 100, model_name: Optional[str] = None, application_id: Optional[str] = None) -> Dict[str, Any]:
        """Get predictions."""
        params = {"skip": skip, "limit": limit}
        if model_name:
            params["model_name"] = model_name
        if application_id:
            params["application_id"] = application_id
        return self._make_request("GET", "/predictions/", params=params)
    
    def get_prediction(self, prediction_id: str) -> Dict[str, Any]:
        """Get a specific prediction."""
        return self._make_request("GET", f"/predictions/{prediction_id}")
    
    def get_predictions_for_application(self, application_id: str) -> List[Dict[str, Any]]:
        """Get all predictions for a specific application."""
        return self._make_request("GET", f"/predictions/application/{application_id}")
    
    def get_prediction_statistics(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Get prediction statistics."""
        params = {}
        if model_name:
            params["model_name"] = model_name
        return self._make_request("GET", "/predictions/statistics/summary", params=params)

# Global API client instance
api_client = APIClient()

def get_api_client() -> APIClient:
    """Get the global API client instance."""
    return api_client

def is_authenticated() -> bool:
    """Check if user is authenticated."""
    return api_client.token is not None

def login_user(username: str, password: str) -> bool:
    """Login user and store token in session state."""
    response = api_client.login(username, password)
    
    if "access_token" in response:
        st.session_state["access_token"] = response["access_token"]
        st.session_state["user"] = response.get("user", {})
        return True
    else:
        st.error("Login failed. Please check your credentials.")
        return False

def logout_user():
    """Logout user and clear session state."""
    api_client.clear_token()
    if "access_token" in st.session_state:
        del st.session_state["access_token"]
    if "user" in st.session_state:
        del st.session_state["user"]

def get_current_user() -> Optional[Dict[str, Any]]:
    """Get current user from session state."""
    return st.session_state.get("user")

def get_user_role() -> Optional[str]:
    """Get current user role."""
    user = get_current_user()
    return user.get("role") if user else None

def require_auth():
    """Require authentication for a page."""
    if not is_authenticated():
        st.error("Please log in to access this page.")
        st.stop()

def require_role(required_role: str):
    """Require specific role for a page."""
    require_auth()
    user_role = get_user_role()
    
    if user_role not in [required_role, "admin"]:
        st.error(f"Access denied. {required_role.capitalize()} role required.")
        st.stop()
