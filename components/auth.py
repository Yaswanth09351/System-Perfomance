import streamlit as st
import hashlib
import json
import os
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
from database import DatabaseStorage

class AuthenticationManager:
    """Handles user authentication, sign-up, and login functionality"""
    
    def __init__(self):
        self.session_timeout = 24  # hours
        self.db_storage = DatabaseStorage()
        self.load_users()
    
    def load_users(self) -> Dict:
        """Load users from database"""
        try:
            self.users = self.db_storage.get_all_users()
            return self.users
        except Exception as e:
            st.error(f"Error loading users: {e}")
            self.users = {}
            return {}
    
    def save_users(self) -> bool:
        """Save users to database (legacy method - now handled by database)"""
        # This method is kept for compatibility but actual saves happen in database methods
        return True
    
    def hash_password(self, password: str) -> str:
        """Hash password using SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def validate_email(self, email: str) -> bool:
        """Basic email validation"""
        import re
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    def validate_password(self, password: str) -> Tuple[bool, str]:
        """Validate password strength"""
        if len(password) < 6:
            return False, "Password must be at least 6 characters long"
        if not any(c.isdigit() for c in password):
            return False, "Password must contain at least one number"
        if not any(c.isalpha() for c in password):
            return False, "Password must contain at least one letter"
        return True, "Password is valid"
    
    def create_user(self, email: str, password: str, full_name: str) -> Tuple[bool, str]:
        """Create a new user account"""
        try:
            # Validate email
            if not self.validate_email(email):
                return False, "Invalid email format"
            
            # Check if user already exists
            existing_user = self.db_storage.get_user(email)
            if existing_user:
                return False, "User already exists"
            
            # Validate password
            is_valid, message = self.validate_password(password)
            if not is_valid:
                return False, message
            
            # Create user in database
            password_hash = self.hash_password(password)
            if self.db_storage.create_user(email, password_hash, full_name):
                # Reload users cache
                self.load_users()
                return True, "Account created successfully"
            else:
                return False, "Error saving account"
                
        except Exception as e:
            return False, f"Error creating account: {str(e)}"
    
    def authenticate_user(self, email: str, password: str) -> Tuple[bool, str, Optional[Dict]]:
        """Authenticate user login"""
        try:
            user = self.db_storage.get_user(email)
            if not user:
                return False, "User not found", None
            
            if user['password'] != self.hash_password(password):
                return False, "Invalid password", None
            
            # Update last login in database
            self.db_storage.update_user_login(email)
            
            # Reload users cache
            self.load_users()
            
            return True, "Login successful", user
            
        except Exception as e:
            return False, f"Authentication error: {str(e)}", None
    
    def is_session_valid(self) -> bool:
        """Check if current session is valid"""
        if 'authenticated' not in st.session_state:
            return False
        
        if 'login_time' not in st.session_state:
            return False
        
        login_time = datetime.fromisoformat(st.session_state.login_time)
        if datetime.now() - login_time > timedelta(hours=self.session_timeout):
            self.logout()
            return False
        
        return st.session_state.authenticated
    
    def login_user(self, user_data: Dict) -> None:
        """Set session state for logged-in user"""
        st.session_state.authenticated = True
        st.session_state.user_email = user_data['email']
        st.session_state.user_name = user_data['full_name']
        st.session_state.user_role = user_data.get('role', 'user')
        st.session_state.login_time = datetime.now().isoformat()
        st.session_state.user_preferences = user_data.get('preferences', {})
    
    def logout(self) -> None:
        """Clear session state for logout"""
        for key in ['authenticated', 'user_email', 'user_name', 'user_role', 'login_time', 'user_preferences']:
            if key in st.session_state:
                del st.session_state[key]
    
    def get_current_user(self) -> Optional[Dict]:
        """Get current logged-in user data"""
        if not self.is_session_valid():
            return None
        
        email = st.session_state.get('user_email')
        return self.users.get(email)
    
    def update_user_preferences(self, email: str, preferences: Dict) -> bool:
        """Update user preferences"""
        try:
            if self.db_storage.update_user_preferences(email, preferences):
                # Reload users cache
                self.load_users()
                return True
            return False
        except Exception as e:
            st.error(f"Error updating preferences: {e}")
            return False
    
    def render_login_form(self) -> None:
        """Render login form"""
        st.subheader("🔐 Login")
        
        with st.form("login_form"):
            email = st.text_input("Email", placeholder="your.email@example.com")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")
            
            if submitted:
                if not email or not password:
                    st.error("Please fill in all fields")
                else:
                    success, message, user_data = self.authenticate_user(email, password)
                    if success and user_data:
                        self.login_user(user_data)
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)
    
    def render_signup_form(self) -> None:
        """Render sign-up form"""
        st.subheader("👤 Create Account")
        
        with st.form("signup_form"):
            full_name = st.text_input("Full Name", placeholder="John Doe")
            email = st.text_input("Email", placeholder="your.email@example.com")
            password = st.text_input("Password", type="password", help="At least 6 characters with letters and numbers")
            confirm_password = st.text_input("Confirm Password", type="password")
            
            submitted = st.form_submit_button("Create Account")
            
            if submitted:
                if not all([full_name, email, password, confirm_password]):
                    st.error("Please fill in all fields")
                elif password != confirm_password:
                    st.error("Passwords do not match")
                else:
                    success, message = self.create_user(email, password, full_name)
                    if success:
                        st.success(message)
                        st.info("Please login with your new account")
                    else:
                        st.error(message)
    
    def render_user_profile(self) -> None:
        """Render user profile and preferences"""
        user = self.get_current_user()
        if not user:
            return
        
        st.subheader(f"👋 Welcome, {user['full_name']}")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.write(f"**Email:** {user['email']}")
            st.write(f"**Role:** {user['role'].title()}")
            st.write(f"**Member since:** {user['created_at'][:10]}")
            if user['last_login']:
                st.write(f"**Last login:** {user['last_login'][:10]}")
        
        with col2:
            if st.button("🚪 Logout"):
                self.logout()
                st.rerun()
        
        # User preferences
        st.subheader("⚙️ Preferences")
        
        with st.form("preferences_form"):
            prefs = user.get('preferences', {})
            
            theme = st.selectbox("Theme", ["light", "dark"], 
                               index=0 if prefs.get('theme', 'light') == 'light' else 1)
            auto_refresh = st.checkbox("Auto-refresh dashboard", 
                                     value=prefs.get('auto_refresh', True))
            refresh_interval = st.slider("Refresh interval (seconds)", 
                                       1, 60, prefs.get('refresh_interval', 5))
            
            if st.form_submit_button("Save Preferences"):
                new_prefs = {
                    'theme': theme,
                    'auto_refresh': auto_refresh,
                    'refresh_interval': refresh_interval
                }
                
                if self.update_user_preferences(user['email'], new_prefs):
                    st.session_state.user_preferences = new_prefs
                    st.success("Preferences updated successfully")
                    st.rerun()
                else:
                    st.error("Error updating preferences")
    
    def render_auth_page(self) -> bool:
        """Render authentication page and return if user is authenticated"""
        if self.is_session_valid():
            return True
        
        st.title("🖥️ System Performance Monitor")
        st.markdown("---")
        
        # Authentication tabs
        tab1, tab2 = st.tabs(["Login", "Sign Up"])
        
        with tab1:
            self.render_login_form()
        
        with tab2:
            self.render_signup_form()
        
        st.markdown("---")
        st.markdown("### 📊 Dashboard Features")
        st.markdown("""
        - **Real-time System Monitoring**: CPU, Memory, and Disk usage
        - **Performance Predictions**: ML-powered forecasting
        - **AI Assistant**: Ask questions about your system performance
        - **Custom Alerts**: Configurable thresholds and notifications
        - **Data Export**: Export performance data in multiple formats
        """)
        
        return False
    
    def get_user_stats(self) -> Dict:
        """Get user statistics for admin dashboard"""
        total_users = len(self.users)
        active_users = sum(1 for user in self.users.values() 
                          if user.get('last_login') and 
                          datetime.fromisoformat(user['last_login']) > 
                          datetime.now() - timedelta(days=30))
        
        return {
            'total_users': total_users,
            'active_users': active_users,
            'newest_user': max(self.users.values(), 
                             key=lambda x: x['created_at'])['full_name'] if self.users else None
        }