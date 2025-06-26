import streamlit as st
import json
from datetime import datetime, date

def get_authenticated_user():
    """Get authenticated user from session state or authentication parameters."""
    try:
        # Check if user is already in session state
        if hasattr(st.session_state, 'user') and st.session_state.user:
            return st.session_state.user
        
        # Check for authentication parameters in session state
        if hasattr(st.session_state, 'auth_token') and st.session_state.auth_token:
            user_data = {
                'id': st.session_state.get('user_id'),
                'email': st.session_state.get('user_email'),
                'auth_token': st.session_state.auth_token
            }
            st.session_state.user = user_data
            return user_data
        
        # Check URL parameters for authentication (iframe context)
        try:
            import streamlit.components.v1 as components
            
            # JavaScript to get authentication from sessionStorage or URL params
            auth_js = """
            <script>
                // Check sessionStorage first
                const authToken = sessionStorage.getItem('vocam_auth_token');
                const userEmail = sessionStorage.getItem('vocam_user_email');
                const userId = sessionStorage.getItem('vocam_user_id');
                
                if (authToken && userEmail && userId) {
                    // Send to Streamlit
                    window.parent.postMessage({
                        type: 'auth_data',
                        auth_token: authToken,
                        user_email: userEmail,
                        user_id: userId
                    }, '*');
                }
                
                // Also check URL parameters
                const urlParams = new URLSearchParams(window.location.search);
                const urlAuthToken = urlParams.get('auth_token');
                const urlUserEmail = urlParams.get('user_email');
                const urlUserId = urlParams.get('user_id');
                
                if (urlAuthToken && urlUserEmail && urlUserId) {
                    // Store in sessionStorage
                    sessionStorage.setItem('vocam_auth_token', urlAuthToken);
                    sessionStorage.setItem('vocam_user_email', urlUserEmail);
                    sessionStorage.setItem('vocam_user_id', urlUserId);
                    
                    // Send to Streamlit
                    window.parent.postMessage({
                        type: 'auth_data',
                        auth_token: urlAuthToken,
                        user_email: urlUserEmail,
                        user_id: urlUserId
                    }, '*');
                }
            </script>
            """
            
            components.html(auth_js, height=0)
        except:
            pass
        
        # Return None if no authentication found
        return None
        
    except Exception as e:
        print(f"❌ Error getting authenticated user: {e}")
        return None

def require_authentication():
    """Require user authentication - redirect if not authenticated."""
    user = get_authenticated_user()
    
    if not user:
        st.error("🔐 Authentication required")
        st.info("Please log in through the main Vocam website to access this app.")
        
        # Add authentication form as fallback
        with st.expander("Alternative: Manual Authentication", expanded=False):
            st.warning("⚠️ This is for development/testing only")
            
            with st.form("auth_form"):
                user_id = st.text_input("User ID")
                user_email = st.text_input("Email")
                auth_token = st.text_input("Auth Token", type="password")
                
                if st.form_submit_button("Set Authentication"):
                    if user_id and user_email and auth_token:
                        st.session_state.user_id = user_id
                        st.session_state.user_email = user_email
                        st.session_state.auth_token = auth_token
                        st.session_state.user = {
                            'id': user_id,
                            'email': user_email,
                            'auth_token': auth_token
                        }
                        st.success("✅ Authentication set! Please refresh the page.")
                        st.experimental_rerun()
                    else:
                        st.error("Please fill in all fields")
        
        st.stop()
    
    return user

def get_supabase_user_id():
    """Get user ID for Supabase operations."""
    user = get_authenticated_user()
    if user:
        return str(user.get('id'))
    return None

def initialize_user_session():
    """Initialize user session with default values."""
    try:
        user = get_authenticated_user()
        if not user:
            return False
        
        # Initialize session state defaults if not present
        defaults = {
            'words_learned': 0,
            'total_words_learned': 0,
            'level': 1,
            'points': 0,
            'streak_days': 0,
            'last_active_date': date.today(),
            'streak_savers': 0,
            'achievements': {},
            'badges': {},
            'daily_challenges': [],
            'word_of_the_day': None,
            'category_progress': {},
            'vocabulary_tree': {},
            'current_session_id': None,
            'data_loaded': False
        }
        
        for key, default_value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
        
        print(f"✅ User session initialized for: {user.get('email')}")
        return True
        
    except Exception as e:
        print(f"❌ Error initializing user session: {e}")
        return False

def check_authentication_status():
    """Check if user is properly authenticated and session is valid."""
    try:
        user = get_authenticated_user()
        if not user:
            return False, "No authentication found"
        
        required_fields = ['id', 'email', 'auth_token']
        missing_fields = [field for field in required_fields if not user.get(field)]
        
        if missing_fields:
            return False, f"Missing authentication fields: {', '.join(missing_fields)}"
        
        # Test connection to Supabase
        try:
            from main import get_user_database
            db = get_user_database()
            
            # Try a simple query to test authentication
            if db.supabase:
                response = db.supabase.table('vocabulary').select('id').limit(1).execute()
                if hasattr(response, 'data'):
                    return True, "Authentication valid"
            else:
                # Test with requests
                headers = db.get_headers()
                import requests
                response = requests.get(
                    f'{db.supabase_url}/rest/v1/vocabulary?limit=1',
                    headers=headers,
                    timeout=10
                )
                if response.status_code in [200, 401]:  # 401 is expected if no data
                    return True, "Authentication valid"
            
            return False, "Database connection failed"
            
        except Exception as db_error:
            return False, f"Database error: {str(db_error)}"
        
    except Exception as e:
        return False, f"Authentication check failed: {str(e)}"

def refresh_user_data():
    """Refresh user data from Supabase."""
    try:
        user = get_authenticated_user()
        if not user:
            return False
        
        from main import load_user_data_from_supabase
        success = load_user_data_from_supabase()
        
        if success:
            print("✅ User data refreshed successfully")
        else:
            print("⚠️ Failed to refresh user data")
        
        return success
        
    except Exception as e:
        print(f"❌ Error refreshing user data: {e}")
        return False

# Event handler for authentication messages from iframe
def handle_auth_message():
    """Handle authentication messages from iframe JavaScript."""
    try:
        # This would be called when receiving postMessage from JavaScript
        # In practice, Streamlit doesn't have direct postMessage handling
        # So we rely on sessionStorage and URL parameters instead
        pass
    except Exception as e:
        print(f"❌ Error handling auth message: {e}")

# Authentication debugging functions
def debug_authentication():
    """Debug authentication state - useful for troubleshooting."""
    st.write("### 🔍 Authentication Debug Info")
    
    user = get_authenticated_user()
    if user:
        st.success("✅ User authenticated")
        st.json({
            "user_id": user.get('id'),
            "email": user.get('email'),
            "has_token": bool(user.get('auth_token'))
        })
    else:
        st.error("❌ No authentication found")
    
    # Check session state
    st.write("#### Session State:")
    auth_keys = [key for key in st.session_state.keys() if 'auth' in key.lower() or 'user' in key.lower()]
    if auth_keys:
        for key in auth_keys:
            st.write(f"- {key}: {bool(st.session_state.get(key))}")
    else:
        st.write("No authentication keys in session state")
    
    # Test database connection
    auth_valid, message = check_authentication_status()
    if auth_valid:
        st.success(f"✅ Database connection: {message}")
    else:
        st.error(f"❌ Database connection: {message}")

def clear_authentication():
    """Clear all authentication data - useful for logout."""
    try:
        # Clear session state
        auth_keys = [key for key in st.session_state.keys() if any(term in key.lower() for term in ['auth', 'user', 'token'])]
        for key in auth_keys:
            del st.session_state[key]
        
        # Clear user-specific data
        data_keys = [
            'words_learned', 'total_words_learned', 'level', 'points',
            'streak_days', 'last_active_date', 'streak_savers',
            'achievements', 'badges', 'daily_challenges',
            'word_of_the_day', 'category_progress', 'vocabulary_tree',
            'current_session_id', 'data_loaded'
        ]
        for key in data_keys:
            if key in st.session_state:
                del st.session_state[key]
        
        st.success("✅ Authentication cleared")
        
    except Exception as e:
        st.error(f"❌ Error clearing authentication: {e}")