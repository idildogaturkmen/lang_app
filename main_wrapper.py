import streamlit as st
import os
import sys
import importlib.util
import numpy as np
import time

# Set up the page
st.set_page_config(
    page_title="AI Language Learning App",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Display Python version info
st.sidebar.markdown(f"**Python Version:** {sys.version}")

# Create compatibility layers for problematic modules
# 1. OpenCV (cv2) compatibility
class DummyCV2:
    def __init__(self):
        self.__version__ = "Fallback 4.6.0"
        
    def __getattr__(self, name):
        def dummy_method(*args, **kwargs):
            return None
        return dummy_method
        
    def cvtColor(self, img, code):
        return img  # Return the input image unchanged
        
    @staticmethod
    def imread(path):
        try:
            from PIL import Image
            import numpy as np
            img = Image.open(path)
            return np.array(img)
        except Exception:
            return None
            
    @staticmethod
    def imwrite(path, img):
        try:
            from PIL import Image
            import numpy as np
            Image.fromarray(img).save(path)
            return True
        except Exception:
            return False

# 2. PyTorch compatibility
class DummyTorch:
    def __init__(self):
        self.hub = type('obj', (object,), {
            'load': lambda *args, **kwargs: DummyModel()
        })
    
    def __getattr__(self, name):
        def dummy_method(*args, **kwargs):
            return None
        return dummy_method

class DummyModel:
    def __init__(self):
        self.names = {0: "object", 1: "person", 2: "dog", 3: "cat", 4: "car", 5: "chair"}
        
    def __call__(self, image):
        # Create a dummy result
        class DummyResult:
            def __init__(self):
                self.xyxy = [[np.array([100, 100, 200, 200, 0.9, 1])]]
                
            def render(self):
                # Return the original image with a frame
                if isinstance(image, np.ndarray):
                    img = image.copy()
                else:
                    img = np.array(image)
                # Add a colored border to show "detection"
                h, w = img.shape[:2]
                img[0:5, :] = [0, 255, 0]  # Top border
                img[-5:, :] = [0, 255, 0]  # Bottom border
                img[:, 0:5] = [0, 255, 0]  # Left border
                img[:, -5:] = [0, 255, 0]  # Right border
                return [img]
        return DummyResult()
        
    def eval(self):
        return self

# Inject our dummy modules into sys.modules
try:
    import cv2
    st.sidebar.success("✅ OpenCV imported successfully")
except ImportError:
    sys.modules['cv2'] = DummyCV2()
    st.sidebar.info("⚠️ Using fallback implementation for OpenCV")

try:
    import torch
    st.sidebar.success("✅ PyTorch imported successfully")
except ImportError:
    sys.modules['torch'] = DummyTorch()
    st.sidebar.info("⚠️ Using fallback implementation for PyTorch")

# Fix Google Cloud credentials handling
def prepare_google_credentials():
    import json
    import tempfile
    
    try:
        if 'gcp_service_account' in st.secrets:
            # Create a temporary file to store credentials
            credentials_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.json')
            credentials_path = credentials_temp.name
            
            # Get a copy of all secret key-value pairs
            credentials_dict = {}
            for key in st.secrets["gcp_service_account"]:
                credentials_dict[key] = st.secrets["gcp_service_account"][key]
            
            # Write the credentials to the temporary file
            with open(credentials_path, 'w') as f:
                json.dump(credentials_dict, f)
            
            # Set environment variable to point to this file
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
            st.sidebar.success("Google Cloud credentials loaded from secrets!")
        else:
            # Local development fallback (use a placeholder for Streamlit Cloud)
            dummy_creds = {
                "type": "service_account",
                "project_id": "dummy-project",
                "private_key_id": "dummy",
                "private_key": "dummy",
                "client_email": "dummy@example.com",
                "client_id": "dummy",
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                "client_x509_cert_url": "dummy",
                "universe_domain": "googleapis.com"
            }
            
            credentials_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.json')
            credentials_path = credentials_temp.name
            
            with open(credentials_path, 'w') as f:
                json.dump(dummy_creds, f)
                
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
            st.sidebar.info("Using dummy Google Cloud credentials (limited features)")
    except Exception as e:
        st.sidebar.error(f"Error setting up credentials: {e}")

# Prepare Google credentials
prepare_google_credentials()

# Add CSS to ensure compatibility
st.markdown("""
<style>
    .stApp {
        max-width: 100%;
    }
    .app-container {
        margin: 0;
        padding: 0;
    }
    @media (max-width: 768px) {
        button {
            font-size: 1.1rem !important;
            padding: 0.5rem !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# The key is to import main.py as a module, not execute it directly
try:
    import main_backup
    st.sidebar.success("✅ Main application loaded successfully!")
except Exception as e:
    st.error(f"Error loading main application: {e}")
    st.markdown(f"""
    ## Main Application Error
    
    There was an error loading the main application:
    ```
    {str(e)}
    ```
    
    Please try using the simplified app version (full_app.py) instead.
    """)