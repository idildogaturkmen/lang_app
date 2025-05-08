import streamlit as st
import sys
import os
import importlib

# Set up the page
st.set_page_config(
    page_title="AI Language Learning App",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.sidebar.markdown(f"**Python Version:** {sys.version}")

# Try to fix the numpy/opencv compatibility issue
try:
    # First, make sure numpy is properly imported
    import numpy as np
    st.sidebar.success("NumPy imported successfully")
    
    # Then try to import OpenCV with specific error handling
    try:
        import cv2
        st.sidebar.success(f"OpenCV imported successfully: v{cv2.__version__}")
    except ImportError as e:
        st.sidebar.error(f"OpenCV import error: {e}")
        
        # Create a dummy cv2 module for fallback
        class DummyCV2:
            def __init__(self):
                pass
                
            def __getattr__(self, name):
                def dummy_method(*args, **kwargs):
                    return None
                return dummy_method
                
            def cvtColor(self, *args, **kwargs):
                return args[0]
                
            @staticmethod
            def imread(path):
                try:
                    img = np.array(importlib.import_module('PIL.Image').open(path))
                    return img
                except Exception:
                    return None
                    
            @staticmethod
            def imwrite(path, img):
                try:
                    importlib.import_module('PIL.Image').fromarray(img).save(path)
                    return True
                except Exception:
                    return None
        
        # Replace cv2 module with our dummy
        sys.modules['cv2'] = DummyCV2()
        st.sidebar.warning("Using fallback implementation for OpenCV")
        
except Exception as e:
    st.sidebar.error(f"Error setting up dependencies: {e}")

# Now import and run the main application
try:
    # Check if the main module file exists
    if os.path.exists("main.py"):
        # Import the main module
        import main
        
        # If we got here, the main app imported successfully
        st.sidebar.success("Main application loaded successfully")
    else:
        st.error("main.py file not found!")
except Exception as e:
    st.error(f"Error loading main application: {e}")
    st.markdown(f"""
    ## Application Error
    
    There was an error loading the main application:
    ```
    {str(e)}
    ```
    
    Please contact the application developer with this error message.
    """)