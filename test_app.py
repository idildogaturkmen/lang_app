import streamlit as st
import sys
import os

st.title("System Diagnostic App")

st.write(f"Python version: {sys.version}")
st.write(f"Current working directory: {os.getcwd()}")
st.write(f"Directory contents: {os.listdir()}")

# Try importing core dependencies
st.subheader("Testing dependencies")

try:
    import numpy as np
    st.success("✅ NumPy imported successfully")
except ImportError as e:
    st.error(f"❌ NumPy import failed: {e}")

try:
    import pandas as pd
    st.success("✅ Pandas imported successfully")
except ImportError as e:
    st.error(f"❌ Pandas import failed: {e}")

try:
    from PIL import Image
    st.success("✅ PIL/Pillow imported successfully")
except ImportError as e:
    st.error(f"❌ PIL import failed: {e}")

try:
    import cv2
    st.success(f"✅ OpenCV imported successfully: {cv2.__version__}")
except ImportError as e:
    st.error(f"❌ OpenCV import failed: {e}")
    
# Check for system files
st.subheader("System Environment")
env_vars = [
    "PYTHONPATH",
    "PATH",
    "STREAMLIT_SERVER_PORT"
]
for var in env_vars:
    st.write(f"**{var}**: {os.environ.get(var, 'Not set')}")

# Add a check for our specific error path
path_to_check = "/mount/admin/install_path"
st.write(f"Checking path: {path_to_check}")
if os.path.exists(path_to_check):
    st.success(f"✅ Path exists: {path_to_check}")
else:
    st.error(f"❌ Path does not exist: {path_to_check}")