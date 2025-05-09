import streamlit as st
import os
import sys

st.title("Diagnostic App")
st.write(f"Python version: {sys.version}")
st.write(f"Current working directory: {os.getcwd()}")

# Try importing dependencies one by one
import_results = {}

# Base libraries
for lib in ["numpy", "pandas", "matplotlib", "PIL"]:
    try:
        if lib == "PIL":
            from PIL import Image
            import_results[lib] = "✅ Success"
        else:
            __import__(lib)
            import_results[lib] = "✅ Success"
    except Exception as e:
        import_results[lib] = f"❌ Failed: {str(e)}"

# Computer vision
try:
    import cv2
    import_results["OpenCV"] = f"✅ Success (v{cv2.__version__})"
except Exception as e:
    import_results["OpenCV"] = f"❌ Failed: {str(e)}"

# Deep learning
try:
    import torch
    import_results["PyTorch"] = f"✅ Success (v{torch.__version__})"
except Exception as e:
    import_results["PyTorch"] = f"❌ Failed: {str(e)}"

# Display results
st.write("## Import Status")
for package, status in import_results.items():
    st.write(f"**{package}**: {status}")