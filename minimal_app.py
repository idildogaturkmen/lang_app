import streamlit as st

st.title("Minimal Test App")
st.write("This is a minimal test app to debug installation issues.")

# Test imports one by one
import_statuses = {}

try:
    import numpy as np
    import_statuses["numpy"] = "✅ Success"
except Exception as e:
    import_statuses["numpy"] = f"❌ Failed: {str(e)}"

try:
    import pandas as pd
    import_statuses["pandas"] = "✅ Success"
except Exception as e:
    import_statuses["pandas"] = f"❌ Failed: {str(e)}"

try:
    import matplotlib.pyplot as plt
    import_statuses["matplotlib"] = "✅ Success"
except Exception as e:
    import_statuses["matplotlib"] = f"❌ Failed: {str(e)}"

try:
    from PIL import Image
    import_statuses["Pillow"] = "✅ Success"
except Exception as e:
    import_statuses["Pillow"] = f"❌ Failed: {str(e)}"

try:
    import cv2
    import_statuses["OpenCV"] = "✅ Success"
except Exception as e:
    import_statuses["OpenCV"] = f"❌ Failed: {str(e)}"

try:
    import torch
    import_statuses["PyTorch"] = "✅ Success"
except Exception as e:
    import_statuses["PyTorch"] = f"❌ Failed: {str(e)}"

st.write("## Import Status")
for package, status in import_statuses.items():
    st.write(f"**{package}**: {status}")