import streamlit as st
import os
import sys

st.set_page_config(
    page_title="Language Learning App - Basic Version",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 Language Learning App")
st.subheader("Basic Version")

st.write(f"Python version: {sys.version}")
st.write(f"Current working directory: {os.getcwd()}")
st.write(f"Directory contents: {os.listdir()}")

# Create tabs for different sections
tab1, tab2 = st.tabs(["App Info", "Troubleshooting"])

with tab1:
    st.markdown("""
    ## Language Learning Application
    
    This is a simplified version of the language learning application.
    
    ### Available Features in Full Version:
    - Object recognition with camera
    - Translation to multiple languages
    - Quiz mode for vocabulary practice
    - Progress tracking and statistics
    
    Currently working on resolving deployment issues for the full application.
    """)
    
    # Display some sample translations
    st.subheader("Sample Translations")
    sample_words = {
        "Apple": {"es": "Manzana", "fr": "Pomme", "de": "Apfel"},
        "Book": {"es": "Libro", "fr": "Livre", "de": "Buch"},
        "Water": {"es": "Agua", "fr": "Eau", "de": "Wasser"}
    }
    
    for english, translations in sample_words.items():
        with st.expander(f"English: {english}"):
            for lang, word in translations.items():
                st.write(f"{lang.upper()}: {word}")

with tab2:
    st.markdown("""
    ## Troubleshooting Information
    
    The full application requires several libraries that are currently not compatible with Python 3.12:
    
    1. **PyTorch**: No wheels available for Python 3.12
    2. **OpenCV**: Compatibility issues with Python 3.12
    3. **NumPy**: Build problems due to missing 'distutils' in Python 3.12
    
    We're working on creating a compatible version that will run on Streamlit Cloud.
    """)
    
    # Import diagnostics
    st.subheader("Import Status")
    imports = ["numpy", "pandas", "matplotlib", "PIL"]
    
    for imp in imports:
        try:
            if imp == "PIL":
                from PIL import Image
                st.success(f"✅ {imp} imported successfully")
            else:
                __import__(imp)
                st.success(f"✅ {imp} imported successfully")
        except ImportError as e:
            st.error(f"❌ {imp} import failed: {e}")