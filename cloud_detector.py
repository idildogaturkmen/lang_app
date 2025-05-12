import os
import sys

# Try to detect if we're running on Streamlit Cloud
def detect_streamlit_cloud():
    """Detect if we're running on Streamlit Cloud environment"""
    # Check for environment variables that might indicate Streamlit Cloud
    in_streamlit_cloud = False
    
    # These environment variables might be present in Streamlit Cloud
    streamlit_env_vars = [
        'STREAMLIT_SHARING',
        'IS_STREAMLIT_CLOUD',
        'STREAMLIT_SERVER_URL'
    ]
    
    for var in streamlit_env_vars:
        if os.environ.get(var):
            in_streamlit_cloud = True
            break
            
    # Check for paths that might indicate Streamlit deployment
    streamlit_paths = [
        '/app/streamlit',
        '/mount/src/streamlit'
    ]
    
    for path in streamlit_paths:
        if os.path.exists(path):
            in_streamlit_cloud = True
            break
    
    # Set an environment variable so other modules can check
    if in_streamlit_cloud:
        os.environ['IS_STREAMLIT_CLOUD'] = 'true'
        
    return in_streamlit_cloud

# Run detection
is_cloud = detect_streamlit_cloud()

# Print detection result (for debugging)
if is_cloud:
    print("Detected Streamlit Cloud environment")
    # Disable features that require PyAudio
    os.environ['DISABLE_PRONUNCIATION_PRACTICE'] = 'true'
else:
    print("Running in local environment")