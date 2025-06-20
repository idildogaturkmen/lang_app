import subprocess
import threading
import time
import signal
import sys
import os
import requests
from flask import Flask, request, jsonify
from werkzeug.middleware.proxy_fix import ProxyFix
import logging

# Import the auth API
from auth_api import api_app

# Create main Flask app for routing
app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
streamlit_process = None
STREAMLIT_PORT = 8501
API_PORT = 5001
MAIN_PORT = int(os.environ.get('PORT', 10000))

def start_streamlit():
    """Start Streamlit app."""
    global streamlit_process
    
    cmd = [
        "streamlit", "run", "main.py",
        "--server.port", str(STREAMLIT_PORT),
        "--server.address", "0.0.0.0",
        "--server.headless", "true",
        "--server.fileWatcherType", "none",
        "--browser.gatherUsageStats", "false",
        "--server.maxUploadSize", "10"
    ]
    
    logger.info(f"Starting Streamlit on port {STREAMLIT_PORT}")
    streamlit_process = subprocess.Popen(cmd)
    
    # Wait for Streamlit to be ready
    max_retries = 30
    for i in range(max_retries):
        try:
            response = requests.get(f'http://localhost:{STREAMLIT_PORT}/healthz', timeout=2)
            if response.status_code == 200:
                logger.info("Streamlit is ready!")
                break
        except:
            pass
        time.sleep(1)
    else:
        logger.warning("Streamlit may not have started properly")

def start_auth_api():
    """Start authentication API in a thread."""
    def run_api():
        api_app.run(host='0.0.0.0', port=API_PORT, debug=False, threaded=True)
    
    api_thread = threading.Thread(target=run_api, daemon=True)
    api_thread.start()
    logger.info(f"Auth API started on port {API_PORT}")
    
    # Wait for API to be ready
    max_retries = 10
    for i in range(max_retries):
        try:
            response = requests.get(f'http://localhost:{API_PORT}/api/health', timeout=2)
            if response.status_code == 200:
                logger.info("Auth API is ready!")
                break
        except:
            pass
        time.sleep(1)

# Route for health check
@app.route('/health')
def health():
    """Health check for the main server."""
    return jsonify({
        'status': 'healthy',
        'services': {
            'streamlit': streamlit_process is not None and streamlit_process.poll() is None,
            'auth_api': True
        }
    })

# Proxy API requests to the auth API
@app.route('/api/<path:path>', methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'])
def proxy_api(path):
    """Proxy API requests to the auth API."""
    try:
        url = f'http://localhost:{API_PORT}/api/{path}'
        
        # Forward the request
        if request.method == 'GET':
            response = requests.get(url, params=request.args, headers=dict(request.headers), timeout=10)
        elif request.method == 'POST':
            response = requests.post(url, json=request.get_json(), headers=dict(request.headers), timeout=10)
        elif request.method == 'OPTIONS':
            response = requests.options(url, headers=dict(request.headers), timeout=10)
        else:
            response = requests.request(
                method=request.method,
                url=url,
                json=request.get_json() if request.is_json else None,
                data=request.get_data() if not request.is_json else None,
                headers=dict(request.headers),
                timeout=10
            )
        
        # Return the response
        from flask import Response
        return Response(
            response.content,
            status=response.status_code,
            headers=dict(response.headers)
        )
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error proxying API request: {e}")
        return jsonify({'error': 'API unavailable'}), 503

# Proxy all other requests to Streamlit
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def proxy_streamlit(path):
    """Proxy requests to Streamlit app."""
    try:
        url = f'http://localhost:{STREAMLIT_PORT}/{path}'
        
        # Forward query parameters
        if request.query_string:
            url += f'?{request.query_string.decode()}'
        
        # Forward the request
        response = requests.request(
            method=request.method,
            url=url,
            headers={k: v for k, v in request.headers if k.lower() != 'host'},
            data=request.get_data(),
            stream=True,
            timeout=30
        )
        
        # Return the response
        from flask import Response
        return Response(
            response.iter_content(chunk_size=8192),
            status=response.status_code,
            headers=dict(response.headers)
        )
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error proxying to Streamlit: {e}")
        return jsonify({
            'error': 'Streamlit app unavailable',
            'message': 'The learning app is starting up. Please wait a moment and refresh.'
        }), 503

def signal_handler(signum, frame):
    """Handle shutdown signals."""
    global streamlit_process
    logger.info("Shutting down services...")
    
    if streamlit_process:
        streamlit_process.terminate()
        streamlit_process.wait()
    
    sys.exit(0)

def main():
    """Main function to start all services."""
    global streamlit_process
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info("Starting Vocam unified server...")
    
    # Start authentication API
    start_auth_api()
    
    # Start Streamlit
    start_streamlit()
    
    # Start main proxy server
    logger.info(f"Starting main server on port {MAIN_PORT}")
    app.run(host='0.0.0.0', port=MAIN_PORT, debug=False, threaded=True)

if __name__ == "__main__":
    main()