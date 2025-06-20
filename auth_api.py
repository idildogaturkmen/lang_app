from flask import Flask, request, jsonify
from flask_cors import CORS
from database import LanguageLearningDB, initialize_demo_users
import os
import re
import threading

# Create Flask app for API only
api_app = Flask(__name__)
api_app.secret_key = os.environ.get('SECRET_KEY', 'vocam_secret_key_change_in_production')

# Enable CORS for Netlify domain
CORS(api_app, origins=[
    'https://vocam.app', 
    'https://www.vocam.app',
    'https://*.netlify.app',
    'http://localhost:*'
], supports_credentials=True)

# Initialize database
db = LanguageLearningDB("language_learning.db")
initialize_demo_users(db)

def validate_email(email):
    """Validate email format."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def validate_username(username):
    """Validate username format."""
    pattern = r'^[a-zA-Z0-9_]{3,20}$'
    return re.match(pattern, username) is not None

@api_app.route('/api/auth/register', methods=['POST', 'OPTIONS'])
def register():
    """Register a new user."""
    if request.method == 'OPTIONS':
        return '', 204
        
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'success': False, 'message': 'No data provided'}), 400
        
        username = data.get('username', '').strip()
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')
        display_name = data.get('display_name', username)
        
        # Validation
        if not all([username, email, password]):
            return jsonify({'success': False, 'message': 'All fields are required'}), 400
        
        if not validate_username(username):
            return jsonify({'success': False, 'message': 'Username must be 3-20 characters, letters, numbers, and underscores only'}), 400
        
        if not validate_email(email):
            return jsonify({'success': False, 'message': 'Invalid email format'}), 400
        
        if len(password) < 6:
            return jsonify({'success': False, 'message': 'Password must be at least 6 characters long'}), 400
        
        # Create user
        user_id = db.create_user(username, email, password, display_name)
        
        if user_id:
            # Create session token
            session_token = db.create_session_token(user_id)
            
            if session_token:
                user = db.get_user_by_id(user_id)
                
                return jsonify({
                    'success': True,
                    'message': 'Registration successful',
                    'user': {
                        'id': user['id'],
                        'username': user['username'],
                        'email': user['email'],
                        'display_name': user['display_name'],
                        'total_points': user['total_points'],
                        'current_level': user['current_level'],
                        'words_learned_total': user['words_learned_total']
                    },
                    'session_token': session_token
                }), 201
            else:
                return jsonify({'success': False, 'message': 'Failed to create session'}), 500
        else:
            return jsonify({'success': False, 'message': 'Registration failed'}), 500
            
    except ValueError as e:
        return jsonify({'success': False, 'message': str(e)}), 400
    except Exception as e:
        print(f"Registration error: {e}")
        return jsonify({'success': False, 'message': 'Internal server error'}), 500

@api_app.route('/api/auth/login', methods=['POST', 'OPTIONS'])
def login():
    """Authenticate a user."""
    if request.method == 'OPTIONS':
        return '', 204
        
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'success': False, 'message': 'No data provided'}), 400
        
        username_or_email = data.get('username', '').strip()
        password = data.get('password', '')
        
        if not all([username_or_email, password]):
            return jsonify({'success': False, 'message': 'Username and password are required'}), 400
        
        # Authenticate user
        user = db.authenticate_user(username_or_email, password)
        
        if user:
            # Create session token
            session_token = db.create_session_token(user['id'])
            
            if session_token:
                return jsonify({
                    'success': True,
                    'message': 'Login successful',
                    'user': {
                        'id': user['id'],
                        'username': user['username'],
                        'email': user['email'],
                        'display_name': user['display_name'],
                        'total_points': user['total_points'],
                        'current_level': user['current_level'],
                        'words_learned_total': user['words_learned_total']
                    },
                    'session_token': session_token
                }), 200
            else:
                return jsonify({'success': False, 'message': 'Failed to create session'}), 500
        else:
            return jsonify({'success': False, 'message': 'Invalid username or password'}), 401
            
    except Exception as e:
        print(f"Login error: {e}")
        return jsonify({'success': False, 'message': 'Internal server error'}), 500

@api_app.route('/api/auth/validate', methods=['POST', 'OPTIONS'])
def validate_session():
    """Validate a session token."""
    if request.method == 'OPTIONS':
        return '', 204
        
    try:
        data = request.get_json()
        session_token = data.get('session_token') if data else None
        
        if not session_token:
            return jsonify({'success': False, 'message': 'Session token required'}), 400
        
        user = db.validate_session_token(session_token)
        
        if user:
            return jsonify({
                'success': True,
                'user': {
                    'id': user['id'],
                    'username': user['username'],
                    'email': user['email'],
                    'display_name': user['display_name'],
                    'total_points': user['total_points'],
                    'current_level': user['current_level'],
                    'words_learned_total': user['words_learned_total']
                }
            }), 200
        else:
            return jsonify({'success': False, 'message': 'Invalid or expired session'}), 401
            
    except Exception as e:
        print(f"Session validation error: {e}")
        return jsonify({'success': False, 'message': 'Internal server error'}), 500

@api_app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'service': 'vocam-auth-api'}), 200

# Error handlers
@api_app.errorhandler(404)
def not_found(error):
    return jsonify({'success': False, 'message': 'Endpoint not found'}), 404

@api_app.errorhandler(500)
def internal_error(error):
    return jsonify({'success': False, 'message': 'Internal server error'}), 500

def run_auth_api():
    """Run the authentication API server."""
    port = int(os.environ.get('AUTH_PORT', 5001))
    api_app.run(host='0.0.0.0', port=port, debug=False, threaded=True)

if __name__ == '__main__':
    run_auth_api()