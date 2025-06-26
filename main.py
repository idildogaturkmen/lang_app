import streamlit as st
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import base64
import time
import sqlite3
from datetime import datetime, date
import re
from PIL import Image
from io import BytesIO
from gamification import GamificationSystem
import random
import io
from vocam_ui import apply_custom_css
from streamlit.components.v1 import components
import hashlib
from example_sentences import ExampleSentenceGenerator
import requests
from deep_translator import GoogleTranslator
from database import LanguageLearningDB
import json

# First, display Python version for
st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded"
)

def setup_iframe_embedding():
    """Setup proper iframe embedding configuration."""
    # Inject JavaScript to handle iframe communication
    iframe_js = """
    <script>
    // Prevent redirect loops in iframe
    if (window.location !== window.parent.location) {
        console.log('🖼️ Running in iframe mode');
        
        // Disable automatic redirects that cause loops
        window.addEventListener('beforeunload', function(e) {
            e.preventDefault();
            return '';
        });
        
        // Handle authentication in iframe context
        if (window.parent) {
            window.parent.postMessage({
                type: 'streamlit_ready',
                source: 'vocam_app'
            }, '*');
        }
    }
    
    // Handle authentication parameters from URL
    const urlParams = new URLSearchParams(window.location.search);
    const authToken = urlParams.get('auth_token');
    const userEmail = urlParams.get('user_email');
    const userId = urlParams.get('user_id');
    
    if (authToken && userEmail && userId) {
        console.log('🔐 Authentication parameters received');
        // Store in sessionStorage for use by Streamlit
        sessionStorage.setItem('vocam_auth_token', authToken);
        sessionStorage.setItem('vocam_user_email', userEmail);
        sessionStorage.setItem('vocam_user_id', userId);
    }
    </script>
    """
    
    st.components.v1.html(iframe_js, height=0)

# Call this function early in your app
setup_iframe_embedding()


class SupabaseDB:
    def __init__(self):
        self.supabase_url = "https://csszlzpsfwmsezursivk.supabase.co"
        self.supabase_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImNzc3psenBzZndtc2V6dXJzaXZrIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTA1Mjg1MjEsImV4cCI6MjA2NjEwNDUyMX0.gIi0Q_pifYpXeM1r8kWlgTO1LD8bc91lQ3suH8OWDKI"
        
    def get_user_id(self):
        """Get current user ID from authentication."""
        user = get_authenticated_user()
        if user:
            return str(user.get('id'))
        return None
    
    def get_headers(self):
        """Get headers with proper authentication."""
        user = get_authenticated_user()
        auth_token = user.get('auth_token') if user else None
        
        headers = {
            'apikey': self.supabase_key,
            'Content-Type': 'application/json',
            'Prefer': 'return=representation'
        }
        
        if auth_token:
            headers['Authorization'] = f'Bearer {auth_token}'
        else:
            headers['Authorization'] = f'Bearer {self.supabase_key}'
        
        return headers
    
    def check_word_exists(self, word_original, word_translated, language_translated):
        """Check if a word already exists in the user's vocabulary."""
        user_id = self.get_user_id()
        if not user_id:
            return False
            
        try:
            import requests
            
            headers = self.get_headers()
            
            # Query to check if word already exists for this user
            url = f'{self.supabase_url}/rest/v1/vocabulary?user_id=eq.{user_id}&word_original=eq.{word_original}&word_translated=eq.{word_translated}&language_translated=eq.{language_translated}'
            
            response = requests.get(url, headers=headers, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                return len(result) > 0  # Returns True if word exists
            return False
                
        except Exception as e:
            print(f"Error checking for existing word: {e}")
            return False
    
    def add_vocabulary(self, word_original, word_translated, language_translated, category=None, image_path=None):
        """Add vocabulary to Supabase - with duplicate prevention."""
        user_id = self.get_user_id()
        if not user_id:
            return None
        
        # Check if word already exists
        if self.check_word_exists(word_original, word_translated, language_translated):
            print(f"⚠️ Word '{word_original}' -> '{word_translated}' already exists in vocabulary")
            return 'duplicate'  # Return special value to indicate duplicate
            
        try:
            import requests
            
            headers = self.get_headers()
            
            data = {
                'user_id': user_id,
                'word_original': word_original,
                'word_translated': word_translated,
                'language_translated': language_translated,
                'category': category or 'other',
                'image_path': image_path,
                'source': f'user_{user_id}'
            }
            
            response = requests.post(
                f'{self.supabase_url}/rest/v1/vocabulary',
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code in [200, 201]:
                result = response.json()
                if result and len(result) > 0:
                    return result[0].get('id')
            return None
                
        except Exception as e:
            print(f"Error adding vocabulary: {e}")
            return None
    
    def get_all_vocabulary(self):
        """Get all vocabulary for current user."""
        user_id = self.get_user_id()
        if not user_id:
            return []
            
        try:
            import requests
            
            headers = self.get_headers()
            url = f'{self.supabase_url}/rest/v1/vocabulary?user_id=eq.{user_id}&order=date_added.desc'
            
            response = requests.get(url, headers=headers, timeout=30)
            
            if response.status_code == 200:
                return response.json()
            return []
                
        except Exception as e:
            print(f"Error getting vocabulary: {e}")
            return []
    
    def start_session(self):
        """Start a new learning session."""
        user_id = self.get_user_id()
        if not user_id:
            return None
            
        try:
            import requests
            from datetime import datetime
            
            headers = self.get_headers()
            
            data = {
                'user_id': user_id,
                'start_time': datetime.now().isoformat(),
                'words_studied': 0,
                'words_learned': 0
            }
            
            response = requests.post(
                f'{self.supabase_url}/rest/v1/sessions',
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code in [200, 201]:
                result = response.json()
                if result and len(result) > 0:
                    return result[0].get('id')
            return None
                
        except Exception as e:
            print(f"Error starting session: {e}")
            return None
    
    def end_session(self, session_id, words_studied, words_learned):
        """End a learning session."""
        if not session_id:
            return False
            
        try:
            import requests
            from datetime import datetime
            
            headers = self.get_headers()
            
            data = {
                'end_time': datetime.now().isoformat(),
                'words_studied': words_studied,
                'words_learned': words_learned
            }
            
            response = requests.patch(
                f'{self.supabase_url}/rest/v1/sessions?id=eq.{session_id}',
                headers=headers,
                json=data,
                timeout=30
            )
            
            return response.status_code in [200, 204]
                
        except Exception as e:
            print(f"Error ending session: {e}")
            return False
        
    def get_user_streak_data(self):
        """Get user's streak data from Supabase."""
        user_id = self.get_user_id()
        if not user_id:
            return None
            
        try:
            import requests
            
            headers = self.get_headers()
            url = f'{self.supabase_url}/rest/v1/user_streaks?user_id=eq.{user_id}'
            
            response = requests.get(url, headers=headers, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                return result[0] if result else None
            return None
                
        except Exception as e:
            print(f"Error getting streak data: {e}")
            return None

    def update_user_streak_data(self, streak_days, last_active_date, streak_savers=0):
        """Update user's streak data in Supabase."""
        user_id = self.get_user_id()
        if not user_id:
            return False
            
        try:
            import requests
            from datetime import datetime
            
            headers = self.get_headers()
            
            # Convert date to string if needed
            if hasattr(last_active_date, 'isoformat'):
                date_str = last_active_date.isoformat()
            else:
                date_str = str(last_active_date)
            
            data = {
                'user_id': user_id,
                'streak_days': streak_days,
                'last_active_date': date_str,
                'streak_savers': streak_savers,
                'updated_at': datetime.now().isoformat()
            }
            
            # Try to update first, then insert if not exists
            update_url = f'{self.supabase_url}/rest/v1/user_streaks?user_id=eq.{user_id}'
            update_response = requests.patch(update_url, headers=headers, json=data, timeout=30)
            
            if update_response.status_code in [200, 204]:
                print(f"✅ Streak updated: {streak_days} days")
                return True
            
            # If update failed, try insert (user doesn't exist yet)
            insert_url = f'{self.supabase_url}/rest/v1/user_streaks'
            insert_response = requests.post(insert_url, headers=headers, json=data, timeout=30)
            
            if insert_response.status_code in [200, 201]:
                print(f"✅ Streak created: {streak_days} days")
                return True
            else:
                print(f"❌ Failed to save streak: {insert_response.status_code} - {insert_response.text}")
                return False
                
        except Exception as e:
            print(f"Error updating streak data: {e}")
            return False

        
# Authentication Functions for Supabase
def get_url_params():
    """Get URL parameters from Streamlit."""
    try:
        # Try the newer method first
        query_params = st.query_params
        return {key: [value] for key, value in query_params.items()}
    except:
        try:
            # Fallback to experimental method
            query_params = st.experimental_get_query_params()
            return query_params
        except:
            # Last resort - parse manually
            import urllib.parse
            url = st.query_params
            if hasattr(url, 'to_dict'):
                return {key: [value] for key, value in url.to_dict().items()}
            return {}

def get_authenticated_user():
    """Get authenticated user from URL parameters or session state."""
    if 'user' not in st.session_state:
        try:
            # Use experimental_get_query_params for Streamlit 1.29.0
            query_params = st.experimental_get_query_params()
            
            # Debug: Print what we're getting
            st.write("🔍 Debug - Query params received:", query_params)  # Remove this after testing
            
            auth_token = query_params.get('auth_token')
            user_email = query_params.get('user_email') 
            user_id = query_params.get('user_id')
            auth_provider = query_params.get('auth_provider')
            
            if auth_token and user_email and user_id:
                # Extract values from lists (query params come as lists)
                token = auth_token[0] if isinstance(auth_token, list) else auth_token
                email = user_email[0] if isinstance(user_email, list) else user_email
                uid = user_id[0] if isinstance(user_id, list) else user_id
                provider = auth_provider[0] if isinstance(auth_provider, list) and auth_provider else 'supabase'
                
                # Create user object from URL parameters
                st.session_state.user = {
                    'id': uid,
                    'email': email,
                    'auth_token': token,
                    'provider': provider
                }
                print(f"✅ Authenticated user from URL: {email}")
                return st.session_state.user
            else:
                print("❌ Authentication failed - missing params")
                print(f"Token: {bool(auth_token)}, Email: {bool(user_email)}, ID: {bool(user_id)}")
                return None
                
        except Exception as e:
            print(f"❌ Error getting query params: {e}")
            st.write(f"❌ Auth error: {e}")  # Remove this after testing
            return None
    
    return st.session_state.get('user')

def require_authentication():
    """Require user authentication before proceeding."""
    user = get_authenticated_user()
    
    if not user:
        st.error("🔐 Authentication Required")
        
        # Show debug info temporarily
        query_params = st.experimental_get_query_params()
        st.write("Debug - All URL params:", query_params)  # Remove this after testing
        
        st.markdown("""
        **Please access this app through the proper authentication flow.**
        
        If you're seeing this page, please:
        1. Go back to [vocam.app/web](https://vocam.app/web)
        2. Sign in with your account
        3. You'll be automatically redirected here
        
        If you don't have an account, you can create one at the link above.
        """)
        
        st.markdown("""
        <div style="text-align: center; margin: 2rem 0;">
            <a href="https://vocam.app/web" 
               style="background: #1679AB; color: white; padding: 12px 24px; 
                      border-radius: 8px; text-decoration: none; font-weight: 600;">
                🚀 Go to Vocam Login
            </a>
        </div>
        """, unsafe_allow_html=True)
        
        st.stop()
    
    return user

def debug_authentication():
    """Debug version to see what we're receiving."""
    st.title("🔍 Authentication Debug")
    
    # Get query parameters
    try:
        query_params = st.experimental_get_query_params()
        st.write("**All Query Parameters:**", query_params)
        
        # Check each parameter
        auth_token = query_params.get('auth_token')
        user_email = query_params.get('user_email')
        user_id = query_params.get('user_id')
        auth_provider = query_params.get('auth_provider')
        
        st.write("**Individual Parameters:**")
        st.write(f"- auth_token: {auth_token}")
        st.write(f"- user_email: {user_email}")  
        st.write(f"- user_id: {user_id}")
        st.write(f"- auth_provider: {auth_provider}")
        
        if auth_token and user_email and user_id:
            st.success("✅ All required parameters present!")
            
            # Extract from lists if needed
            token = auth_token[0] if isinstance(auth_token, list) else auth_token
            email = user_email[0] if isinstance(user_email, list) else user_email
            uid = user_id[0] if isinstance(user_id, list) else user_id
            
            st.write("**Processed Values:**")
            st.write(f"- Token (first 20 chars): {token[:20]}...")
            st.write(f"- Email: {email}")
            st.write(f"- User ID: {uid}")
            
            return {
                'id': uid,
                'email': email,
                'auth_token': token,
                'provider': 'supabase'
            }
        else:
            st.error("❌ Missing required parameters")
            return None
            
    except Exception as e:
        st.error(f"❌ Error: {e}")
        return None
    
user = debug_authentication()

if user:
    st.success(f"🎉 Successfully authenticated: {user['email']}")
    st.write("You can now proceed with the app!")
else:
    st.error("Authentication failed - check the debug info above")

def sync_user_data_to_supabase():
    """Comprehensive function to sync all user data to Supabase."""
    try:
        user = get_authenticated_user()
        if not user:
            print("❌ No authenticated user for data sync")
            return False
        
        user_id = user.get('id')
        if not user_id:
            print("❌ No user ID for data sync")
            return False
        
        # Get actual vocabulary count from Supabase
        vocabulary = get_all_vocabulary_direct()
        actual_word_count = len(vocabulary) if vocabulary else 0
        
        # Calculate level and points based on actual vocabulary
        calculated_level = max(1, actual_word_count // 10 + 1)
        calculated_points = actual_word_count * 10
        
        # Update session state with actual data
        st.session_state.words_learned = actual_word_count
        st.session_state.total_words_learned = actual_word_count
        st.session_state.level = calculated_level
        st.session_state.points = calculated_points
        
        # Prepare comprehensive user data
        user_data = {
            'user_id': user_id,
            'words_learned': actual_word_count,
            'total_words_learned': actual_word_count,
            'level': calculated_level,
            'points': calculated_points,
            'streak_days': st.session_state.get('streak_days', 0),
            'last_active_date': str(date.today()),
            'streak_savers': st.session_state.get('streak_savers', 0),
            'achievements': json.dumps(st.session_state.get('achievements', {})),
            'badges': json.dumps(st.session_state.get('badges', {})),
            'daily_challenges': json.dumps(st.session_state.get('daily_challenges', [])),
            'word_of_the_day': json.dumps(st.session_state.get('word_of_the_day')),
            'category_progress': json.dumps(st.session_state.get('category_progress', {})),
            'vocabulary_tree': json.dumps({'size': actual_word_count, 'level': calculated_level}),
            'updated_at': datetime.now().isoformat()
        }
        
        # Save to Supabase user_game_state table
        db = get_user_database()
        response = db.supabase.table('user_game_state').upsert(
            user_data,
            on_conflict='user_id'
        ).execute()
        
        if response.data:
            print(f"✅ User data synced to Supabase: {actual_word_count} words, Level {calculated_level}, {calculated_points} points")
            return True
        else:
            print(f"❌ Failed to sync user data: {response}")
            return False
            
    except Exception as e:
        print(f"❌ Error syncing user data: {e}")
        return False

def load_user_data_from_supabase():
    """Load all user data from Supabase on app startup."""
    try:
        user = get_authenticated_user()
        if not user:
            return False
        
        user_id = user.get('id')
        if not user_id:
            return False
        
        # First, get actual vocabulary count
        vocabulary = get_all_vocabulary_direct()
        actual_word_count = len(vocabulary) if vocabulary else 0
        
        # Load from Supabase
        db = get_user_database()
        response = db.supabase.table('user_game_state').select('*').eq('user_id', user_id).execute()
        
        if response.data and len(response.data) > 0:
            user_data = response.data[0]
            
            # Load data but prioritize actual vocabulary count
            st.session_state.words_learned = actual_word_count
            st.session_state.total_words_learned = actual_word_count
            st.session_state.level = max(user_data.get('level', 1), max(1, actual_word_count // 10 + 1))
            st.session_state.points = max(user_data.get('points', 0), actual_word_count * 10)
            st.session_state.streak_days = user_data.get('streak_days', 0)
            st.session_state.last_active_date = user_data.get('last_active_date')
            st.session_state.streak_savers = user_data.get('streak_savers', 0)
            
            # Load JSON data safely
            try:
                st.session_state.achievements = json.loads(user_data.get('achievements', '{}'))
                st.session_state.badges = json.loads(user_data.get('badges', '{}'))
                st.session_state.daily_challenges = json.loads(user_data.get('daily_challenges', '[]'))
                st.session_state.word_of_the_day = json.loads(user_data.get('word_of_the_day', 'null'))
                st.session_state.category_progress = json.loads(user_data.get('category_progress', '{}'))
                st.session_state.vocabulary_tree = json.loads(user_data.get('vocabulary_tree', '{}'))
            except:
                # If JSON parsing fails, use defaults
                st.session_state.achievements = {}
                st.session_state.badges = {}
                st.session_state.daily_challenges = []
                st.session_state.word_of_the_day = None
                st.session_state.category_progress = {}
                st.session_state.vocabulary_tree = {'size': actual_word_count, 'level': st.session_state.level}
            
            print(f"✅ User data loaded: {actual_word_count} words, Level {st.session_state.level}, {st.session_state.points} points")
            return True
        else:
            # Initialize with actual vocabulary count
            st.session_state.words_learned = actual_word_count
            st.session_state.total_words_learned = actual_word_count
            st.session_state.level = max(1, actual_word_count // 10 + 1)
            st.session_state.points = actual_word_count * 10
            st.session_state.streak_days = 0
            st.session_state.last_active_date = None
            st.session_state.streak_savers = 0
            st.session_state.achievements = {}
            st.session_state.badges = {}
            st.session_state.daily_challenges = []
            st.session_state.word_of_the_day = None
            st.session_state.category_progress = {}
            st.session_state.vocabulary_tree = {'size': actual_word_count, 'level': st.session_state.level}
            
            # Save initial data
            sync_user_data_to_supabase()
            print(f"🆕 Initialized user data with {actual_word_count} existing words")
            return True
            
    except Exception as e:
        print(f"❌ Error loading user data: {e}")
        return False

def detect_objects_google_vision(image, confidence_threshold=0.5):
    """Google Cloud Vision-based object detection."""
    try:
        import os
        from google.cloud import vision
        from google.oauth2 import service_account
        import io
        
        # Get API key from environment
        api_key = os.getenv('GOOGLE_CLOUD_VISION_API_KEY')
        if not api_key:
            print("❌ Google Cloud Vision API key not found")
            return [], np.array(image)
        
        # Initialize the client with API key
        client = vision.ImageAnnotatorClient(
            client_options={"api_key": api_key}
        )
        
        # Convert PIL image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG', quality=90)
        img_byte_arr.seek(0)
        
        # Create Vision API image object
        vision_image = vision.Image(content=img_byte_arr.getvalue())
        
        # Perform object localization
        objects = client.object_localization(image=vision_image).localized_object_annotations
        
        # Convert to our detection format
        detections = []
        img_array = np.array(image)
        height, width = img_array.shape[:2]
        
        for obj in objects:
            # Get confidence score
            confidence = obj.score
            
            if confidence < confidence_threshold:
                continue
            
            # Get object name and clean it
            label = obj.name.lower()
            
            # Map Google Vision labels to our categories
            mapped_label = map_google_vision_label(label)
            
            # Get bounding box (normalized coordinates)
            vertices = obj.bounding_poly.normalized_vertices
            
            # Convert normalized coordinates to pixel coordinates
            x_coords = [v.x * width for v in vertices]
            y_coords = [v.y * height for v in vertices]
            
            left = min(x_coords)
            top = min(y_coords)
            right = max(x_coords)
            bottom = max(y_coords)
            
            detections.append({
                'label': mapped_label,
                'confidence': confidence,
                'bbox': [left, top, right, bottom],
                'original_label': label  # Keep original for reference
            })
        
        print(f"✅ Google Vision detected {len(detections)} objects")
        
        # Draw detections on image
        result_image = draw_detections(img_array, detections)
        
        return detections, result_image
        
    except ImportError:
        print("❌ Google Cloud Vision library not installed")
        return [], np.array(image)
    except Exception as e:
        print(f"❌ Google Vision detection error: {e}")
        return [], np.array(image)

def map_google_vision_label(vision_label):
    """Map Google Vision labels to our existing label system."""
    # Mapping from Google Vision labels to our expected labels
    label_mapping = {
        # Electronics
        'mobile phone': 'cell phone',
        'smartphone': 'cell phone', 
        'telephone': 'cell phone',
        'computer': 'laptop',
        'laptop computer': 'laptop',
        'television': 'tv',
        'television set': 'tv',
        'computer monitor': 'tv',
        'computer mouse': 'mouse',
        'computer keyboard': 'keyboard',
        'remote control': 'remote',
        
        # Food & Drinks
        'drinking glass': 'cup',
        'coffee cup': 'cup',
        'tea cup': 'cup',
        'wine glass': 'wine glass',
        'water bottle': 'bottle',
        'plastic bottle': 'bottle',
        'glass bottle': 'bottle',
        'food': 'food',
        'fruit': 'fruit',
        'apple': 'apple',
        'banana': 'banana',
        'orange': 'orange',
        
        # Furniture
        'chair': 'chair',
        'armchair': 'chair',
        'office chair': 'chair',
        'sofa': 'couch',
        'couch': 'couch',
        'table': 'dining table',
        'desk': 'dining table',
        'bed': 'bed',
        'toilet': 'toilet',
        
        # Vehicles
        'car': 'car',
        'automobile': 'car',
        'vehicle': 'car',
        'bicycle': 'bicycle',
        'motorcycle': 'motorcycle',
        'bus': 'bus',
        'truck': 'truck',
        'airplane': 'airplane',
        'aircraft': 'airplane',
        
        # Animals
        'dog': 'dog',
        'cat': 'cat',
        'bird': 'bird',
        'horse': 'horse',
        
        # Personal items
        'handbag': 'handbag',
        'backpack': 'backpack',
        'suitcase': 'suitcase',
        'umbrella': 'umbrella',
        'tie': 'tie',
        
        # Sports
        'ball': 'sports ball',
        'football': 'sports ball',
        'basketball': 'sports ball',
        'tennis ball': 'sports ball',
        'baseball': 'sports ball',
        'soccer ball': 'sports ball',
        
        # Kitchen items
        'knife': 'knife',
        'fork': 'fork',
        'spoon': 'spoon',
        'bowl': 'bowl',
        'plate': 'bowl',
        
        # Household
        'book': 'book',
        'clock': 'clock',
        'vase': 'vase',
        'plant': 'potted plant',
        'houseplant': 'potted plant',
    }
    
    # Try exact match first
    if vision_label in label_mapping:
        return label_mapping[vision_label]
    
    # Try partial matches
    for vision_key, our_label in label_mapping.items():
        if vision_key in vision_label or vision_label in vision_key:
            return our_label
    
    # If no mapping found, clean the label and return it
    cleaned_label = vision_label.replace('_', ' ').strip()
    return cleaned_label

def get_google_vision_status():
    """Check Google Cloud Vision API status and configuration."""
    try:
        import os
        from google.cloud import vision
        
        api_key = os.getenv('GOOGLE_CLOUD_VISION_API_KEY')
        
        if not api_key:
            return "❌ API key not found in environment variables"
        
        # Test API connection with a simple request
        try:
            client = vision.ImageAnnotatorClient(
                client_options={"api_key": api_key}
            )
            
            # Create a small test image (1x1 pixel)
            import io
            from PIL import Image as PILImage
            test_img = PILImage.new('RGB', (1, 1), color='white')
            img_bytes = io.BytesIO()
            test_img.save(img_bytes, format='JPEG')
            img_bytes.seek(0)
            
            vision_image = vision.Image(content=img_bytes.getvalue())
            
            # Test with label detection (simpler than object detection)
            response = client.label_detection(image=vision_image)
            
            if response.error.message:
                return f"❌ API Error: {response.error.message}"
            
            return "✅ Google Cloud Vision API is working correctly"
            
        except Exception as e:
            return f"❌ API Connection Error: {str(e)}"
            
    except ImportError:
        return "❌ Google Cloud Vision library not installed"
    except Exception as e:
        return f"❌ Configuration Error: {str(e)}"

    
def display_quiz_image(word, caption=""):
    """Display an image for quiz questions, prioritizing cropped versions."""
    image_path = word.get('image_path', '')
    if not image_path:
        return False
    
    try:
        # Use cropped version if available
        display_image_path = get_cropped_image_path(image_path)
        
        if display_image_path.startswith('vocabulary-images/'):
            # Private Supabase Storage
            signed_url = get_signed_image_url(display_image_path)
            if signed_url:
                st.image(signed_url, caption=caption, width=400)
                st.markdown("*🎯 Focused view of detected object*")
                return True
        elif display_image_path.startswith('http'):
            # Legacy public URL
            st.image(display_image_path, caption=caption, width=400)
            if "_cropped" in display_image_path:
                st.markdown("*🎯 Focused view of detected object*")
            return True
        elif os.path.exists(display_image_path):
            # Local file
            image = Image.open(display_image_path)
            st.image(image, caption=caption, width=400)
            if "_cropped.jpg" in display_image_path:
                st.markdown("*🎯 Focused view of detected object*")
            return True
        
        return False
    except Exception as e:
        print(f"Error displaying quiz image: {e}")
        return False
    

def get_all_vocabulary_direct():
    """Get all vocabulary using Supabase."""
    user = get_authenticated_user()
    if not user:
        print("❌ No authenticated user for vocabulary retrieval")
        return []
    
    try:
        db = get_user_database()
        vocabulary = db.get_all_vocabulary()
        
        # Convert to the format expected by the app
        formatted_vocab = []
        for item in vocabulary:
            # Flatten user_progress data if it exists
            progress = item.get('user_progress', [])
            progress_data = progress[0] if progress else {}
            
            formatted_item = {
                'id': item['id'],
                'word_original': item['word_original'],
                'word_translated': item['word_translated'],
                'language_translated': item['language_translated'],
                'category': item.get('category'),
                'image_path': item.get('image_path'),
                'date_added': item['date_added'],
                'source': item.get('source'),
                'proficiency_level': progress_data.get('proficiency_level', 0),
                'review_count': progress_data.get('review_count', 0),
                'correct_count': progress_data.get('correct_count', 0),
                'last_reviewed': progress_data.get('last_reviewed')
            }
            formatted_vocab.append(formatted_item)
        
        print(f"📊 Retrieved {len(formatted_vocab)} vocabulary items for user")
        return formatted_vocab
        
    except Exception as e:
        print(f"❌ Error getting vocabulary: {e}")
        return []

user = require_authentication()

if 'data_loaded' not in st.session_state:
    load_user_data_from_supabase()
    st.session_state.data_loaded = True

# Sync data periodically (every 10 vocabulary additions)
if 'last_sync_count' not in st.session_state:
    st.session_state.last_sync_count = 0

current_word_count = len(get_all_vocabulary_direct()) if get_all_vocabulary_direct() else 0
if current_word_count != st.session_state.last_sync_count:
    sync_user_data_to_supabase()
    st.session_state.last_sync_count = current_word_count

def check_and_update_user_streak():
    """Check and update user's daily streak with Supabase storage."""
    try:
        from datetime import date, datetime, timedelta
        
        user = get_authenticated_user()
        if not user:
            return
        
        db = get_user_database()
        today = date.today()
        
        print(f"🔥 Checking streak for date: {today}")
        
        # Get current streak data from Supabase
        streak_data = db.get_user_streak_data()
        
        if not streak_data:
            # First time user - start streak
            print("🔥 First time user - starting streak")
            st.session_state.streak_days = 1
            st.session_state.last_active_date = today
            st.session_state.streak_savers = 0
            
            # Save to Supabase
            db.update_user_streak_data(1, today, 0)
            
            # Show welcome message
            st.toast("🔥 Welcome! Your learning streak has started!")
            return
        
        # Parse existing data
        current_streak = streak_data.get('streak_days', 0)
        last_active_str = streak_data.get('last_active_date', '')
        streak_savers = streak_data.get('streak_savers', 0)
        
        # Parse last active date
        try:
            if 'T' in last_active_str:  # ISO format with time
                last_active = datetime.fromisoformat(last_active_str.replace('Z', '+00:00')).date()
            else:  # Date only
                last_active = datetime.strptime(last_active_str, '%Y-%m-%d').date()
        except:
            last_active = today - timedelta(days=2)  # Force streak reset if can't parse
        
        print(f"🔥 Current streak: {current_streak}, Last active: {last_active}")
        
        # Calculate days since last activity
        days_passed = (today - last_active).days
        
        # Update session state with current values
        st.session_state.streak_days = current_streak
        st.session_state.last_active_date = last_active
        st.session_state.streak_savers = streak_savers
        
        # If already active today, no change needed
        if days_passed == 0:
            print("🔥 Already active today - no streak change")
            return
        
        # Streak continues (visited yesterday)
        if days_passed == 1:
            new_streak = current_streak + 1
            st.session_state.streak_days = new_streak
            st.session_state.last_active_date = today
            
            print(f"🔥 Streak continued! New streak: {new_streak}")
            
            # Save to Supabase
            db.update_user_streak_data(new_streak, today, streak_savers)
            
            # Check for streak milestones
            if new_streak == 3:
                st.toast("🔥 3-day streak! You're on fire!")
            elif new_streak == 7:
                st.session_state.streak_savers += 1
                db.update_user_streak_data(new_streak, today, streak_savers + 1)
                st.toast("🔥 7-day streak! You earned a Streak Saver! 🛟")
            elif new_streak == 30:
                st.session_state.streak_savers += 2
                db.update_user_streak_data(new_streak, today, streak_savers + 2)
                st.toast("🔥 30-day streak! Amazing dedication! You earned 2 Streak Savers! 🛟")
            elif new_streak % 7 == 0 and new_streak > 7:  # Every week after first
                st.toast(f"🔥 {new_streak}-day streak! Keep it up!")
            
        # Missed a day but have streak saver
        elif days_passed == 2 and streak_savers > 0:
            new_savers = streak_savers - 1
            st.session_state.streak_savers = new_savers
            st.session_state.last_active_date = today
            
            print(f"🛟 Used streak saver! Remaining: {new_savers}")
            
            # Save to Supabase
            db.update_user_streak_data(current_streak, today, new_savers)
            
            st.toast("🛟 Used a Streak Saver to maintain your streak!")
            
        # Streak broken
        else:
            print(f"💔 Streak broken! Days passed: {days_passed}")
            
            st.session_state.streak_days = 1  # Start new streak
            st.session_state.last_active_date = today
            
            # Save to Supabase
            db.update_user_streak_data(1, today, streak_savers)
            
            if current_streak > 1:
                st.toast(f"💔 Your {current_streak}-day streak ended, but you're starting fresh!")
            
    except Exception as e:
        print(f"❌ Error in streak check: {e}")
        # Initialize with safe defaults
        st.session_state.streak_days = 1
        st.session_state.last_active_date = date.today()
        st.session_state.streak_savers = 0

# CHECK AND UPDATE STREAK - Add this new section
try:
    check_and_update_user_streak()
except Exception as e:
    print(f"⚠️ Streak check error (non-critical): {e}")

def get_user_database():
    """Get Supabase database instance."""
    if 'supabase_db' not in st.session_state:
        st.session_state.supabase_db = SupabaseDB()
        print("✅ Supabase database connection initialized")
    return st.session_state.supabase_db



# Import the UI enhancement module
from vocam_ui import (
    apply_custom_css, 
    success_message, 
    info_message, 
    warning_message, 
    error_message,
    show_loading_spinner, 
    vocam_card, 
    word_card,
    add_result_separator,
    add_scroll_indicator,
    style_title,
    style_section_title,
    add_footer
)

apply_custom_css()

# Try importing OCR with fallback
try:
    import pytesseract
    has_tesseract = True
except ImportError as e:
    has_tesseract = False
    # Dummy implementation
    class DummyTesseract:
        def image_to_string(self, *args, **kwargs):
            return "OCR requires pytesseract. Install with: pip install pytesseract"
    pytesseract = DummyTesseract()

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
    has_transformers = True
    print("✅ Transformers loaded successfully")
except ImportError as e:
    has_transformers = False
    print(f"❌ Transformers not available: {e}")
    
    # Create dummy classes for fallback
    class DummyPipeline:
        def __init__(self, *args, **kwargs):
            pass
        def __call__(self, text):
            return [{"translation_text": f"[Translation unavailable - transformers not installed]"}]
    
    class DummyTokenizer:
        def __init__(self, *args, **kwargs):
            pass
    
    class DummyModel:
        def __init__(self, *args, **kwargs):
            pass
    
    pipeline = DummyPipeline
    AutoTokenizer = DummyTokenizer
    AutoModelForSeq2SeqLM = DummyModel

try:
    from pronunciation_practice import create_pronunciation_practice
    has_pronunciation_practice = True
    print("✅ Enhanced pronunciation practice with AI feedback loaded")
except ImportError as e:
    has_pronunciation_practice = False
    print(f"❌ Pronunciation practice not available: {e}")

# Try importing OpenCV with robust fallback mechanism
try:
    import cv2
except ImportError as e:
    # Create dummy CV2 class to prevent crashes
    class DummyCV2:
        def __init__(self):
            pass
            
        def __getattr__(self, name):
            def dummy_method(*args, **kwargs):
                return None
            return dummy_method
            
        def cvtColor(self, *args, **kwargs):
            return args[0]  # Return the input image unchanged
            
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
            except False:
                return False
    
    # Replace cv2 with our dummy implementation
    cv2 = DummyCV2()

# Try importing gTTS
try:
    from gtts import gTTS
except ImportError as e:
    # Create a dummy gTTS class
    class DummyGTTS:
        def __init__(self, text="", lang="en", slow=False):
            self.text = text
            self.lang = lang
            
        def write_to_fp(self, fp):
            fp.write(b'dummy audio data')
    
    gTTS = DummyGTTS

# Import database module with error handling
try:
    from database import LanguageLearningDB
except ImportError as e:
    # Define a basic LanguageLearningDB class for fallback
    class LanguageLearningDB:
        def __init__(self, db_path):
            self.db_path = db_path
            
        def start_session(self, user_id):
            return None
            
        def end_session(self, session_id, words_studied, words_learned):
            return True

# Import custom audio recorder
try:
    from custom_audio_recorder import audio_recorder
    has_custom_recorder = True
    print("Custom audio recorder imported successfully")
except ImportError as e:
    has_custom_recorder = False
    print(f"Custom audio recorder not available: {e}")

def check_pronunciation_dependencies():
    """Check and report pronunciation practice dependencies"""
    dependencies = {
        'streamlit_webrtc': False,
        'speech_recognition': False,
        'librosa': False,
        'Levenshtein': False,
        'av': False
    }
    
    try:
        import streamlit_webrtc
        dependencies['streamlit_webrtc'] = True
    except ImportError:
        pass
    
    try:
        import speech_recognition
        dependencies['speech_recognition'] = True
    except ImportError:
        pass
    
    try:
        import librosa
        dependencies['librosa'] = True
    except ImportError:
        pass
    
    try:
        import Levenshtein
        dependencies['Levenshtein'] = True
    except ImportError:
        pass
    
    try:
        import av
        dependencies['av'] = True
    except ImportError:
        pass
    
    return dependencies


def draw_detections(image_np, detections):
    """Draw bounding boxes and labels on the image."""
    result_image = image_np.copy()
    
    for detection in detections:
        bbox = detection['bbox']
        left, top, right, bottom = [int(x) for x in bbox]
        label = detection['label']
        confidence = detection['confidence']
        
        # Use different colors for different object types
        color = get_detection_color(label)
        
        # Draw bounding box
        cv2.rectangle(result_image, (left, top), (right, bottom), color, 3)
        
        # Prepare label text
        label_text = f"{label} {confidence:.2f}"
        label_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        
        # Draw background for text
        cv2.rectangle(result_image, 
                     (left, top - label_size[1] - 10), 
                     (left + label_size[0], top), 
                     color, -1)
        
        # Draw text (white or black depending on background)
        text_color = (255, 255, 255) if sum(color) < 400 else (0, 0, 0)
        cv2.putText(result_image, label_text,
                   (left, top - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
    
    return result_image


def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) of two bounding boxes."""
    # box format: [left, top, right, bottom]
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def get_detection_color(label):
    """Get a consistent color for each object type."""
    # Color mapping for different object categories
    color_map = {
        # Electronics - Blue shades
        'cell phone': (255, 100, 100),
        'laptop': (255, 150, 100),
        'tv': (255, 200, 100),
        'mouse': (200, 255, 100),
        'keyboard': (150, 255, 100),
        'remote': (100, 255, 100),
        
        # People - Green shades
        'person': (100, 255, 150),
        
        # Furniture - Purple shades
        'chair': (150, 100, 255),
        'couch': (200, 100, 255),
        'bed': (255, 100, 255),
        
        # Food - Orange/Red shades
        'bottle': (100, 150, 255),
        'cup': (100, 200, 255),
        'bowl': (100, 255, 255),
        
        # Default color
        'default': (0, 255, 0)
    }
    
    return color_map.get(label, color_map['default'])



def show_detection_settings():
    """Show detection settings in the sidebar."""
    with st.sidebar.expander("🎛️ Detection Settings"):
        st.markdown("**Non-Maximum Suppression (NMS)**")
        st.markdown("✅ Enabled - Removes duplicate detections")
        
        # Allow user to adjust IOU threshold
        iou_threshold = st.slider(
            "Overlap Threshold", 
            min_value=0.1, 
            max_value=0.9, 
            value=0.45,
            step=0.05,
            help="Lower values = fewer duplicates, Higher values = more detections"
        )
        
        st.markdown(f"**Current Settings:**")
        st.markdown(f"- Overlap: {iou_threshold:.2f}")
        st.markdown("- Confidence: Set below ⬇️")
        
        return iou_threshold
    
# Helper function to convert AttrDict to a regular dict recursively
def convert_to_dict(obj):
    if isinstance(obj, dict):
        return {key: convert_to_dict(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_dict(item) for item in obj]
    else:
        return obj


# Define object categories for better organization
OBJECT_CATEGORIES = {
    "food": ["banana", "apple", "sandwich", "orange", "broccoli", "carrot", 
             "hot dog", "pizza", "donut", "cake", "bottle", "wine glass", 
             "cup", "fork", "knife", "spoon", "bowl"],
    
    "animals": ["bird", "cat", "dog", "horse", "sheep", "cow", "elephant", 
                "bear", "zebra", "giraffe"],
    
    "vehicles": ["bicycle", "car", "motorcycle", "airplane", "bus", "train", 
                 "truck", "boat"],
    
    "electronics": ["tv", "laptop", "mouse", "remote", "keyboard", "cell phone", 
                   "microwave", "oven", "toaster", "refrigerator"],
    
    "furniture": ["chair", "couch", "potted plant", "bed", "dining table", 
                  "toilet", "bench"],
    
    "personal": ["backpack", "umbrella", "handbag", "tie", "suitcase"],
    
    "sports": ["frisbee", "skis", "snowboard", "sports ball", "kite", 
              "baseball bat", "baseball glove", "skateboard", "surfboard", 
              "tennis racket"],
    
    "household": ["bottle", "wine glass", "cup", "fork", "knife", "spoon", 
                 "bowl", "book", "clock", "vase", "scissors", "teddy bear", 
                 "hair drier", "toothbrush", "sink"]
}

COCO_CLASS_NAMES = {
    1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
    6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
    11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
    16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep',
    21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe',
    27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase',
    34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball',
    38: 'kite', 39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard',
    42: 'surfboard', 43: 'tennis racket', 44: 'bottle', 46: 'wine glass',
    47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl',
    52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli',
    57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut', 61: 'cake',
    62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table',
    70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote',
    76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven',
    80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock',
    86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'
}

# Define question types
QUESTION_TYPES = [
    "translation_to_target",     # English → Target language
    "translation_to_english",    # Target language → English
    "image_recognition",         # Show image, select correct word
    "category_match",            # Match word to correct category
    "sentence_completion",       # Fill in blank in a sentence
    "multiple_choice_category",  # Choose words from same category
    "audio_recognition"          # Hear word, select correct option
]


def get_object_category(label):
    """Get the category for a detected object label - updated for Google Vision."""
    label = label.lower()
    
    # Electronics
    if any(term in label for term in ['phone', 'cell phone', 'mobile', 'laptop', 'computer', 'tv', 'television', 'mouse', 'keyboard', 'remote']):
        return "electronics"
    
    # Food & Drinks  
    elif any(term in label for term in ['bottle', 'cup', 'glass', 'food', 'fruit', 'apple', 'banana', 'orange', 'sandwich', 'pizza']):
        return "food"
    
    # Furniture
    elif any(term in label for term in ['chair', 'couch', 'sofa', 'table', 'desk', 'bed', 'toilet']):
        return "furniture"
    
    # Vehicles
    elif any(term in label for term in ['car', 'bicycle', 'motorcycle', 'bus', 'truck', 'airplane']):
        return "vehicles"
    
    # Animals
    elif any(term in label for term in ['dog', 'cat', 'bird', 'horse', 'animal']):
        return "animals"
    
    # Personal items
    elif any(term in label for term in ['bag', 'backpack', 'suitcase', 'umbrella', 'tie']):
        return "personal"
    
    # Sports
    elif any(term in label for term in ['ball', 'sports', 'football', 'basketball', 'tennis']):
        return "sports"
    
    # Household
    elif any(term in label for term in ['book', 'clock', 'vase', 'plant', 'knife', 'fork', 'spoon', 'bowl']):
        return "household"
    
    else:
        return "other"


def get_image_hash(image):
    """Create a hash of an image for caching purposes."""
    img_byte_arr = io.BytesIO()
    
    # Fix RGBA to JPEG conversion issue
    if image.mode in ('RGBA', 'LA', 'P'):
        # Convert to RGB if image has transparency
        rgb_image = Image.new('RGB', image.size, (255, 255, 255))
        if image.mode == 'P':
            image = image.convert('RGBA')
        rgb_image.paste(image, mask=image.split()[-1] if image.mode in ('RGBA', 'LA') else None)
        image = rgb_image
    
    image.save(img_byte_arr, format='JPEG', quality=70)
    return hashlib.md5(img_byte_arr.getvalue()).hexdigest()

# Function to detect objects in image
def detect_objects(image, confidence_threshold=0.5, iou_threshold=0.45):
    """Main detection function - Google Cloud Vision only."""
    return detect_objects_google_vision(image, confidence_threshold)


# Function to enhance image quality
def enhance_image(image, enhance_type="auto"):
    """Enhance the image to improve object detection."""
    try:
        # Convert PIL image to numpy array
        img_array = np.array(image)
        
        if enhance_type == "auto" or enhance_type == "brightness":
            # Auto-adjust brightness
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            mean_brightness = np.mean(gray)
            
            if mean_brightness < 100:  # Image is too dark
                # Increase brightness
                hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
                h, s, v = cv2.split(hsv)
                
                # Calculate how much to increase brightness (more for darker images)
                brightness_factor = max(1.0, (130 - mean_brightness) / 80)
                v = cv2.add(v, np.array([brightness_factor * 30.0], dtype=np.uint8))
                
                final_hsv = cv2.merge((h, s, v))
                img_array = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB)
            
            elif mean_brightness > 200:  # Image is too bright
                # Decrease brightness
                hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
                h, s, v = cv2.split(hsv)
                
                # Reduce brightness
                v = cv2.subtract(v, np.array([30], dtype=np.uint8))
                
                final_hsv = cv2.merge((h, s, v))
                img_array = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB)
        
        if enhance_type == "auto" or enhance_type == "contrast":
            # Enhance contrast
            lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            
            # Merge the CLAHE enhanced L-channel with the a and b channels
            enhanced_lab = cv2.merge((cl, a, b))
            img_array = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        
        # Convert back to PIL image
        enhanced_image = Image.fromarray(img_array)
        return enhanced_image
    
    except Exception as e:
        error_message(f"Image enhancement error: {e}")
        return image  # Return original image on error

# Function to detect text in image (OCR)
def detect_text_in_image(image):
    """Detect text in image using Google Vision API only."""
    try:
        from google.cloud import vision
        import io
        import json
        import tempfile
        import streamlit as st
        
        # Fix RGBA to JPEG conversion for text detection
        processed_image = image
        if image.mode in ('RGBA', 'LA', 'P'):
            rgb_image = Image.new('RGB', image.size, (255, 255, 255))
            if image.mode == 'P':
                image = image.convert('RGBA')
            rgb_image.paste(image, mask=image.split()[-1] if image.mode in ('RGBA', 'LA') else None)
            processed_image = rgb_image
        
        # Get credentials from Streamlit secrets
        if "gcp_service_account" in st.secrets:
            # Create credentials from Streamlit secrets
            credentials_info = dict(st.secrets["gcp_service_account"])
            
            # Create a temporary file with the credentials
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(credentials_info, f)
                temp_cred_file = f.name
            
            # Set the environment variable to point to the temp file
            import os
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = temp_cred_file
        
        # Initialize the client
        client = vision.ImageAnnotatorClient()
        
        # Convert PIL image to bytes
        img_byte_arr = io.BytesIO()
        processed_image.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()
        
        # Create vision image object
        vision_image = vision.Image(content=img_byte_arr)
        
        # Perform text detection
        response = client.text_detection(image=vision_image)
        texts = response.text_annotations
        
        # Clean up temp file if created
        if "gcp_service_account" in st.secrets and 'temp_cred_file' in locals():
            try:
                import os
                os.unlink(temp_cred_file)
            except:
                pass
        
        if response.error.message:
            raise Exception(f"Google Vision API error: {response.error.message}")
        
        if texts:
            detected_text = texts[0].description.strip()
            print(f"✅ Google Vision detected text: {detected_text}")
            return detected_text
        else:
            print("No text detected in image")
            return "No text found in this image."
            
    except ImportError:
        return "Google Vision API is not installed. Please install google-cloud-vision with compatible protobuf version."
    except Exception as e:
        print(f"❌ Google Vision API error: {e}")
        return f"Text detection failed: {str(e)}."

def display_my_progress():
    """Display user progress with enhanced visuals."""
    try:
        style_title("My Progress")
        
        user = get_authenticated_user()
        if not user:
            st.warning("Please log in to view your progress.")
            return
        
        # Get vocabulary stats
        vocabulary = get_all_vocabulary_direct()
        total_words = len(vocabulary) if vocabulary else 0
        
        # Enhanced stats with better visuals
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 20px; border-radius: 15px; text-align: center; color: white;">
                <h1 style="margin: 0; font-size: 2.5em;">📚</h1>
                <h2 style="margin: 5px 0; color: white;">{}</h2>
                <p style="margin: 0; opacity: 0.9;">Words Learned</p>
            </div>
            """.format(total_words), unsafe_allow_html=True)
        
        with col2:
            current_streak = st.session_state.get('streak_days', 0)
            streak_emoji = "🔥" if current_streak > 0 else "❄️"
            streak_color = "#ff6b6b" if current_streak > 7 else "#4ecdc4"
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, {streak_color} 0%, #45b7d1 100%); 
                        padding: 20px; border-radius: 15px; text-align: center; color: white;">
                <h1 style="margin: 0; font-size: 2.5em;">{streak_emoji}</h1>
                <h2 style="margin: 5px 0; color: white;">{current_streak} days</h2>
                <p style="margin: 0; opacity: 0.9;">Current Streak</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            current_level = st.session_state.get('level', 1)
            level_emoji = "🥇" if current_level >= 10 else "🥈" if current_level >= 5 else "🥉"
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                        padding: 20px; border-radius: 15px; text-align: center; color: white;">
                <h1 style="margin: 0; font-size: 2.5em;">{level_emoji}</h1>
                <h2 style="margin: 5px 0; color: white;">Level {current_level}</h2>
                <p style="margin: 0; opacity: 0.9;">Current Level</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            points = st.session_state.get('points', 0)
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); 
                        padding: 20px; border-radius: 15px; text-align: center; color: #333;">
                <h1 style="margin: 0; font-size: 2.5em;">⭐</h1>
                <h2 style="margin: 5px 0; color: #333;">{points}</h2>
                <p style="margin: 0; opacity: 0.8;">Total Points</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Enhanced Achievements section
        st.markdown("## 🏆 Your Achievement Collection")
        
        achievements = st.session_state.get('achievements', {})
        if achievements:
            # Create achievement cards
            cols = st.columns(3)
            for i, (achievement_id, achievement_data) in enumerate(achievements.items()):
                with cols[i % 3]:
                    if isinstance(achievement_data, dict):
                        title = achievement_data.get('title', achievement_id)
                        description = achievement_data.get('description', 'No description')
                        
                        # Different badge styles for different achievements
                        badge_colors = [
                            "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
                            "linear-gradient(135deg, #f093fb 0%, #f5576c 100%)",
                            "linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)",
                            "linear-gradient(135deg, #43e97b 0%, #38f9d7 100%)",
                            "linear-gradient(135deg, #fa709a 0%, #fee140 100%)"
                        ]
                        badge_color = badge_colors[i % len(badge_colors)]
                        
                        # Achievement emojis
                        achievement_emojis = ["🎯", "🌟", "🚀", "💎", "👑", "🔥", "⚡", "🎨", "🎪", "🎭"]
                        emoji = achievement_emojis[i % len(achievement_emojis)]
                        
                        st.markdown(f"""
                        <div style="background: {badge_color}; 
                                    padding: 15px; border-radius: 12px; text-align: center; 
                                    color: white; margin-bottom: 10px;
                                    box-shadow: 0 4px 8px rgba(0,0,0,0.2);">
                            <h1 style="margin: 0; font-size: 2em;">{emoji}</h1>
                            <h3 style="margin: 5px 0; color: white;">{title}</h3>
                            <p style="margin: 0; opacity: 0.9; font-size: 0.9em;">{description}</p>
                        </div>
                        """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #e0e0e0 0%, #f5f5f5 100%); 
                        padding: 30px; border-radius: 15px; text-align: center;">
                <h1 style="margin: 0; font-size: 3em;">🏆</h1>
                <h3>No achievements yet!</h3>
                <p>Keep learning to unlock amazing badges and achievements!</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Enhanced Vocabulary Tree
        st.markdown("## 🌳 Your Learning Tree")
        
        tree_data = st.session_state.get('vocabulary_tree', {})
        tree_level = tree_data.get('level', 1)
        tree_size = tree_data.get('size', total_words if total_words > 0 else 1)
        
        # Calculate progress for next level
        words_for_next_level = tree_level * 10
        progress = min(tree_size / words_for_next_level, 1.0)
        
        # Visual tree representation
        tree_stages = [
            {"level": 1, "emoji": "🌱", "name": "Seedling", "color": "#8FBC8F"},
            {"level": 3, "emoji": "🌿", "name": "Sprout", "color": "#90EE90"},
            {"level": 5, "emoji": "🌳", "name": "Young Tree", "color": "#32CD32"},
            {"level": 8, "emoji": "🎄", "name": "Mature Tree", "color": "#228B22"},
            {"level": 10, "emoji": "🌲", "name": "Forest Giant", "color": "#006400"}
        ]
        
        current_stage = tree_stages[0]
        for stage in tree_stages:
            if tree_level >= stage["level"]:
                current_stage = stage
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, {current_stage['color']} 0%, #87CEEB 100%); 
                        padding: 25px; border-radius: 15px; text-align: center; color: white;">
                <h1 style="margin: 0; font-size: 4em;">{current_stage['emoji']}</h1>
                <h2 style="margin: 10px 0; color: white;">Level {tree_level}</h2>
                <p style="margin: 0; opacity: 0.9;">{current_stage['name']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"**Tree Growth Progress** (Level {tree_level} → {tree_level + 1})")
            st.progress(progress)
            st.markdown(f"**Words learned:** {tree_size} / {words_for_next_level}")
            
            if progress >= 1.0:
                st.success("🎉 Ready to level up! Keep learning to grow your tree!")
            else:
                remaining = words_for_next_level - tree_size
                st.info(f"📚 Learn {remaining} more words to reach the next level!")
        
        # Category progress with visual elements
        st.markdown("## 📊 Learning Progress by Category")
        
        if vocabulary:
            category_counts = {}
            category_colors = {
                'electronics': '#FF6B6B',
                'food': '#4ECDC4', 
                'sports': '#45B7D1',
                'other': '#96CEB4',
                'text': '#FECA57',
                'manual': '#FF9FF3'
            }
            
            for word in vocabulary:
                category = word.get('category', 'Other')
                category_counts[category] = category_counts.get(category, 0) + 1
            
            if category_counts:
                cols = st.columns(min(len(category_counts), 3))
                for i, (category, count) in enumerate(category_counts.items()):
                    with cols[i % 3]:
                        color = category_colors.get(category.lower(), '#95A5A6')
                        
                        st.markdown(f"""
                        <div style="background: {color}; padding: 15px; border-radius: 10px; 
                                    text-align: center; color: white; margin-bottom: 10px;">
                            <h2 style="margin: 0; color: white;">{count}</h2>
                            <p style="margin: 5px 0; opacity: 0.9;">{category.title()} Words</p>
                        </div>
                        """, unsafe_allow_html=True)
        else:
            st.info("🚀 Start learning to see your progress by category!")
            
    except Exception as e:
        print(f"❌ Error in display_my_progress: {e}")
        st.error("❌ There was an error displaying progress. Please refresh the page.")

# Function to get example sentence
def get_example_sentence(word, target_language):
    """Generate an example sentence using the word via the example generator."""
    # Try to determine category from OBJECT_CATEGORIES
    category = None
    for cat_name, items in OBJECT_CATEGORIES.items():
        if word.lower() in [item.lower() for item in items]:
            category = cat_name
            break
    
    # Call the generator with the category hint
    return example_generator.get_example_sentence(word, target_language, category)
        


# Function to get pronunciation guide
def get_pronunciation_guide(word, language_code):
    """Generate a simple pronunciation guide for the word."""
    try:
        # Map of common sounds in different languages
        pronunciation_maps = {
            "es": {  # Spanish
                'j': 'h', 'll': 'y', 'ñ': 'ny', 'rr': 'rolled r'
            },
            "fr": {  # French
                'eau': 'oh', 'au': 'oh', 'ai': 'eh', 'ou': 'oo', 'u': 'ü', 'r': 'guttural r'
            },
            "de": {  # German
                'sch': 'sh', 'ch': 'kh/sh', 'ei': 'eye', 'ie': 'ee', 'ä': 'eh', 'ö': 'er', 'ü': 'ü'
            },
            "it": {  # Italian
                'gn': 'ny', 'gli': 'ly', 'ch': 'k', 'c+e/i': 'ch', 'c+a/o/u': 'k'
            }
        }
        
        # Get pronunciation map for this language
        sound_map = pronunciation_maps.get(language_code, {})
        
        # Build pronunciation guide
        notes = []
        
        for sound, pronunciation in sound_map.items():
            if sound in word.lower():
                notes.append(f"'{sound}' sounds like '{pronunciation}'")
        
        return notes
    except Exception as e:
        return [f"Pronunciation guide unavailable: {str(e)}"]

def display_vocabulary_image(image_path, word_original):
    """Display image with proper handling for all storage types."""
    if not image_path:
        return False
    
    try:
        # Use cropped version if available
        display_image_path = get_cropped_image_path(image_path)
        
        print(f"🔍 Attempting to display image: {display_image_path}")
        
        # Check if it's a new private Supabase storage path
        if display_image_path.startswith('vocabulary-images/'):
            print(f"🔍 Supabase storage path detected")
            signed_url = get_signed_image_url(display_image_path)
            if signed_url:
                st.image(signed_url, caption=f"📷 {word_original} - Cropped from detection")
                st.markdown("*🎯 Showing focused crop of detected object*")
                st.markdown("*🔒 Private image - only visible to you*")
                return True
            else:
                print(f"❌ Failed to get signed URL for: {display_image_path}")
                
        # Check if it's a legacy public URL (starts with http)
        elif display_image_path.startswith('http'):
            print(f"🔍 Public URL detected: {display_image_path}")
            st.image(display_image_path, caption=f"📷 {word_original}")
            if "_cropped" in display_image_path:
                st.markdown("*🎯 Showing focused crop of detected object*")
            else:
                st.markdown("*📸 Showing full original image*")
            return True
            
        # Check if it's a local file path
        elif os.path.exists(display_image_path):
            print(f"🔍 Local file detected: {display_image_path}")
            image = Image.open(display_image_path)
            st.image(image, caption=f"📷 {word_original}")
            
            if "_cropped.jpg" in display_image_path:
                st.markdown("*🎯 Showing focused crop of detected object*")
            else:
                st.markdown("*📸 Showing full original image*")
            return True
            
        # If none of the above work, it might be a legacy path format
        else:
            print(f"🔍 Trying to find legacy image: {display_image_path}")
            
            # Try different legacy path formats
            possible_paths = [
                display_image_path,
                f"object_images/{display_image_path}",
                display_image_path.replace("vocabulary-images/", "object_images/"),
                display_image_path.replace("vocabulary-images/", "")
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    print(f"✅ Found legacy image at: {path}")
                    image = Image.open(path)
                    st.image(image, caption=f"📷 {word_original}")
                    st.markdown("*📸 Legacy image*")
                    return True
        
        print(f"❌ Could not display image: {display_image_path}")
        return False
        
    except Exception as e:
        print(f"❌ Error displaying image: {e}")
        return False
    
def add_vocabulary_direct(word_original, word_translated, language_translated, category=None, image_path=None):
    """Add vocabulary using Supabase - Production version with enhanced logging."""
    user = get_authenticated_user()

    if vocab_id:
        # Sync user data after successful vocabulary addition
        sync_user_data_to_supabase()

    if not user:
        error_message("Please log in to save vocabulary.")
        return None
    
    try:
        print(f"🔄 Attempting to save: {word_original} → {word_translated}")
        print(f"📁 Image path: {image_path}")
        print(f"👤 User ID: {user.get('id')}")
        
        db = get_user_database()
        vocab_id = db.add_vocabulary(word_original, word_translated, language_translated, category, image_path)
        
        if vocab_id == 'duplicate':
            print(f"⚠️ Duplicate word detected: {word_original}")
            warning_message(f"'{word_original}' → '{word_translated}' is already in your vocabulary!")
            return 'duplicate'
        elif vocab_id:
            print(f"✅ Successfully saved to Supabase with ID: {vocab_id}")
            
            # Verify the save by checking if we can retrieve it
            verification = verify_last_save_operation()
            if isinstance(verification, dict):
                print(f"✅ Verification successful: {verification['word']}")
            else:
                print(f"⚠️ Verification failed: {verification}")
            
            try:
                gamification.check_achievements(
                    "word_learned",
                    word=word_original,
                    category=category,
                    language=language_translated
                )
            except Exception as e:
                print(f"⚠️ Gamification error (non-critical): {e}")
            
            return vocab_id
        else:
            print(f"❌ Save failed - no vocab_id returned")
            error_message("Unable to save vocabulary. Please try again.")
            return None
            
    except Exception as e:
        print(f"❌ Save error: {e}")
        error_message("There was a problem saving your vocabulary. Please check your connection and try again.")
        return None
    

    
def create_session_direct():
    """Create session using Supabase - Production version."""
    user = get_authenticated_user()
    if not user:
        return None
    
    try:
        db = get_user_database()
        session_id = db.start_session()
        return session_id
    except Exception as e:
        error_message("Unable to start learning session. Please try again.")
        return None

# Function to get session statistics
def get_session_stats_direct(days=30):
    """Get session statistics from Supabase."""
    user = get_authenticated_user()
    if not user:
        return {}
    
    try:
        import requests
        from datetime import datetime, timedelta
        
        db = get_user_database()
        headers = db.get_headers()
        user_id = db.get_user_id()
        
        # Calculate date filter
        start_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        # Get sessions from Supabase
        sessions_url = f'{db.supabase_url}/rest/v1/sessions?user_id=eq.{user_id}&start_time=gte.{start_date}'
        sessions_response = requests.get(sessions_url, headers=headers)
        
        if sessions_response.status_code == 200:
            sessions = sessions_response.json()
            
            total_sessions = len(sessions)
            total_words_studied = sum(s.get('words_studied', 0) for s in sessions)
            total_words_learned = sum(s.get('words_learned', 0) for s in sessions)
            
            avg_words_per_session = total_words_studied / total_sessions if total_sessions > 0 else 0
            
            # Calculate session durations
            total_minutes = 0
            session_count = 0
            
            for session in sessions:
                if session.get('end_time') and session.get('start_time'):
                    try:
                        start = datetime.fromisoformat(session['start_time'].replace('Z', '+00:00'))
                        end = datetime.fromisoformat(session['end_time'].replace('Z', '+00:00'))
                        duration = (end - start).total_seconds() / 60
                        total_minutes += duration
                        session_count += 1
                    except:
                        pass
            
            avg_session_minutes = total_minutes / session_count if session_count > 0 else 0
            
            return {
                'total_sessions': total_sessions,
                'total_words_studied': total_words_studied,
                'total_words_learned': total_words_learned,
                'avg_words_per_session': avg_words_per_session,
                'avg_session_minutes': avg_session_minutes
            }
        else:
            print(f"Error getting sessions: {sessions_response.text}")
            return {}
            
    except Exception as e:
        print(f"Error getting session stats: {e}")
        return {}

# Function to check if database is properly set up
def check_database_setup():
    """Check if the database is properly set up and try to fix if needed."""
    try:
        conn = sqlite3.connect("language_learning.db")
        cursor = conn.cursor()
        
        # Check if tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [table[0] for table in cursor.fetchall()]
        
        required_tables = ['vocabulary', 'user_progress', 'sessions', 'camera_translations']
        missing_tables = [table for table in required_tables if table not in tables]
        
        if missing_tables:
            # Create missing tables with ORIGINAL schema (no user_id in sessions)
            if 'vocabulary' in missing_tables:
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS vocabulary (
                    id INTEGER PRIMARY KEY,
                    word_original TEXT NOT NULL,
                    word_translated TEXT NOT NULL,
                    language_translated TEXT NOT NULL,
                    category TEXT,
                    image_path TEXT,
                    date_added TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    source TEXT DEFAULT 'manual'
                );
                ''')
            
            if 'user_progress' in missing_tables:
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_progress (
                    id INTEGER PRIMARY KEY,
                    vocabulary_id INTEGER,
                    review_count INTEGER DEFAULT 0,
                    correct_count INTEGER DEFAULT 0,
                    last_reviewed TIMESTAMP,
                    proficiency_level INTEGER DEFAULT 0,
                    FOREIGN KEY (vocabulary_id) REFERENCES vocabulary (id)
                );
                ''')
            
            if 'sessions' in missing_tables:
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY,
                    start_time TIMESTAMP,
                    end_time TIMESTAMP,
                    words_studied INTEGER DEFAULT 0,
                    words_learned INTEGER DEFAULT 0
                );
                ''')
            
            if 'camera_translations' in missing_tables:
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS camera_translations (
                    id INTEGER PRIMARY KEY,
                    image_path TEXT,
                    detected_text TEXT,
                    translated_text TEXT,
                    source_language TEXT,
                    target_language TEXT,
                    date_captured TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_saved_to_vocabulary BOOLEAN DEFAULT 0
                );
                ''')
            
            conn.commit()
        
        conn.close()
        return True
    except Exception as e:
        error_message(f"Database error: {e}")
        return False

def get_database_path():
    """Get the correct database path for the environment."""
    if os.environ.get('RENDER'):
        # On Render, use /opt/render/project/src/ for persistence
        db_path = "/opt/render/project/src/language_learning.db"
    else:
        # Local development
        db_path = "language_learning.db"
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    return db_path
  
def prepare_vocabulary_for_diverse_questions(vocabulary, languages):
    """Enhance vocabulary data to support diverse question types."""
    total_words = len(vocabulary)
    words_with_categories = 0
    words_with_images = 0
    words_with_examples = 0
    
    # Count and prepare vocabulary for diverse questions
    for word in vocabulary:
        # Check/count category
        if word.get('category') and word['category'] not in ['other', 'manual', '']:
            words_with_categories += 1
        
        # Check/count image
        if word.get('image_path') and os.path.exists(word.get('image_path', '')):
            words_with_images += 1
        
        # Test for example sentence
        try:
            example = get_example_sentence(word.get('word_original', ''), word.get('language_translated', 'en'))
            if example and example.get('translated'):
                words_with_examples += 1
        except:
            pass
    
    return vocabulary

if 'db_checked' not in st.session_state:
    st.session_state.db_checked = check_database_setup()

# Initialize database
@st.cache_resource
def get_database():
    return LanguageLearningDB("language_learning.db")

db = get_database()


# Initialize session state for manual mode
if 'manual_mode' not in st.session_state:
    st.session_state.manual_mode = False
if 'manual_label' not in st.session_state:
    st.session_state.manual_label = ""

# Initialize session state variables
if 'target_language' not in st.session_state:
    st.session_state.target_language = "es"  # Default to Spanish
if 'session_id' not in st.session_state:
    st.session_state.session_id = None
if 'words_studied' not in st.session_state:
    st.session_state.words_studied = 0
if 'words_learned' not in st.session_state:
    st.session_state.words_learned = 0
if 'quiz_score' not in st.session_state:
    st.session_state.quiz_score = 0
if 'quiz_total' not in st.session_state:
    st.session_state.quiz_total = 0
if 'current_quiz_word' not in st.session_state:
    st.session_state.current_quiz_word = None
if 'quiz_options' not in st.session_state:
    st.session_state.quiz_options = []
if 'answered' not in st.session_state:
    st.session_state.answered = False
if 'detection_checkboxes' not in st.session_state:
    st.session_state.detection_checkboxes = {}
# Ensure session state variables are initialized first
if 'level' not in st.session_state:
    st.session_state.level = 1
if 'points' not in st.session_state:
    st.session_state.points = 0
if 'streak_days' not in st.session_state:
    st.session_state.streak_days = 0
if 'daily_challenges' not in st.session_state:
    st.session_state.daily_challenges = []
if 'word_of_the_day' not in st.session_state:
    st.session_state.word_of_the_day = None
# Add these initializations with your other session state initializations
if 'audio_data' not in st.session_state:
    st.session_state.audio_data = None
if 'audio_data_received' not in st.session_state:
    st.session_state.audio_data_received = False
if 'current_recording_word' not in st.session_state:
    st.session_state.current_recording_word = None
st.session_state.use_vision_api = True
# Always force it to be True
st.session_state.use_vision_api = True
if 'app_mode' not in st.session_state:
    st.session_state.app_mode = "Camera Mode"
# Add flag to track save button state
if 'save_button_clicked' not in st.session_state:
    st.session_state.save_button_clicked = False
if 'words_just_saved' not in st.session_state:
    st.session_state.words_just_saved = False
if 'saved_count' not in st.session_state:
    st.session_state.saved_count = 0
if 'saved_items' not in st.session_state:
    st.session_state.saved_items = []
if 'faster_rcnn_model_loaded' not in st.session_state:
    st.session_state.faster_rcnn_model_loaded = False


def get_gamification():
    # Initialize GamificationSystem without the translate function for now
    return GamificationSystem()

def get_user_scoped_gamification():
    """Get gamification instance with Supabase data."""
    if 'gamification' not in st.session_state:
        db = get_user_database()  # This gets your SupabaseDB instance
        st.session_state.gamification = GamificationSystem(db_instance=db)
    return st.session_state.gamification

# Initialize user-scoped gamification
gamification = get_user_scoped_gamification()


# Function to translate text
class FreeTranslationService:
    def __init__(self):
        self.translation_cache = {}
        self.last_request_time = 0
        self.rate_limit_delay = 1.0  # seconds between requests
        
    def translate_text(self, text, target_language, source_language='en'):
        """
        Translate text using multiple free services with fallbacks
        """
        # Check cache first
        cache_key = f"{text}_{source_language}_{target_language}"
        if cache_key in self.translation_cache:
            return self.translation_cache[cache_key]
        
        # Rate limiting
        current_time = time.time()
        if current_time - self.last_request_time < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - (current_time - self.last_request_time))
        
        translation = None
        
        # Method 1: Deep Translator (Free Google Translate web interface)
        try:
            translator = GoogleTranslator(source=source_language, target=target_language)
            translation = translator.translate(text)
            if translation and translation != text:
                self.translation_cache[cache_key] = translation
                self.last_request_time = time.time()
                return translation
        except Exception as e:
            print(f"Deep Translator failed: {e}")
        
        # Method 2: MyMemory Translation API (Free tier: 10,000 chars/day)
        try:
            translation = self._translate_with_mymemory(text, source_language, target_language)
            if translation:
                self.translation_cache[cache_key] = translation
                self.last_request_time = time.time()
                return translation
        except Exception as e:
            print(f"MyMemory failed: {e}")
        
        # Method 3: LibreTranslate (if you have a server)
        try:
            translation = self._translate_with_libretranslate(text, source_language, target_language)
            if translation:
                self.translation_cache[cache_key] = translation
                self.last_request_time = time.time()
                return translation
        except Exception as e:
            print(f"LibreTranslate failed: {e}")
        
        # Method 4: Hugging Face models (for specific language pairs)
        try:
            translation = self._translate_with_huggingface(text, source_language, target_language)
            if translation:
                self.translation_cache[cache_key] = translation
                self.last_request_time = time.time()
                return translation
        except Exception as e:
            print(f"Hugging Face translation failed: {e}")
        
        # Fallback: Return formatted message
        return f"[Translation to {target_language} unavailable]"
    
    def _translate_with_mymemory(self, text, source_lang, target_lang):
        """MyMemory Translation API - Free tier"""
        url = "https://api.mymemory.translated.net/get"
        params = {
            'q': text,
            'langpair': f"{source_lang}|{target_lang}"
        }
        
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get('responseStatus') == 200:
                return data['responseData']['translatedText']
        return None
    
    def _translate_with_libretranslate(self, text, source_lang, target_lang):
        """LibreTranslate - Free self-hosted option"""
        # You can use the free public instance (limited) or host your own
        url = "https://libretranslate.de/translate"  # Public instance
        
        data = {
            'q': text,
            'source': source_lang,
            'target': target_lang,
            'format': 'text'
        }
        
        response = requests.post(url, data=data, timeout=10)
        if response.status_code == 200:
            result = response.json()
            return result.get('translatedText')
        return None
    
    def _translate_with_huggingface(self, text, source_lang, target_lang):
        """Hugging Face translation models - Completely free"""
        try:
            # Check if transformers is available
            if not has_transformers:
                print("Transformers not available, skipping Hugging Face translation")
                return None
                
            # Map language codes to model names (add more as needed)
            model_map = {
                ('en', 'es'): 'Helsinki-NLP/opus-mt-en-es',
                ('en', 'fr'): 'Helsinki-NLP/opus-mt-en-fr',
                ('en', 'de'): 'Helsinki-NLP/opus-mt-en-de',
                ('en', 'it'): 'Helsinki-NLP/opus-mt-en-it',
                ('en', 'pt'): 'Helsinki-NLP/opus-mt-en-pt',
                ('en', 'ru'): 'Helsinki-NLP/opus-mt-en-ru',
                # Add reverse translations
                ('es', 'en'): 'Helsinki-NLP/opus-mt-es-en',
                ('fr', 'en'): 'Helsinki-NLP/opus-mt-fr-en',
                ('de', 'en'): 'Helsinki-NLP/opus-mt-de-en',
            }
            
            model_name = model_map.get((source_lang, target_lang))
            if not model_name:
                return None
            
            # Load model and tokenizer
            translator = pipeline(
                "translation", 
                model=model_name,
                return_all_scores=False,
                max_length=512
            )
            
            result = translator(text)
            return result[0]['translation_text']
            
        except Exception as e:
            print(f"Hugging Face model error: {e}")
            return None

# Initialize the service
free_translator = FreeTranslationService()

# Replace your translate_text function with this:
def translate_text(text, target_language, source_language='en'):
    return free_translator.translate_text(text, target_language, source_language)


# Connect the translation function to gamification system after both are initialized
gamification.set_translate_func(translate_text)

@st.cache_resource
def get_example_generator():
    """Initialize and cache the example sentence generator."""
    return ExampleSentenceGenerator(translate_func=translate_text, debug=True)

example_generator = get_example_generator()

# Function for text-to-speech
def text_to_speech(text, lang):
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        mp3_fp = BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        audio_bytes = mp3_fp.read()
        return audio_bytes
    except Exception as e:
        error_message(f"Text-to-speech error: {e}")
        return None

# Function to generate HTML for audio playback
def get_audio_html(audio_bytes):
    """Generate HTML for audio playback without autoplay."""
    audio_base64 = base64.b64encode(audio_bytes).decode()
    # Remove the autoplay attribute - only keep controls
    audio_tag = f'<audio src="data:audio/mp3;base64,{audio_base64}" controls></audio>'
    return audio_tag


# Function to start or end a learning session
def manage_session(action):
    """Session management with Supabase."""
    try:
        user = get_authenticated_user()
        if not user:
            error_message("No authenticated user found")
            return False
            
        if action == "start":
            try:
                session_id = create_session_direct()
                
                if session_id:
                    st.session_state.session_id = session_id
                    st.session_state.words_studied = 0
                    st.session_state.words_learned = 0
                    success_message("Started new learning session!")
                    return True
                else:
                    error_message("Failed to create session.")
                    return False
                    
            except Exception as e:
                error_message(f"Error starting session: {str(e)}")
                return False
                
        elif action == "end" and st.session_state.session_id:
            try:
                db = get_user_database()
                success = db.end_session(
                    st.session_state.session_id, 
                    st.session_state.words_studied, 
                    st.session_state.words_learned
                )
                
                if success:
                    success_message(f"Session completed! Words studied: {st.session_state.words_studied}, Words learned: {st.session_state.words_learned}")
                    
                    st.session_state.session_id = None
                    st.session_state.words_studied = 0
                    st.session_state.words_learned = 0
                    
                    return True
                else:
                    error_message("Failed to end session properly")
                    return False
                    
            except Exception as e:
                error_message(f"Error ending session: {str(e)}")
                return False
        
        return False
        
    except Exception as e:
        error_message(f"Session management error: {str(e)}")
        return False
    
def save_image_to_supabase(image, label, detection_bbox=None):
    """Save image to private Supabase Storage with proper authentication."""
    try:
        import requests
        import io
        import uuid
        from datetime import datetime
        import numpy as np
        from PIL import Image as PILImage
        
        user = get_authenticated_user()
        if not user:
            print("❌ No authenticated user for image upload")
            return None
        
        user_id = user.get('id')
        auth_token = user.get('auth_token')
        
        if not auth_token:
            print("❌ No auth token available for image upload")
            return None
        
        print(f"🔄 Starting Supabase upload for user: {user_id}")
        
        # Process image (crop if bbox provided)
        processed_image = image
        if detection_bbox:
            left, top, right, bottom = [int(x) for x in detection_bbox]
            img_array = np.array(image)
            
            # Add padding
            height, width = img_array.shape[:2]
            obj_width = right - left
            obj_height = bottom - top
            
            padding_x = max(10, int(obj_width * 0.1))
            padding_y = max(10, int(obj_height * 0.1))
            
            crop_left = max(0, left - padding_x)
            crop_top = max(0, top - padding_y)
            crop_right = min(width, right + padding_x)
            crop_bottom = min(height, bottom + padding_y)
            
            cropped_img = img_array[crop_top:crop_bottom, crop_left:crop_right]
            processed_image = PILImage.fromarray(cropped_img)
            print(f"🎯 Image cropped to: {cropped_img.shape}")
        
        # Fix RGBA to JPEG conversion
        if processed_image.mode in ('RGBA', 'LA', 'P'):
            rgb_image = PILImage.new('RGB', processed_image.size, (255, 255, 255))
            if processed_image.mode == 'P':
                processed_image = processed_image.convert('RGBA')
            rgb_image.paste(processed_image, mask=processed_image.split()[-1] if processed_image.mode in ('RGBA', 'LA') else None)
            processed_image = rgb_image
            print(f"🔧 Converted {processed_image.mode} to RGB")
        
        # Create a unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        filename = f"{user_id}/{label}_{timestamp}_{unique_id}.jpg"
        
        # Convert image to bytes
        img_bytes = io.BytesIO()
        processed_image.save(img_bytes, format='JPEG', quality=85)
        img_bytes.seek(0)
        
        # Upload to Supabase Storage using the correct endpoint
        supabase_url = "https://csszlzpsfwmsezursivk.supabase.co"
        
        # Use the standard file upload endpoint
        upload_url = f"{supabase_url}/storage/v1/object/vocabulary-images/{filename}"
        
        headers = {
            'Authorization': f'Bearer {auth_token}',
            'Content-Type': 'image/jpeg',
            'x-upsert': 'true'  # Allow overwrite if exists
        }
        
        print(f"🔄 Uploading to: {upload_url}")
        print(f"📋 Headers: {headers}")
        
        response = requests.post(
            upload_url,
            headers=headers,
            data=img_bytes.getvalue()
        )
        
        print(f"📋 Upload response status: {response.status_code}")
        print(f"📋 Upload response: {response.text}")
        
        if response.status_code in [200, 201]:
            # Return the storage path that we'll use for retrieval
            storage_path = f"vocabulary-images/{filename}"
            print(f"✅ Image uploaded successfully to: {storage_path}")
            return storage_path
        else:
            print(f"❌ Failed to upload image: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Error uploading image to Supabase: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_supabase_storage_permissions():
    """Test Supabase Storage permissions and setup - FIXED VERSION."""
    try:
        import requests
        
        user = get_authenticated_user()
        if not user:
            return "❌ No authenticated user"
        
        supabase_url = "https://csszlzpsfwmsezursivk.supabase.co"
        supabase_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImNzc3psenBzZndtc2V6dXJzaXZrIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTA1Mjg1MjEsImV4cCI6MjA2NjEwNDUyMX0.gIi0Q_pifYpXeM1r8kWlgTO1LD8bc91lQ3suH8OWDKI"
        
        headers = {
            'apikey': supabase_key,
            'Authorization': f'Bearer {supabase_key}',
        }
        
        results = []
        
        # Test 1: List buckets
        buckets_url = f"{supabase_url}/storage/v1/bucket"
        response = requests.get(buckets_url, headers=headers, timeout=10)
        
        results.append(f"Buckets test: {response.status_code}")
        if response.status_code == 200:
            buckets = response.json()
            bucket_names = [b.get('name', 'unknown') for b in buckets]
            results.append(f"Available buckets: {bucket_names}")
            
            # Check if vocabulary-images bucket exists
            if 'vocabulary-images' in bucket_names:
                results.append("✅ vocabulary-images bucket exists")
            else:
                results.append("❌ vocabulary-images bucket NOT found")
                results.append("🔧 Create bucket in Supabase Dashboard: Storage → Create bucket → name: 'vocabulary-images'")
        
        # Test 2: Try to list files in vocabulary-images (FIXED)
        list_url = f"{supabase_url}/storage/v1/object/list/vocabulary-images"
        list_payload = {'prefix': ''}  # FIX: Add empty prefix to list all files
        response = requests.post(list_url, headers=headers, json=list_payload, timeout=10)
        
        results.append(f"List files test: {response.status_code}")
        if response.status_code == 200:
            files = response.json()
            results.append(f"Files in bucket: {len(files)}")
            if files:
                results.append(f"Sample files: {[f.get('name', 'unknown')[:50] for f in files[:3]]}")
        else:
            results.append(f"List files error: {response.text}")
        
        # Test 3: Check bucket policies (FIXED)
        policy_url = f"{supabase_url}/storage/v1/bucket/vocabulary-images"
        response = requests.get(policy_url, headers=headers, timeout=10)
        
        results.append(f"Bucket info test: {response.status_code}")
        if response.status_code == 200:
            bucket_info = response.json()
            results.append(f"Bucket public: {bucket_info.get('public', 'unknown')}")
            results.append(f"Bucket allowed mime types: {bucket_info.get('allowed_mime_types', 'any')}")
        elif response.status_code == 404:
            results.append("❌ Bucket not found - create it in Supabase Dashboard")
        else:
            results.append(f"Bucket info error: {response.text}")
        
        return "\n".join(results)
        
    except Exception as e:
        return f"❌ Storage test error: {e}"


def try_alternative_upload(supabase_url, supabase_key, filename, image_data):
    """Try alternative upload method with service role."""
    try:
        import requests
        
        # Use service role for upload
        headers = {
            'apikey': supabase_key,
            'Authorization': f'Bearer {supabase_key}',
            'Content-Type': 'image/jpeg',
        }
        
        upload_url = f"{supabase_url}/storage/v1/object/vocabulary-images/{filename}"
        
        response = requests.post(
            upload_url,
            headers=headers,
            data=image_data,
            timeout=30
        )
        
        if response.status_code in [200, 201]:
            storage_path = f"vocabulary-images/{filename}"
            print(f"✅ Alternative upload successful: {storage_path}")
            return storage_path
        else:
            print(f"❌ Alternative upload failed: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Alternative upload error: {e}")
        return None

# Update the save_image function call in your vocabulary saving:
def save_image(image, label, detection_bbox=None):
    """Save image with cropping support - tries Supabase first, allows graceful fallback."""
    try:
        # Fix RGBA conversion first
        processed_image = image
        if image.mode in ('RGBA', 'LA', 'P'):
            rgb_image = Image.new('RGB', image.size, (255, 255, 255))
            if image.mode == 'P':
                image = image.convert('RGBA')
            rgb_image.paste(image, mask=image.split()[-1] if image.mode in ('RGBA', 'LA') else None)
            processed_image = rgb_image
        
        # Try Supabase Storage first
        supabase_path = save_image_to_supabase(processed_image, label, detection_bbox)
        if supabase_path:
            print(f"✅ Image saved to Supabase: {supabase_path}")
            return supabase_path
        
        # Fallback to local storage
        print(f"⚠️ Supabase save failed, trying local storage for {label}")
        
        import numpy as np
        import cv2
        import os
        import time
        
        img_array = np.array(processed_image)
        os.makedirs("object_images", exist_ok=True)
        
        timestamp = int(time.time())
        
        # Save original image
        original_filename = f"object_images/{label}_{timestamp}_original.jpg"
        img_cv_original = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        cv2.imwrite(original_filename, img_cv_original)
        
        # Save cropped image if bbox provided
        if detection_bbox:
            left, top, right, bottom = [int(x) for x in detection_bbox]
            height, width = img_array.shape[:2]
            
            # Add padding
            obj_width = right - left
            obj_height = bottom - top
            padding_x = max(10, int(obj_width * 0.1))
            padding_y = max(10, int(obj_height * 0.1))
            
            crop_left = max(0, left - padding_x)
            crop_top = max(0, top - padding_y)
            crop_right = min(width, right + padding_x)
            crop_bottom = min(height, bottom + padding_y)
            
            # Crop the image
            cropped_img = img_array[crop_top:crop_bottom, crop_left:crop_right]
            
            # Save cropped image
            cropped_filename = f"object_images/{label}_{timestamp}_cropped.jpg"
            img_cv_cropped = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(cropped_filename, img_cv_cropped)
            
            print(f"✅ Local images saved: {original_filename} and {cropped_filename}")
            return cropped_filename  # Return cropped version as primary
        else:
            print(f"✅ Local image saved: {original_filename}")
            return original_filename
            
    except Exception as e:
        print(f"❌ All image save methods failed for {label}: {e}")
        # Return None but don't raise exception - allow vocabulary saving to continue
        return None

def get_signed_image_url(storage_path, expires_in=3600):
    """Get a signed URL for private image access."""
    try:
        import requests
        
        user = get_authenticated_user()
        if not user:
            print("❌ No authenticated user for signed URL")
            return None
        
        auth_token = user.get('auth_token')
        if not auth_token:
            print("❌ No auth token for signed URL")
            return None
        
        supabase_url = "https://csszlzpsfwmsezursivk.supabase.co"
        
        headers = {
            'Authorization': f'Bearer {auth_token}',
            'Content-Type': 'application/json',
        }
        
        # Remove 'vocabulary-images/' prefix if present for the API call
        clean_path = storage_path.replace('vocabulary-images/', '') if storage_path.startswith('vocabulary-images/') else storage_path
        
        # Create signed URL for private access
        signed_url_endpoint = f"{supabase_url}/storage/v1/object/sign/vocabulary-images/{clean_path}"
        
        data = {
            'expiresIn': expires_in
        }
        
        print(f"🔄 Requesting signed URL from: {signed_url_endpoint}")
        
        response = requests.post(
            signed_url_endpoint,
            headers=headers,
            json=data
        )
        
        print(f"📋 Signed URL response status: {response.status_code}")
        print(f"📋 Signed URL response: {response.text}")
        
        if response.status_code == 200:
            result = response.json()
            signed_token = result.get('signedURL')
            if signed_token:
                # Return full signed URL
                full_signed_url = f"{supabase_url}/storage/v1{signed_token}"
                print(f"✅ Generated signed URL: {full_signed_url}")
                return full_signed_url
        
        print(f"❌ Failed to get signed URL: {response.status_code} - {response.text}")
        return None
        
    except Exception as e:
        print(f"❌ Error getting signed URL: {e}")
        return None

def debug_supabase_connection():
    """Debug function to verify Supabase connection and data."""
    try:
        import requests
        
        user = get_authenticated_user()
        if not user:
            return "❌ No authenticated user"
        
        db = get_user_database()
        headers = db.get_headers()
        user_id = db.get_user_id()
        
        # Test 1: Check if we can connect to Supabase
        test_url = f'{db.supabase_url}/rest/v1/'
        test_response = requests.get(test_url, headers=headers, timeout=10)
        
        connection_status = "✅ Connected" if test_response.status_code == 200 else f"❌ Connection failed: {test_response.status_code}"
        
        # Test 2: Get vocabulary count directly from Supabase
        vocab_url = f'{db.supabase_url}/rest/v1/vocabulary?user_id=eq.{user_id}&select=count'
        vocab_response = requests.get(vocab_url, headers=headers, timeout=10)
        
        if vocab_response.status_code == 200:
            vocab_count = len(vocab_response.json())
        else:
            vocab_count = f"Error: {vocab_response.status_code}"
        
        # Test 3: Get latest vocabulary entries
        latest_url = f'{db.supabase_url}/rest/v1/vocabulary?user_id=eq.{user_id}&order=date_added.desc&limit=5'
        latest_response = requests.get(latest_url, headers=headers, timeout=10)
        
        latest_words = []
        if latest_response.status_code == 200:
            latest_data = latest_response.json()
            latest_words = [f"{item['word_original']} → {item['word_translated']}" for item in latest_data]
        
        # Test 4: Test image storage (FIXED)
        storage_url = f'{db.supabase_url}/storage/v1/object/list/vocabulary-images'
        storage_payload = {'prefix': f'{user_id}/'}  # FIX: Add required prefix
        storage_response = requests.post(
            storage_url,
            headers=headers,
            json=storage_payload,
            timeout=10
        )
        
        image_count = 0
        if storage_response.status_code == 200:
            image_count = len(storage_response.json())
        else:
            print(f"Storage list error: {storage_response.status_code} - {storage_response.text}")
        
        return {
            'connection': connection_status,
            'user_id': user_id,
            'vocab_count': vocab_count,
            'latest_words': latest_words,
            'image_count': image_count,
            'auth_token_exists': bool(user.get('auth_token'))
        }
        
    except Exception as e:
        return f"❌ Debug error: {e}"
    
def debug_user_info():
    """Debug function to check user info format."""
    user = st.session_state.get('user')
    if user:
        user_id = user.get('id')
        print(f"🔍 User ID: {user_id}")
        print(f"🔍 User ID type: {type(user_id)}")
        
        if isinstance(user_id, str):
            try:
                import uuid
                uuid.UUID(user_id)
                print(f"✅ Valid UUID format")
            except ValueError:
                print(f"❌ Invalid UUID format")
        
        # Print full user object structure
        print(f"🔍 Full user object: {user}")
    else:
        print("❌ No user in session state")

debug_user_info()

def verify_last_save_operation():
    """Verify the last vocabulary save operation."""
    try:
        user = get_authenticated_user()
        if not user:
            return "No user authenticated"
        
        db = get_user_database()
        headers = db.get_headers()
        user_id = db.get_user_id()
        
        # Get the most recent vocabulary entry
        url = f'{db.supabase_url}/rest/v1/vocabulary?user_id=eq.{user_id}&order=date_added.desc&limit=1'
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data:
                latest = data[0]
                return {
                    'word': f"{latest['word_original']} → {latest['word_translated']}",
                    'language': latest['language_translated'],
                    'category': latest.get('category', 'No category'),
                    'date_added': latest['date_added'],
                    'has_image': bool(latest.get('image_path')),
                    'image_path': latest.get('image_path', 'No image')
                }
            else:
                return "No vocabulary found in database"
        else:
            return f"Database query failed: {response.status_code}"
            
    except Exception as e:
        return f"Verification error: {e}"
    
def debug_storage_status():
    """Debug function to check Supabase Storage status."""
    try:
        import requests
        
        user = get_authenticated_user()
        if not user:
            return "No authenticated user"
        
        supabase_url = "https://csszlzpsfwmsezursivk.supabase.co"
        supabase_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImNzc3psenBzZndtc2V6dXJzaXZrIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTA1Mjg1MjEsImV4cCI6MjA2NjEwNDUyMX0.gIi0Q_pifYpXeM1r8kWlgTO1LD8bc91lQ3suH8OWDKI"
        
        headers = {
            'apikey': supabase_key,
            'Authorization': f'Bearer {supabase_key}',
        }
        
        # List files in bucket
        list_url = f"{supabase_url}/storage/v1/object/list/vocabulary-images"
        
        response = requests.post(
            list_url,
            headers=headers,
            json={},
            timeout=10
        )
        
        if response.status_code == 200:
            files = response.json()
            return f"Storage Status: {len(files)} files found"
        else:
            return f"Storage Error: {response.status_code} - {response.text}"
            
    except Exception as e:
        return f"Debug Error: {e}"
    
def display_vocabulary_image(image_path, word_original):
    """Display image with proper private access."""
    if not image_path:
        return False
    
    try:
        print(f"🔍 Attempting to display image: {image_path}")
        
        # Check if it's a Supabase storage path
        if image_path.startswith('vocabulary-images/'):
            print(f"🔍 Supabase storage path detected")
            signed_url = get_signed_image_url(image_path)
            if signed_url:
                st.image(signed_url, caption=f"📷 {word_original}", width=300)
                st.markdown("*🔒 Private image - only visible to you*")
                return True
            else:
                st.markdown("*Image temporarily unavailable - please try refreshing*")
                return False
        
        # Legacy handling for old image paths
        elif image_path.startswith('http'):
            print(f"🔍 Public URL detected")
            st.image(image_path, caption=f"📷 {word_original}", width=300)
            return True
        
        elif os.path.exists(image_path):
            print(f"🔍 Local file detected")
            image = Image.open(image_path)
            st.image(image, caption=f"📷 {word_original}", width=300)
            return True
        
        else:
            print(f"❌ Could not find image: {image_path}")
            return False
            
    except Exception as e:
        print(f"❌ Error displaying image: {e}")
        st.markdown("*Error loading image*")
        return False

def get_cropped_image_path(image_path):
    """Get the cropped version of an image path - improved legacy support."""
    if not image_path:
        return image_path
    
    print(f"🔍 Processing image path: {image_path}")
    
    # For new Supabase storage paths
    if image_path.startswith('vocabulary-images/'):
        return image_path  # Already processed during upload
    
    # For legacy public URLs
    if image_path.startswith('http'):
        if "_cropped" in image_path:
            return image_path
        return image_path.replace(".jpg", "_cropped.jpg")
    
    # For local file paths
    if "_cropped.jpg" in image_path:
        return image_path
    
    # Try to find cropped version of local file
    cropped_path = image_path.replace(".jpg", "_cropped.jpg")
    if os.path.exists(cropped_path):
        return cropped_path
    
    return image_path

# Function to start a new quiz
def start_new_quiz(vocabulary, num_questions=5):
    """Start a new quiz with proper question limiting."""
    # Reset quiz state
    st.session_state.quiz_score = 0
    st.session_state.quiz_total = 0
    st.session_state.current_question_number = 1  # ADD THIS LINE
    st.session_state.quiz_target_questions = num_questions  # Store target
    st.session_state.answered = False
    st.session_state.quiz_completed = False
    
    if not vocabulary or len(vocabulary) < 4:
        warning_message("Not enough vocabulary words for a quiz (need at least 4).")
        return False
    
    # Start a new session if needed
    if not st.session_state.session_id:
        st.session_state.session_id = create_session_direct()
        st.session_state.words_studied = 0
        st.session_state.words_learned = 0
    
    # Set up first question
    setup_new_question(vocabulary)
    return True


# Function to set up a new quiz question
def setup_new_question(vocabulary):
    """Set up a new quiz question with unique options."""
    if not vocabulary or len(vocabulary) < 4:
        return False
    
    # Select a random word as the question
    st.session_state.current_quiz_word = random.choice(vocabulary)
    
    # Get the correct answer
    correct_word = st.session_state.current_quiz_word
    
    # Create options starting with correct answer
    options = [correct_word]
    
    # Get wrong options (different words with different translations)
    available_wrong_options = [
        word for word in vocabulary 
        if (word['id'] != correct_word['id'] and 
            word['word_translated'].lower() != correct_word['word_translated'].lower() and
            word['word_original'].lower() != correct_word['word_original'].lower())
    ]
    
    # If we don't have enough unique wrong options, duplicate the vocabulary
    while len(available_wrong_options) < 3 and len(vocabulary) >= 2:
        available_wrong_options.extend([
            word for word in vocabulary 
            if word['id'] != correct_word['id']
        ])
    
    # Select 3 unique wrong options
    wrong_options = []
    for word in available_wrong_options:
        if len(wrong_options) >= 3:
            break
        
        # Check if this translation is already in our options
        word_translation = word['word_translated'].lower()
        existing_translations = [opt['word_translated'].lower() for opt in options + wrong_options]
        
        if word_translation not in existing_translations:
            wrong_options.append(word)
    
    # If still not enough unique options, create some variations
    while len(wrong_options) < 3:
        for word in vocabulary:
            if word['id'] != correct_word['id'] and len(wrong_options) < 3:
                wrong_options.append(word)
    
    # Add wrong options to the list
    options.extend(wrong_options[:3])
    
    # Shuffle the options
    random.shuffle(options)
    
    st.session_state.quiz_options = options
    st.session_state.answered = False
    
    return True

# Function to update word progress in the database
def update_word_progress_direct(vocab_id, is_correct):
    """Update word progress for the authenticated user."""
    try:
        user = get_authenticated_user()
        if not user:
            return False
        
        db = get_user_database()
        return db.update_word_progress(vocab_id, is_correct)
    except Exception as e:
        print(f"Error updating word progress: {e}")
        return False

# Function to check quiz answer
def check_answer(selected_index):
    """Check if selected quiz answer is correct and update progress."""
    if st.session_state.answered:
        return
    
    selected_word = st.session_state.quiz_options[selected_index]
    is_correct = selected_word['id'] == st.session_state.current_quiz_word['id']
    
    # Update database
    update_word_progress_direct(st.session_state.current_quiz_word['id'], is_correct)
    
    # Update session stats
    st.session_state.words_studied += 1
    if is_correct:
        st.session_state.words_learned += 1
        st.session_state.quiz_score += 1
    
    st.session_state.quiz_total += 1
    st.session_state.answered = True
    
    # FIX: Check if quiz should be completed WITHOUT immediate rerun
    target_questions = st.session_state.get('quiz_target_questions', 5)
    if st.session_state.quiz_total >= target_questions:
        # Don't set quiz_completed here - let the main logic handle it
        pass
    
    # Gamification check
    try:
        gamification.check_challenge_progress(
            quiz_score=st.session_state.quiz_score,
            quiz_total=st.session_state.quiz_total
        )
    except Exception as e:
        print(f"Gamification error: {e}")
    
    return is_correct

def sync_user_data_to_supabase():
    """Sync all user data to Supabase immediately."""
    try:
        user = get_authenticated_user()
        if not user:
            return False
        
        user_id = user.get('id')
        if not user_id:
            return False
        
        # Get actual vocabulary count
        vocabulary = get_all_vocabulary_direct()
        actual_word_count = len(vocabulary) if vocabulary else 0
        
        # Calculate consistent values
        calculated_level = max(1, actual_word_count // 10 + 1)
        calculated_points = actual_word_count * 10
        
        # Update session state
        st.session_state.words_learned = actual_word_count
        st.session_state.total_words_learned = actual_word_count
        st.session_state.level = calculated_level
        st.session_state.points = calculated_points
        
        # Prepare data for Supabase
        user_data = {
            'user_id': user_id,
            'words_learned': actual_word_count,
            'total_words_learned': actual_word_count,
            'level': calculated_level,
            'points': calculated_points,
            'streak_days': st.session_state.get('streak_days', 0),
            'last_active_date': str(date.today()),
            'streak_savers': st.session_state.get('streak_savers', 0),
            'achievements': json.dumps(st.session_state.get('achievements', {})),
            'badges': json.dumps(st.session_state.get('badges', {})),
            'updated_at': datetime.now().isoformat()
        }
        
        # Save to Supabase
        db = get_user_database()
        response = db.supabase.table('user_game_state').upsert(
            user_data,
            on_conflict='user_id'
        ).execute()
        
        if response.data:
            print(f"✅ Data synced: {actual_word_count} words, Level {calculated_level}, {calculated_points} points")
            return True
        else:
            print(f"❌ Sync failed: {response}")
            return False
            
    except Exception as e:
        print(f"❌ Sync error: {e}")
        return False

def load_user_data_from_supabase():
    """Load user data from Supabase on startup."""
    try:
        user = get_authenticated_user()
        if not user:
            return False
        
        user_id = user.get('id')
        if not user_id:
            return False
        
        # Get actual vocabulary first
        vocabulary = get_all_vocabulary_direct()
        actual_word_count = len(vocabulary) if vocabulary else 0
        
        # Load saved data from Supabase
        db = get_user_database()
        response = db.supabase.table('user_game_state').select('*').eq('user_id', user_id).execute()
        
        if response.data and len(response.data) > 0:
            user_data = response.data[0]
            
            # Use actual vocabulary count but load other data from Supabase
            st.session_state.words_learned = actual_word_count
            st.session_state.total_words_learned = actual_word_count
            st.session_state.level = max(user_data.get('level', 1), max(1, actual_word_count // 10 + 1))
            st.session_state.points = max(user_data.get('points', 0), actual_word_count * 10)
            st.session_state.streak_days = user_data.get('streak_days', 0)
            st.session_state.last_active_date = user_data.get('last_active_date')
            st.session_state.streak_savers = user_data.get('streak_savers', 0)
            
            # Load JSON data safely
            try:
                st.session_state.achievements = json.loads(user_data.get('achievements', '{}'))
                st.session_state.badges = json.loads(user_data.get('badges', '{}'))
            except:
                st.session_state.achievements = {}
                st.session_state.badges = {}
            
            print(f"✅ Data loaded: {actual_word_count} words, Level {st.session_state.level}")
            return True
        else:
            # Initialize with actual vocabulary count
            st.session_state.words_learned = actual_word_count
            st.session_state.total_words_learned = actual_word_count
            st.session_state.level = max(1, actual_word_count // 10 + 1)
            st.session_state.points = actual_word_count * 10
            st.session_state.streak_days = 0
            st.session_state.achievements = {}
            st.session_state.badges = {}
            
            # Save initial data
            sync_user_data_to_supabase()
            print(f"🆕 Initialized with {actual_word_count} words")
            return True
            
    except Exception as e:
        print(f"❌ Load error: {e}")
        return False

# Global counter for truly unique widget IDs
if 'widget_counter' not in st.session_state:
    st.session_state.widget_counter = 0


# Main sidebar for navigation
app_mode_options = ["Camera Mode", "My Vocabulary", "Quiz Mode", "Statistics", "My Progress", "Pronunciation Practice"]

# Get current mode or default
if 'app_mode' in st.session_state:
    current_index = app_mode_options.index(st.session_state.app_mode) if st.session_state.app_mode in app_mode_options else 0
else:
    current_index = 0
    st.session_state.app_mode = app_mode_options[0]

# Create selectbox with key to track changes
new_app_mode = st.sidebar.selectbox(
    "Choose a mode",
    app_mode_options,
    index=current_index,
    key="app_mode_selector"
)

# Force immediate update when selection changes
if new_app_mode != st.session_state.app_mode:
    st.session_state.app_mode = new_app_mode
    # Clear any mode-specific state that might interfere
    mode_specific_keys = [
        'quiz_completed', 'current_quiz_word', 'quiz_options', 
        'words_just_saved', 'detection_checkboxes'
    ]
    for key in mode_specific_keys:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()  # Force immediate rerun

# Use the updated app_mode
app_mode = st.session_state.app_mode

# Add gamification info to the sidebar
try:
    gamification.update_sidebar()
except Exception as e:
    st.sidebar.markdown('<div style="background-color: #1679AB; padding: 10px; border-radius: 5px; margin-top: 10px;">'
                        '<h3 style="color: #C5FF95; margin: 0;">🏆 Gamification</h3>'
                        '<p style="color: white; margin-top: 5px;">System initializing...</p>'
                        '</div>', unsafe_allow_html=True)
    print(f"Sidebar update error: {e}")

# Language selection
languages = {
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Portuguese": "pt",
    "Russian": "ru",
    "Japanese": "ja",
    "Chinese (Simplified)": "zh-CN"
}

selected_language = st.sidebar.selectbox(
    "Select target language",
    list(languages.keys()),
    index=list(languages.values()).index(st.session_state.target_language) if st.session_state.target_language in languages.values() else 0
)

# Get the language code from the dropdown selection
st.session_state.target_language = languages[selected_language]

# Force update word of the day translation immediately (UPDATED APPROACH)
if 'word_of_the_day' in st.session_state and st.session_state.word_of_the_day:
    wotd = st.session_state.word_of_the_day
    original_word = wotd.get('original', '')
    
    # Hard-coded translations for "book" in different languages
    book_translations = {
        "es": "libro",
        "fr": "livre",
        "de": "Buch",
        "it": "libro",
        "pt": "livro",
        "ru": "книга",
        "ja": "本",
        "zh-CN": "书"
    }
    
    # Check if the word is "book" and update translation
    if original_word.lower() == "book" and st.session_state.target_language in book_translations:
        wotd['translated'] = book_translations[st.session_state.target_language]
        st.session_state.word_of_the_day = wotd

        
# Add this right after your language selection code
# Force update word of the day when language changes
if 'previous_language' not in st.session_state:
    st.session_state.previous_language = st.session_state.target_language
    
# Check if language has changed
if st.session_state.previous_language != st.session_state.target_language:
    # Language changed - force update word of the day
    if 'word_of_the_day' in st.session_state and st.session_state.word_of_the_day:
        wotd = st.session_state.word_of_the_day
        original = wotd.get('original', '')
        
        # Translation dictionary for common words
        translations = {
            "book": {
                "es": "libro", "fr": "livre", "de": "Buch", "it": "libro",
                "pt": "livro", "ru": "книга", "ja": "本", "zh-CN": "书"
            },
            "hello": {
                "es": "hola", "fr": "bonjour", "de": "hallo", "it": "ciao",
                "pt": "olá", "ru": "привет", "ja": "こんにちは", "zh-CN": "你好"
            },
            # Include all other translations from the previous solution
        }
        
        # Get updated translation for current language
        new_lang = st.session_state.target_language
        if original.lower() in translations and new_lang in translations[original.lower()]:
            wotd['translated'] = translations[original.lower()][new_lang]
            wotd['language'] = new_lang
            st.session_state.word_of_the_day = wotd
    
    # Update previous language to current
    st.session_state.previous_language = st.session_state.target_language
    # Force rerun to update UI
    st.rerun()

# Add help section to the sidebar
with st.sidebar.expander("ℹ️ Need Help?"):
    st.markdown("""
    **Quick Tips:**
    - 📸 Use **Camera Mode** to capture objects and learn new words
    - 📚 Review your words in **My Vocabulary**
    - 🎮 Test yourself in **Quiz Mode**
    - 📊 Track your progress in **Statistics**
    
    **On Mobile:**
    - After taking a picture, scroll down to see results
    - Tap buttons to navigate between sections
    """)

# Display appropriate content based on selected mode
if app_mode == "Camera Mode":
    style_title("Camera Mode")

    # Use the enhanced info message
    info_message("Take a photo or upload an image to identify objects and learn new vocabulary.")
    
    # Session management
    session_container = st.container()
    with session_container:
        col1, col2 = st.columns(2)
        # FIXED: Properly close the with statement
        with col1:
            # Always create a placeholder for the start button
            start_button_placeholder = st.empty()
            
            # Conditionally show the button or message
            if st.session_state.session_id is None:
                if start_button_placeholder.button("Start Learning Session", key="start_session_btn"):
                    if manage_session("start"):
                        st.rerun()
            else:
                # Show session info in the same place
                start_button_placeholder.markdown(
                    f'<div class="info-box" style="margin: 0.5rem 0; height: 38px; display: flex; align-items: center;">'
                    f'Session in progress - Words: {st.session_state.words_learned}'
                    f'</div>', 
                    unsafe_allow_html=True
                )
        
        with col2:
            # Always create a placeholder for the end button
            end_button_placeholder = st.empty()
            
            # Only show the end button when a session is active
            if st.session_state.session_id is not None:
                if end_button_placeholder.button("End Session", key="end_session_btn"):
                    if manage_session("end"):
                        st.rerun()
            else:
                # Empty space with same height to maintain layout
                end_button_placeholder.markdown(
                    '<div style="height: 38px;"></div>', 
                    unsafe_allow_html=True
                )
    
    # Image input options
    image_tab1, image_tab2 = st.tabs(["📷 Take a Photo", "📁 Upload Image"])

    image = None
    # First tab: Camera
    with image_tab1:
        picture = st.camera_input("Take a picture", key="camera_input")
        if picture is not None:
            image = Image.open(picture)

    # Second tab: Upload
    with image_tab2:
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="file_uploader")
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
            except Exception as e:
                st.error(f"Error loading image: {e}")
                image = None
    
    # Detection options
    detection_type = st.radio(
        "What would you like to detect?",
        ["Objects", "Text (OCR)"],
        index=0
    )

    # Detection settings for objects
    if detection_type == "Objects":
        confidence_threshold = st.slider(
            "Detection Confidence", 
            min_value=0.3, 
            max_value=0.9, 
            value=0.5,
            step=0.05
        )
        # Set iou_threshold for optimal detection (balance between precision and maximum detection)
        iou_threshold = 0.45  # Using a lower threshold to detect more objects while maintaining precision
        
        # Auto-enhancement is always applied
        enhancement_type = "auto"
                    
    # Process image if available
    if image is not None:
        # Process based on detection type
        if detection_type == "Objects":
            # Use a placeholder for the spinner that we can clear later
            spinner_placeholder = st.empty()
            with spinner_placeholder.container():
                show_loading_spinner("Detecting objects... This may take a few seconds.")
            
            # Add visual separator for mobile
            separator_placeholder = st.empty()
            separator_placeholder.markdown('<div class="result-separator"></div>', unsafe_allow_html=True)
            
            try:
                # Always apply enhancement for object detection
                enhanced_image = enhance_image(image, "auto")
                if enhanced_image is None:
                    raise Exception("Image enhancement failed")
                
                # Use Google Cloud Vision
                detections, result_image = detect_objects_google_vision(
                    enhanced_image, confidence_threshold
                )
                
            except Exception as e:
                error_message(f"Detection error: {str(e)}")
                # Return empty results on error
                detections, result_image = [], np.array(image)
            
            # Clear the spinner and separator completely
            spinner_placeholder.empty()
            separator_placeholder.empty()
        
            # Display results
            if detections:
                style_section_title("✨ Detected Objects")
                
                # Display image with detection boxes
                st.image(result_image, caption="Detected Objects")
                    
                # Display selection prompt
                st.write("Select objects to save to your vocabulary:")
                    
                # Group detections by category
                # Group detections by label to avoid duplicates
                unique_detections = {}
                for i, detection in enumerate(detections):
                    label = detection['label']
                    confidence = detection['confidence']
                    
                    # If label already exists and new confidence is lower, skip
                    if label in unique_detections and unique_detections[label][1]['confidence'] >= confidence:
                        continue
                    
                    # Otherwise add/update this label with highest confidence detection
                    unique_detections[label] = (i, detection)

                # Now group the unique detections by category
                categorized_detections = {}
                for i, detection in unique_detections.values():
                    label = detection['label']
                    category = get_object_category(label)
                    
                    if category not in categorized_detections:
                        categorized_detections[category] = []
                    
                    categorized_detections[category].append((i, detection))
                
                # Clear detection checkboxes for new image
                if 'last_image_hash' not in st.session_state or st.session_state.last_image_hash != get_image_hash(image):
                    st.session_state.detection_checkboxes = {}
                    st.session_state.last_image_hash = get_image_hash(image)
                
                # Display objects by category in expandable sections
                for category, category_detections in categorized_detections.items():
                    with st.expander(f"{category.title()} ({len(category_detections)} items)", expanded=True):
                        # Process each detection in this category
                        for i, detection in category_detections:
                            label = detection['label']
                            confidence = detection['confidence']
                            checkbox_key = f"detect_{i}"
                            
                            # Translate the label
                            translated_label = translate_text(label, st.session_state.target_language)
                            
                            # Create a container for this object
                            with st.container():
                                # Display the object info
                                st.markdown(f"**{label}** ({confidence:.2f})")
                                st.markdown(f"→ **{translated_label}**")
                                
                                # Create columns for audio, example, checkbox
                                col1, col2, col3 = st.columns([2, 2, 1])
                                
                                with col1:
                                    # Generate audio for the translated word
                                    audio_bytes = text_to_speech(translated_label, st.session_state.target_language)
                                    if audio_bytes:
                                        st.markdown(get_audio_html(audio_bytes), unsafe_allow_html=True)
                                    
                                    # Add pronunciation helpers
                                    pronunciation_tips = get_pronunciation_guide(translated_label, st.session_state.target_language)
                                    if pronunciation_tips:
                                        st.markdown("**Pronunciation Tips:**")
                                        for tip in pronunciation_tips:
                                            st.markdown(f"- {tip}")
                                
                                with col2:
                                    # Add example sentence directly
                                    example = get_example_sentence(label, st.session_state.target_language)
                                    st.markdown("**Example:**")
                                    st.markdown(f"EN: {example['english']}")
                                    
                                    # Only display translated example if available
                                    if example['translated']:
                                        source = example.get('source', 'unknown')
                                        source_name = source.replace('_', ' ').replace('api', 'API').title()
                                        st.markdown(f"{selected_language}: {example['translated']}")
                                        st.markdown(f"<small><i>Source: {source_name}</i></small>", unsafe_allow_html=True)
                                        
                                        # Only generate audio if there's text to speak
                                        example_audio = text_to_speech(example['translated'], st.session_state.target_language)
                                        if example_audio:
                                            st.markdown(get_audio_html(example_audio), unsafe_allow_html=True)
                                    else:
                                        st.markdown("*Translation not available. Please install deep-translator package.*")
                                
                                with col3:
                                    # Default to checked
                                    if checkbox_key not in st.session_state.detection_checkboxes:
                                        st.session_state.detection_checkboxes[checkbox_key] = True
                                        
                                    # Add checkbox for this object
                                    st.session_state.detection_checkboxes[checkbox_key] = st.checkbox(
                                        "Save", 
                                        value=st.session_state.detection_checkboxes[checkbox_key],
                                        key=checkbox_key
                                    )
                                
                                st.markdown("---")  # Add separator

                # Create a stable persistent key for our save button
                save_button_id = "save_objects_button_fixed"
                
                # Display save button if objects haven't been saved yet
                if not st.session_state.words_just_saved:
                    # Create a button with a fixed, consistent key
                    if st.button("Save Selected Objects to Vocabulary", key=save_button_id):
                        # Auto-start session if needed
                        if st.session_state.session_id is None:
                            if manage_session("start"):
                                success_message("Created a new learning session!")
                            else:
                                error_message("Failed to create a session. Please check database connection.")
                                st.stop()
                        
                        # Count selected objects
                        selected_objects = []
                        for i in range(len(detections)):
                            if st.session_state.detection_checkboxes.get(f"detect_{i}", False):
                                selected_objects.append(i)
                        
                        if not selected_objects:
                            warning_message("No objects were selected to save. Please check at least one 'Save' box.")
                        else:
                            # Save the selected objects
                            saved_count = 0
                            saved_items = []
                            failed_items = []
                            
                            for i in selected_objects:
                                vocab_id = None  # Initialize vocab_id for each iteration
                                try:
                                    detection = detections[i]
                                    label = detection['label']
                                    
                                    print(f"🔄 Processing {label}...")
                                    
                                    # Translate the label
                                    translated_label = translate_text(label, st.session_state.target_language)
                                    if not translated_label or translated_label == label:
                                        print(f"❌ Translation failed for {label}")
                                        failed_items.append(label)
                                        continue
                                    
                                    # Try to save image (allow this to fail without stopping vocabulary save)
                                    image_path = None
                                    try:
                                        image_path = save_image(image, label, detection['bbox'])
                                        if image_path:
                                            print(f"✅ Image saved: {image_path}")
                                        else:
                                            print(f"⚠️ Image save failed for {label}, continuing without image")
                                    except Exception as img_error:
                                        print(f"⚠️ Image save error for {label}: {img_error}, continuing without image")
                                    
                                    # Get object category
                                    category = get_object_category(label)
                                    
                                    # Save vocabulary regardless of image save status
                                    vocab_id = add_vocabulary_direct(
                                        word_original=label,
                                        word_translated=translated_label,
                                        language_translated=st.session_state.target_language,
                                        category=category,
                                        image_path=image_path  # This can be None
                                    )
                                    
                                    if vocab_id:
                                        saved_count += 1
                                        saved_items.append(f"{label} → {translated_label}")
                                        # Update session stats
                                        st.session_state.words_studied += 1
                                        st.session_state.words_learned += 1
                                        print(f"✅ Vocabulary saved: {label}")
                                        
                                        # Sync data after each successful save
                                        try:
                                            sync_user_data_to_supabase()
                                        except Exception as sync_error:
                                            print(f"⚠️ Data sync failed: {sync_error}")
                                    else:
                                        failed_items.append(label)
                                        print(f"❌ Vocabulary save failed: {label}")
                                        
                                except Exception as e:
                                    print(f"❌ Error saving {label}: {e}")
                                    failed_items.append(label)
                            
                            # Display results
                            if saved_count > 0:
                                success_message(f"Successfully saved {saved_count} words to vocabulary!")
                                st.session_state.saved_items = saved_items
                                st.session_state.words_just_saved = True
                                st.session_state.saved_count = saved_count
                            
                            if failed_items:
                                error_message(f"Failed to save: {', '.join(failed_items)}")
                            
                            if saved_count == 0:
                                error_message("Failed to save any words. Please check database connection.")
                            
                            # Clear the checkboxes and rerun
                            st.session_state.detection_checkboxes = {}
                            time.sleep(1.5)
                            st.rerun()
                # Show success message and navigation AFTER saving (persists across reruns)
                if st.session_state.words_just_saved:
                    # Create a container for the success message
                    success_container = st.container()
                    
                    with success_container:
                        if st.session_state.saved_count > 0:
                            success_message(f"Successfully added {st.session_state.saved_count} new words to your vocabulary!")
                            
                            # Show saved words in a visually appealing list
                            st.markdown('<h4 style="color: #1679AB;">New words saved:</h4>', unsafe_allow_html=True)
                            for item in st.session_state.saved_items:
                                st.markdown(f"✅ {item}")
                        
                        if st.session_state.get('duplicate_count', 0) > 0:
                            warning_message(f"{st.session_state.duplicate_count} words were already in your vocabulary.")
                            
                            # Show duplicate words
                            st.markdown('<h4 style="color: #FF9800;">Already in vocabulary:</h4>', unsafe_allow_html=True)
                            for item in st.session_state.get('duplicate_items', []):
                                st.markdown(f"⚠️ {item}")
                        
                        # Show navigation options only if something was processed
                        if st.session_state.saved_count > 0 or st.session_state.get('duplicate_count', 0) > 0:
                            st.markdown("### What would you like to do next?")
                            next_col1, next_col2, next_col3 = st.columns(3)
                            
                            # Define navigation callback functions
                            def go_to_quiz_mode():
                                st.session_state.words_just_saved = False  # Reset the saved state
                                st.session_state.app_mode = "Quiz Mode"
                                st.session_state.detection_checkboxes = {}  # Clear checkboxes
                                # Clear duplicate tracking
                                if 'duplicate_count' in st.session_state:
                                    del st.session_state.duplicate_count
                                if 'duplicate_items' in st.session_state:
                                    del st.session_state.duplicate_items
                                st.rerun()

                            def go_to_vocabulary():
                                st.session_state.words_just_saved = False  # Reset the saved state
                                st.session_state.app_mode = "My Vocabulary"
                                st.session_state.detection_checkboxes = {}  # Clear checkboxes
                                # Clear duplicate tracking
                                if 'duplicate_count' in st.session_state:
                                    del st.session_state.duplicate_count
                                if 'duplicate_items' in st.session_state:
                                    del st.session_state.duplicate_items
                                st.rerun()

                            def continue_capturing():
                                st.session_state.words_just_saved = False
                                st.session_state.detection_checkboxes = {}  # Clear checkboxes
                                # Clear duplicate tracking
                                if 'duplicate_count' in st.session_state:
                                    del st.session_state.duplicate_count
                                if 'duplicate_items' in st.session_state:
                                    del st.session_state.duplicate_items
                                st.rerun()
                            
                            # Each button with fixed keys
                            with next_col1:
                                if st.button("🎮 Go to Quiz Mode", key="quiz_nav_button"):
                                    go_to_quiz_mode()
                                    
                            with next_col2:
                                if st.button("📚 View My Vocabulary", key="vocab_nav_button"):
                                    go_to_vocabulary()
                                    
                            with next_col3:
                                if st.button("📸 Continue Capturing", key="continue_button"):
                                    continue_capturing()
                
            # Add manual selection UI if enabled
            if st.session_state.manual_mode:
                st.subheader("Manual Object Selection")
                st.write("Enter a label for the object you want to learn.")
                
                # Object label input
                st.session_state.manual_label = st.text_input("Object Label:", 
                                                            value=st.session_state.manual_label,
                                                            placeholder="e.g., cup, book, chair")
                
                # Translate the label
                if st.session_state.manual_label:
                    translated_label = translate_text(st.session_state.manual_label, 
                                                    st.session_state.target_language)
                    
                    st.write(f"Original: **{st.session_state.manual_label}**")
                    st.write(f"Translation: **{translated_label}**")
                    
                    # Generate audio for the translated word
                    audio_bytes = text_to_speech(translated_label, st.session_state.target_language)
                    if audio_bytes:
                        st.markdown(get_audio_html(audio_bytes), unsafe_allow_html=True)
                    
                    # Save button for manual selection
                    if st.button("Save to Vocabulary"):
                        # Auto-start session if needed
                        if st.session_state.session_id is None:
                            if manage_session("start"):
                                success_message("Created a new learning session!")
                            else:
                                error_message("Failed to create a session.")
                                st.stop()
                        
                        # Save the image
                        image_path = save_image(image, st.session_state.manual_label)
                        
                        # Add to database
                        vocab_id = add_vocabulary_direct(
                            word_original=st.session_state.manual_label,
                            word_translated=translated_label,
                            language_translated=st.session_state.target_language,
                            category="manual",
                            image_path=image_path
                        )
                        
                        if vocab_id:
                            success_message(f"Successfully added '{st.session_state.manual_label}' to your vocabulary!")
                            st.session_state.words_studied += 1
                            st.session_state.words_learned += 1
                            
                            # Reset manual mode
                            st.session_state.manual_mode = False
                            st.session_state.manual_label = ""
                            time.sleep(1.5)
                            st.rerun()
                        else:
                            error_message("Failed to save word to vocabulary.")
                
                # Button to exit manual mode
                if st.button("Cancel Manual Selection"):
                    st.session_state.manual_mode = False
                    st.session_state.manual_label = ""
                    st.rerun()
        
        # Text OCR mode
        elif detection_type == "Text (OCR)":  # Text OCR mode
            # Create container for loading spinner
            spinner_container = st.empty()
            with spinner_container:
                show_loading_spinner("Detecting text... This may take a few seconds.")
                
            # Add visual separator for mobile
            add_result_separator()
            
            with st.spinner("Detecting text..."):
                detected_text = detect_text_in_image(image)
                spinner_container.empty()
                add_scroll_indicator()
                
                if detected_text and detected_text != "No text found in this image.":
                    style_section_title("📝 Detected Text")
                    st.write(f"**Full Text:** {detected_text}")
                    
                    # Split into words for learning (filter out short words and numbers)
                    import re
                    words = [word.strip() for word in re.split(r'[^\w]', detected_text) 
                            if word.strip() and len(word.strip()) > 2 and not word.strip().isdigit()]
                    
                    if words:
                        st.subheader("📚 Words Available for Learning")
                        
                        # Display all words without checkboxes
                        for i, word in enumerate(words[:10]):  # Limit to 10 words
                            st.markdown("---")
                            
                            col1, col2 = st.columns([1, 1])
                            
                            with col1:
                                st.markdown(f"### 📖 {word.lower()}")
                                
                                # Get translation
                                translation_result = translate_text(word.lower(), st.session_state.target_language)
                                if translation_result and translation_result != word.lower():
                                    st.markdown(f"**Translation:** {translation_result}")
                                    
                                    # Pronunciation guide
                                    pronunciation_notes = get_pronunciation_guide(translation_result, st.session_state.target_language)
                                    if pronunciation_notes:
                                        st.markdown("**Pronunciation tips:**")
                                        for note in pronunciation_notes[:2]:
                                            st.markdown(f"• {note}")
                                    
                                    # Audio pronunciation
                                    try:
                                        audio_bytes = text_to_speech(translation_result, st.session_state.target_language)
                                        if audio_bytes:
                                            st.markdown("**🔊 Listen:**")
                                            audio_html = get_audio_html(audio_bytes)
                                            st.markdown(audio_html, unsafe_allow_html=True)
                                    except Exception as e:
                                        print(f"Audio generation error: {e}")
                                else:
                                    st.markdown("*Translation not available*")
                            
                            with col2:
                                # Example sentence
                                try:
                                    example = get_example_sentence(word.lower(), st.session_state.target_language)
                                    if example and example.get('translated'):
                                        st.markdown("**Example:**")
                                        st.markdown(f"*{example['translated']}*")
                                        
                                        if example.get('english'):
                                            st.markdown(f"*{example['english']}*")
                                        
                                        # Example audio
                                        try:
                                            example_audio_bytes = text_to_speech(example['translated'], st.session_state.target_language)
                                            if example_audio_bytes:
                                                st.markdown("**🔊 Example audio:**")
                                                example_audio_html = get_audio_html(example_audio_bytes)
                                                st.markdown(example_audio_html, unsafe_allow_html=True)
                                        except Exception as e:
                                            print(f"Example audio generation error: {e}")
                                    else:
                                        st.markdown("*Example sentence not available*")
                                except Exception as e:
                                    print(f"Example sentence error: {e}")
                                    st.markdown("*Example sentence not available*")
                            
                            # Save button for this word
                            save_key = f"save_text_word_{i}_{word}_{hash(word)}"
                            if st.button(f"💾 Save '{word.lower()}' to vocabulary", key=save_key):
                                if translation_result and translation_result != word.lower():
                                    vocab_id = add_vocabulary_direct(
                                        word.lower(), 
                                        translation_result, 
                                        st.session_state.target_language, 
                                        category="text",
                                        image_path=None
                                    )
                                    
                                    if vocab_id:
                                        success_message(f"✅ '{word.lower()}' saved to vocabulary!")
                                        # Update counters and save to Supabase
                                        sync_user_data_to_supabase()
                                        time.sleep(1)
                                        st.rerun()
                                    else:
                                        error_message("Failed to save word to vocabulary.")
                                else:
                                    error_message("Cannot save - translation not available.")
                    else:
                        st.info("No meaningful words found for vocabulary learning.")
                else:
                    st.warning("No text detected in the image. Try an image with clearer text.")

elif app_mode == "My Vocabulary":
    style_title("My Vocabulary")
    st.markdown("Review all the words you've learned so far.")
    
    # Get vocabulary from database
    vocabulary = get_all_vocabulary_direct()
    
    # Filter options
    col1, col2 = st.columns(2)
    with col1:
        filter_language = st.selectbox(
            "Filter by language:",
            ["All"] + list(languages.keys()),
            index=0
        )
    with col2:
        sort_by = st.selectbox(
            "Sort by:",
            ["Date added (newest first)", "Date added (oldest first)", "Proficiency (low to high)", "Proficiency (high to low)"]
        )
    
    # Apply filters
    filtered_vocab = []
    for word in vocabulary:
        # Skip any None entries or entries without required fields
        if word is None or 'language_translated' not in word:
            continue
            
        if filter_language == "All" or languages.get(filter_language) == word['language_translated']:
            # Make sure proficiency_level exists
            if 'proficiency_level' not in word or word['proficiency_level'] is None:
                word['proficiency_level'] = 0
                
            filtered_vocab.append(word)
    
    # Sort vocabulary
    if filtered_vocab:
        if sort_by == "Date added (newest first)":
            filtered_vocab.sort(key=lambda x: x.get('date_added', ''), reverse=True)
        elif sort_by == "Date added (oldest first)":
            filtered_vocab.sort(key=lambda x: x.get('date_added', ''))
        elif sort_by == "Proficiency (low to high)":
            filtered_vocab.sort(key=lambda x: x.get('proficiency_level', 0))
        elif sort_by == "Proficiency (high to low)":
            filtered_vocab.sort(key=lambda x: x.get('proficiency_level', 0), reverse=True)
    
    # Display vocabulary
    if filtered_vocab:
        st.markdown(f"**Found {len(filtered_vocab)} words in your vocabulary collection.**")
        
        # Create data for table view
        table_data = []
        for word in filtered_vocab:
            # Skip entries without required fields
            if not all(k in word for k in ['word_original', 'word_translated', 'language_translated']):
                continue
                
            # Get language name from code
            lang_code = word.get('language_translated', '')
            lang_name = next((k for k, v in languages.items() if v == lang_code), lang_code)
            
            # Format proficiency level
            proficiency = word.get('proficiency_level', 0) or 0
            proficiency_display = "⭐" * proficiency
            
            # Format date
            date_added = word.get('date_added', '')
            if date_added and isinstance(date_added, str):
                date_display = date_added.split()[0] if ' ' in date_added else date_added
            else:
                date_display = "Unknown"
            
            table_data.append({
                "Original": word.get('word_original', ''),
                "Translation": word.get('word_translated', ''),
                "Language": lang_name,
                "Proficiency": proficiency_display,
                "Date Added": date_display
            })
        
        # Display as a dataframe
        if table_data:
            st.dataframe(pd.DataFrame(table_data))
            
            # Detailed view
            st.subheader("Word Details")
            selected_word_index = st.selectbox(
                "Select a word to review:",
                range(len(filtered_vocab)),
                format_func=lambda i: f"{filtered_vocab[i].get('word_original', '')} → {filtered_vocab[i].get('word_translated', '')}"
            )
            
            word = filtered_vocab[selected_word_index]
            
            # Display word details
            col1, col2 = st.columns([2, 3])
            with col1:
                st.markdown(f"**Original:** {word.get('word_original', '')}")
                st.markdown(f"**Translation:** {word.get('word_translated', '')}")
                
                # Language name from code
                lang_code = word.get('language_translated', '')
                lang_name = next((k for k, v in languages.items() if v == lang_code), lang_code)
                st.markdown(f"**Language:** {lang_name}")
                
                if word.get('category'):
                    st.markdown(f"**Category:** {word.get('category', '')}")
                
                # Generate pronunciation audio
                st.markdown("**Listen to pronunciation:**")
                audio_bytes = text_to_speech(word.get('word_translated', ''), word.get('language_translated', ''))
                if audio_bytes:
                    st.markdown(get_audio_html(audio_bytes), unsafe_allow_html=True)
                
                # Show proficiency if available
                proficiency = word.get('proficiency_level', 0) or 0
                st.markdown("**Learning progress:**")
                st.progress(proficiency / 5)
                review_count = word.get('review_count', 0) or 0
                st.markdown(f"Proficiency: {proficiency}/5 (based on {review_count} reviews)")
                
                # Add pronunciation helpers
                pronunciation_tips = get_pronunciation_guide(word.get('word_translated', ''), word.get('language_translated', ''))
                if pronunciation_tips:
                    st.markdown("**Pronunciation tips:**")
                    for tip in pronunciation_tips:
                        st.markdown(f"- {tip}")
            
            with col2:
                # Display image if available
                image_path = word.get('image_path', '')
                if image_path:
                    try:
                        # Use cropped version if available (maintaining existing functionality)
                        display_image_path = get_cropped_image_path(image_path)
                        
                        # Handle different image storage types
                        image_displayed = False
                        
                        if display_image_path.startswith('vocabulary-images/'):
                            # Private Supabase Storage - get signed URL
                            signed_url = get_signed_image_url(display_image_path)
                            if signed_url:
                                st.image(signed_url, caption=f"📷 {word.get('word_original', '')} - Cropped from detection")
                                st.markdown("*Showing focused crop of detected object*")
                                image_displayed = True
                        elif display_image_path.startswith('http'):
                            # Legacy public URL (for old images)
                            st.image(display_image_path, caption=f"📷 {word.get('word_original', '')} - Cropped from detection")
                            if "_cropped" in display_image_path:
                                st.markdown("*Showing focused crop of detected object*")
                            else:
                                st.markdown("*Showing full original image*")
                            image_displayed = True
                        elif os.path.exists(display_image_path):
                            # Local file (fallback)
                            image = Image.open(display_image_path)
                            st.image(image, caption=f"📷 {word.get('word_original', '')} - Cropped from detection")
                            
                            if "_cropped.jpg" in display_image_path:
                                st.markdown("*Showing focused crop of detected object*")
                            else:
                                st.markdown("*Showing full original image*")
                            image_displayed = True
                        
                        if not image_displayed:
                            st.markdown("*Image temporarily unavailable*")
                            
                    except Exception as e:
                        error_message(f"Error loading image: {e}")
                        st.markdown("*Error loading image*")
                else:
                    st.markdown("*No image available for this word*")
                
                # Add example sentence directly (no expander)
                example = get_example_sentence(word.get('word_original', ''), word.get('language_translated', ''))
                st.markdown(f"**Example in context:**")
                st.markdown(f"**English:** {example['english']}")

                if example['translated']:
                    source = example.get('source', 'unknown')
                    source_name = source.replace('_', ' ').replace('api', 'API').title()
                    st.markdown(f"**{lang_name}:** {example['translated']}")
                    st.markdown(f"<small><i>Source: {source_name}</i></small>", unsafe_allow_html=True)
                    
                    # Only generate audio if there's text to speak
                    example_audio = text_to_speech(example['translated'], word.get('language_translated', ''))
                    if example_audio:
                        st.markdown(get_audio_html(example_audio), unsafe_allow_html=True)

        else:
            warning_message("There was an issue with the vocabulary data format.")
    else:
        info_message("No vocabulary words found with current filter. Go to Camera Mode to start learning new words!")

elif app_mode == "Quiz Mode":
    style_title("Quiz Mode")
    st.markdown("Test your vocabulary knowledge with interactive quizzes.")
    # Get vocabulary from database
    vocabulary = get_all_vocabulary_direct()
    
    # Initialize quiz state variables if they don't exist
    if 'quiz_completed' not in st.session_state:
        st.session_state.quiz_completed = False
    if 'current_quiz_word' not in st.session_state:
        st.session_state.current_quiz_word = None
    if 'quiz_options' not in st.session_state:
        st.session_state.quiz_options = []
    if 'current_question_number' not in st.session_state:
        st.session_state.current_question_number = 1

    # Check if quiz is in progress
    quiz_in_progress = (st.session_state.current_quiz_word is not None and 
                       st.session_state.quiz_options and 
                       not st.session_state.quiz_completed)

    if quiz_in_progress:
        # Check if quiz should be completed BEFORE displaying question
        target_questions = st.session_state.get('quiz_target_questions', 5)
        
        if st.session_state.quiz_total >= target_questions:
            # Quiz is complete
            st.session_state.quiz_completed = True
            st.rerun()
        
        # Quiz is in progress - display current question
        current_word = st.session_state.current_quiz_word
        
        # Display question number
        st.markdown(f"### 🎯 Quiz Question {st.session_state.current_question_number}/{target_questions}")
        
        # Display image if available
        image_path = current_word.get('image_path', '')
        has_image = False
        if image_path:
            if display_quiz_image(current_word, "What is this object?"):
                has_image = True
        
        # Display the question
        if has_image:
            st.markdown(f"### What is this object in {selected_language}?")
        else:
            st.markdown(f"### Translate: **{current_word['word_original']}** to {selected_language}")
        
        # Display answer options - ONLY if not answered yet
        if not st.session_state.answered:
            st.markdown("**Choose the correct answer:**")
            
            for i, option in enumerate(st.session_state.quiz_options):
                button_key = f"quiz_option_{i}_{st.session_state.quiz_total}_{target_questions}"
                
                if st.button(f"{chr(65+i)}. {option['word_translated']}", 
                            key=button_key, 
                            use_container_width=True):
                    # Store the selected answer
                    st.session_state.selected_answer_index = i
                    st.session_state.selected_answer = option
                    
                    is_correct = check_answer(i)
                    st.rerun()
        else:
            # Show comprehensive results after answering
            correct_answer = current_word['word_translated']
            selected_answer = st.session_state.get('selected_answer', {}).get('word_translated', 'Unknown')
            
            # Show what user selected vs correct answer
            if st.session_state.get('selected_answer_index') is not None:
                if selected_answer == correct_answer:
                    st.success(f"✅ Correct! You selected: **{selected_answer}**")
                else:
                    st.error(f"❌ Incorrect. You selected: **{selected_answer}**")
                    st.info(f"💡 The correct answer was: **{correct_answer}**")
            
            # Show comprehensive learning information
            st.markdown("---")
            st.markdown("### 📚 Learn More About This Word")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**English:** {current_word['word_original']}")
                st.markdown(f"**{selected_language}:** {correct_answer}")
                st.markdown(f"**Category:** {current_word.get('category', 'Unknown')}")
                
                # Pronunciation audio
                st.markdown("**🔊 Pronunciation:**")
                audio_bytes = text_to_speech(correct_answer, current_word['language_translated'])
                if audio_bytes:
                    st.markdown(get_audio_html(audio_bytes), unsafe_allow_html=True)
                
                # Pronunciation tips
                pronunciation_tips = get_pronunciation_guide(correct_answer, current_word['language_translated'])
                if pronunciation_tips:
                    st.markdown("**💡 Pronunciation Tips:**")
                    for tip in pronunciation_tips:
                        st.markdown(f"- {tip}")
            
            with col2:
                # Example sentences
                example = get_example_sentence(current_word['word_original'], current_word['language_translated'])
                st.markdown("**📝 Example Sentences:**")
                
                st.markdown(f"**English:** {example['english']}")
                
                if example['translated']:
                    st.markdown(f"**{selected_language}:** {example['translated']}")
                    
                    # Audio for example sentence
                    example_audio = text_to_speech(example['translated'], current_word['language_translated'])
                    if example_audio:
                        st.markdown("**🔊 Example Audio:**")
                        st.markdown(get_audio_html(example_audio), unsafe_allow_html=True)
                else:
                    st.markdown("*Example translation not available*")
            
            # Next question or finish quiz
            target_questions = st.session_state.get('quiz_target_questions', 5)
            
            if st.session_state.quiz_total >= target_questions:
                # Quiz complete - show finish button
                if st.button("🏁 Finish Quiz", key=f"finish_quiz_{st.session_state.quiz_total}"):
                    st.session_state.quiz_completed = True
                    st.rerun()
            else:
                # More questions available
                if st.button("➡️ Next Question", key=f"next_q_{st.session_state.quiz_total}"):
                    # Clear selected answer for next question
                    if 'selected_answer_index' in st.session_state:
                        del st.session_state.selected_answer_index
                    if 'selected_answer' in st.session_state:
                        del st.session_state.selected_answer
                    
                    # INCREMENT the question number when moving to next question
                    st.session_state.current_question_number += 1
                    
                    if setup_new_question(vocabulary):
                        st.rerun()
                    else:
                        st.session_state.quiz_completed = True
                        st.rerun()
        
        # Display current score in sidebar
        st.sidebar.markdown(f"### Current Score: {st.session_state.quiz_score}/{st.session_state.quiz_total}")
        if st.session_state.quiz_total > 0:
            accuracy = (st.session_state.quiz_score / st.session_state.quiz_total) * 100
            st.sidebar.markdown(f"**Accuracy:** {accuracy:.1f}%")
            
    # Display quiz results if quiz is completed
    elif st.session_state.quiz_completed and st.session_state.quiz_total > 0:
        st.markdown("### 🎉 Quiz Results")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Final Score", f"{st.session_state.quiz_score}/{st.session_state.quiz_total}")
        with col2:
            accuracy = (st.session_state.quiz_score / st.session_state.quiz_total) * 100
            st.metric("Accuracy", f"{accuracy:.1f}%")
        with col3:
            if accuracy >= 80:
                grade = "🏆 Excellent"
            elif accuracy >= 60:
                grade = "👍 Good"
            else:
                grade = "📚 Keep practicing"
            st.metric("Grade", grade)
        
        # Reset quiz button
        if st.button("🔄 Start New Quiz"):
            # Reset quiz state
            st.session_state.quiz_completed = False
            st.session_state.current_quiz_word = None
            st.session_state.quiz_options = []
            st.session_state.quiz_score = 0
            st.session_state.quiz_total = 0
            st.session_state.current_question_number = 1
            st.session_state.answered = False
            st.rerun()
        
    # Display quiz setup (this is what was missing!)
    else:
        # Introduction
        st.markdown("""
        Choose your quiz settings below to test your vocabulary knowledge.
        The quiz will randomly include different types of questions:
        
        - 🔄 Translation (both directions)
        - 🖼️ Image recognition (with focused object views)
        - 📝 Sentence completion
        - 🎯 Category matching
        - 📊 Related words identification
        - 🔊 Audio recognition
        
        Start with a small number of questions and work your way up!
        """)
        
        # Quiz settings in columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            quiz_language = st.selectbox(
                "Quiz language:",
                list(languages.keys()),
                index=list(languages.values()).index(st.session_state.target_language) if st.session_state.target_language in languages.values() else 0
            )
            quiz_lang_code = languages[quiz_language]
        
        with col2:
            num_questions = st.number_input("Number of questions:", min_value=1, max_value=20, value=5)
        
        with col3:
            # Get all categories from vocabulary
            categories = set()
            for word in vocabulary:
                if word and 'category' in word and word['category'] and word['category'] not in ['other', 'manual']:
                    categories.add(word['category'])
            
            if categories:
                category_filter = st.selectbox(
                    "Category filter (optional):",
                    ["All Categories"] + sorted(list(categories))
                )
            else:
                category_filter = "All Categories"
        
        # Filter vocabulary by selected language
        filtered_vocab = [word for word in vocabulary if word['language_translated'] == quiz_lang_code]
        
        # Further filter by category if selected
        if category_filter != "All Categories":
            filtered_vocab = [word for word in filtered_vocab if word.get('category') == category_filter]
        
        filtered_vocab = prepare_vocabulary_for_diverse_questions(filtered_vocab, languages)
        
        # Display information about available words
        if filtered_vocab:
            st.markdown(f"**{len(filtered_vocab)} words available** for your quiz in {quiz_language}" + 
                        (f" ({category_filter} category)" if category_filter != "All Categories" else ""))
            
            # Count words with images
            words_with_images = sum(1 for word in filtered_vocab 
                                  if word.get('image_path') and os.path.exists(word.get('image_path', '')))
            
            # Show details on available question types
            st.markdown(f"*{words_with_images} words have images for visual recognition questions*")
            
            # Start quiz button with dynamic label
            start_label = "🚀 Start Quiz" if len(filtered_vocab) >= 4 else f"Need {4-len(filtered_vocab)} More Word(s)"
            if st.button(start_label, disabled=len(filtered_vocab) < 4):
                if start_new_quiz(filtered_vocab, num_questions):
                    st.rerun()
            
            # Show word preview 
            if st.checkbox("Preview Available Words"):
                # Create a simple table of words
                preview_data = []
                for word in filtered_vocab[:20]:  # Limit preview to 20 words
                    has_image = "✅" if (word.get('image_path') and os.path.exists(word.get('image_path', ''))) else "❌"
                    preview_data.append({
                        "Original": word.get('word_original', ''),
                        "Translation": word.get('word_translated', ''),
                        "Category": word.get('category', ''),
                        "Has Image": has_image
                    })
                
                st.dataframe(pd.DataFrame(preview_data))
                
                if len(filtered_vocab) > 20:
                    st.markdown(f"*...and {len(filtered_vocab) - 20} more words*")
        else:
            warning_message(f"No vocabulary words found with current filter. Go to Camera Mode to add words in {quiz_language}" +
                      (f" for the {category_filter} category" if category_filter != "All Categories" else "") + ".")
            
            # Show a specific message for empty vocabulary
            if not vocabulary:
                info_message("Start by learning some words in Camera Mode to build your vocabulary!")
            elif not any(word['language_translated'] == quiz_lang_code for word in vocabulary):
                info_message(f"You don't have any words in {quiz_language} yet. Try selecting a different language or add some new words.")
            else:
                info_message(f"No words found in the {category_filter} category. Try selecting 'All Categories' or add words in this category.")

elif app_mode == "Statistics":
    style_title("Learning Statistics")
    st.markdown("Track your progress and learning habits.")
    
    # Get session stats for the last 30 days
    stats = get_session_stats_direct(30)
    
    # Check if stats exist and have total_sessions
    if stats and stats.get('total_sessions'):
        # Display overall statistics
        st.subheader("Overall Statistics (Last 30 Days)")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Sessions", stats.get('total_sessions', 0) or 0)
        with col2:
            st.metric("Words Studied", stats.get('total_words_studied', 0) or 0)
        with col3:
            st.metric("Words Learned", stats.get('total_words_learned', 0) or 0)
        
        # Learning efficiency
        st.subheader("Learning Efficiency")
        col1, col2 = st.columns(2)
        
        with col1:
            avg_words = stats.get('avg_words_per_session', 0) or 0
            st.metric("Avg Words per Session", f"{avg_words:.1f}")
        
        with col2:
            avg_time = stats.get('avg_session_minutes', 0) or 0
            st.metric("Avg Session Length", f"{avg_time:.1f} min")
        
        # Vocabulary distribution by language
        st.subheader("Vocabulary by Language")
        
        # Get all vocabulary items
        vocabulary = get_all_vocabulary_direct()
        
        # Count words per language
        language_counts = {}
        for word in vocabulary:
            if word is None or 'language_translated' not in word:
                continue
                
            lang = word['language_translated']
            if lang in language_counts:
                language_counts[lang] += 1
            else:
                language_counts[lang] = 1
        
        # Convert language codes to names
        language_names = {}
        for name, code in languages.items():
            if code in language_counts:
                language_names[name] = language_counts[code]
        
        # Create chart data
        if language_names:
            chart_data = pd.DataFrame({
                'Language': list(language_names.keys()),
                'Word Count': list(language_names.values())
            })
            
            # Plot bar chart
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(chart_data['Language'], chart_data['Word Count'], color='skyblue')
            
            # Add count labels on top of bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{int(height)}', ha='center', va='bottom')
            
            ax.set_xlabel('Language')
            ax.set_ylabel('Number of Words')
            ax.set_title('Vocabulary Distribution by Language')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            st.pyplot(fig)
        
        # Get proficiency distribution
        st.subheader("Proficiency Level Distribution")
        
        proficiency_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        for word in vocabulary:
            if word is None:
                continue
            level = word.get('proficiency_level', 0) or 0
            proficiency_counts[level] += 1
        
        # Create proficiency chart
        prof_data = pd.DataFrame({
            'Level': [f"Level {lvl}" for lvl in proficiency_counts.keys()],
            'Words': list(proficiency_counts.values())
        })
        
        if sum(proficiency_counts.values()) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = ['#FFCCCC', '#FFE5CC', '#FFFFCC', '#E5FFCC', '#CCFFCC', '#CCFFEF']
            bars = ax.bar(prof_data['Level'], prof_data['Words'], color=colors)
            
            # Add count labels
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                            f'{int(height)}', ha='center', va='bottom')
            
            ax.set_xlabel('Proficiency Level')
            ax.set_ylabel('Number of Words')
            ax.set_title('Word Distribution by Proficiency Level')
            plt.tight_layout()
            
            st.pyplot(fig)
            
            # Add explanation of proficiency levels
            st.markdown("""
            **Proficiency Level Guide:**
            - **Level 0**: New words or words answered incorrectly multiple times
            - **Level 1**: Basic recognition (20% correct answers)
            - **Level 2**: Beginning to remember (40% correct answers)
            - **Level 3**: Moderate proficiency (60% correct answers)
            - **Level 4**: Good proficiency (80% correct answers)
            - **Level 5**: Mastered (90-100% correct answers)
            """)
        
        # Learning suggestions section
        st.subheader("Learning Suggestions")
        st.markdown("""
        Based on your learning patterns, here are some suggestions:
        
        1. **Words to Review**: Focus on lower proficiency words
        2. **Optimal Session Length**: Aim for 10-15 minute learning sessions
        3. **Learning Frequency**: Try to complete at least one session per day
        """)
        
    else:
        info_message("No learning statistics available yet. Complete some learning sessions to see your progress!")
        
        if st.button("Generate Sample Statistics (Demo)"):
            # Create sample data for demonstration
            st.subheader("Sample Statistics (Demo)")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Sessions", 5)
            with col2:
                st.metric("Words Studied", 42)
            with col3:
                st.metric("Words Learned", 38)
                
            # Sample chart
            sample_data = pd.DataFrame({
                'Language': ['Spanish', 'French', 'German', 'Italian'],
                'Word Count': [15, 12, 8, 7]
            })
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(sample_data['Language'], sample_data['Word Count'], color='lightgray')
            ax.set_xlabel('Language')
            ax.set_ylabel('Number of Words (Sample Data)')
            ax.set_title('Example: Vocabulary Distribution by Language')
            plt.tight_layout()
            
            st.pyplot(fig)
            
            st.markdown("*This is sample data. Start learning with Camera Mode to begin tracking your real progress!*")
            
elif app_mode == "My Progress":
    try:
        sync_user_data_to_supabase()
    except:
        pass
    
    display_my_progress()

elif app_mode == "Pronunciation Practice":
    style_title("Pronunciation Practice")
    
    # Check if we have vocabulary
    vocabulary = get_all_vocabulary_direct()
    filtered_vocab = [word for word in vocabulary if word.get('language_translated') == st.session_state.target_language]
    
    if filtered_vocab:
        st.info("🎤 **Enhanced pronunciation practice is temporarily unavailable in the cloud version.**")
        st.markdown("### 📚 Basic Pronunciation Practice")
        st.markdown("You can still practice pronunciation using the audio features in:")
        st.markdown("- **My Vocabulary** - Listen to pronunciations of saved words")
        st.markdown("- **Camera Mode** - Hear pronunciations when learning new words")
        st.markdown("- **Quiz Mode** - Audio questions and pronunciation guides")

        # Add this to your pronunciation practice section
        st.markdown("### 🎯 Pronunciation Challenge")
        st.markdown("**How to practice:**")
        st.markdown("1. 🔊 Listen to the word pronunciation")
        st.markdown("2. 🗣️ Say the word out loud")
        st.markdown("3. 🔄 Listen again to compare")
        st.markdown("4. ✅ Mark if you got it right")

        practice_word = st.selectbox("Choose a word to practice:", 
                                    [f"{w['word_translated']} ({w['word_original']})" 
                                    for w in filtered_vocab])

        if practice_word:
            # Show pronunciation practice UI
            word_data = next(w for w in filtered_vocab if f"{w['word_translated']} ({w['word_original']})" == practice_word)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"### 🎯 Practice: **{word_data['word_translated']}**")
                audio_bytes = text_to_speech(word_data['word_translated'], st.session_state.target_language)
                if audio_bytes:
                    audio_html = get_audio_html(audio_bytes)
                    st.markdown(audio_html, unsafe_allow_html=True)
            
            with col2:
                if st.button("✅ I got it right!"):
                    st.success("Great job! 🎉")
                if st.button("🔄 Let me try again"):
                    st.info("Keep practicing! 💪")
        
        # Show basic word list with audio
        st.markdown("### 🔊 Your Vocabulary with Audio")
        for word in filtered_vocab[:10]:  # Show first 10 words
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.write(f"**{word.get('word_original', '')}** → {word.get('word_translated', '')}")
            
            with col2:
                # Add audio
                try:
                    audio_bytes = text_to_speech(word.get('word_translated', ''), st.session_state.target_language)
                    if audio_bytes:
                        audio_html = get_audio_html(audio_bytes)
                        st.markdown(audio_html, unsafe_allow_html=True)
                except:
                    st.write("🔇")
    else:
        warning_message("No vocabulary words found. Go to Camera Mode to add words first.")


try:
    vocabulary = get_all_vocabulary_direct()
    actual_word_count = len(vocabulary) if vocabulary else 0
    
    # Update session state to match reality
    st.session_state.words_learned = actual_word_count
    st.session_state.total_words_learned = actual_word_count
    
    if actual_word_count > 0:
        st.sidebar.success(f"Session active")
        st.sidebar.info(f"Words studied: {actual_word_count}")
        st.sidebar.info(f"Words learned: {actual_word_count}")
    else:
        st.sidebar.warning("No vocabulary yet")
        st.sidebar.markdown("*Start learning in Camera Mode*")
except Exception as e:
    st.sidebar.error("Error loading session data")
    print(f"Session data error: {e}")

# Get actual vocabulary count for display
vocabulary = get_all_vocabulary_direct()
actual_word_count = len(vocabulary) if vocabulary else 0

# Always show as active if user has vocabulary
if actual_word_count > 0:
    st.sidebar.success(f"Session active")
    st.sidebar.info(f"Words studied: {actual_word_count}")
    st.sidebar.info(f"Words learned: {actual_word_count}")
else:
    st.sidebar.warning("No vocabulary yet")
    st.sidebar.markdown("*Start learning in Camera Mode*")


add_footer()
