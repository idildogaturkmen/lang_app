import streamlit as st
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import base64
import time
import sqlite3
from datetime import datetime
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
    """Get the current authenticated user from Supabase."""
    if 'authenticated_user' not in st.session_state:
        # Get URL parameters
        params = get_url_params()
        
        # Check for Supabase authentication parameters
        auth_token = params.get('auth_token', [None])[0]
        auth_provider = params.get('auth_provider', [None])[0]
        user_email = params.get('user_email', [None])[0]
        user_id = params.get('user_id', [None])[0]
        
        print(f"🔍 Auth params - Token: {auth_token is not None}, Provider: {auth_provider}, Email: {user_email}, ID: {user_id}")
        
        if auth_token and auth_provider == 'supabase' and user_email and user_id:
            # Create user data from Supabase parameters
            user_data = {
                'id': user_id,
                'email': user_email,
                'username': user_email.split('@')[0] if user_email else 'user',
                'displayName': user_email.split('@')[0] if user_email else 'User',
                'auth_token': auth_token,
                'timestamp': datetime.now().timestamp() * 1000
            }
            st.session_state.authenticated_user = user_data
            print(f"✅ User authenticated: {user_email} with ID: {user_id}")
        else:
            print(f"❌ Authentication failed - missing params")
            st.session_state.authenticated_user = None
    
    return st.session_state.authenticated_user

def require_authentication():
    """Require user authentication to access the app."""
    user = get_authenticated_user()
    
    if not user:
        st.error("🔒 Authentication Required")
        st.info("Please log in through the main website to access Vocam.")
        st.markdown("**[← Login Here](https://vocam.app/web)**")
        
        # Show a simple demo mode option
        st.markdown("---")
        st.markdown("### Demo Mode")
        if st.button("Continue as Demo User"):
            # Set demo user data
            demo_user = {
                'id': 'demo_user_999',
                'username': 'demo',
                'displayName': 'Demo User',
                'email': 'demo@vocam.app',
                'timestamp': datetime.now().timestamp() * 1000
            }
            st.session_state.authenticated_user = demo_user
            st.rerun()
        
        st.stop()
    
    return user

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
    
def clear_user_session_data():
    """Clear all session state data for a clean user experience."""
    user = get_authenticated_user()
    if not user:
        return
    
    # Get current user identifier
    current_user_id = user.get('id', user.get('email', 'unknown'))
    
    # Check if this is a different user than last time
    if 'last_user_id' not in st.session_state or st.session_state.last_user_id != current_user_id:
        print(f"🔄 New user detected: {current_user_id}")
        print(f"🧹 Clearing session data...")
        
        # List of all session state keys to clear
        keys_to_clear = [
            'level', 'points', 'streak_days', 'daily_challenges', 'word_of_the_day',
            'achievements', 'badges', 'quiz_score', 'quiz_total', 'words_studied',
            'words_learned', 'user_progress', 'gamification_data', 'learning_stats',
            'vocabulary_tree', 'category_progress', 'total_words_learned'
        ]
        
        # Clear specific gamification keys
        for key in list(st.session_state.keys()):
            if any(x in key.lower() for x in ['gamification', 'achievement', 'badge', 'progress', 'level', 'point']):
                del st.session_state[key]
                print(f"🗑️ Cleared: {key}")
        
        # Reset core learning variables
        st.session_state.level = 1
        st.session_state.points = 0
        st.session_state.streak_days = 0
        st.session_state.daily_challenges = []
        st.session_state.word_of_the_day = None
        st.session_state.words_studied = 0
        st.session_state.words_learned = 0
        st.session_state.quiz_score = 0
        st.session_state.quiz_total = 0
        
        # Mark this user as the current one
        st.session_state.last_user_id = current_user_id
        
        print(f"✅ Session data cleared for user: {current_user_id}")

user = require_authentication()
clear_user_session_data()

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

# First, display Python version for
st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    image.save(img_byte_arr, format='JPEG', quality=70)  # Lower quality for hash stability
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
    """Detect text in image using Google Vision API with EasyOCR fallback."""
    try:
        # Method 1: Try Google Vision API first
        try:
            from google.cloud import vision
            import io
            
            # Initialize the client
            client = vision.ImageAnnotatorClient()
            
            # Convert PIL image to bytes
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='JPEG')
            img_byte_arr = img_byte_arr.getvalue()
            
            # Create vision image object
            vision_image = vision.Image(content=img_byte_arr)
            
            # Perform text detection
            response = client.text_detection(image=vision_image)
            texts = response.text_annotations
            
            if texts:
                # Return the first (most comprehensive) text detection
                detected_text = texts[0].description
                print(f"✅ Google Vision detected: {detected_text}")
                return detected_text
            
        except ImportError:
            print("Google Vision API not available, trying EasyOCR...")
        except Exception as e:
            print(f"Google Vision API failed: {e}, trying EasyOCR...")
        
        # Method 2: Fallback to EasyOCR
        try:
            import easyocr
            
            # Initialize EasyOCR reader
            reader = easyocr.Reader(['en'], gpu=False)
            
            # Convert PIL image to numpy array
            import numpy as np
            img_array = np.array(image)
            
            # Use EasyOCR to detect text
            results = reader.readtext(img_array)
            
            # Extract text from results
            detected_texts = []
            for (bbox, text, confidence) in results:
                if confidence > 0.5:  # Only include confident detections
                    detected_texts.append(text)
            
            if detected_texts:
                combined_text = ' '.join(detected_texts)
                print(f"✅ EasyOCR detected: {combined_text}")
                return combined_text
                
        except ImportError:
            print("EasyOCR not available")
        except Exception as e:
            print(f"EasyOCR failed: {e}")
        
        # Method 3: Fallback message
        return "Text detection requires Google Vision API or EasyOCR. Please set up the necessary credentials or install EasyOCR."
        
    except Exception as e:
        return f"Text detection error: {str(e)}"
    
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
    """Get gamification instance with Supabase data - FIXED VERSION."""
    user = get_authenticated_user()
    if not user:
        return get_gamification()
    
    # Get actual vocabulary count from Supabase
    actual_vocabulary = get_all_vocabulary_direct()
    actual_word_count = len(actual_vocabulary)
    
    print(f"🔍 Actual vocabulary count from Supabase: {actual_word_count}")
    
    # FORCE UPDATE session state with actual data
    st.session_state.words_learned = actual_word_count
    st.session_state.total_words_learned = actual_word_count
    
    # Calculate proper level and points
    st.session_state.level = max(1, actual_word_count // 10 + 1)
    st.session_state.points = actual_word_count * 10
    
    # Initialize/update streak (you can calculate this from session data)
    if 'streak_days' not in st.session_state:
        st.session_state.streak_days = 1 if actual_word_count > 0 else 0
    
    # Create fresh gamification instance
    fresh_gamification = get_gamification()
    fresh_gamification.initialize_state()
    
    # FORCE UPDATE the gamification system with real vocabulary data
    fresh_gamification.actual_vocabulary = actual_vocabulary
    fresh_gamification.actual_word_count = actual_word_count
    
    # Update category progress with real data
    fresh_gamification.update_category_progress_with_real_data(actual_vocabulary)
    
    # Check and update achievements based on real data
    fresh_gamification.check_real_achievements(actual_vocabulary, actual_word_count)
    
    return fresh_gamification

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
    """Save image to private Supabase Storage with detailed debugging."""
    try:
        import requests
        import io
        import uuid
        from datetime import datetime
        
        print(f"🔄 Starting Supabase image upload for: {label}")
        
        user = get_authenticated_user()
        if not user:
            print("❌ No authenticated user for image upload")
            return None
        
        user_id = user.get('id')
        auth_token = user.get('auth_token')
        
        print(f"👤 User ID: {user_id}")
        print(f"🔑 Auth token exists: {bool(auth_token)}")
        
        # Create a unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        filename = f"{user_id}/{label}_{timestamp}_{unique_id}.jpg"
        
        print(f"📁 Target filename: {filename}")
        
        # Process image (crop if bbox provided)
        processed_image = image
        if detection_bbox:
            print(f"✂️ Cropping image with bbox: {detection_bbox}")
            left, top, right, bottom = [int(x) for x in detection_bbox]
            img_array = np.array(image)
            
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
            processed_image = Image.fromarray(cropped_img)
            print(f"✅ Image cropped to size: {processed_image.size}")
        
        # Convert image to bytes
        img_bytes = io.BytesIO()
        if processed_image.width > 800 or processed_image.height > 800:
            processed_image.thumbnail((800, 800), Image.Resampling.LANCZOS)
            print(f"📏 Image resized to: {processed_image.size}")
        
        processed_image.save(img_bytes, format='JPEG', quality=85, optimize=True)
        img_bytes.seek(0)
        file_size = len(img_bytes.getvalue())
        print(f"💾 Image size: {file_size} bytes")
        
        # Supabase configuration
        supabase_url = "https://csszlzpsfwmsezursivk.supabase.co"
        supabase_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImNzc3psenBzZndtc2V6dXJzaXZrIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTA1Mjg1MjEsImV4cCI6MjA2NjEwNDUyMX0.gIi0Q_pifYpXeM1r8kWlgTO1LD8bc91lQ3suH8OWDKI"
        
        # Test different header combinations
        header_combinations = [
            # Method 1: With auth token
            {
                'apikey': supabase_key,
                'Authorization': f'Bearer {auth_token}' if auth_token else f'Bearer {supabase_key}',
                'Content-Type': 'image/jpeg',
            },
            # Method 2: Service role only
            {
                'apikey': supabase_key,
                'Authorization': f'Bearer {supabase_key}',
                'Content-Type': 'image/jpeg',
            },
            # Method 3: With x-upsert header
            {
                'apikey': supabase_key,
                'Authorization': f'Bearer {supabase_key}',
                'Content-Type': 'image/jpeg',
                'x-upsert': 'true',
            }
        ]
        
        upload_url = f"{supabase_url}/storage/v1/object/vocabulary-images/{filename}"
        print(f"🌐 Upload URL: {upload_url}")
        
        # Try each header combination
        for i, headers in enumerate(header_combinations):
            print(f"🔄 Trying upload method {i+1}/3")
            print(f"📋 Headers: {list(headers.keys())}")
            
            try:
                response = requests.post(
                    upload_url,
                    headers=headers,
                    data=img_bytes.getvalue(),
                    timeout=30
                )
                
                print(f"📤 Response status: {response.status_code}")
                print(f"📤 Response headers: {dict(response.headers)}")
                print(f"📤 Response text: {response.text[:200]}...")
                
                if response.status_code in [200, 201]:
                    storage_path = f"vocabulary-images/{filename}"
                    print(f"✅ Upload successful! Path: {storage_path}")
                    return storage_path
                elif response.status_code == 409:
                    print("⚠️ File already exists, but that's okay")
                    storage_path = f"vocabulary-images/{filename}"
                    return storage_path
                else:
                    print(f"❌ Upload failed with status {response.status_code}")
                    continue
                    
            except Exception as e:
                print(f"❌ Upload attempt {i+1} failed: {e}")
                continue
        
        print("❌ All upload methods failed")
        return None
        
    except Exception as e:
        print(f"❌ Critical error in save_image_to_supabase: {e}")
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
    """Save image with cropping support - tries Supabase first, falls back to local."""
    # Try Supabase Storage first (private)
    supabase_path = save_image_to_supabase(image, label, detection_bbox)
    if supabase_path:
        return supabase_path
    
    # Fallback to local storage with cropping
    try:
        img_array = np.array(image)
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
            
            print(f"✅ Saved images: {original_filename} and {cropped_filename}")
            return cropped_filename  # Return cropped version as primary
        else:
            print(f"✅ Saved original image: {original_filename}")
            return original_filename
            
    except Exception as e:
        error_message(f"Error saving image: {e}")
        return None

def get_signed_image_url(storage_path, expires_in=3600):
    """Get a signed URL for private image access."""
    try:
        import requests
        
        user = get_authenticated_user()
        if not user:
            return None
        
        auth_token = user.get('auth_token')
        if not auth_token:
            return None
        
        supabase_url = "https://csszlzpsfwmsezursivk.supabase.co"
        
        headers = {
            'Authorization': f'Bearer {auth_token}',
            'Content-Type': 'application/json',
        }
        
        # Fix: Correct endpoint for creating signed URLs
        signed_url_endpoint = f"{supabase_url}/storage/v1/object/sign/{storage_path}"
        
        data = {
            'expiresIn': expires_in
        }
        
        response = requests.post(
            signed_url_endpoint,
            headers=headers,
            json=data
        )
        
        if response.status_code == 200:
            result = response.json()
            # Fix: Get the correct signed URL
            signed_url = result.get('signedURL')
            if signed_url:
                return f"{supabase_url}/storage/v1{signed_url}"
        
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
        if image_path.startswith('vocabulary-images/'):
            # Supabase Storage path - get signed URL
            signed_url = get_signed_image_url(image_path)
            if signed_url:
                st.image(signed_url, caption=f"📷 {word_original}", width=300)
                return True
            else:
                st.markdown("*Image temporarily unavailable*")
                return False
        elif image_path.startswith('http'):
            # Legacy public URL (for old images)
            st.image(image_path, caption=f"📷 {word_original}", width=300)
            return True
        elif os.path.exists(image_path):
            # Local file (fallback)
            image = Image.open(image_path)
            st.image(image, caption=f"📷 {word_original}", width=300)
            return True
        else:
            return False
    except Exception as e:
        print(f"Error displaying image: {e}")
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
                show_loading_spinner("Detecting objects with Google AI... This may take a few seconds.")
            
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
                    if st.button("💾 Save Selected Objects to Vocabulary", key=save_button_id):
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
                            duplicate_count = 0
                            duplicate_items = []
                            
                            for i in selected_objects:
                                try:
                                    detection = detections[i]
                                    label = detection['label']
                                    translated_label = translate_text(label, st.session_state.target_language)
                                    
                                    # Save the image with bounding box info for cropping
                                    print(f"🔄 Saving image for label: {label}")
                                    image_path = save_image(image, label, detection['bbox'])
                                    print(f"📁 Image saved to: {image_path}")

                                    if image_path and image_path.startswith('vocabulary-images/'):
                                        print(f"✅ Image successfully saved to Supabase: {image_path}")
                                    elif image_path:
                                        print(f"⚠️ Image saved locally (Supabase failed): {image_path}")
                                    else:
                                        print(f"❌ Image save failed completely")
                                    
                                    # Get object category
                                    category = get_object_category(label)
                                    
                                    # Add to database using direct method
                                    vocab_id = add_vocabulary_direct(
                                        word_original=label,
                                        word_translated=translated_label,
                                        language_translated=st.session_state.target_language,
                                        category=category,
                                        image_path=image_path
                                    )
                                    
                                    if vocab_id == 'duplicate':
                                        duplicate_count += 1
                                        duplicate_items.append(f"{label} → {translated_label}")
                                    elif vocab_id:
                                        saved_count += 1
                                        saved_items.append(f"{label} → {translated_label}")
                                        # Update session stats
                                        st.session_state.words_studied += 1
                                        st.session_state.words_learned += 1
                                    else:
                                        error_message(f"Failed to save {label} to vocabulary.")
                                except Exception as e:
                                    error_message(f"Error saving {label}: {str(e)}")
                            
                            # Update success message to handle duplicates
                            if saved_count > 0 or duplicate_count > 0:
                                # Store the saved state and items in session state
                                st.session_state.words_just_saved = True
                                st.session_state.saved_count = saved_count
                                st.session_state.saved_items = saved_items
                                st.session_state.duplicate_count = duplicate_count
                                st.session_state.duplicate_items = duplicate_items
                                st.rerun()  # Rerun once to update the UI
                            else:
                                error_message("Failed to save any words. Please check database connection.")

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
                
                # Clear the spinner
                spinner_container.empty()
                
                # Add scroll indicator for mobile
                add_scroll_indicator()
                
                # FIX: Only process actual detected text, not error messages
                if detected_text and isinstance(detected_text, str) and len(detected_text.strip()) > 0:
                    # Additional validation - make sure it's not an error message
                    error_indicators = [
                        "no clear text", "text detection", "error", "failed", "try using",
                        "not detected", "please try", "install", "requires", "unavailable"
                    ]
                    
                    is_error_message = any(indicator in detected_text.lower() for indicator in error_indicators)
                    
                    if not is_error_message:
                        style_section_title("📝 Detected Text")
                        st.success(f"Found text: **{detected_text}**")
                            
                        # Split into meaningful words for learning (filter out very short words)
                        import re
                        words = [word.strip() for word in re.split(r'[^\w]', detected_text) 
                                if word.strip() and len(word.strip()) > 2]
                            
                        if words:
                            st.subheader("Words to Learn")
                                
                            # Create containers for each word
                            for i, word in enumerate(words):
                                # Skip very common words that aren't useful for learning
                                skip_words = {'the', 'and', 'this', 'that', 'with', 'for', 'are', 'was', 'were', 'been'}
                                if word.lower() in skip_words:
                                    continue
                                        
                                # Translate the word
                                translated_word = translate_text(word, st.session_state.target_language)
                                    
                                # Display in a container
                                with st.container():
                                    cols = st.columns([3, 1])
                                
                                    with cols[0]:
                                        st.write(f"**{word}** → {translated_word}")
                                        # Add audio
                                        audio_bytes = text_to_speech(translated_word, st.session_state.target_language)
                                        if audio_bytes:
                                            st.markdown(get_audio_html(audio_bytes), unsafe_allow_html=True)
                                        
                                    with cols[1]:
                                        # Add save button for each word
                                        if st.button(f"Save", key=f"save_text_{i}"):
                                            # Auto-start session if needed
                                            if st.session_state.session_id is None:
                                                manage_session("start")
                                                
                                            # Save to vocabulary
                                            vocab_id = add_vocabulary_direct(
                                                word_original=word,
                                                word_translated=translated_word,
                                                language_translated=st.session_state.target_language,
                                                category="text",
                                                image_path=None
                                            )
                                                
                                            if vocab_id and vocab_id != 'duplicate':
                                                success_message(f"Added '{word}' to vocabulary!")
                                                st.session_state.words_studied += 1
                                                st.session_state.words_learned += 1
                                            elif vocab_id == 'duplicate':
                                                warning_message(f"'{word}' is already in your vocabulary!")
                                            else:
                                                error_message(f"Failed to save '{word}'")
                                        
                                    st.markdown("---")
                        else:
                            info_message("The detected text doesn't contain meaningful words to learn.")
                    else:
                        # It's an error message, so show failure
                        warning_message("No clear text was detected in this image.")
                        st.info("💡 **Tips for better text detection:**")
                        st.markdown("""
                        - Use images with **large, clear text**
                        - Ensure **good lighting** and contrast
                        - Try **zooming in** on the text
                        - Use **simple fonts** (avoid decorative text)
                        - Make sure text is **horizontal** (not rotated)
                        """)
                else:
                    # No text detected at all
                    warning_message("No text was detected in this image.")
                    st.info("💡 **Try these tips:**")
                    st.markdown("""
                    - Take a photo with **clear, readable text**
                    - Ensure the text is **well-lit** and **in focus**
                    - Try **getting closer** to the text
                    - Use **high contrast** text (dark text on light background or vice versa)
                    - Or switch to **Object Detection** mode to learn vocabulary from objects
                    """)

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
    style_title("My Progress")
    
    try:
        # Force fresh user data
        user = get_authenticated_user()
        
        # Get user's actual vocabulary count from Supabase
        actual_vocabulary = get_all_vocabulary_direct()
        actual_word_count = len(actual_vocabulary)
        
        print(f"📊 Progress page - vocabulary count: {actual_word_count}")
        
        # FORCE UPDATE session state with actual data BEFORE creating gamification
        st.session_state.words_learned = actual_word_count
        st.session_state.total_words_learned = actual_word_count
        
        # Calculate proper level and points
        st.session_state.level = max(1, actual_word_count // 10 + 1)
        st.session_state.points = actual_word_count * 10
        
        # Initialize/update streak
        if 'streak_days' not in st.session_state:
            st.session_state.streak_days = 1 if actual_word_count > 0 else 0
        
        # Get or create gamification instance with forced refresh
        user_gamification = get_user_scoped_gamification()
        
        # Use the full dashboard
        if actual_word_count == 0:
            st.info("🌱 Start learning words in Camera Mode to see your progress!")
            
            # Show minimal progress for new users
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Level", "1")
            with col2:
                st.metric("Words Learned", "0")
            with col3:
                st.metric("Points", "0")
        else:
            # Show main metrics first (guaranteed to show real data)
            st.markdown("### 🏆 Your Learning Progress")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Level", st.session_state.level)
            with col2:
                st.metric("Words Learned", actual_word_count)
            with col3:
                st.metric("Points", st.session_state.points)
            with col4:
                st.metric("Streak", f"{st.session_state.streak_days} days")
            
            # Progress to next level
            current_level_min = (st.session_state.level - 1) * 10
            next_level_min = st.session_state.level * 10
            progress = min((actual_word_count - current_level_min) / 10, 1.0)
            
            st.markdown("### 📈 Progress to Next Level")
            st.progress(progress)
            st.markdown(f"**{actual_word_count}/{next_level_min} words** to reach Level {st.session_state.level + 1}")
            
            # Show achievements
            st.markdown("### 🏅 Your Achievements")
            
            if st.session_state.get('achievements'):
                # Display achievements in a nice grid
                achievement_cols = st.columns(min(3, len(st.session_state.achievements)))
                
                for i, achievement in enumerate(st.session_state.achievements[-6:]):  # Show last 6
                    col_idx = i % len(achievement_cols)
                    with achievement_cols[col_idx]:
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                    color: white; padding: 15px; border-radius: 10px; margin: 5px; text-align: center;">
                            <div style="font-size: 2em;">{achievement['icon']}</div>
                            <div style="font-weight: bold; margin: 5px 0;">{achievement['name']}</div>
                            <div style="font-size: 0.9em; opacity: 0.9;">{achievement['description']}</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                st.markdown(f"**Total achievements unlocked: {len(st.session_state.achievements)}**")
            else:
                st.info("Keep learning to unlock achievements!")
            
            # Language breakdown
            if actual_vocabulary:
                st.markdown("### 🌍 Learning Progress by Language")
                
                language_stats = {}
                for word in actual_vocabulary:
                    lang = word.get('language_translated', 'unknown')
                    lang_name = next((k for k, v in languages.items() if v == lang), lang)
                    language_stats[lang_name] = language_stats.get(lang_name, 0) + 1
                
                # Create a nice progress display for each language
                for lang, count in language_stats.items():
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"**{lang}**")
                        # Create a progress bar based on words learned
                        progress = min(count / 20, 1.0)  # 20 words = 100%
                        st.progress(progress)
                    with col2:
                        st.markdown(f"**{count} words**")
            
            # Try to show the full gamification dashboard as additional content
            try:
                st.markdown("---")
                st.markdown("### 📊 Detailed Progress")
                user_gamification.render_dashboard()
            except Exception as e:
                print(f"Dashboard error (non-critical): {e}")
                # Don't show error to user since we already have the main content above
                pass
                
    except Exception as e:
        error_message("There was an error displaying progress.")
        print(f"Progress error: {e}")
        
        # Fallback display
        st.markdown("### 🏆 Your Learning Progress")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Level", "1")
        with col2:
            st.metric("Words Learned", "0")
        with col3:
            st.metric("Points", "0")

elif app_mode == "Pronunciation Practice":
    style_title("Pronunciation Practice")
    st.markdown("Practice your pronunciation with real-time AI feedback and comprehensive analysis.")

    # Session management
    col1, col2 = st.columns(2)
    with col1:
        if st.session_state.session_id is None:
            if st.button("Start Learning Session", key="start_pron_session"):
                if manage_session("start"):
                    st.rerun()
        else:
            info_message(f"Session in progress - Words studied: {st.session_state.words_studied}")
    with col2:
        if st.session_state.session_id is not None:
            if st.button("End Session", key="end_pron_session"):
                if manage_session("end"):
                    st.rerun()
    
    if has_pronunciation_practice:
        try:
            # Initialize pronunciation practice if not already initialized
            if 'pronunciation_practice' not in st.session_state:
                # Initialize the enhanced pronunciation practice module
                st.session_state.pronunciation_practice = create_pronunciation_practice(
                    text_to_speech_func=text_to_speech, 
                    get_audio_html_func=get_audio_html,
                    translate_text_func=translate_text,
                    get_example_sentence_func=get_example_sentence
                )
                print("✅ Enhanced pronunciation practice initialized with AI feedback")

                # Add pronunciation practice capabilities to session state
                st.session_state.pronunciation_capabilities = {
                    'realtime_feedback': True,
                    'ai_analysis': True,
                    'visual_feedback': True,
                    'progress_tracking': True
                }
            
            # Check for custom recorder availability
            try:
                from custom_audio_recorder import audio_recorder
                st.session_state.pronunciation_practice.has_custom_recorder = True
                print("✅ Custom audio recorder available")
            except ImportError:
                st.session_state.pronunciation_practice.has_custom_recorder = False
                print("ℹ️ Using fallback recording methods")
            # Enhanced pronunciation practice interface
            st.markdown("""
            ### 🎯 Features Available:
            - **Real-time feedback** during recording
            - **AI-powered analysis** of your pronunciation 
            - **Visual spectrograms** showing sound patterns
            - **Progress tracking** across practice sessions
            - **Language-specific tips** for difficult sounds
            """)
            
            # Get vocabulary from database
            vocabulary = get_all_vocabulary_direct()
            
            # Language selection
            practice_language = st.selectbox(
                "Select practice language:",
                list(languages.keys()),
                index=list(languages.values()).index(st.session_state.target_language) 
                    if st.session_state.target_language in languages.values() else 0,
                key="pron_lang_select"
            )
            practice_language_code = languages[practice_language]
            
            # Filter vocabulary for the selected language
            filtered_vocab = [word for word in vocabulary if word['language_translated'] == practice_language_code]
            
            if filtered_vocab:
                # Practice mode selection
                practice_mode = st.radio(
                    "Choose practice mode:",
                    [
                        "📚 Individual Word Practice", 
                        "🎯 Focused Practice Session",
                        "🏆 Challenge Mode"
                    ],
                    key="practice_mode_select"
                )
                
                if practice_mode == "📚 Individual Word Practice":
                    # Individual word practice
                    word_index = st.selectbox(
                        "Select a word to practice:",
                        range(len(filtered_vocab)),
                        format_func=lambda i: f"{filtered_vocab[i].get('word_translated', '')} ({filtered_vocab[i].get('word_original', '')})",
                        key="word_select"
                    )
                    
                    selected_word = filtered_vocab[word_index]
                    st.session_state.pronunciation_practice.render_practice_ui(selected_word)
                
                elif practice_mode == "🎯 Focused Practice Session":
                    # Focused session mode
                    if 'practice_session_words' not in st.session_state:
                        session_size = st.slider("Number of words to practice:", 3, 10, 5)
                        
                        if st.button("🚀 Start Focused Session", type="primary"):
                            import random
                            st.session_state.practice_session_words = random.sample(
                                filtered_vocab, min(session_size, len(filtered_vocab))
                            )
                            st.session_state.current_session_index = 0
                            st.session_state.session_scores = []
                            st.rerun()
                    else:
                        # Session in progress
                        current_index = st.session_state.current_session_index
                        total_words = len(st.session_state.practice_session_words)
                        
                        # Progress display
                        progress = current_index / total_words
                        st.progress(progress)
                        st.markdown(f"**Word {current_index + 1} of {total_words}**")
                        
                        if current_index < total_words:
                            current_word = st.session_state.practice_session_words[current_index]
                            st.session_state.pronunciation_practice.render_practice_ui(current_word)
                            
                            # Session navigation
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                if current_index > 0:
                                    if st.button("⬅️ Previous Word"):
                                        st.session_state.current_session_index -= 1
                                        st.rerun()
                            
                            with col2:
                                if st.button("⏭️ Skip Word"):
                                    st.session_state.current_session_index += 1
                                    st.rerun()
                            
                            with col3:
                                if current_index < total_words - 1:
                                    if st.button("➡️ Next Word"):
                                        st.session_state.current_session_index += 1
                                        st.rerun()
                        else:
                            # Session completed
                            st.success("🎉 Practice session completed!")
                            
                            # Show session results
                            if 'session_scores' in st.session_state and st.session_state.session_scores:
                                avg_score = sum(st.session_state.session_scores) / len(st.session_state.session_scores)
                                st.metric("Average Score", f"{avg_score:.0f}%")
                                
                                # Progress chart
                                fig, ax = plt.subplots(figsize=(8, 4))
                                ax.plot(range(1, len(st.session_state.session_scores) + 1), 
                                       st.session_state.session_scores, 
                                       marker='o', linestyle='-')
                                ax.set_xlabel('Word Number')
                                ax.set_ylabel('Score (%)')
                                ax.set_title('Session Progress')
                                ax.grid(True, alpha=0.3)
                                st.pyplot(fig)
                            
                            # Reset session
                            if st.button("🔄 Start New Session"):
                                for key in ['practice_session_words', 'current_session_index', 'session_scores']:
                                    if key in st.session_state:
                                        del st.session_state[key]
                                st.rerun()
                
                elif practice_mode == "🏆 Challenge Mode":
                    # Challenge mode - rapid pronunciation assessment
                    st.markdown("### 🏆 Pronunciation Challenge")
                    st.markdown("Quick-fire pronunciation assessment - get scored on speed and accuracy!")
                    
                    if 'challenge_mode' not in st.session_state:
                        difficulty = st.selectbox(
                            "Select difficulty:",
                            ["🟢 Easy (3 words)", "🟡 Medium (5 words)", "🔴 Hard (8 words)"]
                        )
                        
                        word_count = {"🟢 Easy (3 words)": 3, "🟡 Medium (5 words)": 5, "🔴 Hard (8 words)": 8}[difficulty]
                        
                        if st.button("🚀 Start Challenge!", type="primary"):
                            import random
                            st.session_state.challenge_words = random.sample(
                                filtered_vocab, min(word_count, len(filtered_vocab))
                            )
                            st.session_state.challenge_index = 0
                            st.session_state.challenge_scores = []
                            st.session_state.challenge_start_time = time.time()
                            st.session_state.challenge_mode = True
                            st.rerun()
                    else:
                        # Challenge in progress
                        current_index = st.session_state.challenge_index
                        total_words = len(st.session_state.challenge_words)
                        
                        if current_index < total_words:
                            # Timer display
                            elapsed_time = time.time() - st.session_state.challenge_start_time
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Time Elapsed", f"{elapsed_time:.1f}s")
                            with col2:
                                st.metric("Words Remaining", total_words - current_index)
                            
                            # Current word practice
                            current_word = st.session_state.challenge_words[current_index]
                            st.markdown(f"### Challenge Word: {current_word.get('word_translated', '')}")
                            
                            # Quick practice interface
                            st.session_state.pronunciation_practice.render_practice_ui(current_word)
                            
                        else:
                            # Challenge completed
                            total_time = time.time() - st.session_state.challenge_start_time
                            st.success(f"🏆 Challenge completed in {total_time:.1f} seconds!")
                            
                            # Challenge results
                            if st.session_state.challenge_scores:
                                avg_score = sum(st.session_state.challenge_scores) / len(st.session_state.challenge_scores)
                                speed_bonus = max(0, 100 - total_time)  # Bonus for speed
                                final_score = (avg_score * 0.8) + (speed_bonus * 0.2)
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Average Accuracy", f"{avg_score:.0f}%")
                                with col2:
                                    st.metric("Speed Bonus", f"{speed_bonus:.0f}")
                                with col3:
                                    st.metric("Final Score", f"{final_score:.0f}%")
                                
                                # Achievement badges
                                if final_score >= 90:
                                    st.markdown("🏆 **PRONUNCIATION MASTER!**")
                                elif final_score >= 80:
                                    st.markdown("🥇 **EXCELLENT PERFORMANCE!**")
                                elif final_score >= 70:
                                    st.markdown("🥈 **GREAT JOB!**")
                                else:
                                    st.markdown("🥉 **KEEP PRACTICING!**")
                            
                            # Reset challenge
                            if st.button("🔄 New Challenge"):
                                for key in ['challenge_words', 'challenge_index', 'challenge_scores', 
                                          'challenge_start_time', 'challenge_mode']:
                                    if key in st.session_state:
                                        del st.session_state[key]
                                st.rerun()
            else:
                warning_message(f"No vocabulary words found for {practice_language}. Go to Camera Mode to add words first.")
            
        except Exception as e:
            error_message(f"Error in pronunciation practice: {str(e)}")
            st.info("Try refreshing the page or check the pronunciation practice module.")
    else:
        # Show what's available vs what's missing
        st.warning("🎤 Some pronunciation features require additional packages.")
        
        # Basic pronunciation practice without advanced features
        st.markdown("### 🎯 Basic Pronunciation Practice")
        st.markdown("You can still practice pronunciation with the available features:")
        
        # Get vocabulary
        vocabulary = get_all_vocabulary_direct()
        practice_language = st.selectbox(
            "Select practice language:",
            list(languages.keys()),
            index=list(languages.values()).index(st.session_state.target_language) 
                if st.session_state.target_language in languages.values() else 0,
            key="basic_pron_lang_select"
        )
        practice_language_code = languages[practice_language]
        
        # Filter vocabulary
        filtered_vocab = [word for word in vocabulary if word['language_translated'] == practice_language_code]
        
        if filtered_vocab:
            word_index = st.selectbox(
                "Select a word to practice:",
                range(len(filtered_vocab)),
                format_func=lambda i: f"{filtered_vocab[i].get('word_translated', '')} ({filtered_vocab[i].get('word_original', '')})",
                key="basic_word_select"
            )
            
            selected_word = filtered_vocab[word_index]
            word_translated = selected_word.get('word_translated', '')
            
            st.markdown(f"### Practice: {word_translated}")
            
            # Play pronunciation
            st.markdown("**🔊 Listen and repeat:**")
            audio_bytes = text_to_speech(word_translated, practice_language_code)
            if audio_bytes:
                st.markdown(get_audio_html(audio_bytes), unsafe_allow_html=True)
            
            # Show pronunciation tips
            pronunciation_tips = get_pronunciation_guide(word_translated, practice_language_code)
            if pronunciation_tips:
                st.markdown("**💡 Pronunciation Tips:**")
                for tip in pronunciation_tips:
                    st.markdown(f"- {tip}")
            
            # File upload for basic feedback
            st.markdown("**📁 Upload your recording for basic analysis:**")
            uploaded_audio = st.file_uploader(
                "Record yourself saying the word and upload the audio file", 
                type=["wav", "mp3", "ogg", "m4a"],
                key="basic_audio_upload"
            )
            
            if uploaded_audio:
                # Basic analysis without advanced features
                st.audio(uploaded_audio)
                
                # Simple feedback
                st.markdown("### 📝 Basic Feedback")
                st.success("✅ Audio received! Keep practicing by:")
                st.markdown("- 🔄 Comparing your pronunciation with the correct audio")
                st.markdown("- 📚 Focusing on the pronunciation tips above")
                st.markdown("- 🎯 Recording multiple attempts to improve")
                
                # Save to vocabulary option
                if st.button("💾 Save to Vocabulary", key="basic_save_vocab"):
                    if st.session_state.session_id is None:
                        manage_session("start")
                    
                    vocab_id = add_vocabulary_direct(
                        word_original=selected_word.get('word_original', ''),
                        word_translated=word_translated,
                        language_translated=practice_language_code,
                        category="pronunciation_practice",
                        image_path=None
                    )
                    
                    if vocab_id:
                        st.success("✅ Word saved to vocabulary!")
                        st.session_state.words_studied += 1
                        st.session_state.words_learned += 1
        else:
            warning_message(f"No vocabulary words found for {practice_language}. Go to Camera Mode to add words first.")
        
        # Installation guide
        st.markdown("### 🛠️ For Advanced AI Feedback")
        st.markdown("Install these packages for real-time AI pronunciation analysis:")
        st.code("pip install streamlit-webrtc speech-recognition librosa python-Levenshtein av")

st.sidebar.markdown("---")
st.sidebar.markdown("### Session Info")
if st.session_state.session_id:
    st.sidebar.success(f"Session active")
    st.sidebar.info(f"Words studied: {st.session_state.words_studied}")
    st.sidebar.info(f"Words learned: {st.session_state.words_learned}")
else:
    st.sidebar.warning("No active session")
    st.sidebar.markdown("*Start a session in Camera Mode to track progress*")


add_footer()
