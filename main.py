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
from collections import defaultdict
import io
from vocam_ui import apply_custom_css
from streamlit.components.v1 import components
import hashlib
from example_sentences import ExampleSentenceGenerator
import requests
from deep_translator import GoogleTranslator
from ultralytics import YOLO
import json
from urllib.parse import parse_qs
from database import LanguageLearningDB

st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded"
)

# Supabase Configuration
SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://csszlzpsfwmsezursivk.supabase.co")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImNzc3psenBzZndtc2V6dXJzaXZrIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTA1Mjg1MjEsImV4cCI6MjA2NjEwNDUyMX0.gIi0Q_pifYpXeM1r8kWlgTO1LD8bc91lQ3suH8OWDKI")

def validate_environment():
    """Validate required environment variables are present."""
    required_vars = ['SUPABASE_URL', 'SUPABASE_ANON_KEY']
    missing_vars = []
    for var in required_vars:
        if not os.environ.get(var):
            missing_vars.append(var)
    
    if missing_vars:
        raise EnvironmentError(f"Missing required environment variables: {missing_vars}")
    
    return True

# Call this at startup
validate_environment()

# Authentication Functions
def get_url_params():
    """Get URL parameters from Streamlit."""
    try:
        query_params = st.experimental_get_query_params()
        return query_params
    except:
        return {}

def validate_supabase_token(token):
    """Validate Supabase access token and return user data."""
    try:
        import base64
        
        parts = token.split('.')
        if len(parts) != 3:
            print("Invalid token format")
            return None
        
        payload_encoded = parts[1]
        payload_encoded += '=' * (4 - len(payload_encoded) % 4)
        
        try:
            payload_json = base64.b64decode(payload_encoded).decode('utf-8')
            payload = json.loads(payload_json)
        except Exception as e:
            print(f"Error decoding payload: {e}")
            return None
        
        exp = payload.get('exp', 0)
        if exp < datetime.now().timestamp():
            print("Token has expired")
            return None
        
        user_data = {
            'id': payload.get('sub', ''),
            'email': payload.get('email', ''),
            'aud': payload.get('aud', ''),
            'role': payload.get('role', ''),
            'username': payload.get('email', '').split('@')[0] if payload.get('email') else 'user',
            'displayName': payload.get('email', 'User'),
            'timestamp': datetime.now().timestamp() * 1000
        }
        
        return user_data
        
    except Exception as e:
        print(f"Error validating Supabase token: {e}")
        return None

def get_authenticated_user():
    """Get the current authenticated user from Supabase."""
    if 'authenticated_user' not in st.session_state:
        params = get_url_params()
        auth_token = params.get('auth_token', [None])[0]
        auth_provider = params.get('auth_provider', [None])[0]
        user_email = params.get('user_email', [None])[0]
        user_id = params.get('user_id', [None])[0]
        
        if auth_token and auth_provider == 'supabase':
            user_data = validate_supabase_token(auth_token)
            if user_data:
                user_data['email'] = user_email or user_data.get('email', '')
                user_data['id'] = user_id or user_data.get('id', '')
                
                st.session_state.authenticated_user = user_data
                st.session_state.supabase_token = auth_token
            else:
                st.session_state.authenticated_user = None
        else:
            st.session_state.authenticated_user = None
    
    return st.session_state.authenticated_user

def require_authentication():
    """Require user authentication to access the app."""
    user = get_authenticated_user()
    
    if not user:
        st.error("🔒 Authentication Required")
        st.info("Please log in through the main website to access Vocam.")
        st.markdown("**[← Login Here](https://vocam.app/web)**")
        
        st.markdown("---")
        st.markdown("### Demo Mode (Development Only)")
        if st.button("Continue as Demo User"):
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

def get_user_database():
    """Get database instance for the authenticated user."""
    try:
        if 'user_db' not in st.session_state:
            print("🔄 Creating new database instance...")
            st.session_state.user_db = LanguageLearningDB("language_learning.db")
            print("✅ Database instance created successfully")
        else:
            print("✅ Using existing database instance")
        
        return st.session_state.user_db
        
    except Exception as e:
        print(f"❌ Error getting database: {e}")
        try:
            db = LanguageLearningDB("language_learning.db")
            st.session_state.user_db = db
            return db
        except Exception as e2:
            print(f"❌ Failed to create fallback database: {e2}")
            return None

def create_session_direct():
    """Create a session for the authenticated user."""
    try:
        user = get_authenticated_user()
        if not user:
            print("❌ No authenticated user found in create_session_direct")
            return None
        
        print(f"Creating session for user: {user.get('id', 'Unknown ID')}")
        
        db = get_user_database()
        if db is None:
            print("❌ Database is None")
            return None
        
        try:
            test_query = db.cursor.execute("SELECT 1").fetchone()
            print("✅ Database connection test successful")
        except Exception as db_error:
            print(f"❌ Database connection error: {db_error}")
            return None
        
        session_id = db.start_session(user['id'])
        print(f"Database returned session_id: {session_id}")
        
        return session_id
        
    except Exception as e:
        print(f"❌ Error in create_session_direct: {e}")
        import traceback
        traceback.print_exc()
        return None

def add_vocabulary_direct(word_original, word_translated, language_translated, category=None, image_path=None):
    """Add vocabulary for the authenticated user."""
    user = get_authenticated_user()
    if not user:
        return None
    
    db = get_user_database()
    vocab_id = db.add_vocabulary(
        user_id=user['id'],
        word_original=word_original,
        word_translated=word_translated,
        language_translated=language_translated,
        category=category,
        image_path=image_path
    )
    
    if vocab_id:
        try:
            gamification.check_achievements(
                "word_learned",
                word=word_original,
                category=category,
                language=language_translated
            )
        except Exception as e:
            print(f"Gamification error: {e}")
    
    return vocab_id

def get_all_vocabulary_direct():
    """Get all vocabulary for the authenticated user."""
    user = get_authenticated_user()
    if not user:
        return []
    
    db = get_user_database()
    vocabulary = db.get_all_vocabulary(user_id=user['id'])
    
    result = []
    for row in vocabulary:
        result.append(dict(row))
    
    return result

def get_session_stats_direct(days=30):
    """Get session statistics for the authenticated user."""
    user = get_authenticated_user()
    if not user:
        return {}
    
    db = get_user_database()
    stats = db.get_session_stats(user['id'], days)
    return dict(stats) if stats else {}

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

def manage_session(action):
    """Session management with Supabase user context."""
    try:
        user = get_authenticated_user()
        if not user:
            error_message("No authenticated user found")
            return False
            
        if action == "start":
            try:
                print(f"Starting session for user: {user.get('id', 'Unknown ID')}")
                
                session_id = create_session_direct()
                
                if session_id:
                    st.session_state.session_id = session_id
                    st.session_state.words_studied = 0
                    st.session_state.words_learned = 0
                    success_message("Started new learning session!")
                    print(f"✅ Session started successfully with ID: {session_id}")
                    return True
                else:
                    error_message("Failed to create session. Please check database connection.")
                    print("❌ create_session_direct() returned None")
                    return False
                    
            except Exception as e:
                error_message(f"Error starting session: {str(e)}")
                print(f"❌ Session start error: {e}")
                import traceback
                traceback.print_exc()
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
                print(f"❌ Session end error: {e}")
                return False
        
        return False
        
    except Exception as e:
        error_message(f"Session management error: {str(e)}")
        print(f"❌ Session management error: {e}")
        import traceback
        traceback.print_exc()
        return False

# Initialize the user and require authentication
user = require_authentication()

# Display user info in sidebar for debugging
st.sidebar.markdown("---")
st.sidebar.markdown("### User Info")
st.sidebar.markdown(f"**Email:** {user.get('email', 'Unknown')}")
st.sidebar.markdown(f"**User ID:** {user.get('id', 'Unknown')[:8]}...")

# Import dependencies with error handling
@st.cache_resource
def load_yolov8_nano():
    """Load ultra-lightweight YOLOv8 Nano - only 4MB!"""
    try:
        print("Loading YOLOv8 Nano detector...")
        model = YOLO('yolov8n.pt')
        model.to('cpu')
        print("✅ YOLOv8 Nano loaded successfully!")
        return model
    except Exception as e:
        print(f"❌ Error loading YOLOv8: {e}")
        return None

def detect_objects_yolov8(image, confidence_threshold=0.5):
    """YOLOv8-based object detection with better error handling"""
    try:
        model = load_yolov8_nano()
        if model is None:
            return [], np.array(image)
        
        if hasattr(image, 'convert'):
            image = image.convert('RGB')
            image_np = np.array(image)
        else:
            image_np = np.array(image)
        
        results = model.predict(image_np, conf=confidence_threshold, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    confidence = float(box.conf[0].cpu().numpy())
                    class_name = model.names[class_id]
                    
                    detections.append({
                        'label': class_name.lower(),
                        'confidence': confidence,
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'class_id': class_id
                    })
        
        result_image = draw_detections(image_np, detections)
        
        return detections, result_image
        
    except Exception as e:
        print(f"YOLOv8 detection error: {e}")
        return [], np.array(image)

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

# Import dependencies with fallbacks
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
    has_transformers = True
    print("✅ Transformers loaded successfully")
except ImportError as e:
    has_transformers = False
    print(f"❌ Transformers not available: {e}")
    
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

try:
    import pytesseract
    has_tesseract = True
except ImportError as e:
    has_tesseract = False
    class DummyTesseract:
        def image_to_string(self, *args, **kwargs):
            return "OCR requires pytesseract. Install with: pip install pytesseract"
    pytesseract = DummyTesseract()

try:
    import cv2
except ImportError as e:
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
    
    cv2 = DummyCV2()

try:
    from gtts import gTTS
except ImportError as e:
    class DummyGTTS:
        def __init__(self, text="", lang="en", slow=False):
            self.text = text
            self.lang = lang
            
        def write_to_fp(self, fp):
            fp.write(b'dummy audio data')
    
    gTTS = DummyGTTS

try:
    from custom_audio_recorder import audio_recorder
    has_custom_recorder = True
    print("Custom audio recorder imported successfully")
except ImportError as e:
    has_custom_recorder = False
    print(f"Custom audio recorder not available: {e}")

def draw_detections(image_np, detections):
    """Draw bounding boxes and labels on the image."""
    result_image = image_np.copy()
    
    for detection in detections:
        bbox = detection['bbox']
        left, top, right, bottom = [int(x) for x in bbox]
        label = detection['label']
        confidence = detection['confidence']
        
        color = get_detection_color(label)
        
        cv2.rectangle(result_image, (left, top), (right, bottom), color, 3)
        
        label_text = f"{label} {confidence:.2f}"
        label_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        
        cv2.rectangle(result_image, 
                     (left, top - label_size[1] - 10), 
                     (left + label_size[0], top), 
                     color, -1)
        
        text_color = (255, 255, 255) if sum(color) < 400 else (0, 0, 0)
        cv2.putText(result_image, label_text,
                   (left, top - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
    
    return result_image

def get_detection_color(label):
    """Get a consistent color for each object type."""
    color_map = {
        'cell phone': (255, 100, 100),
        'laptop': (255, 150, 100),
        'tv': (255, 200, 100),
        'mouse': (200, 255, 100),
        'keyboard': (150, 255, 100),
        'remote': (100, 255, 100),
        'person': (100, 255, 150),
        'chair': (150, 100, 255),
        'couch': (200, 100, 255),
        'bed': (255, 100, 255),
        'bottle': (100, 150, 255),
        'cup': (100, 200, 255),
        'bowl': (100, 255, 255),
        'default': (0, 255, 0)
    }
    
    return color_map.get(label, color_map['default'])

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

def get_object_category(label):
    """Get the category for a detected object label."""
    label = label.lower()
    for category, items in OBJECT_CATEGORIES.items():
        if label in items:
            return category
    return "other"

def get_image_hash(image):
    """Create a hash of an image for caching purposes."""
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG', quality=70)
    return hashlib.md5(img_byte_arr.getvalue()).hexdigest()

def detect_objects(image, confidence_threshold=0.5, iou_threshold=0.45):
    """Main detection function - now using YOLOv8"""
    return detect_objects_yolov8(image, confidence_threshold)

def enhance_image(image, enhance_type="auto"):
    """Enhance the image to improve object detection."""
    try:
        img_array = np.array(image)
        
        if enhance_type == "auto" or enhance_type == "brightness":
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            mean_brightness = np.mean(gray)
            
            if mean_brightness < 100:
                hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
                h, s, v = cv2.split(hsv)
                brightness_factor = max(1.0, (130 - mean_brightness) / 80)
                v = cv2.add(v, np.array([brightness_factor * 30.0], dtype=np.uint8))
                final_hsv = cv2.merge((h, s, v))
                img_array = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB)
            
            elif mean_brightness > 200:
                hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
                h, s, v = cv2.split(hsv)
                v = cv2.subtract(v, np.array([30], dtype=np.uint8))
                final_hsv = cv2.merge((h, s, v))
                img_array = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB)
        
        if enhance_type == "auto" or enhance_type == "contrast":
            lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            enhanced_lab = cv2.merge((cl, a, b))
            img_array = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        
        enhanced_image = Image.fromarray(img_array)
        return enhanced_image
    
    except Exception as e:
        error_message(f"Image enhancement error: {e}")
        return image

def detect_text_in_image(image):
    """Detect text in image using OCR."""
    try:
        if not has_tesseract:
            return "OCR functionality requires installing pytesseract."
        
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((1, 1), np.uint8)
        processed = cv2.dilate(binary, kernel, iterations=1)
        processed = cv2.erode(processed, kernel, iterations=1)
        processed = cv2.bitwise_not(processed)
        detected_text = pytesseract.image_to_string(processed)
        detected_text = detected_text.strip()
        
        return detected_text
    except Exception as e:
        return f"Text detection error: {e}"

def get_example_sentence(word, target_language):
    """Generate an example sentence using the word via the example generator."""
    category = None
    for cat_name, items in OBJECT_CATEGORIES.items():
        if word.lower() in [item.lower() for item in items]:
            category = cat_name
            break
    
    return example_generator.get_example_sentence(word, target_language, category)

def get_pronunciation_guide(word, language_code):
    """Generate a simple pronunciation guide for the word."""
    try:
        pronunciation_maps = {
            "es": {'j': 'h', 'll': 'y', 'ñ': 'ny', 'rr': 'rolled r'},
            "fr": {'eau': 'oh', 'au': 'oh', 'ai': 'eh', 'ou': 'oo', 'u': 'ü', 'r': 'guttural r'},
            "de": {'sch': 'sh', 'ch': 'kh/sh', 'ei': 'eye', 'ie': 'ee', 'ä': 'eh', 'ö': 'er', 'ü': 'ü'},
            "it": {'gn': 'ny', 'gli': 'ly', 'ch': 'k', 'c+e/i': 'ch', 'c+a/o/u': 'k'}
        }
        
        sound_map = pronunciation_maps.get(language_code, {})
        notes = []
        
        for sound, pronunciation in sound_map.items():
            if sound in word.lower():
                notes.append(f"'{sound}' sounds like '{pronunciation}'")
        
        return notes
    except Exception as e:
        return [f"Pronunciation guide unavailable: {str(e)}"]

# Initialize session state variables
if 'target_language' not in st.session_state:
    st.session_state.target_language = "es"
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
if 'debug_quiz' not in st.session_state:
    st.session_state.debug_quiz = False
if 'audio_data' not in st.session_state:
    st.session_state.audio_data = None
if 'audio_data_received' not in st.session_state:
    st.session_state.audio_data_received = False
if 'current_recording_word' not in st.session_state:
    st.session_state.current_recording_word = None
if 'use_vision_api' not in st.session_state:
    st.session_state.use_vision_api = True
if 'app_mode' not in st.session_state:
    st.session_state.app_mode = "Camera Mode"
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

# Initialize gamification
def get_gamification():
    return GamificationSystem()

gamification = get_gamification()
gamification.initialize_state()

# Translation service
class FreeTranslationService:
    def __init__(self):
        self.translation_cache = {}
        self.last_request_time = 0
        self.rate_limit_delay = 1.0
        
    def translate_text(self, text, target_language, source_language='en'):
        """Translate text using multiple free services with fallbacks"""
        cache_key = f"{text}_{source_language}_{target_language}"
        if cache_key in self.translation_cache:
            return self.translation_cache[cache_key]
        
        current_time = time.time()
        if current_time - self.last_request_time < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - (current_time - self.last_request_time))
        
        translation = None
        
        try:
            translator = GoogleTranslator(source=source_language, target=target_language)
            translation = translator.translate(text)
            if translation and translation != text:
                self.translation_cache[cache_key] = translation
                self.last_request_time = time.time()
                return translation
        except Exception as e:
            print(f"Deep Translator failed: {e}")
        
        return f"[Translation to {target_language} unavailable]"

free_translator = FreeTranslationService()

def translate_text(text, target_language, source_language='en'):
    return free_translator.translate_text(text, target_language, source_language)

gamification.set_translate_func(translate_text)

@st.cache_resource
def get_example_generator():
    """Initialize and cache the example sentence generator."""
    return ExampleSentenceGenerator(translate_func=translate_text, debug=True)

example_generator = get_example_generator()

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

def get_audio_html(audio_bytes):
    """Generate HTML for audio playback without autoplay."""
    audio_base64 = base64.b64encode(audio_bytes).decode()
    audio_tag = f'<audio src="data:audio/mp3;base64,{audio_base64}" controls></audio>'
    return audio_tag

def save_image(image, label):
    try:
        img_array = np.array(image)
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        os.makedirs("object_images", exist_ok=True)
        filename = f"object_images/{label}_{int(time.time())}.jpg"
        cv2.imwrite(filename, img_cv)
        return filename
    except Exception as e:
        error_message(f"Error saving image: {e}")
        return None

# Main sidebar for navigation
app_mode_options = ["Camera Mode", "My Vocabulary", "Quiz Mode", "Statistics", "My Progress", "Pronunciation Practice"]
if 'app_mode' in st.session_state:
    default_index = app_mode_options.index(st.session_state.app_mode) if st.session_state.app_mode in app_mode_options else 0
else:
    default_index = 0

app_mode = st.sidebar.selectbox(
    "Choose a mode",
    app_mode_options,
    index=default_index
)

st.session_state.app_mode = app_mode

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

st.session_state.target_language = languages[selected_language]

# Help section in sidebar
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
    style_title("📸 Camera Mode")
    info_message("Take a photo or upload an image to identify objects and learn new vocabulary.")
    
    # Session management - FIXED VERSION
    session_container = st.container()
    with session_container:
        col1, col2 = st.columns(2)
        
        with col1:
            start_button_placeholder = st.empty()
            
            if st.session_state.session_id is None:
                if start_button_placeholder.button("Start Learning Session", key="start_session_btn"):
                    if manage_session("start"):
                        st.rerun()
            else:
                start_button_placeholder.markdown(
                    f'<div class="info-box" style="margin: 0.5rem 0; height: 38px; display: flex; align-items: center;">'
                    f'Session in progress - Words: {st.session_state.words_learned}'
                    f'</div>', 
                    unsafe_allow_html=True
                )
        
        with col2:
            end_button_placeholder = st.empty()
            
            if st.session_state.session_id is not None:
                if end_button_placeholder.button("End Session", key="end_session_btn"):
                    if manage_session("end"):
                        st.rerun()
            else:
                end_button_placeholder.markdown(
                    '<div style="height: 38px;"></div>', 
                    unsafe_allow_html=True
                )
    
    # Image input options
    image_tab1, image_tab2 = st.tabs(["📷 Take a Photo", "📁 Upload Image"])

    image = None
    with image_tab1:
        picture = st.camera_input("Take a picture", key="camera_input")
        if picture is not None:
            image = Image.open(picture)

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
        iou_threshold = 0.45
        enhancement_type = "auto"
    
    # Process image if available
    if image is not None:
        if detection_type == "Objects":
            spinner_placeholder = st.empty()
            with spinner_placeholder.container():
                show_loading_spinner("Detecting objects... This may take a few seconds.")
            
            separator_placeholder = st.empty()
            separator_placeholder.markdown('<div class="result-separator"></div>', unsafe_allow_html=True)
            
            try:
                enhanced_image = enhance_image(image, "auto")
                if enhanced_image is None:
                    raise Exception("Image enhancement failed")
                
                detections, result_image = detect_objects(
                    enhanced_image, confidence_threshold, iou_threshold
                )
                
            except Exception as e:
                if "memory" in str(e).lower() or "resource" in str(e).lower():
                    error_message("Memory limit reached. Please try a smaller image or refresh the page.")
                else:
                    error_message(f"Detection error: {str(e)}")
                detections, result_image = [], np.array(image)
            
            separator_placeholder.empty()
        
            # Display results
            if detections:
                style_section_title("✨ Detected Objects")
                
                st.image(result_image, caption="Detected Objects")
                st.write("Select objects to save to your vocabulary:")
                
                # Group detections by label to avoid duplicates
                unique_detections = {}
                for i, detection in enumerate(detections):
                    label = detection['label']
                    confidence = detection['confidence']
                    
                    if label in unique_detections and unique_detections[label][1]['confidence'] >= confidence:
                        continue
                    
                    unique_detections[label] = (i, detection)

                # Group by category
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
                        for i, detection in category_detections:
                            label = detection['label']
                            confidence = detection['confidence']
                            checkbox_key = f"detect_{i}"
                            
                            translated_label = translate_text(label, st.session_state.target_language)
                            
                            with st.container():
                                st.markdown(f"**{label}** ({confidence:.2f})")
                                st.markdown(f"→ **{translated_label}**")
                                
                                col1, col2, col3 = st.columns([2, 2, 1])
                                
                                with col1:
                                    audio_bytes = text_to_speech(translated_label, st.session_state.target_language)
                                    if audio_bytes:
                                        st.markdown(get_audio_html(audio_bytes), unsafe_allow_html=True)
                                    
                                    pronunciation_tips = get_pronunciation_guide(translated_label, st.session_state.target_language)
                                    if pronunciation_tips:
                                        st.markdown("**Pronunciation Tips:**")
                                        for tip in pronunciation_tips:
                                            st.markdown(f"- {tip}")
                                
                                with col2:
                                    example = get_example_sentence(label, st.session_state.target_language)
                                    st.markdown("**Example:**")
                                    st.markdown(f"EN: {example['english']}")
                                    
                                    if example['translated']:
                                        source = example.get('source', 'unknown')
                                        source_name = source.replace('_', ' ').replace('api', 'API').title()
                                        st.markdown(f"{selected_language}: {example['translated']}")
                                        st.markdown(f"<small><i>Source: {source_name}</i></small>", unsafe_allow_html=True)
                                        
                                        example_audio = text_to_speech(example['translated'], st.session_state.target_language)
                                        if example_audio:
                                            st.markdown(get_audio_html(example_audio), unsafe_allow_html=True)
                                    else:
                                        st.markdown("*Translation not available.*")
                                
                                with col3:
                                    if checkbox_key not in st.session_state.detection_checkboxes:
                                        st.session_state.detection_checkboxes[checkbox_key] = True
                                        
                                    st.session_state.detection_checkboxes[checkbox_key] = st.checkbox(
                                        "Save", 
                                        value=st.session_state.detection_checkboxes[checkbox_key],
                                        key=checkbox_key
                                    )
                                
                                st.markdown("---")

                save_button_id = "save_objects_button_fixed"
                
                if not st.session_state.words_just_saved:
                    if st.button("💾 Save Selected Objects to Vocabulary", key=save_button_id):
                        if st.session_state.session_id is None:
                            if manage_session("start"):
                                success_message("Created a new learning session!")
                            else:
                                error_message("Failed to create a session. Please check database connection.")
                                st.stop()
                        
                        selected_objects = []
                        for i in range(len(detections)):
                            if st.session_state.detection_checkboxes.get(f"detect_{i}", False):
                                selected_objects.append(i)
                        
                        if not selected_objects:
                            warning_message("No objects were selected to save. Please check at least one 'Save' box.")
                        else:
                            saved_count = 0
                            saved_items = []
                            
                            for i in selected_objects:
                                try:
                                    detection = detections[i]
                                    label = detection['label']
                                    translated_label = translate_text(label, st.session_state.target_language)
                                    
                                    image_path = save_image(image, label)
                                    category = get_object_category(label)
                                    
                                    vocab_id = add_vocabulary_direct(
                                        word_original=label,
                                        word_translated=translated_label,
                                        language_translated=st.session_state.target_language,
                                        category=category,
                                        image_path=image_path
                                    )
                                    
                                    if vocab_id:
                                        saved_count += 1
                                        saved_items.append(f"{label} → {translated_label}")
                                        st.session_state.words_studied += 1
                                        st.session_state.words_learned += 1
                                    else:
                                        error_message(f"Failed to save {label} to vocabulary.")
                                except Exception as e:
                                    error_message(f"Error saving {label}: {str(e)}")
                            
                            if saved_count > 0:
                                st.session_state.words_just_saved = True
                                st.session_state.saved_count = saved_count
                                st.session_state.saved_items = saved_items
                                st.rerun()
                            else:
                                error_message("Failed to save any words. Please check database connection.")

                if st.session_state.words_just_saved:
                    success_container = st.container()
                    
                    with success_container:
                        success_message(f"Successfully added {st.session_state.saved_count} new words to your vocabulary!")
                        
                        st.markdown('<h4 style="color: #1679AB;">Words saved:</h4>', unsafe_allow_html=True)
                        for item in st.session_state.saved_items:
                            st.markdown(f"✅ {item}")
                        
                        st.markdown("### What would you like to do next?")
                        next_col1, next_col2, next_col3 = st.columns(3)
                        
                        def go_to_quiz_mode():
                            st.session_state.words_just_saved = False
                            st.session_state.app_mode = "Quiz Mode"
                            st.session_state.detection_checkboxes = {}
                            st.rerun()

                        def go_to_vocabulary():
                            st.session_state.words_just_saved = False
                            st.session_state.app_mode = "My Vocabulary"
                            st.session_state.detection_checkboxes = {}
                            st.rerun()

                        def continue_capturing():
                            st.session_state.words_just_saved = False
                            st.session_state.detection_checkboxes = {}
                            st.rerun()
                        
                        with next_col1:
                            if st.button("🎮 Go to Quiz Mode", key="quiz_nav_button"):
                                go_to_quiz_mode()
                                
                        with next_col2:
                            if st.button("📚 View My Vocabulary", key="vocab_nav_button"):
                                go_to_vocabulary()
                                
                        with next_col3:
                            if st.button("📸 Continue Capturing", key="continue_button"):
                                continue_capturing()
            else:
                info_message("No objects detected. Try adjusting the confidence threshold or taking a clearer photo.")

        else:  # Text OCR mode
            spinner_container = st.container()
            with spinner_container:
                show_loading_spinner("Detecting text... This may take a few seconds.")
                
            add_result_separator()
            
            with st.spinner("Detecting text..."):
                detected_text = detect_text_in_image(image)
                
                spinner_container.empty()
                add_scroll_indicator()
                
                if detected_text:
                    style_section_title("📝 Detected Text")
                    st.write(detected_text)
                    
                    words = [word.strip() for word in re.split(r'[^\w]', detected_text) if word.strip()]
                    
                    if words:
                        st.subheader("Words to Learn")
                        
                        for i, word in enumerate(words):
                            if len(word) <= 2:
                                continue
                                
                            translated_word = translate_text(word, st.session_state.target_language)
                            
                            with st.container():
                                cols = st.columns([3, 1])
                                
                                with cols[0]:
                                    st.write(f"**{word}** → {translated_word}")
                                    audio_bytes = text_to_speech(translated_word, st.session_state.target_language)
                                    if audio_bytes:
                                        st.markdown(get_audio_html(audio_bytes), unsafe_allow_html=True)
                                
                                with cols[1]:
                                    if st.button(f"Save", key=f"save_text_{i}"):
                                        if st.session_state.session_id is None:
                                            manage_session("start")
                                        
                                        vocab_id = add_vocabulary_direct(
                                            word_original=word,
                                            word_translated=translated_word,
                                            language_translated=st.session_state.target_language,
                                            category="text",
                                            image_path=None
                                        )
                                        
                                        if vocab_id:
                                            success_message(f"Added '{word}' to vocabulary!")
                                            st.session_state.words_studied += 1
                                            st.session_state.words_learned += 1
                                        else:
                                            error_message(f"Failed to save '{word}'")
                                
                                st.markdown("---")
                    else:
                        info_message("No clear words detected in the image.")
                else:
                    info_message("No text detected. Try another image or adjust image clarity.")

elif app_mode == "My Vocabulary":
    style_title("📚 My Vocabulary")
    st.markdown("Review all the words you've learned so far.")
    
    vocabulary = get_all_vocabulary_direct()
    
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
        if word is None or 'language_translated' not in word:
            continue
            
        if filter_language == "All" or languages.get(filter_language) == word['language_translated']:
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
            if not all(k in word for k in ['word_original', 'word_translated', 'language_translated']):
                continue
                
            lang_code = word.get('language_translated', '')
            lang_name = next((k for k, v in languages.items() if v == lang_code), lang_code)
            
            proficiency = word.get('proficiency_level', 0) or 0
            proficiency_display = "⭐" * proficiency
            
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
        
        if table_data:
            st.dataframe(pd.DataFrame(table_data))
            
            st.subheader("Word Details")
            selected_word_index = st.selectbox(
                "Select a word to review:",
                range(len(filtered_vocab)),
                format_func=lambda i: f"{filtered_vocab[i].get('word_original', '')} → {filtered_vocab[i].get('word_translated', '')}"
            )
            
            word = filtered_vocab[selected_word_index]
            
            col1, col2 = st.columns([2, 3])
            with col1:
                st.markdown(f"**Original:** {word.get('word_original', '')}")
                st.markdown(f"**Translation:** {word.get('word_translated', '')}")
                
                lang_code = word.get('language_translated', '')
                lang_name = next((k for k, v in languages.items() if v == lang_code), lang_code)
                st.markdown(f"**Language:** {lang_name}")
                
                if word.get('category'):
                    st.markdown(f"**Category:** {word.get('category', '')}")
                
                st.markdown("**Listen to pronunciation:**")
                audio_bytes = text_to_speech(word.get('word_translated', ''), word.get('language_translated', ''))
                if audio_bytes:
                    st.markdown(get_audio_html(audio_bytes), unsafe_allow_html=True)
                
                proficiency = word.get('proficiency_level', 0) or 0
                st.markdown("**Learning progress:**")
                st.progress(proficiency / 5)
                review_count = word.get('review_count', 0) or 0
                st.markdown(f"Proficiency: {proficiency}/5 (based on {review_count} reviews)")
                
                pronunciation_tips = get_pronunciation_guide(word.get('word_translated', ''), word.get('language_translated', ''))
                if pronunciation_tips:
                    st.markdown("**Pronunciation tips:**")
                    for tip in pronunciation_tips:
                        st.markdown(f"- {tip}")
            
            with col2:
                image_path = word.get('image_path', '')
                if image_path and os.path.exists(image_path):
                    try:
                        image = Image.open(image_path)
                        st.image(image, caption=f"Image for {word.get('word_original', '')}")
                    except Exception as e:
                        error_message(f"Error loading image: {e}")
                else:
                    st.markdown("*No image available for this word*")
                
                example = get_example_sentence(word.get('word_original', ''), word.get('language_translated', ''))
                st.markdown(f"**Example in context:**")
                st.markdown(f"**English:** {example['english']}")

                if example['translated']:
                    source = example.get('source', 'unknown')
                    source_name = source.replace('_', ' ').replace('api', 'API').title()
                    st.markdown(f"**{lang_name}:** {example['translated']}")
                    st.markdown(f"<small><i>Source: {source_name}</i></small>", unsafe_allow_html=True)
                    
                    example_audio = text_to_speech(example['translated'], word.get('language_translated', ''))
                    if example_audio:
                        st.markdown(get_audio_html(example_audio), unsafe_allow_html=True)

        else:
            warning_message("There was an issue with the vocabulary data format.")
    else:
        info_message("No vocabulary words found with current filter. Go to Camera Mode to start learning new words!")

elif app_mode == "Quiz Mode":
    style_title("🎮 Quiz Mode")
    st.markdown("Test your vocabulary knowledge with interactive quizzes.")
    
    if 'quiz_system' not in st.session_state:
        try:
            from quiz_system import QuizSystem
            
            db_functions = {
                'get_all_vocabulary_direct': get_all_vocabulary_direct,
                'update_word_progress_direct': update_word_progress_direct
            }
            
            quiz_system = QuizSystem(
                db_functions=db_functions,
                text_to_speech=text_to_speech,
                get_audio_html=get_audio_html,
                get_example_sentence=get_example_sentence,
                get_pronunciation_guide=get_pronunciation_guide
            )
            
            st.session_state.quiz_system = quiz_system
            st.session_state.gamification = gamification
            
        except ImportError as e:
            error_message(f"Error loading quiz system: {e}")
            info_message("Please make sure quiz_system.py is in the same directory as main.py")
            st.stop()
    
    quiz_system = st.session_state.quiz_system
    vocabulary = get_all_vocabulary_direct()
    
    if 'quiz_completed' not in st.session_state:
        st.session_state.quiz_completed = False
        
    if st.session_state.current_quiz_word and st.session_state.quiz_options:
        quiz_system.display_quiz_question(languages, manage_session)
        
        st.sidebar.markdown(f"### Current Score: {st.session_state.quiz_score}/{st.session_state.quiz_total}")
        if st.session_state.quiz_total > 0:
            accuracy = (st.session_state.quiz_score / st.session_state.quiz_total) * 100
            st.sidebar.markdown(f"**Accuracy:** {accuracy:.1f}%")
            
    elif st.session_state.quiz_completed and st.session_state.quiz_total > 0:
        quiz_system.display_quiz_results()
        
    else:
        st.markdown("""
        Choose your quiz settings below to test your vocabulary knowledge.
        The quiz will randomly include different types of questions:
        
        - 🔄 Translation (both directions)
        - 🖼️ Image recognition
        - 📝 Sentence completion
        - 🎯 Category matching
        - 📊 Related words identification
        - 🔊 Audio recognition
        
        Start with a small number of questions and work your way up!
        """)
        
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
        
        filtered_vocab = [word for word in vocabulary if word['language_translated'] == quiz_lang_code]
        
        if category_filter != "All Categories":
            filtered_vocab = [word for word in filtered_vocab if word.get('category') == category_filter]
        
        if filtered_vocab:
            st.markdown(f"**{len(filtered_vocab)} words available** for your quiz in {quiz_language}" + 
                        (f" ({category_filter} category)" if category_filter != "All Categories" else ""))
            
            words_with_images = sum(1 for word in filtered_vocab 
                                  if word.get('image_path') and os.path.exists(word.get('image_path', '')))
            
            st.markdown(f"*{words_with_images} words have images for image recognition questions*")
            
            start_label = "Start Quiz" if len(filtered_vocab) >= 4 else f"Need {4-len(filtered_vocab)} More Word(s)"
            if st.button(start_label, disabled=len(filtered_vocab) < 4):
                if quiz_system.start_new_quiz(filtered_vocab, languages, num_questions, manage_session):
                    st.rerun()
            
            if st.checkbox("Preview Available Words"):
                preview_data = []
                for word in filtered_vocab[:20]:
                    preview_data.append({
                        "Original": word.get('word_original', ''),
                        "Translation": word.get('word_translated', ''),
                        "Category": word.get('category', '')
                    })
                
                st.dataframe(pd.DataFrame(preview_data))
                
                if len(filtered_vocab) > 20:
                    st.markdown(f"*...and {len(filtered_vocab) - 20} more words*")
        else:
            warning_message(f"No vocabulary words found with current filter. Go to Camera Mode to add words in {quiz_language}" +
                      (f" for the {category_filter} category" if category_filter != "All Categories" else "") + ".")
            
            if not vocabulary:
                info_message("Start by learning some words in Camera Mode to build your vocabulary!")
            elif not any(word['language_translated'] == quiz_lang_code for word in vocabulary):
                info_message(f"You don't have any words in {quiz_language} yet. Try selecting a different language or add some new words.")
            else:
                info_message(f"No words found in the {category_filter} category. Try selecting 'All Categories' or add words in this category.")

elif app_mode == "Statistics":
    style_title("📊 Learning Statistics")
    st.markdown("Track your progress and learning habits.")
    
    stats = get_session_stats_direct(30)
    
    if st.checkbox("Show raw stats data"):
        st.write("Raw stats data from database:")
        st.write(stats)
    
    if stats and stats.get('total_sessions'):
        st.subheader("Overall Statistics (Last 30 Days)")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Sessions", stats.get('total_sessions', 0) or 0)
        with col2:
            st.metric("Words Studied", stats.get('total_words_studied', 0) or 0)
        with col3:
            st.metric("Words Learned", stats.get('total_words_learned', 0) or 0)
        
        st.subheader("Learning Efficiency")
        col1, col2 = st.columns(2)
        
        with col1:
            avg_words = stats.get('avg_words_per_session', 0) or 0
            st.metric("Avg Words per Session", f"{avg_words:.1f}")
        
        with col2:
            avg_time = stats.get('avg_session_minutes', 0) or 0
            st.metric("Avg Session Length", f"{avg_time:.1f} min")
        
        st.subheader("Vocabulary by Language")
        
        vocabulary = get_all_vocabulary_direct()
        
        language_counts = {}
        for word in vocabulary:
            if word is None or 'language_translated' not in word:
                continue
                
            lang = word['language_translated']
            if lang in language_counts:
                language_counts[lang] += 1
            else:
                language_counts[lang] = 1
        
        language_names = {}
        for name, code in languages.items():
            if code in language_counts:
                language_names[name] = language_counts[code]
        
        if language_names:
            chart_data = pd.DataFrame({
                'Language': list(language_names.keys()),
                'Word Count': list(language_names.values())
            })
            
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(chart_data['Language'], chart_data['Word Count'], color='skyblue')
            
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
        
        st.subheader("Proficiency Level Distribution")
        
        proficiency_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        for word in vocabulary:
            if word is None:
                continue
            level = word.get('proficiency_level', 0) or 0
            proficiency_counts[level] += 1
        
        prof_data = pd.DataFrame({
            'Level': [f"Level {lvl}" for lvl in proficiency_counts.keys()],
            'Words': list(proficiency_counts.values())
        })
        
        if sum(proficiency_counts.values()) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = ['#FFCCCC', '#FFE5CC', '#FFFFCC', '#E5FFCC', '#CCFFCC', '#CCFFEF']
            bars = ax.bar(prof_data['Level'], prof_data['Words'], color=colors)
            
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
            
            st.markdown("""
            **Proficiency Level Guide:**
            - **Level 0**: New words or words answered incorrectly multiple times
            - **Level 1**: Basic recognition (20% correct answers)
            - **Level 2**: Beginning to remember (40% correct answers)
            - **Level 3**: Moderate proficiency (60% correct answers)
            - **Level 4**: Good proficiency (80% correct answers)
            - **Level 5**: Mastered (90-100% correct answers)
            """)
        
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
            st.subheader("Sample Statistics (Demo)")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Sessions", 5)
            with col2:
                st.metric("Words Studied", 42)
            with col3:
                st.metric("Words Learned", 38)
                
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
        gamification.render_dashboard()
    except Exception as e:
        error_message("There was an error displaying the Progress. The system might be initializing.")
        info_message("Please try again in a moment or add some vocabulary first to initialize the system.")
        print(f"Dashboard error: {e}")

elif app_mode == "Pronunciation Practice":
    style_title("🤖 AI-Powered Pronunciation Practice")
    st.markdown("Practice your pronunciation with real-time AI feedback and comprehensive analysis.")

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
    
    st.warning("🎤 Pronunciation practice features require additional packages.")
    
    st.markdown("### 🎯 Basic Pronunciation Practice")
    st.markdown("You can still practice pronunciation with the available features:")
    
    vocabulary = get_all_vocabulary_direct()
    practice_language = st.selectbox(
        "Select practice language:",
        list(languages.keys()),
        index=list(languages.values()).index(st.session_state.target_language) 
            if st.session_state.target_language in languages.values() else 0,
        key="basic_pron_lang_select"
    )
    practice_language_code = languages[practice_language]
    
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
        
        st.markdown("**🔊 Listen and repeat:**")
        audio_bytes = text_to_speech(word_translated, practice_language_code)
        if audio_bytes:
            st.markdown(get_audio_html(audio_bytes), unsafe_allow_html=True)
        
        pronunciation_tips = get_pronunciation_guide(word_translated, practice_language_code)
        if pronunciation_tips:
            st.markdown("**💡 Pronunciation Tips:**")
            for tip in pronunciation_tips:
                st.markdown(f"- {tip}")
        
        st.markdown("**📁 Upload your recording for basic analysis:**")
        uploaded_audio = st.file_uploader(
            "Record yourself saying the word and upload the audio file", 
            type=["wav", "mp3", "ogg", "m4a"],
            key="basic_audio_upload"
        )
        
        if uploaded_audio:
            st.audio(uploaded_audio)
            
            st.markdown("### 📝 Basic Feedback")
            st.success("✅ Audio received! Keep practicing by:")
            st.markdown("- 🔄 Comparing your pronunciation with the correct audio")
            st.markdown("- 📚 Focusing on the pronunciation tips above")
            st.markdown("- 🎯 Recording multiple attempts to improve")
            
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

# Add logout functionality
if st.sidebar.button("🚪 Logout"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    
    st.markdown("**Logging out...**")
    st.markdown("[← Return to Login](https://vocam.app/web)")
    st.stop()

add_footer()