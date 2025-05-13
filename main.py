import streamlit as st
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import base64
import time
import sqlite3
import datetime
import sys
import json
import tempfile
import re
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from io import BytesIO
from gamification import GamificationSystem
import random
from collections import defaultdict
import io
from vocam_ui import apply_custom_css

apply_custom_css()

try:
    from cloud_detector import detect_streamlit_cloud
    is_cloud = detect_streamlit_cloud()
except ImportError:
    is_cloud = False

if is_cloud:
    os.environ['IS_STREAMLIT_CLOUD'] = 'true'
    print("Running in Streamlit Cloud - some features may be limited")

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

try:
    from cloud_detector import detect_streamlit_cloud
    is_cloud = detect_streamlit_cloud()
except ImportError:
    is_cloud = False

if is_cloud:
    os.environ['IS_STREAMLIT_CLOUD'] = 'true'
    print("Running in Streamlit Cloud - some features may be limited")

# First, display Python version for debugging
st.set_page_config(
    page_title="Vocam",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Fix the typo in the import statement
try:
    from pronunciation_practice import create_pronunciation_practice
    has_pronunciation_practice = True
except ImportError:
    has_pronunciation_practice = False

    
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
            except Exception:
                return False
    
    # Replace cv2 with our dummy implementation
    cv2 = DummyCV2()

# Import other dependencies with careful error handling
try:
    import torch
except ImportError as e:
    # Dummy torch for fallback
    class DummyTorch:
        def __init__(self):
            self.hub = type('obj', (object,), {
                'load': lambda *args, **kwargs: DummyModel()
            })
            
    class DummyModel:
        def __call__(self, *args, **kwargs):
            return type('obj', (object,), {
                'xyxy': [[]], 
                'render': lambda: [[np.zeros((300, 300, 3), dtype=np.uint8)]],
                'names': {0: 'unknown'}
            })
            
        def eval(self):
            return self
            
    torch = DummyTorch()

# Try importing Google Cloud libraries
try:
    from google.cloud import translate_v2 as translate
except ImportError as e:
    # Dummy translate for fallback
    class DummyTranslate:
        class Client:
            def __init__(self):
                pass
                
            def translate(self, text, target_language=None):
                return {"translatedText": f"[Translation to {target_language} would appear here]"}
    
    translate = DummyTranslate()

# Try importing deep translator with fallback
try:
    from deep_translator import GoogleTranslator
    has_deep_translator = True
except ImportError as e:
    has_deep_translator = False
    # Fallback function
    def dummy_translator(*args, **kwargs):
        return "Could not generate example. Install deep-translator package."
    GoogleTranslator = dummy_translator

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
            
        def start_session(self):
            return None
            
        def end_session(self, session_id, words_studied, words_learned):
            return True

# Helper function to convert AttrDict to a regular dict recursively
def convert_to_dict(obj):
    if isinstance(obj, dict):
        return {key: convert_to_dict(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_dict(item) for item in obj]
    else:
        return obj

# Handle Google Cloud credentials with proper type handling
try:
    if 'gcp_service_account' in st.secrets:
        # Create a temporary file to store credentials
        credentials_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.json')
        credentials_path = credentials_temp.name
        
        # Get a copy of all secret key-value pairs
        credentials_dict = {}
        for key in st.secrets["gcp_service_account"]:
            credentials_dict[key] = st.secrets["gcp_service_account"][key]
        
        # Write the credentials to the temporary file
        with open(credentials_path, 'w') as f:
            json.dump(credentials_dict, f)
        
        # Set environment variable to point to this file
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
    else:
        # Local development fallback
        credentials_path = r'C:\Users\HP\Desktop\Senior Proj\credentials.json'
        if os.path.exists(credentials_path):
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
except Exception as e:
    pass

# Define object categories for better organization
OBJECT_CATEGORIES = {
    "food": ["apple", "banana", "orange", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", 
             "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake"],
    
    "animals": ["bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"],
    
    "vehicles": ["bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"],
    
    "electronics": ["tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", 
                   "toaster", "sink", "refrigerator"],
    
    "furniture": ["chair", "couch", "potted plant", "bed", "dining table", "toilet"],
    
    "personal": ["backpack", "umbrella", "handbag", "tie", "suitcase"],
    
    "sports": ["frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", 
              "baseball glove", "skateboard", "surfboard", "tennis racket"],
    
    "household": ["bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "book", "clock", 
                 "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
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
    """Get the category for a detected object label."""
    label = label.lower()
    for category, items in OBJECT_CATEGORIES.items():
        if label in items:
            return category
    return "other"

# Function to detect objects in image
def detect_objects(image, confidence_threshold=0.5, iou_threshold=0.45):
    try:
        model = load_model()
        
        # Pass IOU threshold to the model
        model.conf = confidence_threshold
        model.iou = iou_threshold
        
        # Run inference
        results = model(image)
        
        # Filter by confidence
        detections = []
        for detection in results.xyxy[0]:
            xmin, ymin, xmax, ymax, confidence, class_idx = detection
            if confidence > confidence_threshold:
                label = results.names[int(class_idx)]
                detections.append({
                    'label': label,
                    'confidence': float(confidence),
                    'bbox': [float(xmin), float(ymin), float(xmax), float(ymax)]
                })
        
        return detections, results.render()[0]
    except Exception as e:
        error_message(f"Database error: {e}")
        dummy_image = np.array(image)
        return [], dummy_image

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
    """Detect text in image using OCR."""
    try:
        if not has_tesseract:
            return "OCR functionality requires installing pytesseract."
        
        # Convert PIL image to numpy array
        img_array = np.array(image)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Apply threshold to get image with only black and white
        _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        
        # Apply dilation and erosion to remove noise
        kernel = np.ones((1, 1), np.uint8)
        processed = cv2.dilate(binary, kernel, iterations=1)
        processed = cv2.erode(processed, kernel, iterations=1)
        
        # Invert back
        processed = cv2.bitwise_not(processed)
        
        # Detect text
        detected_text = pytesseract.image_to_string(processed)
        
        # Clean and process the text
        detected_text = detected_text.strip()
        
        return detected_text
    except Exception as e:
        return f"Text detection error: {e}"

# Function to get example sentence
# If deep-translator isn't available, we can use Google Translate API directly
def get_example_sentence(word, target_language):
    """Generate an example sentence using the word."""
    try:
        # Simple English templates
        templates = [
            f"The {word} is on the table.",
            f"I like this {word} very much.",
            f"Can you see the {word}?",
            f"This {word} is very useful.",
            f"I need a new {word}."
        ]
        
        # Select a random template
        example = np.random.choice(templates)
        
        # Try to translate using the Google Translate API
        try:
            translate_client = translate.Client()
            result = translate_client.translate(example, target_language=target_language)
            translated_example = result.get("translatedText", "")
        except Exception:
            translated_example = ""
        
        return {
            "english": example,
            "translated": translated_example
        }
    except Exception:
        # Return English example but empty translation
        return {
            "english": f"The {word} is on the table.",
            "translated": ""
        }
        
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

# Function to create a database session
def create_session_direct():
    """Create a session directly using SQLite."""
    try:
        # Connect to the database
        conn = sqlite3.connect("language_learning.db")
        cursor = conn.cursor()
        
        # Insert a new session with the current time
        current_time = datetime.datetime.now()
        cursor.execute(
            "INSERT INTO sessions (start_time, words_studied, words_learned) VALUES (?, 0, 0)",
            (current_time,)
        )
        conn.commit()
        
        # Get the last inserted ID
        session_id = cursor.lastrowid
        conn.close()
        
        return session_id
    except Exception as e:
        error_message(f"Direct session creation error: {str(e)}")
        return None

# Function to add vocabulary to the database
def add_vocabulary_direct(word_original, word_translated, language_translated, category=None, image_path=None):
    """Add vocabulary directly using SQLite with improved error handling for duplicates and locks."""
    try:
        # Original function code here...
        # Connect to the database with timeout to handle locks
        conn = sqlite3.connect("language_learning.db", timeout=10.0)
        cursor = conn.cursor()
        
        # Check if this word already exists in this language
        cursor.execute(
            "SELECT id FROM vocabulary WHERE word_original = ? AND language_translated = ?",
            (word_original, language_translated)
        )
        existing_word = cursor.fetchone()
        
        # If word exists, update it rather than inserting a new one
        if existing_word:
            vocab_id = existing_word[0]
            
            # Update the existing word with new translation and image if provided
            cursor.execute(
                "UPDATE vocabulary SET word_translated = ?, category = ?, image_path = ? WHERE id = ?",
                (word_translated, category, image_path, vocab_id)
            )
            
            # Let the user know we're updating
            info_message(f"Word '{word_original}' already exists in {language_translated}. Updating with new information.")
        else:
            # Current time for timestamps
            current_time = datetime.datetime.now()
            
            # Insert a new word
            try:
                # Try with source column
                cursor.execute('''
                INSERT INTO vocabulary 
                (word_original, word_translated, language_translated, category, image_path, date_added, source)
                VALUES (?, ?, ?, ?, ?, ?, 'manual')
                ''', (word_original, word_translated, language_translated, category, image_path, current_time))
            except sqlite3.OperationalError as e:
                if 'no column named source' in str(e):
                    # Try without source column
                    cursor.execute('''
                    INSERT INTO vocabulary 
                    (word_original, word_translated, language_translated, category, image_path, date_added)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ''', (word_original, word_translated, language_translated, category, image_path, current_time))
                else:
                    raise e
            
            # Get the last inserted ID
            vocab_id = cursor.lastrowid
            
            # Check if we need to add user progress
            cursor.execute("SELECT id FROM user_progress WHERE vocabulary_id = ?", (vocab_id,))
            if not cursor.fetchone():
                # Initialize user progress for this vocabulary
                cursor.execute('''
                INSERT INTO user_progress (vocabulary_id, last_reviewed, proficiency_level)
                VALUES (?, ?, 0)
                ''', (vocab_id, current_time))
        
        # Commit changes and close
        conn.commit()
        conn.close()
        
        # NEW CODE: Integration with gamification system - with error handling
        if vocab_id:
            try:
                # Check for gamification achievements
                gamification.check_achievements(
                    "word_learned",
                    word=word_original,
                    category=category,
                    language=language_translated
                )
                
                # Check for daily challenges
                gamification.check_challenge_progress(
                    word_original=word_original,
                    word_translated=word_translated,
                    language=language_translated
                )
            except Exception as e:
                print(f"Gamification error in add_vocabulary_direct: {e}")
        
        return vocab_id
    except sqlite3.OperationalError as e:
        # Handle database locks with specific advice
        if 'database is locked' in str(e):
            error_message("Database is currently locked. Please wait a moment and try again.")
            # Add a small delay to allow the database to unlock
            time.sleep(1.5)
        else:
            error_message(f"Database error: {str(e)}")
        return None
    except Exception as e:
        error_message(f"Direct vocabulary save error: {str(e)}")
        return None
    

# Function to get all vocabulary items from the database
def get_all_vocabulary_direct():
    """Get all vocabulary items directly from SQLite."""
    try:
        # Connect to the database
        conn = sqlite3.connect("language_learning.db")
        
        # Use dictionary cursor for easier access
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get all vocabulary with user progress info
        cursor.execute('''
        SELECT v.id, v.word_original, v.word_translated, v.language_translated,
               v.category, v.image_path, v.date_added,
               up.proficiency_level, up.review_count, up.correct_count, up.last_reviewed
        FROM vocabulary v
        LEFT JOIN user_progress up ON v.id = up.vocabulary_id
        ORDER BY v.date_added DESC
        ''')
        
        # Fetch all results
        results = cursor.fetchall()
        
        # Convert to list of dictionaries
        vocabulary = []
        for row in results:
            # Convert row to dictionary
            word = dict(row)
            vocabulary.append(word)
        
        conn.close()
        return vocabulary
    except Exception as e:
        error_message(f"Error retrieving vocabulary: {str(e)}")
        return []

# Function to get session statistics
def get_session_stats_direct(days=30):
    """Get session statistics directly from SQLite."""
    try:
        # Connect to the database
        conn = sqlite3.connect("language_learning.db")
        cursor = conn.cursor()
        
        # Calculate date for filtering
        current_time = datetime.datetime.now()
        start_date = current_time - datetime.timedelta(days=days)
        
        # Convert to string format
        start_date_str = start_date.strftime("%Y-%m-%d")
        
        # Get total sessions
        cursor.execute(
            "SELECT COUNT(*) FROM sessions WHERE start_time >= ?",
            (start_date_str,)
        )
        total_sessions = cursor.fetchone()[0]
        
        # Get words studied and learned
        cursor.execute(
            "SELECT SUM(words_studied), SUM(words_learned) FROM sessions WHERE start_time >= ?",
            (start_date_str,)
        )
        result = cursor.fetchone()
        total_words_studied = result[0] if result[0] else 0
        total_words_learned = result[1] if result[1] else 0
        
        # Calculate averages
        avg_words_per_session = total_words_studied / total_sessions if total_sessions > 0 else 0
        
        # Get session durations
        cursor.execute(
            """
            SELECT start_time, end_time 
            FROM sessions 
            WHERE start_time >= ? AND end_time IS NOT NULL
            """,
            (start_date_str,)
        )
        
        # Calculate average session length
        total_minutes = 0
        session_count = 0
        
        for start_time_str, end_time_str in cursor.fetchall():
            try:
                # Parse the datetime strings
                start_time = datetime.datetime.fromisoformat(start_time_str.replace(' ', 'T'))
                end_time = datetime.datetime.fromisoformat(end_time_str.replace(' ', 'T'))
                
                # Calculate duration in minutes
                duration = (end_time - start_time).total_seconds() / 60
                total_minutes += duration
                session_count += 1
            except:
                pass
        
        avg_session_minutes = total_minutes / session_count if session_count > 0 else 0
        
        conn.close()
        
        # Return stats dictionary
        return {
            'total_sessions': total_sessions,
            'total_words_studied': total_words_studied,
            'total_words_learned': total_words_learned,
            'avg_words_per_session': avg_words_per_session,
            'avg_session_minutes': avg_session_minutes
        }
    except Exception as e:
        error_message(f"Error retrieving session stats: {str(e)}")
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
            # Create missing tables
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
    
    if st.session_state.debug_quiz:
        st.sidebar.markdown("### Vocabulary Stats")
        st.sidebar.markdown(f"Total words: {total_words}")
        st.sidebar.markdown(f"With categories: {words_with_categories}")
        st.sidebar.markdown(f"With images: {words_with_images}")
        st.sidebar.markdown(f"With examples: {words_with_examples}")
    
    return vocabulary

if 'db_checked' not in st.session_state:
    st.session_state.db_checked = check_database_setup()

# Initialize database
@st.cache_resource
def get_database():
    return LanguageLearningDB("language_learning.db")

db = get_database()

# Initialize processing queue in session state for background tasks
if 'processing_queue' not in st.session_state:
    st.session_state.processing_queue = queue.Queue()
if 'processing_results' not in st.session_state:
    st.session_state.processing_results = {}
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False

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
# For debugging question type selection
if 'debug_quiz' not in st.session_state:
    st.session_state.debug_quiz = False


@st.cache_resource
def get_gamification():
    return GamificationSystem()

# Initialize gamification
gamification = get_gamification()
# Make sure state is explicitly initialized
gamification.initialize_state()


# Function to translate text
def translate_text(text, target_language):
    try:
        translate_client = translate.Client()
        result = translate_client.translate(text, target_language=target_language)
        return result["translatedText"]
    except Exception as e:
        error_message(f"Translation error: {e}")
        return text

# Background worker function for translation
def translate_worker(texts, target_language, task_id):
    """Worker function to translate multiple texts in background."""
    try:
        translate_client = translate.Client()
        results = {}
        
        for key, text in texts.items():
            try:
                result = translate_client.translate(text, target_language=target_language)
                results[key] = result["translatedText"]
            except Exception as e:
                results[key] = f"[Translation error: {str(e)}]"
        
        # Store results
        st.session_state.processing_results[task_id] = results
    except Exception as e:
        # Store error
        st.session_state.processing_results[task_id] = {
            'error': str(e)
        }

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

# Function to load YOLOv5 model
@st.cache_resource
def load_model():
    try:
        # Use a larger model variant for better accuracy
        model = torch.hub.load('ultralytics/yolov5', 'yolov5m', source='github')  # Using medium model for better accuracy
        model.eval()
        return model
    except Exception as e:
        error_message(f"Error loading object detection model: {e}")
        # Return a dummy model that won't cause NoneType errors
        class DummyModel:
            def __init__(self):
                self.names = {0: 'unknown'}
                
            def __call__(self, image):
                # Return a result object with the expected structure
                class DummyResults:
                    def __init__(self):
                        self.xyxy = [[]]
                        self.names = {0: 'unknown'}
                    
                    def render(self):
                        # Return a copy of the input image as a numpy array
                        if isinstance(image, np.ndarray):
                            return [image.copy()]
                        else:
                            return [np.array(image)]
                
                return DummyResults()
        
        return DummyModel()

# Background worker function for object detection
def detect_objects_worker(image, confidence_threshold, iou_threshold, task_id):
    """Worker function to run object detection in background."""
    try:
        model = load_model()
        model.conf = confidence_threshold
        model.iou = iou_threshold
        
        results = model(image)
        
        # Filter by confidence
        detections = []
        for detection in results.xyxy[0]:
            xmin, ymin, xmax, ymax, confidence, class_idx = detection
            if confidence > confidence_threshold:
                label = results.names[int(class_idx)]
                detections.append({
                    'label': label,
                    'confidence': float(confidence),
                    'bbox': [float(xmin), float(ymin), float(xmax), float(ymax)]
                })
        
        rendered_image = results.render()[0]
        
        # Store results
        st.session_state.processing_results[task_id] = {
            'detections': detections,
            'result_image': rendered_image
        }
        
        # Mark task as complete
        st.session_state.processing_complete = True
    except Exception as e:
        # Store error
        st.session_state.processing_results[task_id] = {
            'error': str(e)
        }
        st.session_state.processing_complete = True

# Function to start or end a learning session
def manage_session(action):
    """Start or end learning session with improved error handling."""
    if action == "start":
        try:
            # Try to use the direct method instead of the database object
            session_id = create_session_direct()
            
            if session_id:
                st.session_state.session_id = session_id
                st.session_state.words_studied = 0
                st.session_state.words_learned = 0
                success_message(f"Started new learning session!")
                return True
            else:
                error_message("Failed to create a session directly. Check database permissions.")
                return False
                
        except Exception as e:
            error_message(f"Error starting session: {str(e)}")
            return False
            
    elif action == "end" and st.session_state.session_id:
        try:
            # Connect directly to the database
            conn = sqlite3.connect("language_learning.db")
            cursor = conn.cursor()
            
            # Update the session with end time and stats
            current_time = datetime.datetime.now()
            cursor.execute(
                "UPDATE sessions SET end_time = ?, words_studied = ?, words_learned = ? WHERE id = ?",
                (current_time, st.session_state.words_studied, st.session_state.words_learned, st.session_state.session_id)
            )
            conn.commit()
            conn.close()
            
            success_message(f"Session completed! Words studied: {st.session_state.words_studied}, Words learned: {st.session_state.words_learned}")
            # Clear session state
            st.session_state.session_id = None
            st.session_state.words_studied = 0
            st.session_state.words_learned = 0
            return True
                
        except Exception as e:
            error_message(f"Error ending session: {str(e)}")
            return False
    
    return False

# Function to save image
def save_image(image, label):
    try:
        # Convert PIL Image to OpenCV format
        img_array = np.array(image)
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Create directory if it doesn't exist
        os.makedirs("object_images", exist_ok=True)
        
        # Save image
        filename = f"object_images/{label}_{int(time.time())}.jpg"
        cv2.imwrite(filename, img_cv)
        
        return filename
    except Exception as e:
        error_message(f"Error saving image: {e}")
        return None

# Function to start a new quiz
def start_new_quiz(vocabulary, num_questions=5):
    # Reset quiz state
    st.session_state.quiz_score = 0
    st.session_state.quiz_total = 0
    st.session_state.answered = False
    
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
    if not vocabulary:
        return False
    
    # Select a random word as the question
    st.session_state.current_quiz_word = np.random.choice(vocabulary)
    
    # Create options (3 wrong + 1 correct)
    options = [st.session_state.current_quiz_word]
    while len(options) < 4:
        wrong_option = np.random.choice(vocabulary)
        if wrong_option['id'] != st.session_state.current_quiz_word['id'] and not any(o['id'] == wrong_option['id'] for o in options):
            options.append(wrong_option)
    
    # Shuffle options
    np.random.shuffle(options)
    st.session_state.quiz_options = options
    st.session_state.answered = False
    
    return True

# Function to update word progress in the database
def update_word_progress_direct(vocab_id, is_correct):
    """Update word progress directly using SQLite."""
    try:
        # Connect to the database
        conn = sqlite3.connect("language_learning.db")
        cursor = conn.cursor()
        
        # Current time for timestamp
        current_time = datetime.datetime.now()
        
        # Get current progress
        cursor.execute(
            """
            SELECT review_count, correct_count, proficiency_level 
            FROM user_progress 
            WHERE vocabulary_id = ?
            """,
            (vocab_id,)
        )
        
        result = cursor.fetchone()
        
        if result:
            review_count, correct_count, proficiency_level = result
            
            # Increment counts
            review_count = review_count + 1 if review_count else 1
            correct_count = correct_count + 1 if correct_count and is_correct else (1 if is_correct else 0)
            
            # Calculate proficiency (0-5 scale)
            if review_count > 0:
                accuracy = correct_count / review_count
                if accuracy >= 0.9 and review_count >= 5:
                    proficiency_level = 5
                elif accuracy >= 0.8 and review_count >= 4:
                    proficiency_level = 4
                elif accuracy >= 0.6 and review_count >= 3:
                    proficiency_level = 3
                elif accuracy >= 0.4 and review_count >= 2:
                    proficiency_level = 2
                elif accuracy >= 0.2:
                    proficiency_level = 1
                else:
                    proficiency_level = 0
            
            # Update progress
            cursor.execute(
                """
                UPDATE user_progress 
                SET review_count = ?, correct_count = ?, proficiency_level = ?, last_reviewed = ? 
                WHERE vocabulary_id = ?
                """,
                (review_count, correct_count, proficiency_level, current_time, vocab_id)
            )
        else:
            # Create new progress entry
            cursor.execute(
                """
                INSERT INTO user_progress 
                (vocabulary_id, review_count, correct_count, proficiency_level, last_reviewed)
                VALUES (?, ?, ?, ?, ?)
                """,
                (vocab_id, 1, 1 if is_correct else 0, 1 if is_correct else 0, current_time)
            )
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        error_message(f"Error updating word progress: {str(e)}")
        return False

# Function to check quiz answer
def check_answer(selected_index):
    """Check if selected quiz answer is correct and update progress."""
    if st.session_state.answered:
        return
    
    selected_word = st.session_state.quiz_options[selected_index]
    is_correct = selected_word['id'] == st.session_state.current_quiz_word['id']
    
    # Update database using direct method instead of db class
    update_word_progress_direct(st.session_state.current_quiz_word['id'], is_correct)
    
    # Update session stats
    st.session_state.words_studied += 1
    if is_correct:
        st.session_state.words_learned += 1
        st.session_state.quiz_score += 1
    
    st.session_state.quiz_total += 1
    st.session_state.answered = True
    
    # Check if any challenges are completed - with error handling
    try:
        gamification.check_challenge_progress(
            quiz_score=st.session_state.quiz_score,
            quiz_total=st.session_state.quiz_total
        )
        
        # Check for quiz-related achievements
        if st.session_state.quiz_total >= 5:  # Only check if quiz is substantial
            gamification.check_achievements(
                "quiz_completed",
                score=st.session_state.quiz_score,
                total=st.session_state.quiz_total
            )
    except Exception as e:
        print(f"Gamification error in check_answer: {e}")
    
    return is_correct

# Main sidebar for navigation
st.sidebar.title("🌍 Vocam")
app_mode_options = ["Camera Mode", "My Vocabulary", "Quiz Mode", "Statistics", "My Progress", "Pronunciation Practice"]
app_mode = st.sidebar.selectbox(
    "Choose a mode",
    app_mode_options
)

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
    style_title("📸 Camera Mode")

    # Use the enhanced info message
    info_message("Take a photo or upload an image to identify objects and learn new vocabulary.")
    
    # Session management
    col1, col2 = st.columns(2)
    with col1:
        if st.session_state.session_id is None:
            if st.button("Start Learning Session"):
                if manage_session("start"):
                    st.rerun()
        else:
            info_message(f"Session in progress - Words learned: {st.session_state.words_learned}")
    with col2:
        if st.session_state.session_id is not None:
            if st.button("End Session"):
                if manage_session("end"):
                    st.rerun()
    
    # Image input options
    image_source = st.radio("Select image source:", ["Upload Image", "Take a Photo"])
    
    image = None
    if image_source == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
    else:  # Take a Photo
        picture = st.camera_input("Take a picture")
        if picture is not None:
            image = Image.open(picture)
    
    # Detection options
    detection_type = st.radio(
        "What would you like to detect?",
        ["Objects", "Text (OCR)"],
        index=0
    )
    
    # Detection settings for objects
    if detection_type == "Objects":
        confidence_threshold = st.slider(
            "Detection Confidence Threshold", 
            min_value=0.2, 
            max_value=0.9, 
            value=0.35,  # Optimized default threshold to balance precision and detection
            step=0.05
        )
        
        # Set iou_threshold for optimal detection (balance between precision and maximum detection)
        iou_threshold = 0.3  # Using a lower threshold to detect more objects while maintaining precision
        
        # Auto-enhancement is always applied
        enhancement_type = "auto"
    
    # Process image if available
    if image is not None:
    # Show original image
        st.image(image, caption="Original Image", use_column_width=True)
        
        # Always apply enhancement for object detection
        if detection_type == "Objects":
            with st.spinner("Enhancing image for better detection..."):
                enhanced_image = enhance_image(image, enhancement_type)
                # Use the enhanced image for detection
                image_for_detection = enhanced_image
        else:
            image_for_detection = image
        
        # Process based on detection type
        # Around line 1379-1393, after the loading spinner is shown but before detections are processed, 
# add this code:

# Process based on detection type
        if detection_type == "Objects":
            # Create container for loading spinner
            spinner_container = st.container()
            with spinner_container:
                show_loading_spinner("Detecting objects... This may take a few seconds.")
                
            # Add visual separator for mobile
            add_result_separator()
            
            with st.spinner("Detecting objects..."):
                # ADD THIS LINE TO FIX THE ERROR:
                detections, result_image = detect_objects(image_for_detection, confidence_threshold, iou_threshold)
                
                # Display results
                # Clear the spinner once processing is done
                spinner_container.empty()
                
                # Display results
                if detections:
                    style_section_title("✨ Detected Objects")
                    
                    # Display image with detection boxes
                    st.image(result_image, caption="Detected Objects", use_column_width=True)
                    
                    # Display selection prompt
                    st.write("Select objects to save to your vocabulary:")
                    
                    # Group detections by category
                    categorized_detections = {}
                    for i, detection in enumerate(detections):
                        label = detection['label']
                        category = get_object_category(label)
                        
                        if category not in categorized_detections:
                            categorized_detections[category] = []
                        
                        categorized_detections[category].append((i, detection))
                    
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
                                        # Add example sentence directly (no nested expander)
                                        example = get_example_sentence(label, st.session_state.target_language)
                                        st.markdown("**Example:**")
                                        st.markdown(f"EN: {example['english']}")
                                        
                                        # Only display translated example if available
                                        if example['translated']:
                                            st.markdown(f"{selected_language}: {example['translated']}")
                                            
                                            # Only generate audio if there's text to speak
                                            example_audio = text_to_speech(example['translated'], st.session_state.target_language)
                                            if example_audio:
                                                st.markdown(get_audio_html(example_audio), unsafe_allow_html=True)
                                        else:
                                            st.markdown("*Translation not available. Please install deep-translator package.*")
                                    
                                    with col3:
                                        # Add checkbox for this object
                                        st.session_state.detection_checkboxes[checkbox_key] = st.checkbox(
                                            "Save", 
                                            value=True,
                                            key=checkbox_key
                                        )
                                    
                                    st.markdown("---")  # Add separator
                    
                    # Add a save button with greater prominence
                    if st.button("💾 Save Selected Objects to Vocabulary", type="primary", use_container_width=True):
                        # Auto-start session if needed
                        if st.session_state.session_id is None:
                            if manage_session("start"):
                                success_message("Created a new learning session!")
                            else:
                                error_message("Failed to create a session. Please check database connection.")
                                st.stop()
                        
                        # Count selected objects
                        selected_objects = [i for i in range(len(detections)) 
                                        if st.session_state.detection_checkboxes.get(f"detect_{i}", False)]
                        
                        if not selected_objects:
                            warning_message("No objects were selected to save. Please check at least one 'Save' box.")
                        else:
                            # Save the selected objects
                            saved_count = 0
                            saved_items = []
                            
                            for i in selected_objects:
                                try:
                                    detection = detections[i]
                                    label = detection['label']
                                    translated_label = translate_text(label, st.session_state.target_language)
                                    
                                    # Save the image
                                    image_path = save_image(image, label)
                                    
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
                                    
                                    if vocab_id:
                                        saved_count += 1
                                        saved_items.append(f"{label} → {translated_label}")
                                        # Update session stats
                                        st.session_state.words_studied += 1
                                        st.session_state.words_learned += 1
                                    else:
                                        error_message(f"Failed to save {label} to vocabulary.")
                                except Exception as e:
                                    error_message(f"Error saving {label}: {str(e)}")
                            
                            if saved_count > 0:
                                success_message(f"Successfully added {saved_count} new words to your vocabulary!")
                                
                                # Show saved words in a visually appealing list
                                st.markdown('<h4 style="color: #1679AB;">Words saved:</h4>', unsafe_allow_html=True)
                                for item in saved_items:
                                    st.markdown(f"✅ {item}")
                                
                                # Clear checkboxes after saving
                                st.session_state.detection_checkboxes = {}
                                
                                # Show navigation options
                                next_col1, next_col2 = st.columns(2)
                                with next_col1:
                                    st.button("🎮 Go to Quiz Mode", key="goto_quiz", 
                                            on_click=lambda: setattr(st.session_state, 'app_mode', 'Quiz Mode'),
                                            use_container_width=True)
                                with next_col2:
                                    st.button("📚 View My Vocabulary", key="goto_vocab", 
                                            on_click=lambda: setattr(st.session_state, 'app_mode', 'My Vocabulary'),
                                            use_container_width=True)
                                # Give user a moment to see the success message
                                time.sleep(1)
                                st.rerun()
                            else:
                                error_message("Failed to save any words. Please check database connection.")
                    
                else:
                    info_message("No objects detected. Try another image or adjust the confidence threshold.")
                    
                    # Add manual selection option
                    if st.button("Enable Manual Selection"):
                        st.session_state.manual_mode = True
                        st.rerun()
            
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
                    if st.button("Save to Vocabulary", type="primary"):
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
        # Text OCR mode
        else:  # Text OCR mode
            # Create container for loading spinner
            spinner_container = st.container()
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
                
                if detected_text:
                    style_section_title("📝 Detected Text")
                    st.write(detected_text)
                    
                    # Split into words for learning
                    words = [word.strip() for word in re.split(r'[^\w]', detected_text) if word.strip()]
                    
                    if words:
                        st.subheader("Words to Learn")
                        
                        # Create containers for each word
                        for i, word in enumerate(words):
                            if len(word) <= 2:  # Skip very short words
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
            st.dataframe(pd.DataFrame(table_data), use_container_width=True)
            
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
                if image_path and os.path.exists(image_path):
                    try:
                        image = Image.open(image_path)
                        st.image(image, caption=f"Image for {word.get('word_original', '')}", use_column_width=True)
                    except Exception as e:
                        error_message(f"Error loading image: {e}")
                else:
                    st.markdown("*No image available for this word*")
                
                # Add pronunciation practice if available
                if has_pronunciation_practice:
                    try:
                        # Only initialize if not already initialized
                        if 'pronunciation_practice' not in st.session_state:
                            # Initialize the pronunciation practice module with the functions it needs
                            st.session_state.pronunciation_practice = create_pronunciation_practice(
                                text_to_speech_func=text_to_speech,
                                get_audio_html_func=get_audio_html,
                                translate_text_func=translate_text
                            )
                            print("Successfully initialized pronunciation practice module")
                        
                        # Now use the initialized module
                        st.session_state.pronunciation_practice.render_practice_ui(word)
                    except Exception as e:
                        # Gracefully handle any errors
                        print(f"Error in pronunciation module: {str(e)}")
                        with st.expander("🎤 Practice Pronunciation"):
                            warning_message("Pronunciation practice is temporarily unavailable.")
                            info_message("This feature may not be supported in the current environment.")
                else:
                    # Show a message if pronunciation practice is not available
                    with st.expander("🎤 Practice Pronunciation"):
                        warning_message("Pronunciation practice requires additional packages.")
                        info_message("To enable pronunciation practice, install the following packages:")
                        st.code("pip install SpeechRecognition pydub PyAudio python-Levenshtein")
                        st.markdown("After installing, restart the application to use pronunciation practice.")

                        
                # Add example sentence directly (no expander)
                st.markdown("**Example in context:**")
                example = get_example_sentence(word.get('word_original', ''), word.get('language_translated', ''))
                st.markdown(f"**English:** {example['english']}")
                
                # Only show translated example if available
                if example['translated']:
                    st.markdown(f"**{lang_name}:** {example['translated']}")
                    
                    # Only generate audio if there's text to speak
                    example_audio = text_to_speech(example['translated'], word.get('language_translated', ''))
                    if example_audio:
                        st.markdown(get_audio_html(example_audio), unsafe_allow_html=True)
                else:
                    st.markdown("*Translation not available. Please install deep-translator package.*")

        else:
            warning_message("There was an issue with the vocabulary data format.")
    else:
        info_message("No vocabulary words found with current filter. Go to Camera Mode to start learning new words!")

elif app_mode == "Quiz Mode":
    style_title("🎮 Quiz Mode")
    st.markdown("Test your vocabulary knowledge with interactive quizzes.")
    
    # Import the quiz system if not already imported
    if 'quiz_system' not in st.session_state:
        try:
            # Import quiz system
            from quiz_system import QuizSystem
            
            # Create a dictionary of database functions
            db_functions = {
                'get_all_vocabulary_direct': get_all_vocabulary_direct,
                'update_word_progress_direct': update_word_progress_direct
            }
            
            # Initialize the quiz system
            quiz_system = QuizSystem(
                db_functions=db_functions,
                text_to_speech=text_to_speech,
                get_audio_html=get_audio_html,
                get_example_sentence=get_example_sentence,
                get_pronunciation_guide=get_pronunciation_guide
            )
            
            # Store in session state
            st.session_state.quiz_system = quiz_system
            
            # Add gamification to session state for access by quiz system
            st.session_state.gamification = gamification
            
        except ImportError as e:
            error_message(f"Error loading quiz system: {e}")
            info_message("Please make sure quiz_system.py is in the same directory as main.py")
            st.stop()
    
    # Get the quiz system from session state
    quiz_system = st.session_state.quiz_system
    
    # Get vocabulary from database
    vocabulary = get_all_vocabulary_direct()
    
    # Quiz settings tab and quiz display tab
    if 'quiz_completed' not in st.session_state:
        st.session_state.quiz_completed = False
        
    if st.session_state.current_quiz_word and st.session_state.quiz_options:
        # Quiz is already in progress, display it
        quiz_system.display_quiz_question(languages, manage_session)
        
        # Display current score in sidebar
        st.sidebar.markdown(f"### Current Score: {st.session_state.quiz_score}/{st.session_state.quiz_total}")
        if st.session_state.quiz_total > 0:
            accuracy = (st.session_state.quiz_score / st.session_state.quiz_total) * 100
            st.sidebar.markdown(f"**Accuracy:** {accuracy:.1f}%")
            
    # Display quiz results if quiz is completed
    elif st.session_state.quiz_completed and st.session_state.quiz_total > 0:
        quiz_system.display_quiz_results()
        
    # Display quiz setup
    else:
        # Introduction
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
            st.markdown(f"*{words_with_images} words have images for image recognition questions*")
            
            # Start quiz button with dynamic label
            start_label = "Start Quiz" if len(filtered_vocab) >= 4 else f"Need {4-len(filtered_vocab)} More Word(s)"
            if st.button(start_label, type="primary", disabled=len(filtered_vocab) < 4):
                if quiz_system.start_new_quiz(filtered_vocab, languages, num_questions, manage_session):
                    st.rerun()
            
            # Show word preview 
            if st.checkbox("Preview Available Words"):
                # Create a simple table of words
                preview_data = []
                for word in filtered_vocab[:20]:  # Limit preview to 20 words
                    preview_data.append({
                        "Original": word.get('word_original', ''),
                        "Translation": word.get('word_translated', ''),
                        "Category": word.get('category', '')
                    })
                
                st.dataframe(pd.DataFrame(preview_data), use_container_width=True)
                
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
    style_title("📊 Learning Statistics")
    st.markdown("Track your progress and learning habits.")
    
    # Get session stats for the last 30 days
    stats = get_session_stats_direct(30)
    
    # Debug display for stats
    if st.checkbox("Show raw stats data"):
        st.write("Raw stats data from database:")
        st.write(stats)
    
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
        gamification.render_dashboard()
    except Exception as e:
        error_message("There was an error displaying the Progress. The system might be initializing.")
        info_message("Please try again in a moment or add some vocabulary first to initialize the system.")
        print(f"Dashboard error: {e}")

elif app_mode == "Pronunciation Practice":
    style_title("🎤 Pronunciation Practice")
    st.markdown("Practice your pronunciation and get instant feedback on your speaking skills.")
    
    # Session management
    col1, col2 = st.columns(2)
    with col1:
        if st.session_state.session_id is None:
            if st.button("Start Learning Session"):
                if manage_session("start"):
                    st.rerun()
        else:
            info_message(f"Session in progress - Words studied: {st.session_state.words_studied}")
    with col2:
        if st.session_state.session_id is not None:
            if st.button("End Session"):
                if manage_session("end"):
                    st.rerun()
    
    if has_pronunciation_practice:
        try:
            # Only initialize if not already initialized
            if 'pronunciation_practice' not in st.session_state:
                # Initialize the pronunciation practice module
                st.session_state.pronunciation_practice = create_pronunciation_practice(
                    text_to_speech_func=text_to_speech,
                    get_audio_html_func=get_audio_html,
                    translate_text_func=translate_text
                )
            
            # Get vocabulary from database
            vocabulary = get_all_vocabulary_direct()
            
            # Let user select language
            practice_language = st.selectbox(
                "Select practice language:",
                list(languages.keys()),
                index=list(languages.values()).index(st.session_state.target_language) 
                    if st.session_state.target_language in languages.values() else 0
            )
            practice_language_code = languages[practice_language]
            
            # Filtered vocabulary for the selected language
            filtered_vocab = [word for word in vocabulary if word['language_translated'] == practice_language_code]
            
            if filtered_vocab:
                info_message(f"Found {len(filtered_vocab)} words in {practice_language}. Start a practice session to improve your pronunciation.")
                
                # Start a new practice session button
                if 'practice_words' not in st.session_state:
                    if st.button("Start Practice Session", type="primary"):
                        try:
                            # Import standard library random (not numpy)
                            import random
                            # Select 5 random words for practice (or fewer if not enough words)
                            practice_size = min(5, len(filtered_vocab))
                            st.session_state.practice_words = random.sample(filtered_vocab, practice_size)
                            st.session_state.current_practice_index = 0
                            st.session_state.practice_scores = []
                            st.rerun()
                        except Exception as e:
                            error_message(f"Error starting practice session: {str(e)}")
                
                # Run the practice session if words are selected
                if 'practice_words' in st.session_state:
                    try:
                        st.session_state.pronunciation_practice.render_practice_session(
                            vocabulary, practice_language_code)
                    except Exception as e:
                        error_message(f"Error in practice session: {str(e)}")
                        # Add a reset button
                        if st.button("Reset Practice Session"):
                            if 'practice_words' in st.session_state:
                                del st.session_state.practice_words
                            st.rerun()
            else:
                warning_message(f"No vocabulary words found for {practice_language}. Go to Camera Mode to add words first.")
        
        except Exception as e:
            error_message(f"Error initializing pronunciation practice: {str(e)}")
            warning_message("Pronunciation practice is temporarily unavailable.")
            info_message("This feature may not be supported in the current environment.")
    else:
        warning_message("Pronunciation practice requires additional packages.")
        info_message("To enable pronunciation practice, install the following packages:")
        
        st.markdown("### Python Packages:")
        st.code("pip install SpeechRecognition pydub PyAudio python-Levenshtein")
        
        st.markdown("After installing, restart the application to use pronunciation practice.")
        
        # Add a sample of what the feature will look like
        st.markdown("### Sample Pronunciation Feature")
        st.image("https://i.ibb.co/GTxfJsQ/pronunciation-practice-sample.png", caption="Sample pronunciation practice interface")

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