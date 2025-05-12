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

# First, display Python version for debugging
st.set_page_config(
    page_title="Vocam",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Create a container for technical details that will be hidden by default
tech_details = []
tech_details.append(f"Python Version: {sys.version}")

# Try importing OCR with fallback
try:
    import pytesseract
    tech_details.append("PyTesseract: Imported successfully")
    has_tesseract = True
except ImportError as e:
    tech_details.append(f"PyTesseract: Failed to import ({e})")
    has_tesseract = False
    # Dummy implementation
    class DummyTesseract:
        def image_to_string(self, *args, **kwargs):
            return "OCR requires pytesseract. Install with: pip install pytesseract"
    pytesseract = DummyTesseract()

# Try importing OpenCV with robust fallback mechanism
try:
    import cv2
    tech_details.append("OpenCV: Imported successfully")
except ImportError as e:
    tech_details.append(f"OpenCV: Failed to import ({e})")
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
    tech_details.append("OpenCV: Using fallback implementation")

# Import other dependencies with careful error handling
try:
    import torch
    tech_details.append("PyTorch: Imported successfully")
except ImportError as e:
    tech_details.append(f"PyTorch: Failed to import ({e})")
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
    tech_details.append("Google Cloud Translation: Imported successfully")
except ImportError as e:
    tech_details.append(f"Google Cloud Translation: Failed to import ({e})")
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
    tech_details.append("deep_translator: Imported successfully")
    has_deep_translator = True
except ImportError as e:
    tech_details.append(f"deep_translator: Failed to import ({e})")
    has_deep_translator = False
    # Fallback function
    def dummy_translator(*args, **kwargs):
        return "Could not generate example. Install deep-translator package."
    GoogleTranslator = dummy_translator

# Try importing gTTS
try:
    from gtts import gTTS
    tech_details.append("gTTS: Imported successfully")
except ImportError as e:
    tech_details.append(f"gTTS: Failed to import ({e})")
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
    tech_details.append("Database module: Imported successfully")
except ImportError as e:
    tech_details.append(f"Database module: Failed to import ({e})")
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
        tech_details.append("Google Cloud credentials: Loaded from secrets")
    else:
        # Local development fallback
        credentials_path = r'C:\Users\HP\Desktop\Senior Proj\credentials.json'
        if os.path.exists(credentials_path):
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
            tech_details.append("Google Cloud credentials: Loaded from local file")
        else:
            tech_details.append("Google Cloud credentials: Not found")
except Exception as e:
    tech_details.append(f"Google Cloud credentials: Error setting up ({e})")

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
        st.error(f"Object detection error: {e}")
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
        st.error(f"Image enhancement error: {e}")
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
        st.error(f"Direct session creation error: {str(e)}")
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
            st.info(f"Word '{word_original}' already exists in {language_translated}. Updating with new information.")
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
            st.error("Database is currently locked. Please wait a moment and try again.")
            # Add a small delay to allow the database to unlock
            time.sleep(1.5)
        else:
            st.error(f"Database error: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Direct vocabulary save error: {str(e)}")
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
        st.error(f"Error retrieving vocabulary: {str(e)}")
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
        st.error(f"Error retrieving session stats: {str(e)}")
        return {}

# Function to debug database
def debug_database():
    """Check database tables and content for debugging."""
    # Only create the debug UI after the main UI has been set up
    if 'db_debug_initialized' not in st.session_state:
        st.session_state.db_debug_initialized = True
        return

    st.sidebar.markdown("---")
    if st.sidebar.checkbox("Debug Database"):
        st.sidebar.markdown("### Database Debug")
        try:
            # Check if tables exist
            conn = sqlite3.connect("language_learning.db")
            cursor = conn.cursor()
            
            # Get list of tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            st.sidebar.write(f"Tables in database: {[t[0] for t in tables]}")
            
            # Check vocabulary count
            cursor.execute("SELECT COUNT(*) FROM vocabulary;")
            vocab_count = cursor.fetchone()[0]
            st.sidebar.write(f"Vocabulary entries: {vocab_count}")
            
            # Check session count
            cursor.execute("SELECT COUNT(*) FROM sessions;")
            session_count = cursor.fetchone()[0]
            st.sidebar.write(f"Session entries: {session_count}")
            
            # Show recent vocabulary
            if vocab_count > 0:
                cursor.execute("SELECT id, word_original, word_translated, language_translated, date_added FROM vocabulary ORDER BY date_added DESC LIMIT 5;")
                recent_vocab = cursor.fetchall()
                st.sidebar.write("Recent vocabulary:")
                for item in recent_vocab:
                    st.sidebar.write(f"ID: {item[0]}, {item[1]} → {item[2]} ({item[3]}), {item[4]}")
            
            # Show active sessions
            cursor.execute("SELECT id, start_time, end_time FROM sessions ORDER BY start_time DESC LIMIT 3;")
            recent_sessions = cursor.fetchall()
            st.sidebar.write("Recent sessions:")
            for session in recent_sessions:
                st.sidebar.write(f"ID: {session[0]}, Started: {session[1]}, Ended: {session[2] or 'Active'}")
            
            conn.close()
        except Exception as e:
            st.sidebar.error(f"Database error: {e}")

# Call debug database function
debug_database()

# Add technical details to the sidebar
with st.sidebar.expander("Technical Details", expanded=False):
    for detail in tech_details:
        st.write(detail)

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
        st.error(f"Translation error: {e}")
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
        st.error(f"Text-to-speech error: {e}")
        return None

# Function to generate HTML for audio playback
def get_audio_html(audio_bytes):
    audio_base64 = base64.b64encode(audio_bytes).decode()
    audio_tag = f'<audio autoplay="true" src="data:audio/mp3;base64,{audio_base64}" controls></audio>'
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
        st.error(f"Error loading object detection model: {e}")
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
                st.success(f"Started new learning session!")
                return True
            else:
                st.error("Failed to create a session directly. Check database permissions.")
                return False
                
        except Exception as e:
            st.error(f"Error starting session: {str(e)}")
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
            
            st.success(f"Session completed! Words studied: {st.session_state.words_studied}, Words learned: {st.session_state.words_learned}")
            # Clear session state
            st.session_state.session_id = None
            st.session_state.words_studied = 0
            st.session_state.words_learned = 0
            return True
                
        except Exception as e:
            st.error(f"Error ending session: {str(e)}")
            return False
    
    return False

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
            with st.sidebar.expander("Database Setup", expanded=False):
                st.error(f"Missing tables in database: {', '.join(missing_tables)}")
                st.info("Attempting to create missing tables...")
            
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
            with st.sidebar.expander("Database Setup", expanded=False):
                st.success("Database tables created successfully!")
        
        conn.close()
        return True
    except Exception as e:
        with st.sidebar.expander("Database Setup", expanded=False):
            st.error(f"Database error: {e}")
        return False

if 'db_checked' not in st.session_state:
    st.session_state.db_checked = check_database_setup()

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
        st.error(f"Error saving image: {e}")
        return None

# Function to start a new quiz
def start_new_quiz(vocabulary, num_questions=5):
    # Reset quiz state
    st.session_state.quiz_score = 0
    st.session_state.quiz_total = 0
    st.session_state.answered = False
    
    if not vocabulary or len(vocabulary) < 4:
        st.warning("Not enough vocabulary words for a quiz (need at least 4).")
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
        st.error(f"Error updating word progress: {str(e)}")
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
app_mode_options = ["Camera Mode", "My Vocabulary", "Quiz Mode", "Statistics", "Gamification Dashboard"]
app_mode = st.sidebar.selectbox(
    "Choose a mode",
    app_mode_options
)

# Add gamification info to the sidebar
try:
    gamification.update_sidebar()
except Exception as e:
    st.sidebar.markdown("🏆 **Gamification system is initializing...**")
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

# Display appropriate content based on selected mode
if app_mode == "Camera Mode":
    st.title("📸 Camera Mode")
    st.markdown("Take a photo or upload an image to identify objects and learn their names in your target language.")
    
    # Session management
    col1, col2 = st.columns(2)
    with col1:
        if st.session_state.session_id is None:
            if st.button("Start Learning Session"):
                if manage_session("start"):
                    st.rerun()
        else:
            st.info(f"Session in progress - Words learned: {st.session_state.words_learned}")
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
        if detection_type == "Objects":
            with st.spinner("Detecting objects..."):
                # Perform object detection
                detections, result_image = detect_objects(
                    image_for_detection, 
                    confidence_threshold=confidence_threshold,
                    iou_threshold=iou_threshold
                )
                
                # Display results
                if detections:
                    st.subheader("Detected Objects")
                    
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
                    
                    # Add a save button
                    if st.button("Save Selected Objects to Vocabulary", type="primary"):
                        # Auto-start session if needed
                        if st.session_state.session_id is None:
                            if manage_session("start"):
                                st.success("Created a new learning session!")
                            else:
                                st.error("Failed to create a session. Please check database connection.")
                                st.stop()
                        
                        # Count selected objects
                        selected_objects = [i for i in range(len(detections)) 
                                        if st.session_state.detection_checkboxes.get(f"detect_{i}", False)]
                        
                        if not selected_objects:
                            st.warning("No objects were selected to save. Please check at least one 'Save' box.")
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
                                        st.error(f"Failed to save {label} to vocabulary.")
                                except Exception as e:
                                    st.error(f"Error saving {label}: {str(e)}")
                            
                            if saved_count > 0:
                                st.success(f"Successfully added {saved_count} new words to your vocabulary!")
                                st.write("Words saved:")
                                for item in saved_items:
                                    st.write(f"- {item}")
                                # Clear checkboxes after saving
                                st.session_state.detection_checkboxes = {}
                                # Give user a moment to see the success message
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error("Failed to save any words. Please check database connection.")
                    
                else:
                    st.info("No objects detected. Try another image or adjust the confidence threshold.")
                    
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
                                st.success("Created a new learning session!")
                            else:
                                st.error("Failed to create a session.")
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
                            st.success(f"Successfully added '{st.session_state.manual_label}' to your vocabulary!")
                            st.session_state.words_studied += 1
                            st.session_state.words_learned += 1
                            
                            # Reset manual mode
                            st.session_state.manual_mode = False
                            st.session_state.manual_label = ""
                            time.sleep(1.5)
                            st.rerun()
                        else:
                            st.error("Failed to save word to vocabulary.")
                
                # Button to exit manual mode
                if st.button("Cancel Manual Selection"):
                    st.session_state.manual_mode = False
                    st.session_state.manual_label = ""
                    st.rerun()
        
        # Text OCR mode
        else:  # Text OCR mode
            with st.spinner("Detecting text..."):
                detected_text = detect_text_in_image(image)
                
                if detected_text:
                    st.subheader("Detected Text")
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
                                            st.success(f"Added '{word}' to vocabulary!")
                                            st.session_state.words_studied += 1
                                            st.session_state.words_learned += 1
                                        else:
                                            st.error(f"Failed to save '{word}'")
                                
                                st.markdown("---")
                    else:
                        st.info("No clear words detected in the image.")
                else:
                    st.info("No text detected. Try another image or adjust image clarity.")

elif app_mode == "My Vocabulary":
    st.title("📚 My Vocabulary")
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
                        st.error(f"Error loading image: {e}")
                else:
                    st.markdown("*No image available for this word*")
                
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
            st.warning("There was an issue with the vocabulary data format.")
    else:
        st.info("No vocabulary words found with current filter. Go to Camera Mode to start learning new words!")

elif app_mode == "Quiz Mode":
    st.title("🎮 Quiz Mode")
    st.markdown("Test your vocabulary knowledge with interactive quizzes.")
    
    # Get vocabulary from database
    vocabulary = get_all_vocabulary_direct()
    
    # Quiz settings
    col1, col2 = st.columns(2)
    with col1:
        quiz_language = st.selectbox(
            "Quiz language:",
            list(languages.keys()),
            index=list(languages.values()).index(st.session_state.target_language) if st.session_state.target_language in languages.values() else 0
        )
        quiz_lang_code = languages[quiz_language]
    
    with col2:
        num_questions = st.number_input("Number of questions:", min_value=1, max_value=20, value=5)
    
    # Filter vocabulary by selected language
    filtered_vocab = [word for word in vocabulary if word['language_translated'] == quiz_lang_code]
    
    # Start quiz button
    if st.button("Start New Quiz"):
        if start_new_quiz(filtered_vocab, num_questions):
            st.rerun()
    
    # Display current quiz question if available
    if st.session_state.current_quiz_word and st.session_state.quiz_options:
        word = st.session_state.current_quiz_word
        
        # Create a progress bar for the quiz
        progress = min(st.session_state.quiz_total / num_questions, 1.0)
        st.progress(progress)
        st.markdown(f"**Question {st.session_state.quiz_total + 1}/{num_questions}**")
        
        # Display the question
        st.markdown(f"## What is the translation of '{word['word_original']}'?")
        
        # Display image if available
        if word['image_path'] and os.path.exists(word['image_path']):
            image = Image.open(word['image_path'])
            st.image(image, caption=f"Image for {word['word_original']}", width=300)
        
        # Create answer buttons
        cols = st.columns(len(st.session_state.quiz_options))
        for i, option in enumerate(st.session_state.quiz_options):
            with cols[i]:
                # Determine button appearance based on answer status
                if st.session_state.answered:
                    is_correct_option = option['id'] == word['id']
                    if is_correct_option:
                        st.success(option['word_translated'])
                    else:
                        st.error(option['word_translated'])
                else:
                    if st.button(option['word_translated'], key=f"option_{i}"):
                        is_correct = check_answer(i)
                        # Force rerun to update UI
                        st.rerun()
        
        # Display pronunciation after answering
        if st.session_state.answered:
            st.markdown("### Pronunciation:")
            audio_bytes = text_to_speech(word['word_translated'], word['language_translated'])
            if audio_bytes:
                st.markdown(get_audio_html(audio_bytes), unsafe_allow_html=True)
            
            # Add pronunciation tips
            pronunciation_tips = get_pronunciation_guide(word['word_translated'], word['language_translated'])
            if pronunciation_tips:
                st.markdown("**Tips:**")
                for tip in pronunciation_tips:
                    st.markdown(f"- {tip}")
            
            # Next question or finish quiz button
            if st.session_state.quiz_total < num_questions:
                if st.button("Next Question"):
                    if st.session_state.quiz_total >= num_questions:
                        # End quiz
                        st.session_state.current_quiz_word = None
                        st.session_state.quiz_options = []
                    else:
                        # Setup next question
                        setup_new_question(filtered_vocab)
                    st.rerun()
            else:
                # Finish quiz button:
                if st.button("Finish Quiz"):
                    st.session_state.current_quiz_word = None
                    st.session_state.quiz_options = []
                    # End session
                    if st.session_state.session_id:
                        manage_session("end")
                    st.rerun()
        
        # Display current score
        st.sidebar.markdown(f"### Current Score: {st.session_state.quiz_score}/{st.session_state.quiz_total}")
        if st.session_state.quiz_total > 0:
            accuracy = (st.session_state.quiz_score / st.session_state.quiz_total) * 100
            st.sidebar.markdown(f"**Accuracy:** {accuracy:.1f}%")
    
    # Display quiz results if quiz is finished
    elif st.session_state.quiz_total > 0:
        st.success(f"Quiz completed! Your score: {st.session_state.quiz_score}/{st.session_state.quiz_total}")
        
        # Calculate and display accuracy
        accuracy = (st.session_state.quiz_score / st.session_state.quiz_total) * 100
        st.markdown(f"### Accuracy: {accuracy:.1f}%")
        
        # Display feedback based on score
        if accuracy >= 90:
            st.balloons()
            st.markdown("### 🎖️ Excellent job! You're mastering these words!")
        elif accuracy >= 70:
            st.markdown("### 👍 Good work! Keep practicing to improve.")
        else:
            st.markdown("### 📚 Keep practicing! Review your vocabulary regularly.")
        
        # Reset button
        if st.button("Start Another Quiz"):
            # Reset quiz state
            st.session_state.current_quiz_word = None
            st.session_state.quiz_options = []
            st.session_state.quiz_score = 0
            st.session_state.quiz_total = 0
            st.rerun()
            
    # If not enough vocabulary, show message
    elif not filtered_vocab or len(filtered_vocab) < 4:
        st.warning(f"You need at least 4 words in {quiz_language} to start a quiz. Go to Camera Mode to learn more words!")

elif app_mode == "Statistics":
    st.title("📊 Learning Statistics")
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
        st.info("No learning statistics available yet. Complete some learning sessions to see your progress!")
        
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
elif app_mode == "Gamification Dashboard":
    try:
        gamification.render_dashboard()
    except Exception as e:
        st.error("There was an error displaying the Gamification Dashboard. The system might be initializing.")
        st.info("Please try again in a moment or add some vocabulary first to initialize the system.")
        print(f"Dashboard error: {e}")


st.sidebar.markdown("---")
st.sidebar.markdown("### Session Info")
if st.session_state.session_id:
    st.sidebar.success(f"Session active")
    st.sidebar.info(f"Words studied: {st.session_state.words_studied}")
    st.sidebar.info(f"Words learned: {st.session_state.words_learned}")
else:
    st.sidebar.warning("No active session")
    st.sidebar.markdown("*Start a session in Camera Mode to track progress*")