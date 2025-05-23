import streamlit as st
import os
import cv2
import pygame
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from PIL import Image
from io import BytesIO
from google.cloud import translate_v2 as translate
from gtts import gTTS
import base64
import time
import sqlite3
import datetime  # Add this import if it's missing
from database import LanguageLearningDB

# Add this function AFTER all imports but BEFORE st.set_page_config()
# Make sure sqlite3 is imported at the top of your file

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

def add_vocabulary_direct(word_original, word_translated, language_translated, category=None, image_path=None):
    """Add vocabulary directly using SQLite with improved error handling for duplicates and locks."""
    try:
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

# Then add this function after all imports but before st.set_page_config()
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




# Set page configuration
st.set_page_config(
    page_title="AI Language Learning App",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

debug_database()

# Setup environment variables for Google Cloud
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'C:\Users\HP\Desktop\Senior Proj\credentials.json'

# Initialize database
@st.cache_resource
def get_database():
    return LanguageLearningDB("language_learning.db")

db = get_database()

# Initialize session state
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

# Function to translate text
def translate_text(text, target_language):
    try:
        translate_client = translate.Client()
        result = translate_client.translate(text, target_language=target_language)
        return result["translatedText"]
    except Exception as e:
        st.error(f"Translation error: {e}")
        return text

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
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model.eval()
    return model

# Object detection function
def detect_objects(image, confidence_threshold=0.5):
    model = load_model()
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
    
    return detections, results.render()[0]  # Return detections and rendered image

# Start or end learning session
# Find this function in your app.py and replace it with this improved version
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

# 3. Add this helper function to check if the database is properly set up
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
            st.sidebar.error(f"Missing tables in database: {', '.join(missing_tables)}")
            st.sidebar.info("Attempting to create missing tables...")
            
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
            st.sidebar.success("Database tables created successfully!")
        
        conn.close()
        return True
    except Exception as e:
        st.sidebar.error(f"Database error: {e}")
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
        st.session_state.session_id = db.start_session()
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

# Check quiz answer
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
    
    return is_correct

# Main sidebar for navigation
st.sidebar.title("🌍 Language Learning App")
app_mode = st.sidebar.selectbox(
    "Choose a mode",
    ["Camera Mode", "My Vocabulary", "Quiz Mode", "Statistics"]
)

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
    
    # Initialize session state variables if they don't exist
    if 'detection_checkboxes' not in st.session_state:
        st.session_state.detection_checkboxes = {}
    
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
    
    # Process image if available
    if image is not None:
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        with st.spinner("Detecting objects..."):
            # Perform object detection
            detections, result_image = detect_objects(image)
            
            # Display results
            if detections:
                st.subheader("Detected Objects")
                
                # Display image with detection boxes
                st.image(result_image, caption="Detected Objects", use_column_width=True)
                
                # Display selection prompt
                st.write("Select objects to save to your vocabulary:")
                
                # Process each detection
                for i, detection in enumerate(detections):
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
                        
                        # Create two columns for audio and checkbox
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            # Generate audio for the translated word
                            audio_bytes = text_to_speech(translated_label, st.session_state.target_language)
                            if audio_bytes:
                                st.markdown(get_audio_html(audio_bytes), unsafe_allow_html=True)
                        
                        with col2:
                            # Add checkbox for this object (default to checked)
                            st.session_state.detection_checkboxes[checkbox_key] = st.checkbox(
                                "Save", 
                                value=True,
                                key=checkbox_key
                            )
                        
                        st.markdown("---")  # Add separator
                
                # Add a save button
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
                                
                                # Debug info
                                st.write(f"Saving word: {label} → {translated_label}")
                                st.write(f"Image path: {image_path}")
                                st.write(f"Language: {st.session_state.target_language}")
                                
                                # Add to database using direct method
                                vocab_id = add_vocabulary_direct(
                                    word_original=label,
                                    word_translated=translated_label,
                                    language_translated=st.session_state.target_language,
                                    category="object",
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

elif app_mode == "My Vocabulary":
    st.title("📚 My Vocabulary")
    st.markdown("Review all the words you've learned so far.")
    
    # Get vocabulary from database - debug output
    vocabulary = get_all_vocabulary_direct()
    st.write(f"Found {len(vocabulary)} words in database")
    
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
    # Start quiz button
    if st.button("Start New Quiz"):
        if start_new_quiz(filtered_vocab, num_questions):
            st.rerun()  # Changed from st.experimental_rerun()
    
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
                    st.rerun()  # Changed from st.experimental_rerun()

            else:
                # Finish quiz button:
                if st.button("Finish Quiz"):
                    st.session_state.current_quiz_word = None
                    st.session_state.quiz_options = []
                    # End session
                    if st.session_state.session_id:
                        manage_session("end")
                    st.rerun()  # Changed from st.experimental_rerun()
        
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
            st.rerun()  # Changed from st.experimental_rerun()
            
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

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### Session Info")
if st.session_state.session_id:
    st.sidebar.success(f"Session active")
    st.sidebar.info(f"Words studied: {st.session_state.words_studied}")
    st.sidebar.info(f"Words learned: {st.session_state.words_learned}")
else:
    st.sidebar.warning("No active session")
    st.sidebar.markdown("*Start a session in Camera Mode to track progress*")