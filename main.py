# This file will be imported by main_wrapper.py
# Key imports will be provided by the wrapper
import streamlit as st
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import base64
import time
import sqlite3
import datetime
import json
import tempfile

# These are already imported by the wrapper
import cv2
import torch

# Try importing Google Cloud libraries
try:
    from google.cloud import translate_v2 as translate
    from gtts import gTTS
    CLOUD_AVAILABLE = True
except ImportError:
    # Create dummy classes
    class DummyTranslate:
        class Client:
            def translate(self, text, target_language=None):
                translations = {
                    "dog": {"es": "perro", "fr": "chien", "de": "Hund"},
                    "cat": {"es": "gato", "fr": "chat", "de": "Katze"},
                    "book": {"es": "libro", "fr": "livre", "de": "Buch"},
                    "apple": {"es": "manzana", "fr": "pomme", "de": "Apfel"},
                    "car": {"es": "coche", "fr": "voiture", "de": "Auto"},
                    "house": {"es": "casa", "fr": "maison", "de": "Haus"},
                    "tree": {"es": "árbol", "fr": "arbre", "de": "Baum"},
                    "water": {"es": "agua", "fr": "eau", "de": "Wasser"},
                    "chair": {"es": "silla", "fr": "chaise", "de": "Stuhl"},
                    "table": {"es": "mesa", "fr": "table", "de": "Tisch"},
                    "person": {"es": "persona", "fr": "personne", "de": "Person"},
                }
                text_lower = text.lower()
                if text_lower in translations and target_language in translations[text_lower]:
                    translated = translations[text_lower][target_language]
                else:
                    translated = f"[{text} in {target_language}]"
                return {"translatedText": translated}
    
    class DummyTTS:
        def __init__(self, text="", lang="en", slow=False):
            self.text = text
            self.lang = lang
            
        def write_to_fp(self, fp):
            fp.write(b"DUMMY_AUDIO_DATA")
    
    # Set up dummy modules
    translate = DummyTranslate()
    gTTS = DummyTTS
    CLOUD_AVAILABLE = False

# Import your database module with error handling
try:
    from database import LanguageLearningDB
    DB_MODULE_AVAILABLE = True
except ImportError:
    # Create a basic database class
    class LanguageLearningDB:
        def __init__(self, db_path):
            self.db_path = db_path
            self.ensure_tables_exist()
            
        def ensure_tables_exist(self):
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Create tables if they don't exist
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
                )
                ''')
                
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_progress (
                    id INTEGER PRIMARY KEY,
                    vocabulary_id INTEGER,
                    review_count INTEGER DEFAULT 0,
                    correct_count INTEGER DEFAULT 0,
                    last_reviewed TIMESTAMP,
                    proficiency_level INTEGER DEFAULT 0,
                    FOREIGN KEY (vocabulary_id) REFERENCES vocabulary (id)
                )
                ''')
                
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY,
                    start_time TIMESTAMP,
                    end_time TIMESTAMP,
                    words_studied INTEGER DEFAULT 0,
                    words_learned INTEGER DEFAULT 0
                )
                ''')
                
                conn.commit()
                conn.close()
            except Exception as e:
                st.error(f"Database setup error: {e}")
                
        def start_session(self):
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                current_time = datetime.datetime.now()
                cursor.execute(
                    "INSERT INTO sessions (start_time, words_studied, words_learned) VALUES (?, 0, 0)",
                    (current_time,)
                )
                conn.commit()
                session_id = cursor.lastrowid
                conn.close()
                return session_id
            except Exception as e:
                st.error(f"Error starting session: {e}")
                return None
                
        def end_session(self, session_id, words_studied, words_learned):
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                current_time = datetime.datetime.now()
                cursor.execute(
                    "UPDATE sessions SET end_time = ?, words_studied = ?, words_learned = ? WHERE id = ?",
                    (current_time, words_studied, words_learned, session_id)
                )
                conn.commit()
                conn.close()
                return True
            except Exception as e:
                st.error(f"Error ending session: {e}")
                return False
    
    DB_MODULE_AVAILABLE = False

# Initialize database
@st.cache_resource
def get_database():
    return LanguageLearningDB("language_learning.db")

db = get_database()

# Define helper functions
def translate_text(text, target_language):
    try:
        translate_client = translate.Client()
        result = translate_client.translate(text, target_language=target_language)
        return result["translatedText"]
    except Exception as e:
        st.error(f"Translation error: {e}")
        return f"[{text} in {target_language}]"

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
        return b"DUMMY_AUDIO_DATA"

def get_audio_html(audio_bytes):
    if audio_bytes == b"DUMMY_AUDIO_DATA":
        return "<p><i>Audio playback not available in demo mode</i></p>"
    
    audio_base64 = base64.b64encode(audio_bytes).decode()
    audio_tag = f'<audio autoplay="true" src="data:audio/mp3;base64,{audio_base64}" controls></audio>'
    return audio_tag

@st.cache_resource
def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model.eval()
    return model

def detect_objects(image, confidence_threshold=0.5):
    try:
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
        
        return detections, results.render()[0]
    except Exception as e:
        st.error(f"Object detection error: {e}")
        # Create dummy detections
        detections = [
            {'label': 'chair', 'confidence': 0.82, 'bbox': [50, 50, 150, 200]},
            {'label': 'person', 'confidence': 0.78, 'bbox': [200, 100, 350, 400]}
        ]
        
        # Create a basic image with a frame
        if isinstance(image, np.ndarray):
            img = image.copy()
        else:
            img = np.array(image)
        # Add a colored border
        h, w = img.shape[:2]
        img[0:5, :] = [0, 255, 0]  # Top border
        img[-5:, :] = [0, 255, 0]  # Bottom border
        img[:, 0:5] = [0, 255, 0]  # Left border
        img[:, -5:] = [0, 255, 0]  # Right border
        
        return detections, img

def save_image(image, label):
    try:
        # Convert PIL Image to OpenCV format
        img_array = np.array(image)
        
        # Create directory if it doesn't exist
        os.makedirs("object_images", exist_ok=True)
        
        # Save image
        filename = f"object_images/{label}_{int(time.time())}.jpg"
        Image.fromarray(img_array).save(filename)
        
        return filename
    except Exception as e:
        st.error(f"Error saving image: {e}")
        return None

def create_session_direct():
    try:
        conn = sqlite3.connect("language_learning.db")
        cursor = conn.cursor()
        current_time = datetime.datetime.now()
        cursor.execute(
            "INSERT INTO sessions (start_time, words_studied, words_learned) VALUES (?, 0, 0)",
            (current_time,)
        )
        conn.commit()
        session_id = cursor.lastrowid
        conn.close()
        return session_id
    except Exception as e:
        st.error(f"Session creation error: {e}")
        return None

def add_vocabulary_direct(word_original, word_translated, language_translated, category=None, image_path=None):
    try:
        conn = sqlite3.connect("language_learning.db")
        cursor = conn.cursor()
        
        # Check if word exists
        cursor.execute(
            "SELECT id FROM vocabulary WHERE word_original = ? AND language_translated = ?",
            (word_original, language_translated)
        )
        existing_word = cursor.fetchone()
        
        if existing_word:
            vocab_id = existing_word[0]
            cursor.execute(
                "UPDATE vocabulary SET word_translated = ?, category = ?, image_path = ? WHERE id = ?",
                (word_translated, category, image_path, vocab_id)
            )
            st.info(f"Word '{word_original}' already exists. Updated with new information.")
        else:
            current_time = datetime.datetime.now()
            try:
                cursor.execute('''
                INSERT INTO vocabulary 
                (word_original, word_translated, language_translated, category, image_path, date_added, source)
                VALUES (?, ?, ?, ?, ?, ?, 'manual')
                ''', (word_original, word_translated, language_translated, category, image_path, current_time))
            except sqlite3.OperationalError:
                cursor.execute('''
                INSERT INTO vocabulary 
                (word_original, word_translated, language_translated, category, image_path, date_added)
                VALUES (?, ?, ?, ?, ?, ?)
                ''', (word_original, word_translated, language_translated, category, image_path, current_time))
            
            vocab_id = cursor.lastrowid
            
            # Initialize user progress
            cursor.execute('''
            INSERT INTO user_progress (vocabulary_id, last_reviewed, proficiency_level)
            VALUES (?, ?, 0)
            ''', (vocab_id, current_time))
        
        conn.commit()
        conn.close()
        return vocab_id
    except Exception as e:
        st.error(f"Error adding vocabulary: {e}")
        return None

def get_all_vocabulary_direct():
    try:
        conn = sqlite3.connect("language_learning.db")
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT v.id, v.word_original, v.word_translated, v.language_translated,
               v.category, v.image_path, v.date_added,
               up.proficiency_level, up.review_count, up.correct_count, up.last_reviewed
        FROM vocabulary v
        LEFT JOIN user_progress up ON v.id = up.vocabulary_id
        ORDER BY v.date_added DESC
        ''')
        
        results = cursor.fetchall()
        vocabulary = [dict(row) for row in results]
        
        conn.close()
        return vocabulary
    except Exception as e:
        st.error(f"Error retrieving vocabulary: {e}")
        return []

def manage_session(action):
    if action == "start":
        session_id = create_session_direct()
        if session_id:
            st.session_state.session_id = session_id
            st.session_state.words_studied = 0
            st.session_state.words_learned = 0
            st.success("Started new learning session!")
            return True
        return False
    elif action == "end" and st.session_state.session_id:
        try:
            conn = sqlite3.connect("language_learning.db")
            cursor = conn.cursor()
            current_time = datetime.datetime.now()
            cursor.execute(
                "UPDATE sessions SET end_time = ?, words_studied = ?, words_learned = ? WHERE id = ?",
                (current_time, st.session_state.words_studied, st.session_state.words_learned, st.session_state.session_id)
            )
            conn.commit()
            conn.close()
            
            st.success(f"Session completed! Words studied: {st.session_state.words_studied}, Words learned: {st.session_state.words_learned}")
            st.session_state.session_id = None
            st.session_state.words_studied = 0
            st.session_state.words_learned = 0
            return True
        except Exception as e:
            st.error(f"Error ending session: {e}")
            return False
    return False

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
if 'detection_checkboxes' not in st.session_state:
    st.session_state.detection_checkboxes = {}

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

elif app_mode == "Quiz Mode" or app_mode == "Statistics":
    st.title(f"🚧 {app_mode} - Coming Soon")
    st.markdown("""
    This feature is currently unavailable in the cloud version due to Python 3.12 compatibility issues.
    
    Please try the Camera Mode and My Vocabulary features, which are fully functional.
    
    We're working on making all features available soon!
    """)
    
    # Show a preview image of what this would look like
    if app_mode == "Quiz Mode":
        st.image("https://raw.githubusercontent.com/streamlit/streamlit/develop/examples/quiz-app/streamlit-quiz-app.jpg", 
                caption="Quiz Mode Preview", use_column_width=True)
    else:
        # Create a simple stats preview
        st.subheader("Sample Statistics (Preview Only)")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Sessions", "5")
        with col2:
            st.metric("Words Studied", "42")
        with col3:
            st.metric("Words Learned", "38")

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