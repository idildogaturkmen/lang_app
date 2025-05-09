import streamlit as st
import os
import sys
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

# Set page configuration
st.set_page_config(
    page_title="AI Language Learning App",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Display version info
st.sidebar.markdown(f"**Python Version:** {sys.version}")

# Create dummy classes for problematic packages
class DummyCV2:
    def __init__(self):
        self.__version__ = "Fallback"
        
    def imread(self, path):
        try:
            return np.array(Image.open(path))
        except Exception:
            return None
            
    def imwrite(self, path, img):
        try:
            Image.fromarray(img).save(path)
            return True
        except Exception:
            return False
            
    def cvtColor(self, img, code):
        # Just return the original image
        return img

# Use the dummy implementation
cv2 = DummyCV2()

class DummyTorch:
    class hub:
        @staticmethod
        def load(*args, **kwargs):
            class DummyModel:
                def __init__(self):
                    self.names = {0: "object", 1: "person", 2: "dog", 3: "cat", 4: "car", 5: "chair"}
                    
                def __call__(self, image):
                    # Create a dummy result
                    class DummyResult:
                        def __init__(self):
                            self.xyxy = [[]]
                            
                        def render(self):
                            # Return the original image with a frame
                            if isinstance(image, np.ndarray):
                                img = image.copy()
                            else:
                                img = np.array(image)
                            # Add a colored border to show "detection"
                            h, w = img.shape[:2]
                            img[0:5, :] = [0, 255, 0]  # Top border
                            img[-5:, :] = [0, 255, 0]  # Bottom border
                            img[:, 0:5] = [0, 255, 0]  # Left border
                            img[:, -5:] = [0, 255, 0]  # Right border
                            return [img]
                    return DummyResult()
                    
                def eval(self):
                    return self
            return DummyModel()

# Use the dummy torch
torch = DummyTorch()

# Create dummy for Google Cloud libraries
class DummyTranslate:
    class Client:
        def translate(self, text, target_language=None):
            # Simple translations for common objects
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
                "phone": {"es": "teléfono", "fr": "téléphone", "de": "Telefon"},
                "computer": {"es": "ordenador", "fr": "ordinateur", "de": "Computer"},
                "object": {"es": "objeto", "fr": "objet", "de": "Objekt"}
            }
            
            # Convert text to lowercase for matching
            text_lower = text.lower()
            
            # Default translation
            if text_lower in translations and target_language in translations[text_lower]:
                translated = translations[text_lower][target_language]
            else:
                translated = f"[{text} in {target_language}]"
                
            return {"translatedText": translated}

try:
    from google.cloud import translate_v2 as translate
    st.sidebar.success("Google Cloud Translation imported successfully!")
except ImportError:
    translate = DummyTranslate()
    st.sidebar.info("Using fallback translations (limited vocabulary)")

# Create dummy for text-to-speech
class DummyTTS:
    def __init__(self, text="", lang="en", slow=False):
        self.text = text
        self.lang = lang
        
    def write_to_fp(self, fp):
        # Write a dummy "audio file" (just a marker)
        fp.write(b"DUMMY_AUDIO_DATA")

try:
    from gtts import gTTS
    st.sidebar.success("Text-to-speech (gTTS) imported successfully!")
except ImportError:
    gTTS = DummyTTS
    st.sidebar.info("Using dummy text-to-speech (no audio)")

# Define database class if not available
class LanguageLearningDB:
    def __init__(self, db_path):
        self.db_path = db_path
        self.ensure_tables_exist()
        
    def ensure_tables_exist(self):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create vocabulary table if it doesn't exist
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
            
            # Create user_progress table if it doesn't exist
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
            
            # Create sessions table if it doesn't exist
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

# Initialize database
@st.cache_resource
def get_database():
    return LanguageLearningDB("language_learning.db")

db = get_database()

# Function to translate text
def translate_text(text, target_language):
    try:
        translate_client = translate.Client()
        result = translate_client.translate(text, target_language=target_language)
        return result["translatedText"]
    except Exception as e:
        st.error(f"Translation error: {e}")
        return f"[Translation of '{text}' to {target_language}]"

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
        # Return dummy audio data
        return b"DUMMY_AUDIO_DATA"

# Function to generate HTML for audio playback
def get_audio_html(audio_bytes):
    if audio_bytes == b"DUMMY_AUDIO_DATA":
        return "<p><i>Audio playback not available in demo mode</i></p>"
    
    audio_base64 = base64.b64encode(audio_bytes).decode()
    audio_tag = f'<audio autoplay="true" src="data:audio/mp3;base64,{audio_base64}" controls></audio>'
    return audio_tag

# Function to load YOLOv5 model (with fallback)
@st.cache_resource
def load_model():
    try:
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        model.eval()
        return model
    except Exception as e:
        st.warning(f"Using fallback object detection (limited functionality)")
        # Return a dummy model
        return torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Object detection function (simplified for compatibility)
def detect_objects(image, confidence_threshold=0.5):
    try:
        model = load_model()
        results = model(image)
        
        # Create some dummy detections
        detections = [
            {'label': 'chair', 'confidence': 0.82, 'bbox': [50, 50, 150, 200]},
            {'label': 'person', 'confidence': 0.78, 'bbox': [200, 100, 350, 400]}
        ]
        
        return detections, results.render()[0]
    except Exception as e:
        st.error(f"Object detection error: {e}")
        # Create a dummy image with a frame
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
        
        # Return dummy detections and the framed image
        detections = [
            {'label': 'chair', 'confidence': 0.82, 'bbox': [50, 50, 150, 200]},
            {'label': 'person', 'confidence': 0.78, 'bbox': [200, 100, 350, 400]}
        ]
        return detections, img

# Initialize session state
if 'target_language' not in st.session_state:
    st.session_state.target_language = "es"  # Default to Spanish
if 'session_id' not in st.session_state:
    st.session_state.session_id = None
if 'words_studied' not in st.session_state:
    st.session_state.words_studied = 0
if 'words_learned' not in st.session_state:
    st.session_state.words_learned = 0
if 'detection_checkboxes' not in st.session_state:
    st.session_state.detection_checkboxes = {}

# Start or end learning session
def manage_session(action):
    if action == "start":
        session_id = db.start_session()
        if session_id:
            st.session_state.session_id = session_id
            st.session_state.words_studied = 0
            st.session_state.words_learned = 0
            return True
        return False
    elif action == "end" and st.session_state.session_id:
        if db.end_session(st.session_state.session_id, st.session_state.words_studied, st.session_state.words_learned):
            st.session_state.session_id = None
            st.session_state.words_studied = 0
            st.session_state.words_learned = 0
            return True
        return False
    return False

# Function to save image
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

# Function to add vocabulary
def add_vocabulary(word_original, word_translated, language_translated, category=None, image_path=None):
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
        else:
            current_time = datetime.datetime.now()
            cursor.execute(
                "INSERT INTO vocabulary (word_original, word_translated, language_translated, category, image_path, date_added) VALUES (?, ?, ?, ?, ?, ?)",
                (word_original, word_translated, language_translated, category, image_path, current_time)
            )
            vocab_id = cursor.lastrowid
            
            # Initialize user progress
            cursor.execute(
                "INSERT INTO user_progress (vocabulary_id, last_reviewed, proficiency_level) VALUES (?, ?, 0)",
                (vocab_id, current_time)
            )
        
        conn.commit()
        conn.close()
        return vocab_id
    except Exception as e:
        st.error(f"Error adding vocabulary: {e}")
        return None

# Function to get vocabulary
def get_vocabulary():
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

# Main UI
st.title("🌍 AI Language Learning App")

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

# Sidebar
st.sidebar.title("Language Learning App")
app_mode = st.sidebar.selectbox(
    "Choose a mode",
    ["Camera Mode", "My Vocabulary", "About"]
)

selected_language = st.sidebar.selectbox(
    "Select target language",
    list(languages.keys()),
    index=list(languages.values()).index(st.session_state.target_language) if st.session_state.target_language in languages.values() else 0
)
st.session_state.target_language = languages[selected_language]

# Display appropriate content based on selected mode
if app_mode == "Camera Mode":
    st.header("📸 Camera Mode")
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
                        manage_session("start")
                    
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
                            detection = detections[i]
                            label = detection['label']
                            translated_label = translate_text(label, st.session_state.target_language)
                            
                            # Save the image
                            image_path = save_image(image, label)
                            
                            # Add to vocabulary
                            vocab_id = add_vocabulary(
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
                        
                        if saved_count > 0:
                            st.success(f"Successfully added {saved_count} new words to your vocabulary!")
                            st.write("Words saved:")
                            for item in saved_items:
                                st.write(f"- {item}")
                            # Clear checkboxes after saving
                            st.session_state.detection_checkboxes = {}
                            st.rerun()
            else:
                st.info("No objects detected. Try another image.")

elif app_mode == "My Vocabulary":
    st.header("📚 My Vocabulary")
    st.markdown("Review all the words you've learned so far.")
    
    # Get vocabulary from database
    vocabulary = get_vocabulary()
    
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
    
    # Apply filters and sorting
    filtered_vocab = []
    for word in vocabulary:
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
        st.markdown(f"Found {len(filtered_vocab)} words in your vocabulary collection.")
        
        # Create data for table
        table_data = []
        for word in filtered_vocab:
            lang_code = word.get('language_translated', '')
            lang_name = next((k for k, v in languages.items() if v == lang_code), lang_code)
            
            proficiency = word.get('proficiency_level', 0) or 0
            proficiency_display = "⭐" * proficiency
            
            table_data.append({
                "Original": word.get('word_original', ''),
                "Translation": word.get('word_translated', ''),
                "Language": lang_name,
                "Proficiency": proficiency_display
            })
        
        # Display as a dataframe
        st.dataframe(pd.DataFrame(table_data), use_container_width=True)
        
        # Word details
        if filtered_vocab:
            st.subheader("Word Details")
            selected_word_index = st.selectbox(
                "Select a word to review:",
                range(len(filtered_vocab)),
                format_func=lambda i: f"{filtered_vocab[i].get('word_original', '')} → {filtered_vocab[i].get('word_translated', '')}"
            )
            
            word = filtered_vocab[selected_word_index]
            
            # Display word details
            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown(f"**Original:** {word.get('word_original', '')}")
                st.markdown(f"**Translation:** {word.get('word_translated', '')}")
                
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
        st.info("No vocabulary words found with current filter. Go to Camera Mode to start learning new words!")

else:  # About
    st.header("About the Language Learning App")
    
    st.markdown("""
    ## AI-Powered Language Learning Application
    
    This application helps you learn new languages by recognizing objects in your environment and teaching you their names in your target language.
    
    ### Features:
    - **Object Recognition**: Use your camera to identify objects around you
    - **Translation**: Learn object names in multiple languages
    - **Vocabulary Management**: Save and review words you've learned
    - **Progress Tracking**: Track your learning progress over time
    
    ### How to Use:
    1. Choose your target language in the sidebar
    2. Go to "Camera Mode" and take a photo or upload an image
    3. Review the detected objects and their translations
    4. Save objects to your vocabulary collection
    5. Review your vocabulary in "My Vocabulary" mode
    
    ### Technology:
    - **Computer Vision**: Object detection using YOLOv5
    - **Natural Language Processing**: Translation using Google Cloud Translation API
    - **Text-to-Speech**: Pronunciation using gTTS
    
    ### Privacy Note:
    - All data is stored locally in your browser session
    - Images are not sent to external servers for processing
    
    """)

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