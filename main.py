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
import re
import queue
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from io import BytesIO
from gamification import GamificationSystem
import random
from collections import defaultdict
import io
from vocam_ui import apply_custom_css
from streamlit.components.v1 import components
import hashlib
from functools import lru_cache
from example_sentences import ExampleSentenceGenerator
import tensorflow as tf
import tensorflow_hub as hub
import requests
from deep_translator import GoogleTranslator
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# First, display Python version for
st.set_page_config(
    page_title="Vocam",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

apply_custom_css()

try:
    from cloud_detector import detect_streamlit_cloud
    is_cloud = detect_streamlit_cloud()
except ImportError:
    is_cloud = False

if is_cloud:
    os.environ['IS_STREAMLIT_CLOUD'] = 'true'
    print("Running in Streamlit Cloud - some features may be limited")

try:
    from pronunciation_practice import create_pronunciation_practice
    has_pronunciation_practice = True
    print("✅ Enhanced pronunciation practice with AI feedback loaded")
except ImportError as e:
    has_pronunciation_practice = False
    print(f"❌ Pronunciation practice not available: {e}")
    
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
            except False:
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


def apply_nms(boxes, classes, scores, image_shape, iou_threshold=0.45):
    """Apply Non-Maximum Suppression to remove duplicate detections."""
    
    final_detections = []
    unique_classes = np.unique(classes)
    height, width = image_shape[:2]
    
    for class_id in unique_classes:
        # Get all detections for this class
        class_mask = classes == class_id
        class_boxes = boxes[class_mask]
        class_scores = scores[class_mask]
        
        if len(class_boxes) == 0:
            continue
        
        # Convert normalized coordinates to pixel coordinates
        pixel_boxes = []
        for box in class_boxes:
            ymin, xmin, ymax, xmax = box
            pixel_boxes.append([
                int(xmin * width),   # left
                int(ymin * height),  # top
                int(xmax * width),   # right
                int(ymax * height)   # bottom
            ])
        pixel_boxes = np.array(pixel_boxes)
        
        # Apply simple NMS (since OpenCV might cause issues)
        keep_indices = simple_nms(pixel_boxes, class_scores, iou_threshold)
        
        # Add kept detections to final list
        for idx in keep_indices:
            class_name = COCO_CLASS_NAMES.get(class_id, f"unknown_{class_id}")
            bbox = pixel_boxes[idx]
            
            final_detections.append({
                'label': class_name.lower(),
                'confidence': float(class_scores[idx]),
                'bbox': [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                'class_id': int(class_id)
            })
    
    # Sort detections by confidence (highest first)
    final_detections.sort(key=lambda x: x['confidence'], reverse=True)
    return final_detections

# Alternative simplified NMS function if OpenCV NMS doesn't work
def simple_nms(boxes, scores, iou_threshold=0.5):
    """Simple Non-Maximum Suppression implementation."""
    if len(boxes) == 0:
        return []
    
    # Sort by confidence score (highest first)
    sorted_indices = np.argsort(scores)[::-1]
    
    keep = []
    while len(sorted_indices) > 0:
        # Take the detection with highest confidence
        current = sorted_indices[0]
        keep.append(current)
        
        if len(sorted_indices) == 1:
            break
        
        # Calculate IoU with all other boxes
        current_box = boxes[current]
        remaining_indices = sorted_indices[1:]
        
        # Calculate IoU with remaining boxes
        ious = []
        for idx in remaining_indices:
            iou = calculate_iou(current_box, boxes[idx])
            ious.append(iou)
        
        # Keep only boxes with IoU below threshold
        ious = np.array(ious)
        keep_mask = ious < iou_threshold
        sorted_indices = remaining_indices[keep_mask]
    
    return keep

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
    """Get the category for a detected object label."""
    label = label.lower()
    for category, items in OBJECT_CATEGORIES.items():
        if label in items:
            return category
    return "other"

# Add this to optimize API usage and reduce costs
@lru_cache(maxsize=100)
def cached_vision_detection(image_hash, confidence_threshold):
    """Cache detection results based on image hash to avoid redundant API calls."""
    # This is a placeholder - the actual implementation would be tied to your caching mechanism
    # Return None to indicate cache miss
    return None

def get_image_hash(image):
    """Create a hash of an image for caching purposes."""
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG', quality=70)  # Lower quality for hash stability
    return hashlib.md5(img_byte_arr.getvalue()).hexdigest()

# Add rate limiting to avoid excessive API calls
last_api_call = 0
MIN_API_CALL_INTERVAL = 0.5  # seconds

def rate_limited_detection(image, confidence_threshold=0.5, iou_threshold=0.45):
    """Rate-limited version of detect_objects to avoid excessive API calls."""
    global last_api_call
    
    # Check cache first
    image_hash = get_image_hash(image)
    cached_result = cached_vision_detection(image_hash, confidence_threshold)
    if cached_result:
        return cached_result
    
    # Rate limiting
    current_time = time.time()
    time_since_last_call = current_time - last_api_call
    if time_since_last_call < MIN_API_CALL_INTERVAL:
        time.sleep(MIN_API_CALL_INTERVAL - time_since_last_call)
    
    # Make the API call
    result = detect_objects(image, confidence_threshold, iou_threshold)
    last_api_call = time.time()
    
    return result

# Function to detect objects in image
def detect_objects(image, confidence_threshold=0.5, iou_threshold=0.45):
    """Detect objects using Faster R-CNN with Non-Maximum Suppression to remove duplicates."""
    
    try:
        # Load the Faster R-CNN model
        detector = load_faster_rcnn_model()
        if detector is None:
            error_message("Failed to load Faster R-CNN model")
            return [], np.array(image)
        
        # Convert PIL image to numpy array if needed
        if hasattr(image, 'convert'):
            image_np = np.array(image.convert('RGB'))
        else:
            image_np = np.array(image)
        
        # Convert to tensor
        image_tensor = tf.convert_to_tensor(image_np)
        image_tensor = image_tensor[tf.newaxis, ...]
        
        # Run object detection
        results = detector(image_tensor)
        
        # Extract results
        boxes = results['detection_boxes'][0].numpy()
        classes = results['detection_classes'][0].numpy().astype(int)
        scores = results['detection_scores'][0].numpy()
        
        # Filter by confidence threshold first
        valid_indices = scores >= confidence_threshold
        filtered_boxes = boxes[valid_indices]
        filtered_classes = classes[valid_indices]
        filtered_scores = scores[valid_indices]
        
        if len(filtered_boxes) == 0:
            return [], image_np
        
        # Apply Non-Maximum Suppression to remove duplicate detections
        final_detections = apply_nms(filtered_boxes, filtered_classes, filtered_scores, image_np.shape, iou_threshold)
        
        # Draw bounding boxes on result image
        result_image = draw_detections(image_np, final_detections)
        
        print(f"✅ Faster R-CNN detected {len(final_detections)} unique objects (after NMS)")
        return final_detections, result_image
        
    except Exception as e:
        error_message(f"Faster R-CNN detection error: {str(e)}")
        # Return empty result on error
        dummy_image = np.array(image) if hasattr(image, 'convert') else image
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

if st.sidebar.checkbox("Show Pronunciation Dependencies"):
    deps = check_pronunciation_dependencies()
    st.sidebar.markdown("### Pronunciation Practice Dependencies")
    for dep, available in deps.items():
        status = "✅" if available else "❌"
        st.sidebar.markdown(f"{status} {dep}")
    
    missing = [dep for dep, avail in deps.items() if not avail]
    if missing:
        st.sidebar.markdown("**To install missing dependencies:**")
        st.sidebar.code(f"pip install {' '.join(missing)}")

if 'db_checked' not in st.session_state:
    st.session_state.db_checked = check_database_setup()

def debug_button(label, **kwargs):
    """Debug wrapper that shows what parameters are being passed to a button and ensures uniqueness"""
    import inspect
    import time
    
    # Get the caller info
    caller = inspect.getframeinfo(inspect.currentframe().f_back)
    
    # Create a unique key based on the calling file, line number, and timestamp
    if 'key' not in kwargs:
        caller_id = f"{caller.filename.split('/')[-1]}_{caller.lineno}"
        timestamp = int(time.time() * 1000) % 10000  # Use last 4 digits of timestamp for readability
        unique_key = f"{label.replace(' ', '_')}_{caller_id}_{timestamp}"
        kwargs['key'] = unique_key
    
    # For debugging, uncomment this line to see what keys are being generated
    # print(f"Button: {label}, Key: {kwargs['key']}")
    
    # Remove any problematic parameters if present
    if 'use_column_width' in kwargs:
        del kwargs['use_column_width']
    if 'type' in kwargs and kwargs['type'] == 'primary':
        del kwargs['type']
    
    # Use the cleaned kwargs
    return st.button(label, **kwargs)

def safe_button(label, **kwargs):
    """Safe wrapper for st.button that ensures uniqueness and removes problematic parameters"""
    import time
    
    # Generate a unique key if none provided
    if 'key' not in kwargs:
        # Create unique key based on label and timestamp
        timestamp = int(time.time() * 1000) % 10000  # Use last 4 digits of timestamp for readability
        unique_key = f"{label.replace(' ', '_')}_{timestamp}"
        kwargs['key'] = unique_key
    
    # Remove problematic parameters if present
    if 'use_column_width' in kwargs:
        del kwargs['use_column_width']
    if 'type' in kwargs and kwargs['type'] == 'primary':
        del kwargs['type']
    
    # Use the cleaned kwargs
    return st.button(label, **kwargs)

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
# Add these initializations with your other session state initializations
# Add these initializations with your other session state initializations
if 'audio_data' not in st.session_state:
    st.session_state.audio_data = None
if 'audio_data_received' not in st.session_state:
    st.session_state.audio_data_received = False
if 'current_recording_word' not in st.session_state:
    st.session_state.current_recording_word = None
if 'use_vision_api' not in st.session_state:
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

# Initialize gamification
gamification = get_gamification()
# Make sure state is explicitly initialized
gamification.initialize_state()


def display_model_status():
    """Display the current object detection model status in sidebar."""
    with st.sidebar.expander("🤖 Object Detection Model"):
        st.markdown("**Current Model:** Faster R-CNN")
        st.markdown("**Status:** Active")
        st.markdown("**Source:** TensorFlow Hub")
        st.markdown("**Classes:** 80+ COCO objects")
        
        if st.button("Test Model"):
            try:
                model = load_faster_rcnn_model()
                if model is not None:
                    st.success("✅ Model loaded successfully!")
                else:
                    st.error("❌ Model failed to load")
            except Exception as e:
                st.error(f"❌ Error: {e}")

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


# Function to load RCNN Model
@st.cache_resource
def load_faster_rcnn_model():
    """Load and cache the Faster R-CNN model from TensorFlow Hub."""
    try:
        print("Loading Faster R-CNN model...")
        model_url = "https://tfhub.dev/tensorflow/faster_rcnn/resnet50_v1_640x640/1"
        detector = hub.load(model_url)
        print("✅ Faster R-CNN model loaded successfully!")
        return detector
    except Exception as e:
        print(f"❌ Error loading Faster R-CNN model: {e}")
        return None

# Background worker function for object detection
def detect_objects_worker(image, confidence_threshold, iou_threshold, task_id):
    """Worker function to run detection in background."""
    try:
        # Run detection
        detections, rendered_image = detect_objects(image, confidence_threshold, iou_threshold)
        
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

# Global counter for truly unique widget IDs
if 'widget_counter' not in st.session_state:
    st.session_state.widget_counter = 0

def truly_safe_button(label, **kwargs):
    """Button helper that guarantees unique keys even with rapid clicks"""
    # Increment the global counter
    st.session_state.widget_counter += 1
    
    # Generate a unique key if none provided
    if 'key' not in kwargs:
        # Create unique key based on counter + millisecond timestamp
        import time
        timestamp = int(time.time() * 1000000) % 1000000  # Microsecond part only
        counter = st.session_state.widget_counter
        unique_key = f"{label.replace(' ', '_').lower()}_{counter}_{timestamp}"
        kwargs['key'] = unique_key
    
    # Remove problematic parameters if present
    if 'use_column_width' in kwargs:
        del kwargs['use_column_width']
    if 'type' in kwargs and kwargs['type'] == 'primary':
        kwargs['type'] = None  # Set to None instead of deleting
    
    # Use the cleaned kwargs
    return st.button(label, **kwargs)

def safe_button(label, **kwargs):
    """Alias for truly_safe_button for backward compatibility"""
    return truly_safe_button(label, **kwargs)


# Main sidebar for navigation
st.sidebar.title("🌍 Vocam")
app_mode_options = ["Camera Mode", "My Vocabulary", "Quiz Mode", "Statistics", "My Progress", "Pronunciation Practice"]
if 'app_mode' in st.session_state:
    # Use the session state value as the default index for the selectbox
    default_index = app_mode_options.index(st.session_state.app_mode) if st.session_state.app_mode in app_mode_options else 0
else:
    default_index = 0

app_mode = st.sidebar.selectbox(
    "Choose a mode",
    app_mode_options,
    index=default_index
)

# Update session state with the current selection (might be from selectbox or previous setting)
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
    style_title("📸 Camera Mode")
    # Use the enhanced info message
    info_message("Take a photo or upload an image to identify objects and learn new vocabulary.")
    
    # Session management
    session_container = st.container()
    with session_container:
        col1, col2 = st.columns(2)
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
            image = Image.open(uploaded_file)
    
    # Detection options
    detection_type = st.radio(
        "What would you like to detect?",
        ["Objects", "Text (OCR)"],
        index=0
    )
    
    # Detection settings for objects
    if detection_type == "Objects":
        col1, col2 = st.columns(2)
        with col1:
            confidence_threshold = st.slider(
                "Detection Confidence", 
                min_value=0.3, 
                max_value=0.9, 
                value=0.5,
                step=0.05
            )
        with col2:
            iou_threshold = st.slider(
                "Duplicate Removal", 
                min_value=0.1, 
                max_value=0.9, 
                value=0.45,
                step=0.05,
                help="Lower = fewer duplicates"
            )
        
        # Set iou_threshold for optimal detection (balance between precision and maximum detection)
        iou_threshold = 0.45  # Using a lower threshold to detect more objects while maintaining precision
        
        # Auto-enhancement is always applied
        enhancement_type = "auto"
    
    # Process image if available
    if image is not None:
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
            # Use a placeholder for the spinner that we can clear later
            spinner_placeholder = st.empty()
            with spinner_placeholder.container():
                show_loading_spinner("Detecting objects... This may take a few seconds.")
            
            # Add visual separator for mobile
            separator_placeholder = st.empty()
            separator_placeholder.markdown('<div class="result-separator"></div>', unsafe_allow_html=True)
            
            # Perform the detection while showing a native spinner
            with st.spinner("Processing..."):
                detections, result_image = detect_objects(image_for_detection, confidence_threshold, iou_threshold)
            
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
                                    # Add example sentence directly (no nested expander)
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
                                # Store the saved state and items in session state
                                st.session_state.words_just_saved = True
                                st.session_state.saved_count = saved_count
                                st.session_state.saved_items = saved_items
                                st.rerun()  # Rerun once to update the UI
                            else:
                                error_message("Failed to save any words. Please check database connection.")

                # Show success message and navigation AFTER saving (persists across reruns)
                if st.session_state.words_just_saved:
                    # Create a container for the success message
                    success_container = st.container()
                    
                    with success_container:
                        success_message(f"Successfully added {st.session_state.saved_count} new words to your vocabulary!")
                        
                        # Show saved words in a visually appealing list
                        st.markdown('<h4 style="color: #1679AB;">Words saved:</h4>', unsafe_allow_html=True)
                        for item in st.session_state.saved_items:
                            st.markdown(f"✅ {item}")
                        
                        # Show navigation options
                        st.markdown("### What would you like to do next?")
                        next_col1, next_col2, next_col3 = st.columns(3)
                        
                        # Define navigation callback functions
                        def go_to_quiz_mode():
                            st.session_state.words_just_saved = False  # Reset the saved state
                            st.session_state.app_mode = "Quiz Mode"
                            st.session_state.detection_checkboxes = {}  # Clear checkboxes
                            st.rerun()

                        def go_to_vocabulary():
                            st.session_state.words_just_saved = False  # Reset the saved state
                            st.session_state.app_mode = "My Vocabulary"
                            st.session_state.detection_checkboxes = {}  # Clear checkboxes
                            st.rerun()

                        def continue_capturing():
                            st.session_state.words_just_saved = False
                            st.session_state.detection_checkboxes = {}  # Clear checkboxes
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
                if image_path and os.path.exists(image_path):
                    try:
                        image = Image.open(image_path)
                        st.image(image, caption=f"Image for {word.get('word_original', '')}")
                    except Exception as e:
                        error_message(f"Error loading image: {e}")
                else:
                    st.markdown("*No image available for this word*")
                
                # Add pronunciation practice if available
                if has_pronunciation_practice:
                    try:
                        # Only initialize if not already initialized
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
                            # Handle saving pronunciation words to vocabulary
                            if 'save_pronunciation_word' in st.session_state:
                                word_data = st.session_state.save_pronunciation_word
                                
                                # Auto-start session if needed
                                if st.session_state.session_id is None:
                                    if manage_session("start"):
                                        success_message("Created a new learning session!")
                                
                                # Add to vocabulary
                                vocab_id = add_vocabulary_direct(
                                    word_original=word_data['original'],
                                    word_translated=word_data['translated'],
                                    language_translated=word_data['language'],
                                    category="pronunciation_practice",
                                    image_path=None
                                )
                                
                                if vocab_id:
                                    st.session_state.words_studied += 1
                                    st.session_state.words_learned += 1
                                    
                                    # Check achievements
                                    try:
                                        gamification.check_achievements(
                                            "pronunciation_practice",
                                            word=word_data['original'],
                                            score=word_data['score']
                                        )
                                    except Exception as e:
                                        print(f"Gamification error: {e}")
                                
                                # Clear the save request
                                del st.session_state.save_pronunciation_word
                            
                    except Exception as e:
                        print(f"❌ Error initializing pronunciation practice: {str(e)}")
                        has_pronunciation_practice = False
                        
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
            if st.button(start_label, disabled=len(filtered_vocab) < 4):
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
    style_title("🤖 AI-Powered Pronunciation Practice")
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
    display_model_status()
else:
    st.sidebar.warning("No active session")
    st.sidebar.markdown("*Start a session in Camera Mode to track progress*")
    display_model_status()

add_footer()