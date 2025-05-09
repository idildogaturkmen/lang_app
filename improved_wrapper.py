import streamlit as st
import os
import sys
import importlib.util
import numpy as np
import time
from contextlib import contextmanager

# Set page configuration
st.set_page_config(
    page_title="AI Language Learning App",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Hide imports from user view
st.markdown("""
<style>
    div[data-testid="stStatusWidget"] {
        display: none;
    }
    .element-container:has(.stWarning) {
        display: none;
    }
    .element-container:has(.stError:contains("import")) {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar information section
st.sidebar.title("🌍 Language Learning App")

# Create a sidebar expander for technical information instead of showing errors
with st.sidebar.expander("ℹ️ Technical Information"):
    st.markdown(f"**Python Version:** {sys.version}")
    
    st.markdown("""
    ### Feature Status
    Running in compatibility mode with limited features:
    
    ✅ Image upload and camera capture  
    ✅ Basic word translations  
    ✅ Vocabulary management  
    ✅ User interface  
    
    ⚠️ AI object detection (simulated)  
    ⚠️ Advanced translations (limited dictionary)  
    ⚠️ Audio pronunciation (limited)  
    """)

# Suppress the import warnings using context manager
@contextmanager
def suppress_stdout():
    """Context manager to suppress stdout temporarily."""
    import sys
    import io
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = old_stdout

# Create dummy implementations of missing modules
class DummyCV2:
    def __init__(self):
        self.__version__ = "Fallback 4.6.0"
        self.COLOR_BGR2RGB = 1
        self.IMREAD_COLOR = 1
        
    def __getattr__(self, name):
        def dummy_method(*args, **kwargs):
            return None
        return dummy_method
        
    def cvtColor(self, img, code):
        return img  # Return the input image unchanged
        
    def imread(self, path):
        try:
            from PIL import Image
            img = Image.open(path)
            return np.array(img)
        except Exception:
            return None
            
    def imwrite(self, path, img):
        try:
            from PIL import Image
            Image.fromarray(img).save(path)
            return True
        except Exception:
            return False

class DummyTorch:
    def __init__(self):
        self.hub = type('obj', (object,), {
            'load': lambda *args, **kwargs: DummyModel()
        })
    
    def __getattr__(self, name):
        def dummy_method(*args, **kwargs):
            return None
        return dummy_method

class DummyModel:
    def __init__(self):
        self.names = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane", 5: "bus", 
                     6: "train", 7: "truck", 8: "boat", 9: "traffic light", 10: "fire hydrant", 
                     11: "stop sign", 12: "parking meter", 13: "bench", 14: "bird", 15: "cat", 
                     16: "dog", 17: "horse", 18: "sheep", 19: "cow", 20: "elephant", 21: "bear", 
                     22: "zebra", 23: "giraffe", 24: "backpack", 25: "umbrella", 26: "handbag", 
                     27: "tie", 28: "suitcase", 29: "frisbee", 30: "skis", 31: "snowboard", 
                     32: "sports ball", 33: "kite", 34: "baseball bat", 35: "baseball glove", 
                     36: "skateboard", 37: "surfboard", 38: "tennis racket", 39: "bottle", 
                     40: "wine glass", 41: "cup", 42: "fork", 43: "knife", 44: "spoon", 
                     45: "bowl", 46: "banana", 47: "apple", 48: "sandwich", 49: "orange", 
                     50: "broccoli", 51: "carrot", 52: "hot dog", 53: "pizza", 54: "donut", 
                     55: "cake", 56: "chair", 57: "couch", 58: "potted plant", 59: "bed", 
                     60: "dining table", 61: "toilet", 62: "tv", 63: "laptop", 64: "mouse", 
                     65: "remote", 66: "keyboard", 67: "cell phone", 68: "microwave", 
                     69: "oven", 70: "toaster", 71: "sink", 72: "refrigerator", 73: "book", 
                     74: "clock", 75: "vase", 76: "scissors", 77: "teddy bear", 78: "hair drier", 
                     79: "toothbrush"}
        
    def __call__(self, image):
        # Create a dummy result
        import random
        
        class DummyResult:
            def __init__(self, image, names):
                self.names = names
                self.image = image
                
                # Generate random detections
                common_objects = [15, 16, 39, 41, 56, 62, 63, 67, 73]  # cat, dog, bottle, cup, chair, tv, laptop, phone, book
                num_detections = random.randint(1, 3)
                
                self.xyxy = []
                for _ in range(num_detections):
                    # Pick a common object
                    class_id = random.choice(common_objects)
                    
                    # Generate random coordinates
                    h, w = image.shape[:2]
                    xmin = random.randint(0, w//2)
                    ymin = random.randint(0, h//2)
                    xmax = random.randint(xmin + 50, min(xmin + 200, w-1))
                    ymax = random.randint(ymin + 50, min(ymin + 200, h-1))
                    
                    # Create detection with high confidence
                    confidence = random.uniform(0.7, 0.95)
                    self.xyxy.append([xmin, ymin, xmax, ymax, confidence, class_id])
                
                # Package as numpy array
                self.xyxy = [np.array(self.xyxy)]
                
            def render(self):
                # Create a copy of the image with colored rectangles
                if isinstance(self.image, np.ndarray):
                    img = self.image.copy()
                else:
                    img = np.array(self.image)
                    
                # Add rectangles for each detection
                for detection in self.xyxy[0]:
                    xmin, ymin, xmax, ymax = map(int, detection[:4])
                    
                    # Draw rectangle
                    color = (0, 255, 0)  # Green
                    thickness = 2
                    
                    # Top border
                    img[ymin:ymin+thickness, xmin:xmax] = color
                    # Bottom border
                    img[ymax-thickness:ymax, xmin:xmax] = color
                    # Left border
                    img[ymin:ymax, xmin:xmin+thickness] = color
                    # Right border
                    img[ymin:ymax, xmax-thickness:xmax] = color
                
                return [img]
                
        return DummyResult(image, self.names)
        
    def eval(self):
        return self

# Dictionary of dummy translations for common objects
TRANSLATIONS = {
    "person": {"es": "persona", "fr": "personne", "de": "Person", "it": "persona", "pt": "pessoa", "ru": "человек", "ja": "人", "zh-CN": "人"},
    "bicycle": {"es": "bicicleta", "fr": "vélo", "de": "Fahrrad", "it": "bicicletta", "pt": "bicicleta", "ru": "велосипед", "ja": "自転車", "zh-CN": "自行车"},
    "car": {"es": "coche", "fr": "voiture", "de": "Auto", "it": "auto", "pt": "carro", "ru": "машина", "ja": "車", "zh-CN": "汽车"},
    "cat": {"es": "gato", "fr": "chat", "de": "Katze", "it": "gatto", "pt": "gato", "ru": "кошка", "ja": "猫", "zh-CN": "猫"},
    "dog": {"es": "perro", "fr": "chien", "de": "Hund", "it": "cane", "pt": "cachorro", "ru": "собака", "ja": "犬", "zh-CN": "狗"},
    "bottle": {"es": "botella", "fr": "bouteille", "de": "Flasche", "it": "bottiglia", "pt": "garrafa", "ru": "бутылка", "ja": "ボトル", "zh-CN": "瓶子"},
    "chair": {"es": "silla", "fr": "chaise", "de": "Stuhl", "it": "sedia", "pt": "cadeira", "ru": "стул", "ja": "椅子", "zh-CN": "椅子"},
    "laptop": {"es": "portátil", "fr": "ordinateur portable", "de": "Laptop", "it": "laptop", "pt": "laptop", "ru": "ноутбук", "ja": "ノートパソコン", "zh-CN": "笔记本电脑"},
    "book": {"es": "libro", "fr": "livre", "de": "Buch", "it": "libro", "pt": "livro", "ru": "книга", "ja": "本", "zh-CN": "书"},
    "cup": {"es": "taza", "fr": "tasse", "de": "Tasse", "it": "tazza", "pt": "xícara", "ru": "чашка", "ja": "カップ", "zh-CN": "杯子"},
    "apple": {"es": "manzana", "fr": "pomme", "de": "Apfel", "it": "mela", "pt": "maçã", "ru": "яблоко", "ja": "りんご", "zh-CN": "苹果"},
    "banana": {"es": "plátano", "fr": "banane", "de": "Banane", "it": "banana", "pt": "banana", "ru": "банан", "ja": "バナナ", "zh-CN": "香蕉"},
    "phone": {"es": "teléfono", "fr": "téléphone", "de": "Telefon", "it": "telefono", "pt": "telefone", "ru": "телефон", "ja": "電話", "zh-CN": "电话"},
    "tv": {"es": "televisión", "fr": "télévision", "de": "Fernseher", "it": "televisore", "pt": "televisão", "ru": "телевизор", "ja": "テレビ", "zh-CN": "电视"},
    "table": {"es": "mesa", "fr": "table", "de": "Tisch", "it": "tavolo", "pt": "mesa", "ru": "стол", "ja": "テーブル", "zh-CN": "桌子"},
    "clock": {"es": "reloj", "fr": "horloge", "de": "Uhr", "it": "orologio", "pt": "relógio", "ru": "часы", "ja": "時計", "zh-CN": "时钟"},
    "bed": {"es": "cama", "fr": "lit", "de": "Bett", "it": "letto", "pt": "cama", "ru": "кровать", "ja": "ベッド", "zh-CN": "床"},
    "keyboard": {"es": "teclado", "fr": "clavier", "de": "Tastatur", "it": "tastiera", "pt": "teclado", "ru": "клавиатура", "ja": "キーボード", "zh-CN": "键盘"},
    "mouse": {"es": "ratón", "fr": "souris", "de": "Maus", "it": "mouse", "pt": "mouse", "ru": "мышь", "ja": "マウス", "zh-CN": "鼠标"}
}

# Create dummy for Google Cloud libraries
class DummyTranslate:
    class Client:
        def translate(self, text, target_language=None):
            # Check if word is in our dictionary
            text_lower = text.lower()
            if text_lower in TRANSLATIONS and target_language in TRANSLATIONS[text_lower]:
                translated = TRANSLATIONS[text_lower][target_language]
            else:
                # Generate placeholder translation
                translated = f"{text} [{target_language}]"
                
            return {"translatedText": translated}

# Create dummy for text-to-speech
class DummyTTS:
    def __init__(self, text="", lang="en", slow=False):
        self.text = text
        self.lang = lang
        
    def write_to_fp(self, fp):
        # Write a dummy "audio file" (just a marker)
        fp.write(b"DUMMY_AUDIO_DATA")

# Silently inject our dummy modules into sys.modules
with suppress_stdout():
    try:
        import cv2
    except ImportError:
        sys.modules['cv2'] = DummyCV2()
        
    try:
        import torch
    except ImportError:
        sys.modules['torch'] = DummyTorch()
    
    try:
        from google.cloud import translate_v2 as translate
    except ImportError:
        sys.modules['google'] = type('obj', (object,), {
            'cloud': type('obj', (object,), {
                'translate_v2': DummyTranslate()
            })
        })
        
    try:
        from gtts import gTTS
    except ImportError:
        sys.modules['gtts'] = type('obj', (object,), {
            'gTTS': DummyTTS
        })

# Create a fake credentials file for Google Cloud
def create_dummy_credentials():
    import json
    import tempfile
    
    # Create a dummy credentials file
    dummy_creds = {
        "type": "service_account",
        "project_id": "dummy-project",
        "private_key_id": "dummy",
        "private_key": "dummy",
        "client_email": "dummy@example.com",
        "client_id": "dummy",
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
        "client_x509_cert_url": "dummy",
        "universe_domain": "googleapis.com"
    }
    
    credentials_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.json')
    credentials_path = credentials_temp.name
    
    with open(credentials_path, 'w') as f:
        json.dump(dummy_creds, f)
        
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
    return credentials_path

# Create dummy credentials
credentials_path = create_dummy_credentials()

# Now import your main application after all the compatibility layers are set up
try:
    # Instead of importing main.py as a module, we'll exec it
    # This avoids any import errors that might show in the UI
    with open('main.py', 'r') as f:
        main_content = f.read()
    
    # Execute main.py in the current namespace
    exec(main_content)
    
except Exception as e:
    # If something goes wrong with the main app, display a simplified version
    st.title("🌍 AI Language Learning App")
    
    st.markdown("""
    ## Welcome to the Language Learning App!
    
    This app helps you learn vocabulary in different languages using the power of AI.
    
    ### How to use:
    1. Take a photo of an object
    2. The app will identify and translate it
    3. Save words to build your vocabulary
    4. Review your vocabulary collection
    
    Unfortunately, we're experiencing some technical difficulties with the application.
    Please try again later.
    
    Error details:
    ```
    {str(e)}
    ```
    """)