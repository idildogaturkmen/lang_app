import os
import cv2
import torch
import time
from google.cloud import vision, translate_v2 as translate
from gtts import gTTS
import pygame
from database import LanguageLearningDB

# Initialize pygame for audio playback
pygame.mixer.init()

# Set up Google Cloud credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'C:\Users\HP\Desktop\Senior Proj\credentials.json'

# Initialize the Vision API client
client = vision.ImageAnnotatorClient()

class LanguageLearningApp:
    def __init__(self):
        """Initialize the language learning app."""
        # Connect to database
        self.db = LanguageLearningDB("language_learning.db")
        
        # Default target language
        self.target_language = "es"  # Spanish
        
        # Initialize video capture
        self.cap = None
        
        # Load object detection model
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.model.eval()
        
        # Track objects already recognized in current session
        self.recognized_objects = set()
        
        # Track session stats
        self.session_id = None
        self.words_studied = 0
        self.words_learned = 0
        
    def translate_text(self, text, target_language):
        """Translates text into the target language."""
        try:
            translate_client = translate.Client()
            result = translate_client.translate(text, target_language=target_language)
            return result["translatedText"]
        except Exception as e:
            print(f"Translation error: {e}")
            return text  # Return original text if translation fails
    
    def play_pronunciation(self, text, lang='en'):
        """Converts text to speech and plays it."""
        try:
            tts = gTTS(text=text, lang=lang, slow=False)
            temp_file = "temp.mp3"
            tts.save(temp_file)
            
            # Use pygame for more controlled audio playback
            pygame.mixer.music.load(temp_file)
            pygame.mixer.music.play()
            # Wait for audio to finish playing
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
        except Exception as e:
            print(f"Pronunciation error: {e}")
    
    def save_frame(self, frame, label):
        """Save the current frame as an image for the recognized object."""
        try:
            # Create directory if it doesn't exist
            os.makedirs("object_images", exist_ok=True)
            
            # Create a clean filename from the label
            filename = f"object_images/{label}_{int(time.time())}.jpg"
            
            # Save the image
            cv2.imwrite(filename, frame)
            
            return filename
        except Exception as e:
            print(f"Error saving frame: {e}")
            return None
    
    def start_session(self):
        """Start a new learning session."""
        self.session_id = self.db.start_session()
        self.words_studied = 0
        self.words_learned = 0
        self.recognized_objects = set()
        print(f"Started session {self.session_id}")
    
    def end_session(self):
        """End the current learning session."""
        if self.session_id:
            self.db.end_session(self.session_id, self.words_studied, self.words_learned)
            print(f"Ended session {self.session_id}")
            print(f"Words studied: {self.words_studied}, Words learned: {self.words_learned}")
            self.session_id = None
    
    def run_camera_mode(self):
        """Run the app in camera mode for real-time object recognition."""
        # Start a new session
        self.start_session()
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(0)  # 0 is typically the default camera
        if not self.cap.isOpened():
            print("Error: Could not open webcam.")
            return

        running = True
        try:
            while running:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break

                # Convert frame to RGB for model input
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Perform object detection
                results = self.model(rgb_frame)

                # Process results
                for detection in results.xyxy[0]:
                    xmin, ymin, xmax, ymax, confidence, class_idx = detection
                    label = results.names[int(class_idx)]
                    
                    # Only process high confidence detections
                    if confidence > 0.6:  # Increased threshold for higher accuracy
                        # Draw bounding box
                        cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
                        
                        # Only translate and save new objects
                        if label not in self.recognized_objects:
                            # Translate the label
                            translated_label = self.translate_text(label, self.target_language)
                            
                            # Save the image
                            image_path = self.save_frame(frame, label)
                            
                            # Add to database
                            vocab_id = self.db.add_vocabulary(
                                word_original=label,
                                word_translated=translated_label,
                                language_translated=self.target_language,
                                category="object",
                                image_path=image_path
                            )
                            
                            # Play pronunciation
                            self.play_pronunciation(translated_label, lang=self.target_language)
                            
                            # Update session stats
                            self.recognized_objects.add(label)
                            self.words_studied += 1
                            self.words_learned += 1
                            
                            # Display info on screen with both original and translated words
                            cv2.putText(frame, f"{label}: {translated_label}", 
                                       (int(xmin), int(ymin) - 10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36, 255, 12), 2)
                        else:
                            # Get translation from database for known objects
                            vocab = self.db.get_all_vocabulary(self.target_language)
                            for word in vocab:
                                if word['word_original'] == label:
                                    translated_label = word['word_translated']
                                    break
                            
                            # Display info for recognized objects
                            cv2.putText(frame, f"{label}: {translated_label}", 
                                       (int(xmin), int(ymin) - 10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36, 255, 12), 2)

                # Display session stats on screen
                cv2.putText(frame, f"Session: Words learned: {self.words_learned}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Display the frame
                cv2.imshow('Language Learning App - Press Q to Exit', frame)

                # Check for exit key
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    running = False
                    break
                elif key == ord('c'):  # Change language
                    # Simple cycling through a few languages
                    lang_cycle = ['es', 'fr', 'de', 'it']
                    current_index = lang_cycle.index(self.target_language) if self.target_language in lang_cycle else 0
                    next_index = (current_index + 1) % len(lang_cycle)
                    self.target_language = lang_cycle[next_index]
                    print(f"Changed target language to: {self.target_language}")

        except Exception as e:
            print(f"Error occurred: {e}")
        
        finally:
            # End session and clean up resources
            self.end_session()
            
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
            
            # Remove temporary audio file
            if os.path.exists("temp.mp3"):
                try:
                    os.remove("temp.mp3")
                except:
                    pass
            
            print("Camera mode closed")
    
    def close(self):
        """Close all resources."""
        self.db.close()
        print("App resources closed")


# Main execution
if __name__ == "__main__":
    app = LanguageLearningApp()
    try:
        app.run_camera_mode()
    finally:
        app.close()