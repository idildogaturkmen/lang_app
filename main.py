import os
import cv2
import torch
from google.cloud import vision, translate_v2 as translate
from gtts import gTTS

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'C:\Users\HP\Desktop\Senior Proj\credentials.json'

# Initialize the Vision API client
client = vision.ImageAnnotatorClient()

def translate_text(text, target_language):
    """Translates text into the target language."""
    translate_client = translate.Client()
    result = translate_client.translate(text, target_language=target_language)
    return result["translatedText"]

def play_pronunciation(text, lang='en'):
    """Converts text to speech and plays it."""
    tts = gTTS(text=text, lang=lang, slow=False)
    tts.save("temp.mp3")
    os.system("start temp.mp3")  # For Windows; use 'open' for macOS or 'xdg-open' for Linux

# Initialize video capture
cap = cv2.VideoCapture(0)  # 0 is typically the default camera
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.eval()

target_language = "es"  # Spanish
played_labels = set()  # Set to track played labels

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform object detection
    results = model(rgb_frame)

    # Process results
    for detection in results.xyxy[0]:
        xmin, ymin, xmax, ymax, confidence, class_idx = detection
        label = results.names[int(class_idx)]
        translated_label = translate_text(label, target_language)
        
        # Draw bounding box and label
        cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
        cv2.putText(frame, translated_label, (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        
        # Play pronunciation if not already played
        if translated_label not in played_labels:
            play_pronunciation(translated_label, lang=target_language)
            played_labels.add(translated_label)

    # Display the frame
    cv2.imshow('Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
