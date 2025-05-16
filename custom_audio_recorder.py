# custom_audio_recorder.py

import os
import streamlit.components.v1 as components
import streamlit as st
import base64
import time

# Define the component directory
COMPONENT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "components")

# Create the directory if it doesn't exist
os.makedirs(COMPONENT_DIR, exist_ok=True)

# Create the HTML file for the component
HTML_FILE = os.path.join(COMPONENT_DIR, "audio_recorder.html")

# Write the HTML/JavaScript code to the file
with open(HTML_FILE, "w") as f:
    f.write("""
<!DOCTYPE html>
<html>
<head>
    <title>Audio Recorder</title>
    <style>
        .recorder-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 15px;
            background-color: #f5f5f5;
            border-radius: 8px;
            margin: 10px 0;
        }
        .record-button {
            background-color: #ff4b4b;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 24px;
            cursor: pointer;
            font-size: 16px;
            margin: 10px;
            display: flex;
            align-items: center;
        }
        .record-button.recording {
            background-color: #ff0000;
            animation: pulse 1.5s infinite;
        }
        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(255, 0, 0, 0.7); }
            70% { box-shadow: 0 0 0 10px rgba(255, 0, 0, 0); }
            100% { box-shadow: 0 0 0 0 rgba(255, 0, 0, 0); }
        }
        .status {
            margin-top: 10px;
            font-style: italic;
        }
    </style>
</head>
<body>
    <div class="recorder-container">
        <button id="recordButton" class="record-button">
            <span id="buttonText">Start Recording</span>
        </button>
        <div id="status" class="status">Ready to record</div>
        <audio id="audioPlayback" controls style="display: none; margin-top: 10px; width: 100%;"></audio>
    </div>

    <script>
        // Audio recording functionality
        let audioChunks = [];
        let mediaRecorder;
        let audioBlob;
        let isRecording = false;
        
        // Elements
        const recordButton = document.getElementById('recordButton');
        const buttonText = document.getElementById('buttonText');
        const statusText = document.getElementById('status');
        const audioPlayback = document.getElementById('audioPlayback');
        
        // Set up event listeners
        recordButton.addEventListener('click', toggleRecording);
        
        // Toggle recording state
        function toggleRecording() {
            if (!isRecording) {
                startRecording();
            } else {
                stopRecording();
            }
        }
        
        // Start recording function
        async function startRecording() {
            audioChunks = [];
            audioPlayback.style.display = 'none';
            
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                mediaRecorder = new MediaRecorder(stream);
                
                mediaRecorder.ondataavailable = (event) => {
                    audioChunks.push(event.data);
                };
                
                mediaRecorder.onstop = processRecording;
                
                // Start recording
                mediaRecorder.start();
                isRecording = true;
                recordButton.classList.add('recording');
                buttonText.textContent = 'Stop Recording';
                statusText.textContent = 'Recording in progress...';
            } catch (err) {
                console.error('Error accessing microphone:', err);
                statusText.textContent = 'Error: Could not access microphone';
            }
        }
        
        // Stop recording function
        function stopRecording() {
            if (mediaRecorder && isRecording) {
                mediaRecorder.stop();
                isRecording = false;
                recordButton.classList.remove('recording');
                buttonText.textContent = 'Start Recording';
                statusText.textContent = 'Processing recording...';
            }
        }
        
        // Process the recording after stopping
        function processRecording() {
            audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
            const audioUrl = URL.createObjectURL(audioBlob);
            audioPlayback.src = audioUrl;
            audioPlayback.style.display = 'block';
            
            // Convert to base64 for passing to Streamlit
            const reader = new FileReader();
            reader.readAsDataURL(audioBlob);
            reader.onloadend = function() {
                const base64data = reader.result.split(',')[1];
                // Send to Streamlit
                sendToStreamlit(base64data);
                statusText.textContent = 'Recording complete!';
            };
        }
        
        // Send data to Streamlit
        function sendToStreamlit(base64AudioData) {
            if (window.Streamlit) {
                window.Streamlit.setComponentValue({
                    data: base64AudioData,
                    format: 'audio/wav'
                });
            }
        }
        
        // Initialize communication with Streamlit
        if (window.Streamlit) {
            window.Streamlit.componentReady();
        }
    </script>
</body>
</html>
    """)

# Create a very simple global counter to make each component unique
_recorder_counter = 0

# Create the custom component function
def audio_recorder():
    """Custom audio recorder component with JavaScript"""
    global _recorder_counter
    
    try:
        # Increment the counter
        _recorder_counter += 1
        
        # Get the component value - DO NOT USE THE KEY PARAMETER AT ALL
        component_value = components.html(
            open(HTML_FILE, "r").read(),
            height=200
        )
        
        # Process the returned value
        if component_value and isinstance(component_value, dict) and 'data' in component_value:
            # Decode the base64 audio data
            audio_bytes = base64.b64decode(component_value['data'])
            
            # Store in session state
            st.session_state.audio_data = audio_bytes
            st.session_state.audio_data_received = True
            
            # Return the audio bytes
            return audio_bytes
        
        return None
    except Exception as e:
        st.error(f"Error in audio recorder component: {e}")
        return None