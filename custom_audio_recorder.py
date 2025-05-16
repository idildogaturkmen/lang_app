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
        .visualization {
            width: 100%;
            height: 60px;
            background-color: #eee;
            margin: 10px 0;
            position: relative;
            border-radius: 4px;
            overflow: hidden;
        }
        #volume-meter {
            position: absolute;
            bottom: 0;
            left: 0;
            width: 100%;
            background: linear-gradient(to top, #4CAF50, #8BC34A, #CDDC39, #FFEB3B, #FFC107, #FF9800, #FF5722);
            height: 0%;
            transition: height 0.1s ease;
        }
        .frequency-bars {
            display: flex;
            justify-content: space-between;
            height: 100%;
            width: 100%;
        }
        .frequency-bar {
            width: 3px;
            background-color: #4CAF50;
            margin: 0 1px;
            transform-origin: bottom;
        }
        .pronunciation-feedback {
            margin-top: 10px;
            text-align: center;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <div class="recorder-container">
        <button id="recordButton" class="record-button">
            <span id="buttonText">Start Recording</span>
        </button>
        <div id="status" class="status">Ready to record</div>
        
        <!-- Audio visualization -->
        <div class="visualization">
            <div id="volume-meter"></div>
            <div id="frequency-bars" class="frequency-bars"></div>
        </div>
        
        <!-- Real-time feedback -->
        <div id="feedback" class="pronunciation-feedback"></div>
        
        <audio id="audioPlayback" controls style="display: none; margin-top: 10px; width: 100%;"></audio>
    </div>

    <script>
        // Audio recording functionality
        let audioChunks = [];
        let mediaRecorder;
        let audioBlob;
        let isRecording = false;
        
        // Audio analysis
        let audioContext;
        let analyser;
        let microphoneStream;
        let dataArray;
        
        // Elements
        const recordButton = document.getElementById('recordButton');
        const buttonText = document.getElementById('buttonText');
        const statusText = document.getElementById('status');
        const audioPlayback = document.getElementById('audioPlayback');
        const volumeMeter = document.getElementById('volume-meter');
        const frequencyBars = document.getElementById('frequency-bars');
        const feedbackElement = document.getElementById('feedback');
        
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
        
        // Initialize frequency bars
        function initializeFrequencyBars(numBars = 32) {
            frequencyBars.innerHTML = '';
            for (let i = 0; i < numBars; i++) {
                const bar = document.createElement('div');
                bar.className = 'frequency-bar';
                frequencyBars.appendChild(bar);
            }
        }
        
        // Initialize frequency bars on load
        initializeFrequencyBars();
        
        // Set up audio analysis
        async function setupAudioAnalysis(stream) {
            try {
                audioContext = new (window.AudioContext || window.webkitAudioContext)();
                analyser = audioContext.createAnalyser();
                microphoneStream = audioContext.createMediaStreamSource(stream);
                microphoneStream.connect(analyser);
                
                analyser.fftSize = 256;
                const bufferLength = analyser.frequencyBinCount;
                dataArray = new Uint8Array(bufferLength);
                
                // Start visualization
                visualize();
            } catch (error) {
                console.error('Error setting up audio analysis:', error);
            }
        }
        
        // Visualize audio
        function visualize() {
            if (!isRecording) return;
            
            // Get frequency data
            analyser.getByteFrequencyData(dataArray);
            
            // Calculate average volume
            let sum = 0;
            const bars = document.querySelectorAll('.frequency-bar');
            
            // Update frequency bars
            const barCount = bars.length;
            const step = Math.floor(dataArray.length / barCount) || 1;
            
            for (let i = 0; i < barCount; i++) {
                const dataIndex = i * step;
                const value = dataArray[dataIndex] / 255.0;
                sum += value;
                
                // Update bar height
                if (bars[i]) {
                    bars[i].style.height = `${value * 100}%`;
                }
            }
            
            // Calculate average volume
            const average = sum / barCount;
            
            // Update volume meter
            volumeMeter.style.height = `${average * 100}%`;
            
            // Update feedback based on volume
            updateFeedback(average);
            
            // Send real-time data to Streamlit
            if (window.Streamlit) {
                window.Streamlit.setComponentValue({
                    state: 'analyzing',
                    volume: average,
                    frequencyData: Array.from(dataArray).slice(0, barCount)
                });
            }
            
            // Continue visualization loop
            requestAnimationFrame(visualize);
        }
        
        // Update feedback based on audio analysis
        function updateFeedback(volume) {
            if (volume < 0.05) {
                feedbackElement.textContent = 'Speak louder';
                feedbackElement.style.color = '#FF5722';
            } else if (volume > 0.8) {
                feedbackElement.textContent = 'Too loud!';
                feedbackElement.style.color = '#F44336';
            } else if (volume > 0.4) {
                feedbackElement.textContent = 'Good volume!';
                feedbackElement.style.color = '#4CAF50';
            } else {
                feedbackElement.textContent = 'Speak a bit louder';
                feedbackElement.style.color = '#FFC107';
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
                
                // Set up audio analysis
                setupAudioAnalysis(stream);
                
                // Start recording
                mediaRecorder.start();
                isRecording = true;
                recordButton.classList.add('recording');
                buttonText.textContent = 'Stop Recording';
                statusText.textContent = 'Recording in progress...';
                feedbackElement.textContent = 'Speak clearly...';
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
                
                // Stop visualization
                if (audioContext) {
                    // Close audio context
                    if (audioContext.state !== 'closed') {
                        audioContext.close().catch(console.error);
                    }
                }
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
                feedbackElement.textContent = 'Processing your pronunciation...';
            };
        }
        
        // Send data to Streamlit
        function sendToStreamlit(base64AudioData) {
            if (window.Streamlit) {
                // First send processing state
                window.Streamlit.setComponentValue({
                    status: 'processing',
                    data: null
                });
                
                // Short delay then send complete data
                setTimeout(() => {
                    window.Streamlit.setComponentValue({
                        status: 'complete',
                        data: base64AudioData,
                        format: 'audio/wav'
                    });
                }, 300);
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
        
        # Add a debug message to see what's happening
        debug_container = st.empty()
        
        # Get the component value - DO NOT USE THE KEY PARAMETER AT ALL
        component_value = components.html(
            open(HTML_FILE, "r").read(),
            height=200
        )
        
        # Process the returned value
        if component_value and isinstance(component_value, dict) and 'data' in component_value:
            # Show debug info
            debug_container.info("Audio data received, processing...")
            
            # Decode the base64 audio data
            audio_bytes = base64.b64decode(component_value['data'])
            
            # Store in session state
            st.session_state.audio_data = audio_bytes
            st.session_state.audio_data_received = True
            
            # Add a manual trigger for rerun to ensure the UI updates
            st.session_state.recording_complete = True
            
            # Force a rerun
            time.sleep(0.5)  # Short delay
            st.experimental_rerun()
            
            # Return the audio bytes
            return audio_bytes
        
        # Clear the processing message if no data received yet
        if not component_value:
            debug_container.empty()
        
        return None
    except Exception as e:
        st.error(f"Error in audio recorder component: {e}")
        return None