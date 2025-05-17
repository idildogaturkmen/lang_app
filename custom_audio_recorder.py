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
        #recordingDisplay {
            margin-top: 15px;
            width: 100%;
            text-align: center;
        }
    </style>
</head>
<body>
    <div class="recorder-container">
        <button id="recordButton" class="record-button">
            <span id="buttonText">Start Recording</span>
        </button>
        <div id="status" class="status">Ready to record</div>
        <div id="recordingDisplay">
            <!-- Audio player will be added here -->
        </div>
    </div>

    <script>
        // Audio recording functionality
        let audioChunks = [];
        let mediaRecorder;
        let audioBlob;
        let isRecording = false;
        let audioUrl = null;
        
        // Elements
        const recordButton = document.getElementById('recordButton');
        const buttonText = document.getElementById('buttonText');
        const statusText = document.getElementById('status');
        const recordingDisplay = document.getElementById('recordingDisplay');
        
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
            
            // Clear previous recording display
            recordingDisplay.innerHTML = '';
            
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
                
                // Stop all tracks in the stream to release microphone
                mediaRecorder.stream.getTracks().forEach(track => track.stop());
            }
        }
        
        // Process the recording after stopping
        function processRecording() {
            audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
            audioUrl = URL.createObjectURL(audioBlob);
            
            // Create and display audio player
            const audioPlayer = document.createElement('audio');
            audioPlayer.controls = true;
            audioPlayer.src = audioUrl;
            audioPlayer.style.width = '100%';
            audioPlayer.style.marginTop = '10px';
            audioPlayer.style.borderRadius = '4px';
            
            // Clear previous content and add the player
            recordingDisplay.innerHTML = '';
            recordingDisplay.appendChild(audioPlayer);
            
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
                    format: 'audio/wav',
                    status: 'complete'
                });
            }
        }
        
        // Initialize communication with Streamlit
        if (window.Streamlit) {
            window.Streamlit.componentReady();
        }
            
        // Add real-time visualization function
function setupVisualization() {
    // Create visualization canvas if it doesn't exist
    if (!document.getElementById('audio-visualizer')) {
        const canvas = document.createElement('canvas');
        canvas.id = 'audio-visualizer';
        canvas.width = 300;
        canvas.height = 60;
        canvas.style.marginTop = '10px';
        canvas.style.width = '100%';
        canvas.style.borderRadius = '4px';
        canvas.style.backgroundColor = '#f0f0f0';
        
        // Add it to the container
        recordingDisplay.appendChild(canvas);
        
        // Add volume level indicator
        const volumeIndicator = document.createElement('div');
        volumeIndicator.id = 'volume-level';
        volumeIndicator.style.width = '100%';
        volumeIndicator.style.height = '4px';
        volumeIndicator.style.backgroundColor = '#e0e0e0';
        volumeIndicator.style.position = 'relative';
        volumeIndicator.style.marginTop = '5px';
        volumeIndicator.style.borderRadius = '2px';
        
        const volumeMeter = document.createElement('div');
        volumeMeter.id = 'volume-meter';
        volumeMeter.style.height = '100%';
        volumeMeter.style.width = '0%';
        volumeMeter.style.backgroundColor = '#4CAF50';
        volumeMeter.style.borderRadius = '2px';
        volumeMeter.style.transition = 'width 0.1s ease';
        
        volumeIndicator.appendChild(volumeMeter);
        recordingDisplay.appendChild(volumeIndicator);
        
        // Add feedback text area
        const feedbackText = document.createElement('div');
        feedbackText.id = 'feedback-text';
        feedbackText.style.marginTop = '5px';
        feedbackText.style.fontSize = '14px';
        feedbackText.style.color = '#666';
        feedbackText.style.textAlign = 'center';
        feedbackText.textContent = 'Ready to record';
        
        recordingDisplay.appendChild(feedbackText);
    }
    
    return document.getElementById('audio-visualizer');
}

// Start audio visualization
function startVisualization(stream) {
    const canvas = setupVisualization();
    const canvasCtx = canvas.getContext('2d');
    const volumeMeter = document.getElementById('volume-meter');
    const feedbackText = document.getElementById('feedback-text');
    
    // Set up audio context and analyzer
    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const analyser = audioContext.createAnalyser();
    analyser.fftSize = 256;
    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    
    // Connect the audio stream
    const source = audioContext.createMediaStreamSource(stream);
    source.connect(analyser);
    
    // Visualization function
    function draw() {
        if (!isRecording) return;
        
        requestAnimationFrame(draw);
        
        // Get frequency data
        analyser.getByteFrequencyData(dataArray);
        
        // Calculate volume level (0-100)
        let sum = 0;
        for(let i = 0; i < bufferLength; i++) {
            sum += dataArray[i];
        }
        const average = sum / bufferLength;
        const volumePercentage = (average / 255) * 100;
        
        // Update volume meter
        volumeMeter.style.width = `${volumePercentage}%`;
        
        // Update feedback text based on volume
        if (volumePercentage < 5) {
            feedbackText.textContent = 'Speak louder...';
            feedbackText.style.color = '#F44336';
        } else if (volumePercentage > 80) {
            feedbackText.textContent = 'Too loud!';
            feedbackText.style.color = '#F44336';
        } else if (volumePercentage > 40) {
            feedbackText.textContent = 'Good volume!';
            feedbackText.style.color = '#4CAF50';
        } else {
            feedbackText.textContent = 'Speak a bit louder';
            feedbackText.style.color = '#FFC107';
        }
        
        // Draw visualization
        canvasCtx.fillStyle = 'rgb(240, 240, 240)';
        canvasCtx.fillRect(0, 0, canvas.width, canvas.height);
        
        const barWidth = (canvas.width / bufferLength) * 2.5;
        let x = 0;
        
        for(let i = 0; i < bufferLength; i++) {
            const barHeight = (dataArray[i] / 255) * canvas.height;
            
            // Create gradient color based on frequency
            const hue = i / bufferLength * 240;
            canvasCtx.fillStyle = `hsl(${hue}, 70%, 50%)`;
            
            canvasCtx.fillRect(x, canvas.height - barHeight, barWidth, barHeight);
            x += barWidth + 1;
        }
        
        // Send data to Streamlit
        if (window.Streamlit && isRecording) {
            window.Streamlit.setComponentValue({
                status: 'recording',
                volume: volumePercentage,
                frequencyData: Array.from(dataArray).slice(0, 10) // Send just first 10 values
            });
        }
    }
    
    // Start visualization
    draw();
    
    // Save audio context for cleanup
    window.audioContext = audioContext;
    
    return analyser;
}

// Modify startRecording to include visualization
async function startRecording() {
    audioChunks = [];
    
    // Clear previous recording display
    recordingDisplay.innerHTML = '';
    
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
        
        // Start visualization
        startVisualization(stream);
    } catch (err) {
        console.error('Error accessing microphone:', err);
        statusText.textContent = 'Error: Could not access microphone';
    }
}

// Modify stopRecording to clean up
function stopRecording() {
    if (mediaRecorder && isRecording) {
        mediaRecorder.stop();
        isRecording = false;
        recordButton.classList.remove('recording');
        buttonText.textContent = 'Start Recording';
        statusText.textContent = 'Processing recording...';
        
        // Stop all tracks in the stream to release microphone
        mediaRecorder.stream.getTracks().forEach(track => track.stop());
        
        // Clean up audio context
        if (window.audioContext && window.audioContext.state !== 'closed') {
            window.audioContext.close().catch(console.error);
        }
    }
}


        // Add real-time visualization function
        function setupVisualization() {
            // Create visualization canvas if it doesn't exist
            if (!document.getElementById('audio-visualizer')) {
                const canvas = document.createElement('canvas');
                canvas.id = 'audio-visualizer';
                canvas.width = 300;
                canvas.height = 60;
                canvas.style.marginTop = '10px';
                canvas.style.width = '100%';
                canvas.style.borderRadius = '4px';
                canvas.style.backgroundColor = '#f0f0f0';
                
                // Add it to the container
                recordingDisplay.appendChild(canvas);
                
                // Add volume level indicator
                const volumeIndicator = document.createElement('div');
                volumeIndicator.id = 'volume-level';
                volumeIndicator.style.width = '100%';
                volumeIndicator.style.height = '4px';
                volumeIndicator.style.backgroundColor = '#e0e0e0';
                volumeIndicator.style.position = 'relative';
                volumeIndicator.style.marginTop = '5px';
                volumeIndicator.style.borderRadius = '2px';
                
                const volumeMeter = document.createElement('div');
                volumeMeter.id = 'volume-meter';
                volumeMeter.style.height = '100%';
                volumeMeter.style.width = '0%';
                volumeMeter.style.backgroundColor = '#4CAF50';
                volumeMeter.style.borderRadius = '2px';
                volumeMeter.style.transition = 'width 0.1s ease';
                
                volumeIndicator.appendChild(volumeMeter);
                recordingDisplay.appendChild(volumeIndicator);
                
                // Add feedback text area
                const feedbackText = document.createElement('div');
                feedbackText.id = 'feedback-text';
                feedbackText.style.marginTop = '5px';
                feedbackText.style.fontSize = '14px';
                feedbackText.style.color = '#666';
                feedbackText.style.textAlign = 'center';
                feedbackText.textContent = 'Ready to record';
                
                recordingDisplay.appendChild(feedbackText);
            }
            
            return document.getElementById('audio-visualizer');
        }

        // Start audio visualization
        function startVisualization(stream) {
            const canvas = setupVisualization();
            const canvasCtx = canvas.getContext('2d');
            const volumeMeter = document.getElementById('volume-meter');
            const feedbackText = document.getElementById('feedback-text');
            
            // Set up audio context and analyzer
            const audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const analyser = audioContext.createAnalyser();
            analyser.fftSize = 256;
            const bufferLength = analyser.frequencyBinCount;
            const dataArray = new Uint8Array(bufferLength);
            
            // Connect the audio stream
            const source = audioContext.createMediaStreamSource(stream);
            source.connect(analyser);
            
            // Visualization function
            function draw() {
                if (!isRecording) return;
                
                requestAnimationFrame(draw);
                
                // Get frequency data
                analyser.getByteFrequencyData(dataArray);
                
                // Calculate volume level (0-100)
                let sum = 0;
                for(let i = 0; i < bufferLength; i++) {
                    sum += dataArray[i];
                }
                const average = sum / bufferLength;
                const volumePercentage = (average / 255) * 100;
                
                // Update volume meter
                volumeMeter.style.width = `${volumePercentage}%`;
                
                // Update feedback text based on volume
                if (volumePercentage < 5) {
                    feedbackText.textContent = 'Speak louder...';
                    feedbackText.style.color = '#F44336';
                } else if (volumePercentage > 80) {
                    feedbackText.textContent = 'Too loud!';
                    feedbackText.style.color = '#F44336';
                } else if (volumePercentage > 40) {
                    feedbackText.textContent = 'Good volume!';
                    feedbackText.style.color = '#4CAF50';
                } else {
                    feedbackText.textContent = 'Speak a bit louder';
                    feedbackText.style.color = '#FFC107';
                }
                
                // Draw visualization
                canvasCtx.fillStyle = 'rgb(240, 240, 240)';
                canvasCtx.fillRect(0, 0, canvas.width, canvas.height);
                
                const barWidth = (canvas.width / bufferLength) * 2.5;
                let x = 0;
                
                for(let i = 0; i < bufferLength; i++) {
                    const barHeight = (dataArray[i] / 255) * canvas.height;
                    
                    // Create gradient color based on frequency
                    const hue = i / bufferLength * 240;
                    canvasCtx.fillStyle = `hsl(${hue}, 70%, 50%)`;
                    
                    canvasCtx.fillRect(x, canvas.height - barHeight, barWidth, barHeight);
                    x += barWidth + 1;
                }
                
                // Send data to Streamlit
                if (window.Streamlit && isRecording) {
                    window.Streamlit.setComponentValue({
                        status: 'recording',
                        volume: volumePercentage,
                        frequencyData: Array.from(dataArray).slice(0, 10) // Send just first 10 values
                    });
                }
            }
            
            // Start visualization
            draw();
            
            // Save audio context for cleanup
            window.audioContext = audioContext;
            
            return analyser;
        }

        // Modify startRecording to include visualization
        async function startRecording() {
            audioChunks = [];
            
            // Clear previous recording display
            recordingDisplay.innerHTML = '';
            
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
                
                // Start visualization
                startVisualization(stream);
            } catch (err) {
                console.error('Error accessing microphone:', err);
                statusText.textContent = 'Error: Could not access microphone';
            }
        }

        // Modify stopRecording to clean up
        function stopRecording() {
            if (mediaRecorder && isRecording) {
                mediaRecorder.stop();
                isRecording = false;
                recordButton.classList.remove('recording');
                buttonText.textContent = 'Start Recording';
                statusText.textContent = 'Processing recording...';
                
                // Stop all tracks in the stream to release microphone
                mediaRecorder.stream.getTracks().forEach(track => track.stop());
                
                // Clean up audio context
                if (window.audioContext && window.audioContext.state !== 'closed') {
                    window.audioContext.close().catch(console.error);
                }
            }
        }
            
    </script>
</body>
</html>
    """)

# Create the custom component function
def audio_recorder():
    """Custom audio recorder component with JavaScript"""
    try:
        # Get the component value
        component_value = components.html(
            open(HTML_FILE, "r").read(),
            height=200
        )
        
        # Process the returned value
        if component_value and isinstance(component_value, dict) and 'data' in component_value:
            # Decode the base64 audio data
            audio_bytes = base64.b64decode(component_value['data'])
            
            # Store in session state for persistence
            st.session_state.audio_data = audio_bytes
            st.session_state.audio_data_received = True
            
            # Display the audio player in Streamlit
            audio_container = st.container()
            with audio_container:
                st.audio(audio_bytes, format="audio/wav")
                st.success("Recording saved! You can replay it using the player above.")
            
            # Return the audio bytes
            return audio_bytes
        
        # If we already have audio data in session state, display it
        elif 'audio_data' in st.session_state and st.session_state.audio_data_received:
            st.audio(st.session_state.audio_data, format="audio/wav")
            st.success("Previous recording available. You can replay it or record a new one.")
            return st.session_state.audio_data
        
        return None
    except Exception as e:
        st.error(f"Error in audio recorder component: {e}")
        st.exception(e)  # Show full exception for debugging
        return None