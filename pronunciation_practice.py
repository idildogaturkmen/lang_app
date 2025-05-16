"""
Simplified WebRTC implementation for pronunciation practice
This module focuses on making real-time recording work reliably
"""
"""
Enhanced Pronunciation Practice Module

Required packages for full functionality:
- streamlit-webrtc
- av
- speechrecognition
- pydub
- levenshtein
- epitran
- panphon
- matplotlib
- numpy
- requests

Optional API keys (add to Streamlit secrets):
- azure_speech_key
- azure_region
"""

import streamlit as st
import time
import tempfile
import io
import os
import wave
import re
from datetime import datetime
import tempfile
import io
import os
import matplotlib.pyplot as plt
import numpy as np
import base64
import requests
import json

# Determine Streamlit version
try:
    import streamlit
    STREAMLIT_VERSION = streamlit.__version__
    print(f"Streamlit version: {STREAMLIT_VERSION}")
except Exception as e:
    STREAMLIT_VERSION = "unknown"
    print(f"Error determining Streamlit version: {e}")

# Try importing WebRTC with improved error handling
try:
    from streamlit_webrtc import (
        webrtc_streamer,
        WebRtcMode,
        ClientSettings,
        RTCConfiguration,
        MediaStreamConstraints,
    )
    import av  # Required for WebRTC
    
    # Define a custom audio receiver class
    class AudioProcessor:
        def __init__(self):
            self.frames = []
        
        def recv(self, frame):
            self.frames.append(frame)
            return frame
    
    HAS_WEBRTC = True
except ImportError as e:
    print(f"WebRTC import error: {e}")
    HAS_WEBRTC = False

# Try importing speech recognition
try:
    import speech_recognition as sr
    HAS_SR = True
except ImportError:
    HAS_SR = False

# Try importing Levenshtein
try:
    import Levenshtein
    HAS_LEVENSHTEIN = True
except ImportError:
    HAS_LEVENSHTEIN = False

# Define difficult sounds by language
DIFFICULT_SOUNDS = {
    "es": {  # Spanish
        'j': {'sound': 'h', 'example': 'jalapeño → halapeño'},
        'll': {'sound': 'y', 'example': 'llamar → yamar'},
        'ñ': {'sound': 'ny', 'example': 'niño → ninyo'},
        'rr': {'sound': 'rolled r', 'example': 'perro → pe(rolled r)o'},
        'v': {'sound': 'b/v', 'example': 'vaca sounds like baca'}
    },
    "fr": {  # French
        'r': {'sound': 'guttural r', 'example': 'rouge → (guttural r)oozh'},
        'u': {'sound': 'ü (rounded lips)', 'example': 'tu → tü'},
        'eu': {'sound': 'like "e" in "the"', 'example': 'peu → puh'},
        'ou': {'sound': 'oo', 'example': 'vous → voo'},
        'au/eau': {'sound': 'oh', 'example': 'beau → boh'},
        'ai/è': {'sound': 'eh', 'example': 'mais → meh'}
    },
    "de": {  # German
        'ch': {'sound': 'kh/sh', 'example': 'ich → ish, Bach → bakh'},
        'r': {'sound': 'guttural r', 'example': 'rot → (guttural r)oht'},
        'ü': {'sound': 'ü (rounded lips)', 'example': 'über → üba'},
        'ö': {'sound': 'eu sound', 'example': 'schön → sheun'},
        'ä': {'sound': 'eh', 'example': 'Mädchen → mehdshen'},
        'ei': {'sound': 'eye', 'example': 'nein → nine'},
        'ie': {'sound': 'ee', 'example': 'wie → vee'}
    },
    "it": {  # Italian
        'gli': {'sound': 'ly', 'example': 'figlio → feelyo'},
        'gn': {'sound': 'ny', 'example': 'gnocchi → nyokee'},
        'r': {'sound': 'rolled r', 'example': 'Roma → (rolled r)oma'},
        'c+e/i': {'sound': 'ch', 'example': 'ciao → chow'},
        'c+a/o/u': {'sound': 'k', 'example': 'casa → kaza'},
        'sc+e/i': {'sound': 'sh', 'example': 'scienza → shentsa'}
    }
}

# Language code to name mapping
LANGUAGE_NAMES = {
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "ja": "Japanese",
    "zh-CN": "Chinese"
}

# Language code to speech recognition language mapping
RECOGNITION_LANGUAGES = {
    "es": "es-ES",  # Spanish
    "fr": "fr-FR",  # French
    "de": "de-DE",  # German
    "it": "it-IT",  # Italian
    "pt": "pt-BR",  # Portuguese
    "ru": "ru-RU",  # Russian
    "ja": "ja-JP",  # Japanese
    "zh-CN": "zh-CN",  # Chinese (Simplified)
    "en": "en-US"   # English
}

def create_pronunciation_practice(text_to_speech_func=None, get_audio_html_func=None, translate_text_func=None):
    """
    Create a pronunciation practice module.
    
    Args:
        text_to_speech_func: Function for text-to-speech conversion
        get_audio_html_func: Function to get audio HTML
        translate_text_func: Function to translate text
        
    Returns:
        PronunciationPractice instance
    """
    # Debug information
    st.session_state.pronunciation_debug = {
        "has_webrtc": HAS_WEBRTC,
        "has_sr": HAS_SR,
        "has_levenshtein": HAS_LEVENSHTEIN
    }
    
    return SimplePronunciationPractice(
        text_to_speech_func,
        get_audio_html_func,
        translate_text_func
    )

# Replace the entire SimplePronunciationPractice class with this implementation:

"""
Enhanced pronunciation practice implementation with multi-layered recording approach:
1. Streamlit's native microphone input
2. WebRTC-based recording
3. File upload fallback
"""

"""
Enhanced pronunciation practice implementation with multi-layered recording approach:
1. Streamlit's native microphone input
2. WebRTC-based recording
3. File upload fallback
"""

"""
Enhanced pronunciation practice implementation with multi-layered recording approach:
1. Streamlit's native microphone input
2. WebRTC-based recording
3. File upload fallback
"""

class SimplePronunciationPractice:
    """
    Implementation of pronunciation practice with AI feedback
    """
    
    def __init__(self, text_to_speech_func, get_audio_html_func, translate_text_func):
        """Initialize the pronunciation practice module"""
        self.text_to_speech = text_to_speech_func
        self.get_audio_html = get_audio_html_func
        self.translate_text = translate_text_func
        self.difficult_sounds = DIFFICULT_SOUNDS
        
        # Initialize speech recognition if available
        if HAS_SR:
            self.recognizer = sr.Recognizer()
            self.recognizer.energy_threshold = 300  # Lower threshold to detect speech
        
        # Try to import custom recorder - with more robust error handling
        try:
            import importlib.util
            # Check if the custom_audio_recorder.py file exists
            module_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "custom_audio_recorder.py")
            if os.path.exists(module_path):
                # Dynamically import the module from the file path
                spec = importlib.util.spec_from_file_location("custom_audio_recorder", module_path)
                custom_recorder_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(custom_recorder_module)
                
                # Get the audio_recorder function
                self.custom_recorder = custom_recorder_module.audio_recorder
                self.has_custom_recorder = True
                print("Custom audio recorder loaded successfully")
            else:
                # Fallback to normal import if the file is not found at the expected path
                from custom_audio_recorder import audio_recorder
                self.custom_recorder = audio_recorder
                self.has_custom_recorder = True
                print("Custom audio recorder loaded via import")
        except Exception as e:
            print(f"Custom audio recorder not available: {e}")
            self.has_custom_recorder = False
    
    def _get_streamlit_version(self):
        """Get the current Streamlit version with safer error handling"""
        try:
            import streamlit as st
            # Try to get version from module directly
            if hasattr(st, '__version__'):
                version = st.__version__
            else:
                # Fallback: import main module to check version
                import streamlit
                version = getattr(streamlit, '__version__', "0.0.0")
                
            # Parse version string to get major and minor version numbers
            version_parts = version.split('.')
            if len(version_parts) >= 2:
                try:
                    major, minor = int(version_parts[0]), int(version_parts[1])
                    return (major, minor)
                except ValueError:
                    # Couldn't convert to integer
                    return (0, 0)
            return (0, 0)  # Default if parsing fails
        except:
            return (0, 0)  # Default if import fails
    
    def _add_audio_recorder(self):
        """Add the most appropriate audio recorder based on environment"""
        # Always use the custom recorder if available
        if self.has_custom_recorder:
            self._add_custom_recorder()
        else:
            # Create a container to display any errors
            error_container = st.empty()
            
            try:
                # Fall back to WebRTC if available
                if HAS_WEBRTC:
                    try:
                        self._add_webrtc_recorder()
                    except Exception as e:
                        print(f"WebRTC recorder failed: {e}")
                        # Continue to file upload fallback
                        self._add_upload_recorder()
                else:
                    # Last resort: file upload (always works)
                    self._add_upload_recorder()
                
            except Exception as e:
                # Clear any previous error
                error_container.empty()
                # Show the error and fall back to file upload
                error_container.error(f"Error setting up audio recorder: {str(e)}")
                # Always provide a fallback method
                self._add_upload_recorder()
    
    def _add_custom_recorder(self):
        """Add the custom JavaScript-based audio recorder"""
        try:
            # Only show a single header - moved outside the method
            # st.markdown("### 🎙️ Record Your Pronunciation")
            
            # Use our custom recorder component
            audio_bytes = self.custom_recorder()
            
            # If we have audio data, display it and prepare for analysis
            if audio_bytes or st.session_state.get('recording_complete', False):
                st.success("✅ Recording complete!")
                
                # If we have audio data in session state
                if st.session_state.get('audio_data') is not None:
                    # Play the audio back
                    st.subheader("Your Recording")
                    st.audio(st.session_state.audio_data)
                    
                    # Update current recording word if in practice session
                    self._update_current_recording_word()
                    
                    # Process button (only show if not already processed)
                    if not st.session_state.get('analysis_complete', False):
                        if st.button("✨ Analyze My Pronunciation", key=f"analyze_btn_{int(time.time())}"):
                            # Set flag that analysis is complete
                            st.session_state.analysis_complete = True
                            st.rerun()
                    
                    # If analysis complete flag is set, run the analysis
                    if st.session_state.get('analysis_complete', False):
                        st.session_state.analysis_complete = False  # Reset for next time
                        return True  # Signal to run analysis
                else:
                    st.info("Use the recorder above to practice your pronunciation")
            else:
                st.info("Use the recorder above to practice your pronunciation")
                
            return False  # Don't run analysis yet
        except Exception as e:
            st.error(f"Error using custom recorder: {e}")
            return False
    
    def _add_webrtc_recorder(self):
        """Add WebRTC-based real-time audio recorder"""
        if not HAS_WEBRTC:
            raise ImportError("streamlit-webrtc is not installed")
            
        # Single clear title
        st.markdown("### 🎙️ Record Your Pronunciation")
        st.markdown("Click 'START' below, say the word clearly, then click 'STOP'. After stopping, press 'Process Recording'.")
        
        # Create an empty placeholder for status messages
        status_indicator = st.empty()
        
        # Initialize audio frames in session state if not present
        if 'audio_frames' not in st.session_state:
            st.session_state.audio_frames = []
        
        # Debug information
        debug_mode = False  # Set to True to show debugging information
        if debug_mode:
            st.write(f"Audio frames in session: {len(st.session_state.audio_frames)}")
        
        try:
            # Import required components
            from streamlit_webrtc import webrtc_streamer, WebRtcMode
            import av
            
            def audio_frame_callback(frame):
                """Process incoming audio frames"""
                try:
                    # Convert frame to numpy array and store in session state
                    sound = frame.to_ndarray()
                    st.session_state.audio_frames.append(sound)
                    if debug_mode and len(st.session_state.audio_frames) % 10 == 0:
                        print(f"Added frame, total frames: {len(st.session_state.audio_frames)}")
                except Exception as e:
                    if debug_mode:
                        print(f"Error in audio frame callback: {e}")
                return frame
            
            # Configure WebRTC with explicit settings
            webrtc_ctx = webrtc_streamer(
                key="pronunciation-recorder",
                mode=WebRtcMode.SENDONLY,
                audio_frame_callback=audio_frame_callback,
                rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                media_stream_constraints={"video": False, "audio": True},
            )
            
            # Show different messages based on recording state
            has_frames = len(st.session_state.audio_frames) > 0
            
            if webrtc_ctx.state.playing:
                status_indicator.info("✅ Recording... Speak the word clearly.")
            else:
                if has_frames:
                    status_indicator.success("✅ Recording complete! Click 'Process Recording' below.")
                else:
                    status_indicator.info("ℹ️ Click START above to begin recording.")
            
            # Process button - only show when recording has stopped and we have frames
            if not webrtc_ctx.state.playing and has_frames:
                # Create a clearly visible button to process the recording
                process_btn_container = st.container()
                if process_btn_container.button("💾 Process Recording", type="primary", key="process_webrtc_recording"):
                    with st.spinner("Processing audio..."):
                        try:
                            # Process the collected audio frames
                            import numpy as np
                            import io
                            import wave
                            
                            # Show helpful message while processing
                            process_status = st.empty()
                            process_status.info(f"Processing {len(st.session_state.audio_frames)} audio frames...")
                            
                            # Combine all audio frames
                            all_audio = np.concatenate(st.session_state.audio_frames, axis=0)
                            
                            # Convert to WAV format
                            byte_io = io.BytesIO()
                            with wave.open(byte_io, 'wb') as wf:
                                wf.setnchannels(1)  # Mono audio
                                wf.setsampwidth(2)  # 16-bit audio
                                wf.setframerate(48000)  # 48kHz sampling rate
                                wf.writeframes(all_audio.tobytes())
                            
                            # Get the WAV data
                            byte_io.seek(0)
                            audio_bytes = byte_io.read()
                            
                            # Store in session state
                            st.session_state.audio_data = audio_bytes
                            st.session_state.audio_data_received = True
                            
                            # Update current recording word
                            self._update_current_recording_word()
                            
                            # Clear the frames cache
                            st.session_state.audio_frames = []
                            
                            # Update processing status
                            process_status.success("✅ Audio processed successfully!")
                            
                            # Play the processed audio
                            st.subheader("Your Recording")
                            st.audio(audio_bytes)
                            
                            # Rerun to trigger analysis
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error processing audio: {e}")
                            if debug_mode:
                                st.write(f"Error details: {type(e).__name__}, {str(e)}")
            
            return True  # Successfully used WebRTC recorder
            
        except Exception as e:
            if debug_mode:
                st.error(f"WebRTC setup failed: {e}")
            raise Exception(f"WebRTC setup failed: {e}")
    
    def _add_upload_recorder(self):
        """Add file upload as a fallback recording method"""
        st.markdown("To practice pronunciation:")
        st.markdown("""
        1. Use your device's voice recorder app to record yourself saying the word
        2. Save the recording and upload it below
        3. Click 'Process Recording' to evaluate
        """)
        
        # Create columns for better layout
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Add a file uploader for audio - with unique ID based on time
            timestamp = int(time.time())
            uploaded_file = st.file_uploader(
                "Upload your pronunciation recording (WAV, MP3, etc.)", 
                type=["wav", "mp3", "ogg", "m4a"]
            )
        
        # Process the uploaded file
        if uploaded_file is not None:
            # Read the file
            audio_bytes = uploaded_file.read()
            
            # Store in session state
            st.session_state.audio_data = audio_bytes
            st.session_state.audio_data_received = True
            
            # Update current recording word
            self._update_current_recording_word()
            
            # Display a success message
            st.success("✅ Recording uploaded successfully!")
            
            # Display the audio player
            st.audio(audio_bytes)
            
            with col2:
                # Add a button to process the recording - without key
                if st.button("Process Recording"):
                    st.rerun()
    
    def _update_current_recording_word(self):
        """Update the current recording word in session state if in a practice session"""
        if 'current_practice_index' in st.session_state and 'practice_words' in st.session_state:
            current_index = st.session_state.current_practice_index
            if current_index < len(st.session_state.practice_words):
                current_word = st.session_state.practice_words[current_index]
                st.session_state.current_recording_word = current_word.get('id')
    
    def render_practice_ui(self, word):
        """Render pronunciation practice UI for a word"""
        with st.expander("🎤 Practice Pronunciation"):
            # Get word data
            original_word = word.get('word_original', '')
            translated_word = word.get('word_translated', '')
            language_code = word.get('language_translated', 'en')
            
            # Display the word to practice
            st.subheader(f"Practice: {translated_word}")
            
            # Play the pronunciation
            st.markdown("**Listen to correct pronunciation:**")
            audio_bytes = self.text_to_speech(translated_word, language_code)
            if audio_bytes:
                st.markdown(self.get_audio_html(audio_bytes), unsafe_allow_html=True)
            
            # Show pronunciation tips
            self._show_pronunciation_tips(word)
            
            # Single heading for recording section
            st.markdown("### 🎙️ Record Your Pronunciation")
            
            # Add the appropriate audio recorder - HEADING IS MOVED HERE
            if self.has_custom_recorder:
                self._add_custom_recorder()
            else:
                # Show recording instructions
                st.markdown("Record yourself saying the word above:")
                
                # Try WebRTC first, then fall back to file upload
                try:
                    if HAS_WEBRTC:
                        self._add_webrtc_recorder()
                    else:
                        self._add_upload_recorder()
                except Exception as e:
                    st.error(f"Error with recorder: {e}")
                    self._add_upload_recorder()
            
            # Calculate score based on speech recognition or self-assessment
            if HAS_SR and 'audio_data' in st.session_state and st.session_state.audio_data:
                # Process audio with speech recognition
                similarity_score = self._evaluate_pronunciation(
                    audio_data=st.session_state.audio_data,
                    target_word=translated_word,
                    language_code=language_code
                )
                
                # Show feedback based on AI assessment
                self._show_simple_feedback(translated_word, language_code, similarity_score)
            elif 'audio_data' in st.session_state and st.session_state.audio_data:
                # Fall back to self-assessment if no speech recognition
                st.markdown("### Rate Your Pronunciation")
                st.markdown("After practicing, rate how well you pronounced the word:")
                
                # Create a simple rating system
                rating = st.select_slider(
                    "How well did you pronounce the word?",
                    options=["Poor", "Fair", "Good", "Very Good", "Excellent"],
                    value="Good"
                )
                
                # Calculate score based on rating
                score_map = {
                    "Poor": 20,
                    "Fair": 40,
                    "Good": 60,
                    "Very Good": 80,
                    "Excellent": 95
                }
                
                # Show feedback based on self-rating
                self._show_simple_feedback(translated_word, language_code, score_map.get(rating, 60))
    
    def render_practice_session(self, vocabulary, language_code):
        """Render pronunciation practice session"""
        st.markdown("## Pronunciation Practice")
        
        # If there's no practice session data, show a message
        if 'practice_words' not in st.session_state:
            st.info("Click 'Start Practice Session' to begin.")
            return
        
        # Get current word
        current_index = st.session_state.current_practice_index
        if current_index < len(st.session_state.practice_words):
            current_word = st.session_state.practice_words[current_index]
            
            # Progress bar
            progress = current_index / len(st.session_state.practice_words)
            st.progress(progress)
            st.subheader(f"Word {current_index + 1} of {len(st.session_state.practice_words)}")
            
            # Word info with larger, more visible styling
            st.markdown(f"""
            <div style="background-color: #f0f2f6; padding: 15px; border-radius: 5px; margin-bottom: 20px;">
                <h2 style="margin: 0;">{current_word.get('word_translated', '')}</h2>
                <p style="margin: 5px 0 0 0; color: #666;">English: {current_word.get('word_original', '')}</p>
                <p style="margin: 0; font-weight: bold; color: #1e88e5;">{LANGUAGE_NAMES.get(language_code, language_code)}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Listen to pronunciation
            st.markdown("### 👂 Listen to correct pronunciation")
            audio_bytes = self.text_to_speech(current_word.get('word_translated', ''), language_code)
            if audio_bytes:
                st.markdown(self.get_audio_html(audio_bytes), unsafe_allow_html=True)
            
            # Pronunciation guide
            with st.expander("Pronunciation Guide", expanded=True):
                self._show_pronunciation_tips(current_word)
            
            # Add the recording section - ONE title handled inside this method
            self._add_audio_recorder()
            
            # Check if audio data is available
            has_audio = 'audio_data' in st.session_state and st.session_state.audio_data
            
            if has_audio:
                # Calculate score based on speech recognition or self-assessment
                if HAS_SR:
                    with st.spinner("Analyzing your pronunciation..."):
                        # Process audio with speech recognition
                        similarity_score = self._evaluate_pronunciation(
                            audio_data=st.session_state.audio_data,
                            target_word=current_word.get('word_translated', ''),
                            language_code=language_code
                        )
                else:
                    # Fall back to self-assessment
                    rating = st.select_slider(
                        "How well did you pronounce the word?",
                        options=["Poor", "Fair", "Good", "Very Good", "Excellent"],
                        value="Good"
                    )
                    
                    # Calculate score based on rating
                    score_map = {
                        "Poor": 20,
                        "Fair": 40,
                        "Good": 60,
                        "Very Good": 80,
                        "Excellent": 95
                    }
                    similarity_score = score_map.get(rating, 60)
                
                # Show feedback
                self._show_simple_feedback(
                    current_word.get('word_translated', ''), 
                    language_code, 
                    similarity_score
                )
                
                # Submit button
                if st.button("Submit and Continue", type="primary"):
                    # Store score
                    if 'practice_scores' not in st.session_state:
                        st.session_state.practice_scores = []
                    st.session_state.practice_scores.append(similarity_score)
                    
                    # Clear audio data for next word
                    if 'audio_data' in st.session_state:
                        del st.session_state.audio_data
                    if 'current_recording_word' in st.session_state:
                        del st.session_state.current_recording_word
                    
                    # Move to next word
                    st.session_state.current_practice_index += 1
                    st.rerun()
            
            # Example in context
            example = self._get_example_sentence(
                current_word.get('word_original', ''), 
                current_word.get('language_translated', 'en')
            )
            
            with st.expander("Example in Context"):
                st.markdown(f"**English:** {example['english']}")
                if example.get('translated'):
                    st.markdown(f"**{LANGUAGE_NAMES.get(language_code, language_code)}:** {example['translated']}")
                    example_audio = self.text_to_speech(example['translated'], language_code)
                    if example_audio:
                        st.markdown(self.get_audio_html(example_audio), unsafe_allow_html=True)
        else:
            # Practice session completed
            self._show_practice_results()
            
            # Reset button
            if st.button("Practice Again", type="primary"):
                import random
                
                # Create a new set of practice words
                filtered_vocab = [word for word in vocabulary 
                                if word['language_translated'] == language_code]
                practice_size = min(5, len(filtered_vocab))
                st.session_state.practice_words = random.sample(filtered_vocab, practice_size)
                st.session_state.current_practice_index = 0
                st.session_state.practice_scores = []
                
                # Clear audio data
                if 'audio_data' in st.session_state:
                    del st.session_state.audio_data
                if 'current_recording_word' in st.session_state:
                    del st.session_state.current_recording_word
                
                st.rerun()
                
    def _evaluate_pronunciation(self, audio_data, target_word, language_code):
        """Enhanced pronunciation evaluation with multiple techniques"""
        # Show evaluation status
        status = st.empty()
        status.info("Analyzing your pronunciation in detail... Please wait.")
        
        results = {}
        
        # Basic speech recognition (existing functionality)
        if HAS_SR:
            recognized_text = self._recognize_speech(audio_data, language_code)
            results['recognized_text'] = recognized_text
            
            # Calculate Levenshtein similarity if available
            if HAS_LEVENSHTEIN and recognized_text:
                # Clean up text for comparison
                target_cleaned = self._clean_text_for_comparison(target_word)
                recognized_cleaned = self._clean_text_for_comparison(recognized_text)
                
                # Calculate Levenshtein distance
                distance = Levenshtein.distance(target_cleaned, recognized_cleaned)
                max_len = max(len(target_cleaned), len(recognized_cleaned))
                
                # Convert to similarity percentage
                similarity = max(0, 100 - (distance / max_len * 100))
                results['levenshtein_similarity'] = similarity
        
        # Try advanced API-based analysis if available
        api_results = self._api_based_analysis(audio_data, target_word, language_code)
        if api_results:
            results.update(api_results)
        
        # Perform phoneme-level analysis if possible
        phoneme_results = self._phoneme_analysis(audio_data, target_word, language_code)
        if phoneme_results:
            results.update(phoneme_results)
            
            # Visualize the pronunciation comparison
            self._visualize_pronunciation(
                phoneme_results.get('target_phonemes', ''),
                phoneme_results.get('user_phonemes', '')
            )
        
        # Calculate final score (weighted combination of all available scores)
        final_score = self._calculate_weighted_score(results)
        results['final_score'] = final_score
        
        # Generate detailed feedback based on all analyses
        feedback = self._generate_detailed_feedback(results, language_code)
        results['feedback'] = feedback
        
        status.success("Analysis complete!")
        
        # Modify the result display in _show_simple_feedback
        st.session_state.last_pronunciation_results = results
        
        return final_score
    
    def _clean_text_for_comparison(self, text):
        """Clean text for comparison by removing punctuation and lowercasing"""
        return re.sub(r'[^\w\s]', '', text.lower()).strip()
    
    def _show_pronunciation_tips(self, word):
        """Show pronunciation tips for a word"""
        language_code = word.get('language_translated', 'en')
        translated_word = word.get('word_translated', '')
        
        # Get sounds for this language
        language_sounds = self.difficult_sounds.get(language_code, {})
        
        # Find sounds in this word
        tips = []
        for sound, data in language_sounds.items():
            if sound in translated_word.lower():
                tips.append(f"**'{sound}'** sounds like **'{data['sound']}'** (Example: {data['example']})")
        
        # Display tips
        if tips:
            st.markdown("**Pronunciation tips:**")
            for tip in tips:
                st.markdown(f"- {tip}")
        else:
            st.markdown("*No specific pronunciation tips for this word.*")
    
    def _show_simple_feedback(self, target_word, language_code, similarity_score):
        """Show enhanced pronunciation feedback with detailed results"""
        st.markdown("### Pronunciation Feedback")
        
        # Display score
        st.markdown(f"**Pronunciation accuracy: {similarity_score:.0f}%**")
        st.progress(similarity_score / 100.0)
        
        # Display recognized text if available
        results = getattr(st.session_state, 'last_pronunciation_results', {})
        recognized_text = results.get('recognized_text', '')
        
        if recognized_text:
            st.markdown(f"**Recognized text:** '{recognized_text}'")
            st.markdown(f"**Target word:** '{target_word}'")
        
        # Feedback based on score
        if similarity_score >= 90:
            st.success("✅ Excellent pronunciation!")
        elif similarity_score >= 70:
            st.info("👍 Good pronunciation!")
        elif similarity_score >= 50:
            st.warning("🔄 Fair pronunciation. Keep practicing!")
        else:
            st.error("⚠️ Needs improvement. Listen to the example again.")
        
        # Display detailed feedback
        feedback = results.get('feedback', [])
        if feedback:
            st.markdown("### Detailed Feedback")
            for item in feedback:
                st.markdown(f"- {item}")
        
        # Pronunciation tips section
        if similarity_score < 90:
            st.markdown("### Tips for Improvement")
            
            # Find problematic sounds
            problem_sounds = []
            for sound, data in self.difficult_sounds.get(language_code, {}).items():
                if sound in target_word.lower():
                    problem_sounds.append((sound, data))
            
            # Show tips for each problematic sound
            for sound, data in problem_sounds[:3]:  # Limit to 3 sounds
                st.markdown(f"- Focus on the **'{sound}'** sound: {data['example']}")
            
            # General advice
            st.markdown("""
            **Practice tips:**
            - Listen to the correct pronunciation multiple times
            - Speak slowly and clearly
            - Exaggerate mouth movements at first
            - Practice regularly
            """)
    
    def _get_example_sentence(self, word, language_code):
        """Get example sentence for a word"""
        if self.translate_text:
            import random
            
            # Simple English templates
            templates = [
                f"The {word} is on the table.",
                f"I like this {word} very much.",
                f"Can you see the {word}?",
                f"This {word} is very useful.",
                f"I need a new {word}."
            ]
            
            # Select a random template
            example = random.choice(templates)
            
            # Try to translate
            try:
                translated = self.translate_text(example, language_code)
                return {
                    "english": example,
                    "translated": translated
                }
            except Exception:
                return {"english": example, "translated": ""}
        else:
            return {"english": f"The {word} is on the table.", "translated": ""}
    
    def _show_practice_results(self):
        """Show results of the practice session"""
        st.subheader("🎉 Practice Session Completed!")
        
        # Get scores
        scores = st.session_state.practice_scores if 'practice_scores' in st.session_state else []
        
        if not scores:
            st.info("No pronunciation attempts were recorded.")
            return
        
        # Calculate stats
        avg_score = sum(scores) / len(scores)
        min_score = min(scores)
        max_score = max(scores)
        
        # Display stats
        st.markdown(f"**Average accuracy: {avg_score:.0f}%**")
        st.progress(avg_score / 100.0)
        
        # Min/max scores
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"Best pronunciation: {max_score:.0f}%")
        with col2:
            st.markdown(f"Areas for improvement: {min_score:.0f}%")
        
        # Feedback based on average score
        if avg_score >= 90:
            st.success("Outstanding work! Your pronunciation is excellent.")
        elif avg_score >= 75:
            st.success("Great job! Your pronunciation is very good.")
        elif avg_score >= 60:
            st.info("Good effort! Continue practicing to improve your accent.")
        else:
            st.warning("Keep practicing! Regular practice will improve your pronunciation.")
        
        # Suggestions
        st.markdown("""
        ### Next Steps
        
        1. **Listen carefully** to native speakers
        2. **Record yourself** speaking and compare with native pronunciation
        3. **Practice daily** for best results
        4. **Focus on difficult sounds** specific to this language
        """)

    def _phoneme_analysis(self, audio_data, target_word, language_code):
        """Analyze pronunciation at the phoneme level"""
        try:
            # First check if required libraries are available
            try:
                from epitran import Epitran
                import panphon.distance
            except ImportError:
                st.warning("Enhanced phoneme analysis requires `epitran` and `panphon` libraries")
                return None
                
            # Recognize speech using existing speech recognition
            recognized_text = self._recognize_speech(audio_data, language_code)
            if not recognized_text:
                return None
                
            # Convert text to phonemes
            epi = Epitran(self._map_language_code_for_epitran(language_code))
            target_phonemes = epi.transliterate(target_word)
            recognized_phonemes = epi.transliterate(recognized_text)
            
            # Calculate phonetic distance
            dst = panphon.distance.Distance()
            phoneme_distance = dst.weighted_feature_edit_distance(target_phonemes, recognized_phonemes)
            
            # Calculate similarity score (inverse of distance)
            max_possible_distance = max(len(target_phonemes), len(recognized_phonemes)) * 5  # Rough estimate
            similarity_score = max(0, 100 - (phoneme_distance / max_possible_distance * 100))
            
            return {
                'recognized_text': recognized_text,
                'target_phonemes': target_phonemes,
                'user_phonemes': recognized_phonemes,
                'phoneme_distance': phoneme_distance,
                'phoneme_similarity_score': similarity_score
            }
        except Exception as e:
            print(f"Error in phoneme analysis: {e}")
            return None

    def _map_language_code_for_epitran(self, language_code):
        """Map standard language codes to Epitran-compatible codes"""
        epitran_map = {
            "es": "spa-Latn",
            "fr": "fra-Latn",
            "de": "deu-Latn",
            "it": "ita-Latn",
            "pt": "por-Latn",
            "en": "eng-Latn"
        }
        return epitran_map.get(language_code, "eng-Latn")  # Default to English if unsupported

    def _recognize_speech(self, audio_data, language_code):
        """Recognize speech from audio using speech recognition"""
        if not HAS_SR:
            return ""
            
        try:
            # Prepare the audio data
            audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            audio_file.write(audio_data)
            audio_file.close()
            
            # Use speech recognition to transcribe
            with sr.AudioFile(audio_file.name) as source:
                audio = self.recognizer.record(source)
                
                # Get recognition language code
                rec_lang = RECOGNITION_LANGUAGES.get(language_code, "en-US")
                
                # Recognize speech
                try:
                    recognized_text = self.recognizer.recognize_google(audio, language=rec_lang)
                    return recognized_text.lower()
                except sr.UnknownValueError:
                    return ""
                except sr.RequestError:
                    return ""
        except Exception as e:
            print(f"Error in speech recognition: {e}")
            return ""
        finally:
            # Clean up the temporary file
            try:
                os.unlink(audio_file.name)
            except:
                pass
                
    def _api_based_analysis(self, audio_data, target_word, language_code):
        """Use a specialized API for pronunciation assessment"""
        # Check if we have API keys in the secrets
        if not hasattr(st, 'secrets') or 'azure_speech_key' not in st.secrets:
            return None
            
        try:
            subscription_key = st.secrets["azure_speech_key"]
            region = st.secrets["azure_region"]
            
            # Save audio data to temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            temp_file.write(audio_data)
            temp_file.close()
            
            # Create pronunciation assessment config
            assessment_config = {
                "referenceText": target_word,
                "gradingSystem": "HundredMark",
                "granularity": "Phoneme",
                "enableMiscue": True
            }
            
            # Make API request
            url = f"https://{region}.stt.speech.microsoft.com/speech/recognition/conversation/cognitiveservices/v1"
            headers = {
                "Ocp-Apim-Subscription-Key": subscription_key,
                "Content-Type": "audio/wav",
                "Pronunciation-Assessment": json.dumps(assessment_config)
            }
            
            with open(temp_file.name, "rb") as f:
                response = requests.post(url, headers=headers, data=f.read())
            
            # Process response
            result = response.json()
            
            # Clean up temp file
            os.unlink(temp_file.name)
            
            # Extract scores
            if 'NBest' in result and len(result['NBest']) > 0:
                pronunciation_score = result['NBest'][0].get('PronunciationAssessment', {}).get('PronScore', 0)
                return {
                    'api_pronunciation_score': pronunciation_score,
                    'api_detailed_results': result
                }
            
            return None
        except Exception as e:
            print(f"Error in speech API assessment: {e}")
            return None

    def _visualize_pronunciation(self, target_phonemes, user_phonemes):
        """Create a visual comparison between target and user pronunciation"""
        try:
            # Create a visualization of phoneme matching
            fig, ax = plt.subplots(figsize=(10, 3))
            
            # Split phonemes into characters
            target_chars = list(target_phonemes)
            user_chars = list(user_phonemes)
            
            # Calculate similarity for each character
            max_len = max(len(target_chars), len(user_chars))
            similarities = []
            
            for i in range(max_len):
                if i < len(target_chars) and i < len(user_chars):
                    if target_chars[i] == user_chars[i]:
                        similarities.append(1.0)  # Perfect match
                    else:
                        # Calculate partial match based on phonetic similarity
                        similarities.append(0.3)  # Placeholder - use actual phonetic similarity
                else:
                    similarities.append(0.0)  # Missing phoneme
            
            # Create the visualization
            from matplotlib.colors import ListedColormap
            cmap = ListedColormap(['#ffcccc', '#ffeecc', '#ffffcc', '#eeffcc', '#ccffcc'])
            
            # Plot the data
            ax.imshow([similarities], cmap=cmap, aspect='auto', vmin=0, vmax=1)
            
            # Add text labels
            for i in range(len(target_chars)):
                ax.text(i, -0.2, target_chars[i], ha='center', va='center', fontsize=14)
            
            for i in range(len(user_chars)):
                ax.text(i, 0.2, user_chars[i], ha='center', va='center', fontsize=14)
            
            ax.set_yticks([])
            ax.set_xticks([])
            ax.set_title("Pronunciation Comparison")
            
            # Convert plot to image
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plt.close()
            
            # Display image in Streamlit
            st.image(buf, caption="Phoneme Comparison", use_column_width=True)
            
            # Provide interpretation of the visualization
            st.markdown("""
            **Understanding the comparison:**
            - Green: Correctly pronounced phonemes
            - Yellow: Partially correct phonemes
            - Red: Incorrectly pronounced phonemes
            
            Focus on improving the pronunciation of the red and yellow sections.
            """)
            
        except Exception as e:
            print(f"Error creating visualization: {e}")

    def _calculate_weighted_score(self, results):
        """Calculate a weighted score based on all available pronunciation metrics"""
        # Define weights for different scoring methods
        weights = {
            'levenshtein_similarity': 0.4,  # Original method
            'phoneme_similarity_score': 0.4,  # Phoneme analysis
            'api_pronunciation_score': 0.6   # API assessment (highest weight if available)
        }
        
        total_score = 0
        total_weight = 0
        
        # Add each available score with its weight
        for metric, weight in weights.items():
            if metric in results and results[metric] is not None:
                total_score += results[metric] * weight
                total_weight += weight
        
        # Calculate final score (default to 60 if no metrics available)
        if total_weight > 0:
            return total_score / total_weight
        else:
            return 60.0  # Default score

    def _generate_detailed_feedback(self, results, language_code):
        """Generate detailed pronunciation feedback based on analysis results"""
        feedback = []
        
        # Get recognized text if available
        recognized_text = results.get('recognized_text', '')
        
        # Check if we detected speech
        if not recognized_text:
            feedback.append("No speech detected. Please speak more clearly and ensure your microphone is working properly.")
            return feedback
        
        # Add feedback based on phoneme analysis
        if 'phoneme_similarity_score' in results:
            score = results['phoneme_similarity_score']
            if score < 50:
                feedback.append("Your pronunciation significantly differs from the target. Focus on each sound individually.")
            elif score < 70:
                feedback.append("Your pronunciation has some differences from the target. Pay attention to the highlighted sounds.")
            else:
                feedback.append("Your pronunciation is generally good, with only minor differences from the target.")
        
        # Add language-specific feedback
        language_sounds = self.difficult_sounds.get(language_code, {})
        for sound, data in language_sounds.items():
            # Check if the sound is in both the target and recognized text
            target_phonemes = results.get('target_phonemes', '')
            user_phonemes = results.get('user_phonemes', '')
            
            if sound in target_phonemes and sound not in user_phonemes:
                feedback.append(f"Practice the '{sound}' sound: {data['example']}")
        
        # Add API-specific feedback if available
        if 'api_detailed_results' in results:
            api_results = results['api_detailed_results']
            # Extract additional feedback from API response
            # This depends on the specific API response structure
            
        return feedback