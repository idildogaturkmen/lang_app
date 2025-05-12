"""
WebRTC-based Pronunciation Practice Module for Vocam Language Learning App
This module uses browser-based WebRTC for audio recording, which works on Streamlit Cloud.
"""

import time
import os
import re
from io import BytesIO
import streamlit as st
import numpy as np
import tempfile
import base64
import io
import json
import wave
from datetime import datetime

# Try to import streamlit-webrtc
try:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, ClientSettings
    has_webrtc = True
except ImportError:
    has_webrtc = False

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
        'ai/è': {'sound': 'eh', 'example': 'mais → meh'},
        'an/en': {'sound': 'nasal "ah"', 'example': 'enfant → ahfah (nasal)'}
    },
    "de": {  # German
        'ch': {'sound': 'kh/sh', 'example': 'ich → ish, Bach → bakh'},
        'r': {'sound': 'guttural r', 'example': 'rot → (guttural r)oht'},
        'ü': {'sound': 'ü (rounded lips)', 'example': 'über → üba'},
        'ö': {'sound': 'eu sound', 'example': 'schön → sheun'},
        'ä': {'sound': 'eh', 'example': 'Mädchen → mehdshen'},
        'ei': {'sound': 'eye', 'example': 'nein → nine'},
        'ie': {'sound': 'ee', 'example': 'wie → vee'},
        'z': {'sound': 'ts', 'example': 'zu → tsoo'},
        'v': {'sound': 'f', 'example': 'Vater → fata'}
    },
    "it": {  # Italian
        'gli': {'sound': 'ly', 'example': 'figlio → feelyo'},
        'gn': {'sound': 'ny', 'example': 'gnocchi → nyokee'},
        'r': {'sound': 'rolled r', 'example': 'Roma → (rolled r)oma'},
        'c+e/i': {'sound': 'ch', 'example': 'ciao → chow'},
        'c+a/o/u': {'sound': 'k', 'example': 'casa → kaza'},
        'sc+e/i': {'sound': 'sh', 'example': 'scienza → shentsa'},
        'z': {'sound': 'ts/dz', 'example': 'pizza → peetsa'}
    },
    # Add more languages as needed
}

def create_pronunciation_practice(text_to_speech_func=None, get_audio_html_func=None, translate_text_func=None):
    """
    Factory function to create a pronunciation practice module.
    
    Args:
        text_to_speech_func: Function to convert text to speech audio
        get_audio_html_func: Function to generate HTML for audio playback
        translate_text_func: Function to translate text between languages
        
    Returns:
        An instance of PronunciationPractice class
    """
    try:
        # Check if streamlit-webrtc is available
        if has_webrtc:
            # Create the WebRTC-based practice module
            return WebRTCPronunciationPractice(
                text_to_speech_func,
                get_audio_html_func,
                translate_text_func
            )
        else:
            # Fallback to simplified module
            return SimplifiedPronunciationPractice(
                text_to_speech_func,
                get_audio_html_func,
                translate_text_func
            )
    except Exception as e:
        # Log the error
        print(f"Error creating pronunciation practice module: {str(e)}")
        
        # Create a fallback class that just displays a message
        class FallbackPractice:
            def __init__(self):
                self.difficult_sounds = DIFFICULT_SOUNDS
                
            def render_practice_ui(self, word):
                with st.expander("🎤 Pronunciation Practice"):
                    st.warning("Pronunciation practice is currently unavailable.")
                    st.info("This feature is not supported in the current environment.")
            
            def render_practice_session(self, vocabulary, language_code):
                st.warning("Pronunciation practice is currently unavailable.")
                st.info("This feature is not supported in the current environment.")
                
            def _show_pronunciation_tips(self, word):
                language_code = word.get('language_translated', 'en')
                translated_word = word.get('word_translated', '')
                language_sounds = self.difficult_sounds.get(language_code, {})
                
                # Find sounds in this word
                tips = []
                for sound, data in language_sounds.items():
                    if sound in translated_word.lower():
                        tips.append(f"**'{sound}'** sounds like **'{data['sound']}'**")
                
                # Display tips
                if tips:
                    st.markdown("**Pronunciation tips:**")
                    for tip in tips:
                        st.markdown(f"- {tip}")
                else:
                    st.markdown("*No specific pronunciation tips for this word.*")
        
        return FallbackPractice()

class WebRTCPronunciationPractice:
    """
    Class for pronunciation practice using WebRTC for audio recording.
    Works on Streamlit Cloud without requiring system packages.
    """
    
    def __init__(self, text_to_speech_func, get_audio_html_func, translate_text_func):
        self.text_to_speech = text_to_speech_func
        self.get_audio_html = get_audio_html_func
        self.translate_text = translate_text_func
        self.difficult_sounds = DIFFICULT_SOUNDS
        
        # Set language-specific properties
        self._init_language_settings()
        
        # Initialize webrtc settings
        self._init_webrtc_settings()
        
        # Import speech recognition if available
        try:
            import speech_recognition as sr
            self.recognizer = sr.Recognizer()
            self.has_sr = True
        except ImportError:
            self.has_sr = False
            
        # Import Levenshtein if available
        try:
            import Levenshtein
            self.Levenshtein = Levenshtein
            self.has_levenshtein = True
        except ImportError:
            self.has_levenshtein = False
    
    def _init_language_settings(self):
        """Initialize language-specific settings."""
        # Map language codes to speech recognition language settings
        self.recognition_languages = {
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
    
    def _init_webrtc_settings(self):
        """Initialize WebRTC settings."""
        # Create client settings for WebRTC
        self.client_settings = ClientSettings(
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": False, "audio": True},
        )
    
    def render_practice_ui(self, word):
        """
        Render the pronunciation practice UI for a specific word.
        
        Args:
            word: Dictionary containing the vocabulary word data
        """
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
            
            # WebRTC recorder section
            st.markdown("### Record your pronunciation")
            st.markdown("Click 'START' below to begin recording, then say the word and click 'STOP' when finished.")
            
            # Generate a unique key for this WebRTC instance
            webrtc_key = f"webrtc_recorder_{word['id']}_{int(time.time())}"
            
            # Create a temporary session_state variable to store the recording
            if f"webrtc_audio_{word['id']}" not in st.session_state:
                st.session_state[f"webrtc_audio_{word['id']}"] = None
            
            # Create the WebRTC streamer
            webrtc_ctx = webrtc_streamer(
                key=webrtc_key,
                mode=WebRtcMode.SENDONLY,
                client_settings=self.client_settings,
                video_processor_factory=None,  # No video
                audio_recorder_factory=None,  # Default audio recorder
                async_processing=False,
            )
            
            # If recording was stopped, get the recorded audio
            if webrtc_ctx.state.playing and not st.session_state.get(f"recording_{word['id']}", False):
                st.session_state[f"recording_{word['id']}"] = True
                st.markdown("🎙️ **Recording in progress...**")
            
            elif not webrtc_ctx.state.playing and st.session_state.get(f"recording_{word['id']}", False):
                # Recording was stopped
                st.session_state[f"recording_{word['id']}"] = False
                
                # Get the recorded audio
                if webrtc_ctx.audio_receiver:
                    try:
                        audio_frames = webrtc_ctx.audio_receiver.get_frames()
                        if audio_frames:
                            # Process the frames to get WAV data
                            sound_chunk = AudioProcessor.process_frames(audio_frames)
                            if sound_chunk:
                                # Store the audio in session state
                                st.session_state[f"webrtc_audio_{word['id']}"] = sound_chunk
                                
                                # Create an audio player with the recorded audio
                                st.markdown("**Your recording:**")
                                st.audio(sound_chunk)
                                
                                # Analyze button
                                if st.button("Analyze Pronunciation", key=f"analyze_{word['id']}"):
                                    self._analyze_pronunciation(
                                        sound_chunk, 
                                        translated_word, 
                                        language_code
                                    )
                    except Exception as e:
                        st.error(f"Error processing audio: {str(e)}")
    
    def render_practice_session(self, vocabulary, language_code):
        """
        Render a full pronunciation practice session with multiple words.
        
        Args:
            vocabulary: List of vocabulary words
            language_code: Language code for the practice session
        """
        st.markdown("## Pronunciation Practice")
        st.markdown("""
        Practice your pronunciation and get instant feedback.
        Click 'START' to begin recording, then click 'STOP' when you're finished.
        """)
        
        # If there's no practice session data, show a message
        if 'practice_words' not in st.session_state:
            st.info("Click 'Start Practice Session' to begin pronunciation practice.")
            return
        
        try:
            # Display current practice word
            current_index = st.session_state.current_practice_index
            if current_index < len(st.session_state.practice_words):
                current_word = st.session_state.practice_words[current_index]
                
                # Progress bar
                progress = current_index / len(st.session_state.practice_words)
                st.progress(progress)
                st.subheader(f"Word {current_index + 1} of {len(st.session_state.practice_words)}")
                
                # Create columns for word info
                cols = st.columns([1, 1])
                with cols[0]:
                    st.markdown(f"### {current_word.get('word_original', '')}")
                    st.markdown("**English**")
                with cols[1]:
                    st.markdown(f"### {current_word.get('word_translated', '')}")
                    st.markdown(f"**{self._get_language_name(language_code)}**")
                
                # Audio playback
                audio_bytes = self.text_to_speech(current_word.get('word_translated', ''), language_code)
                if audio_bytes:
                    st.markdown("**Listen to correct pronunciation:**")
                    st.markdown(self.get_audio_html(audio_bytes), unsafe_allow_html=True)
                
                # Detailed pronunciation guide
                with st.expander("Pronunciation Guide", expanded=True):
                    self._show_pronunciation_tips(current_word)
                
                # WebRTC recorder section
                st.markdown("### Record your pronunciation")
                st.markdown("Click 'START' below to begin recording, say the word, then click 'STOP' when finished.")
                
                # Generate a unique key for this WebRTC instance
                webrtc_key = f"webrtc_session_{current_word['id']}_{int(time.time())}"
                
                # Create a temporary session_state variable to store the recording
                if "webrtc_session_audio" not in st.session_state:
                    st.session_state.webrtc_session_audio = None
                    
                # Create the WebRTC streamer
                webrtc_ctx = webrtc_streamer(
                    key=webrtc_key,
                    mode=WebRtcMode.SENDONLY,
                    client_settings=self.client_settings,
                    video_processor_factory=None,  # No video
                    audio_recorder_factory=None,  # Default audio recorder
                    async_processing=False,
                )
                
                # If recording was stopped, get the recorded audio
                if webrtc_ctx.state.playing and not st.session_state.get("session_recording", False):
                    st.session_state.session_recording = True
                    st.markdown("🎙️ **Recording in progress...**")
                
                elif not webrtc_ctx.state.playing and st.session_state.get("session_recording", False):
                    # Recording was stopped
                    st.session_state.session_recording = False
                    
                    # Get the recorded audio
                    if webrtc_ctx.audio_receiver:
                        try:
                            audio_frames = webrtc_ctx.audio_receiver.get_frames()
                            if audio_frames:
                                # Process the frames to get WAV data
                                sound_chunk = AudioProcessor.process_frames(audio_frames)
                                if sound_chunk:
                                    # Store the audio in session state
                                    st.session_state.webrtc_session_audio = sound_chunk
                                    
                                    # Create an audio player with the recorded audio
                                    st.markdown("**Your recording:**")
                                    st.audio(sound_chunk)
                                    
                                    # Analyze button - shows immediately
                                    self._analyze_pronunciation(
                                        sound_chunk, 
                                        current_word.get('word_translated', ''), 
                                        language_code,
                                        show_next_button=True
                                    )
                        except Exception as e:
                            st.error(f"Error processing audio: {str(e)}")
                            # Add the next button anyway if there's an error
                            if st.button("Next Word", type="primary", key="next_error"):
                                st.session_state.current_practice_index += 1
                                # Clear the audio for the next word
                                st.session_state.webrtc_session_audio = None
                                st.rerun()
                
                # Example in context
                example = self._get_example_sentence(
                    current_word.get('word_original', ''), 
                    current_word.get('language_translated', 'en')
                )
                
                with st.expander("Example in Context", expanded=True):
                    st.markdown(f"**English:** {example['english']}")
                    if example.get('translated'):
                        st.markdown(f"**{self._get_language_name(language_code)}:** {example['translated']}")
                        example_audio = self.text_to_speech(example['translated'], language_code)
                        if example_audio:
                            st.markdown(self.get_audio_html(example_audio), unsafe_allow_html=True)
                
                # Skip button
                if st.button("Skip Word", key="skip_word"):
                    st.session_state.current_practice_index += 1
                    # Clear the audio for the next word
                    st.session_state.webrtc_session_audio = None
                    st.rerun()
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
                    st.rerun()
        except Exception as e:
            st.error(f"An error occurred in the practice session: {str(e)}")
            st.info("This may be due to missing functionality. Try using a smaller number of practice words.")
            
            # Add a reset button in case of errors
            if st.button("Reset Practice"):
                if 'practice_words' in st.session_state:
                    del st.session_state.practice_words
                if 'current_practice_index' in st.session_state:
                    del st.session_state.current_practice_index
                if 'practice_scores' in st.session_state:
                    del st.session_state.practice_scores
                if 'webrtc_session_audio' in st.session_state:
                    del st.session_state.webrtc_session_audio
                st.rerun()
    
    def _analyze_pronunciation(self, audio_data, target_word, language_code, show_next_button=False):
        """
        Analyze the recorded pronunciation.
        
        Args:
            audio_data: Audio data to analyze
            target_word: Target word to check pronunciation against
            language_code: Language code for the target word
            show_next_button: Whether to show the next button after analysis
        """
        try:
            # First, try to recognize the speech
            if self.has_sr:
                try:
                    # Create a temporary file for the audio data
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                        temp_file.write(audio_data)
                        temp_filename = temp_file.name
                    
                    # Use speech recognition
                    sr_language = self.recognition_languages.get(language_code, "en-US")
                    
                    with st.spinner("Analyzing pronunciation..."):
                        recognition_result = self._recognize_speech(temp_filename, sr_language)
                    
                    # Clean up the temp file
                    try:
                        os.unlink(temp_filename)
                    except:
                        pass
                    
                    if recognition_result:
                        # Calculate similarity using Levenshtein distance
                        if self.has_levenshtein:
                            similarity = self._calculate_similarity(recognition_result.lower(), target_word.lower())
                        else:
                            # Simple word match check if Levenshtein not available
                            similarity = 100 if recognition_result.lower() == target_word.lower() else 50
                        
                        # Save the score
                        if 'practice_scores' not in st.session_state:
                            st.session_state.practice_scores = []
                        st.session_state.practice_scores.append(similarity)
                        
                        # Show feedback
                        self._show_pronunciation_feedback(
                            target_word, recognition_result, similarity, language_code
                        )
                    else:
                        st.warning("Could not recognize your speech. Please try again and speak clearly.")
                        # Save a low score for failed recognition
                        if 'practice_scores' not in st.session_state:
                            st.session_state.practice_scores = []
                        st.session_state.practice_scores.append(20)  # Low score for failed recognition
                except Exception as e:
                    st.error(f"Speech recognition error: {str(e)}")
                    # Fall back to simplified feedback
                    st.info("We couldn't analyze your pronunciation in detail, but we heard your recording!")
                    # Save a moderate score
                    if 'practice_scores' not in st.session_state:
                        st.session_state.practice_scores = []
                    st.session_state.practice_scores.append(50)  # Moderate score when analysis fails
            else:
                # No speech recognition available, give simplified feedback
                st.info("Your pronunciation was recorded successfully!")
                st.markdown("Speech recognition is not available for detailed feedback.")
                # Save a moderate score
                if 'practice_scores' not in st.session_state:
                    st.session_state.practice_scores = []
                st.session_state.practice_scores.append(60)  # Moderate score when SR not available
            
            # Show next button if requested
            if show_next_button:
                if st.button("Next Word", type="primary", key="next_analyzed"):
                    st.session_state.current_practice_index += 1
                    # Clear the audio for the next word
                    st.session_state.webrtc_session_audio = None
                    st.rerun()
        except Exception as e:
            st.error(f"Error analyzing pronunciation: {str(e)}")
            # Add a fallback score
            if 'practice_scores' not in st.session_state:
                st.session_state.practice_scores = []
            st.session_state.practice_scores.append(40)  # Below average score on error
            
            # Still show next button if requested
            if show_next_button:
                if st.button("Next Word", type="primary", key="next_error"):
                    st.session_state.current_practice_index += 1
                    # Clear the audio for the next word
                    st.session_state.webrtc_session_audio = None
                    st.rerun()
    
    def _recognize_speech(self, audio_file, language):
        """
        Recognize speech in an audio file.
        
        Args:
            audio_file: Path to audio file
            language: Language code for recognition
            
        Returns:
            Recognized text or None
        """
        try:
            with self.recognizer.audio_file(audio_file) as source:
                audio_data = self.recognizer.record(source)
            
            # Try different recognition methods
            try:
                # Try Google Speech Recognition
                result = self.recognizer.recognize_google(audio_data, language=language)
                return result
            except:
                # Try without language specification
                try:
                    result = self.recognizer.recognize_google(audio_data)
                    return result
                except:
                    return None
                
        except Exception as e:
            print(f"Speech recognition error: {e}")
            return None
    
    def _calculate_similarity(self, text1, text2):
        """
        Calculate similarity between two strings.
        
        Args:
            text1: First string
            text2: Second string
            
        Returns:
            Similarity score (0-100%)
        """
        # Remove punctuation and extra spaces
        text1 = re.sub(r'[^\w\s]', '', text1).strip().lower()
        text2 = re.sub(r'[^\w\s]', '', text2).strip().lower()
        
        # If either string is empty, return 0
        if not text1 or not text2:
            return 0
        
        # Calculate Levenshtein distance
        distance = self.Levenshtein.distance(text1, text2)
        
        # Calculate similarity as percentage
        max_len = max(len(text1), len(text2))
        if max_len == 0:
            return 100  # Both strings are empty
        
        similarity = 100 * (1 - distance / max_len)
        return similarity
    
    def _show_pronunciation_feedback(self, target, recognized, similarity, language_code):
        """
        Show feedback on pronunciation with suggestions for improvement.
        
        Args:
            target: Target word or phrase
            recognized: Recognized word or phrase
            similarity: Similarity score (0-100%)
            language_code: Language code
        """
        # Create a feedback container
        with st.container():
            st.subheader("Pronunciation Feedback")
            
            # Display what was heard vs what was expected
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**You said:**")
                st.markdown(f"### {recognized}")
            
            with col2:
                st.markdown("**Correct pronunciation:**")
                st.markdown(f"### {target}")
            
            # Display similarity score
            col1, col2 = st.columns(2)
            with col1:
                # Display score as a gauge
                st.markdown(f"**Accuracy: {similarity:.0f}%**")
                st.progress(similarity / 100.0)
            
            with col2:
                # Display qualitative feedback
                if similarity >= 90:
                    st.success("Excellent pronunciation!")
                elif similarity >= 70:
                    st.info("Good pronunciation!")
                elif similarity >= 50:
                    st.warning("Fair pronunciation.")
                else:
                    st.error("Needs improvement.")
            
            # Show specific feedback based on differences
            if similarity < 90:
                st.markdown("### Tips for Improvement")
                
                # Identify specific sounds to focus on
                problematic_sounds = self._identify_problematic_sounds(target, recognized, language_code)
                
                if problematic_sounds:
                    st.markdown("Focus on these sounds:")
                    for sound in problematic_sounds:
                        # Get example for this sound if available
                        sound_data = self.difficult_sounds.get(language_code, {}).get(sound, None)
                        if sound_data:
                            st.markdown(f"- **'{sound}'** sounds like **'{sound_data['sound']}'** (Example: {sound_data['example']})")
                        else:
                            st.markdown(f"- Practice the **'{sound}'** sound")
                
                # General advice
                st.markdown("""
                **Practice techniques:**
                1. Listen to the correct pronunciation multiple times
                2. Speak slowly and clearly
                3. Exaggerate mouth movements at first
                4. Record yourself and compare with the original
                """)
    
    def _identify_problematic_sounds(self, target, recognized, language_code):
        """
        Identify potentially problematic sounds based on mispronunciation.
        
        Args:
            target: Target word
            recognized: Recognized word
            language_code: Language code
            
        Returns:
            List of problematic sounds
        """
        # Get difficult sounds for this language
        language_sounds = self.difficult_sounds.get(language_code, {})
        
        # If no language-specific sounds, return empty list
        if not language_sounds:
            return []
        
        # Check for these sounds in the target word
        problematic = []
        
        for sound, data in language_sounds.items():
            # If the sound is in the target word but the recognition is different
            if sound in target.lower() and (sound not in recognized.lower()):
                problematic.append(sound)
        
        # If we didn't find specific problems, check for other common issues
        if not problematic:
            for sound in language_sounds.keys():
                if sound in target.lower():
                    problematic.append(sound)
        
        return problematic[:3]  # Limit to top 3 problems
    
    def _show_pronunciation_tips(self, word):
        """
        Show pronunciation tips for a word.
        
        Args:
            word: Word dictionary
        """
        # Get language code
        language_code = word.get('language_translated', 'en')
        
        # Get the word to show tips for
        translated_word = word.get('word_translated', '')
        
        # Get difficult sounds for this language
        language_sounds = self.difficult_sounds.get(language_code, {})
        
        # Find sounds in this word
        tips = []
        for sound, data in language_sounds.items():
            if sound in translated_word.lower():
                tips.append(f"**'{sound}'** sounds like **'{data['sound']}'**")
        
        # Display tips
        if tips:
            st.markdown("**Pronunciation tips:**")
            for tip in tips:
                st.markdown(f"- {tip}")
        else:
            st.markdown("*No specific pronunciation tips for this word.*")
    
    def _get_example_sentence(self, word, language_code):
        """Get an example sentence for the word."""
        if self.translate_text:
            from random import choice
            
            # Simple English templates
            templates = [
                f"The {word} is on the table.",
                f"I like this {word} very much.",
                f"Can you see the {word}?",
                f"This {word} is very useful.",
                f"I need a new {word}."
            ]
            
            # Select a random template
            example = choice(templates)
            
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
    
    def _get_language_name(self, language_code):
        """Convert language code to language name."""
        language_names = {
            "es": "Spanish",
            "fr": "French",
            "de": "German",
            "it": "Italian",
            "pt": "Portuguese",
            "ru": "Russian",
            "ja": "Japanese",
            "zh-CN": "Chinese"
        }
        return language_names.get(language_code, language_code)
    
    def _show_practice_results(self):
        """Show results of the practice session."""
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
        
        # Suggestions for next steps
        st.markdown("""
        ### Next Steps
        
        1. **Listen carefully** to native speakers
        2. **Record yourself** speaking and compare with native pronunciation
        3. **Practice daily** for best results
        4. **Focus on difficult sounds** specific to this language
        """)

class SimplifiedPronunciationPractice:
    """
    Simplified class for pronunciation practice that works on all platforms.
    Focuses on pronunciation tips and listening practice without recording.
    """
    
    def __init__(self, text_to_speech_func, get_audio_html_func, translate_text_func):
        self.text_to_speech = text_to_speech_func
        self.get_audio_html = get_audio_html_func
        self.translate_text = translate_text_func
        self.difficult_sounds = DIFFICULT_SOUNDS
    
    def render_practice_ui(self, word):
        """
        Render the pronunciation practice UI for a specific word.
        
        Args:
            word: Dictionary containing the vocabulary word data
        """
        with st.expander("🎤 Pronunciation Guide"):
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
            
            # Show pronunciation breakdown
            st.markdown("### Pronunciation Breakdown")
            
            # Get language-specific sounds for this word
            language_sounds = self.difficult_sounds.get(language_code, {})
            
            # Find sounds in this word
            tips = []
            for sound, data in language_sounds.items():
                if sound in translated_word.lower():
                    tips.append({
                        'sound': sound,
                        'pronunciation': data['sound'],
                        'example': data['example']
                    })
            
            if tips:
                for tip in tips:
                    with st.container():
                        cols = st.columns([1, 3])
                        with cols[0]:
                            st.markdown(f"### '{tip['sound']}'")
                        with cols[1]:
                            st.markdown(f"**Sounds like:** '{tip['pronunciation']}'")
                            st.markdown(f"**Example:** {tip['example']}")
                            
                # Practice section
                st.markdown("### Practice Steps")
                st.markdown("""
                1. **Listen** to the audio above multiple times
                2. **Repeat** the word out loud, focusing on the highlighted sounds
                3. **Slow down** and exaggerate the movements of your mouth
                4. **Compare** your pronunciation with the audio
                """)
            else:
                st.markdown("*No specific pronunciation tips for this word.*")
                
            # Example sentences
            st.markdown("### Usage Example")
            example = self._get_example_sentence(original_word, language_code)
            st.markdown(f"**English:** {example['english']}")
            
            if example.get('translated'):
                st.markdown(f"**Translation:** {example['translated']}")
                
                # Play example audio if available
                example_audio = self.text_to_speech(example['translated'], language_code)
                if example_audio:
                    st.markdown(self.get_audio_html(example_audio), unsafe_allow_html=True)
    
    def render_practice_session(self, vocabulary, language_code):
        """
        Render a simplified pronunciation practice session.
        
        Args:
            vocabulary: List of vocabulary words
            language_code: Language code for the practice session
        """
        st.markdown("## Pronunciation Practice")
        st.markdown("""
        This simplified pronunciation practice focuses on listening and visual guides.
        """)
        
        # If there's no practice session data, show a message
        if 'practice_words' not in st.session_state:
            st.info("Click 'Start Practice Session' to begin.")
            return
        
        try:
            # Display current practice word
            current_index = st.session_state.current_practice_index
            if current_index < len(st.session_state.practice_words):
                current_word = st.session_state.practice_words[current_index]
                
                # Progress bar
                progress = current_index / len(st.session_state.practice_words)
                st.progress(progress)
                st.subheader(f"Word {current_index + 1} of {len(st.session_state.practice_words)}")
                
                # Create columns for word info
                cols = st.columns([1, 1])
                with cols[0]:
                    st.markdown(f"### {current_word.get('word_original', '')}")
                    st.markdown("**English**")
                with cols[1]:
                    st.markdown(f"### {current_word.get('word_translated', '')}")
                    st.markdown(f"**{self._get_language_name(language_code)}**")
                
                # Audio playback
                audio_bytes = self.text_to_speech(current_word.get('word_translated', ''), language_code)
                if audio_bytes:
                    st.markdown("**Listen and repeat:**")
                    st.markdown(self.get_audio_html(audio_bytes), unsafe_allow_html=True)
                
                # Detailed pronunciation guide
                with st.expander("Pronunciation Guide", expanded=True):
                    self._show_pronunciation_tips(current_word)
                
                # Example in context
                example = self._get_example_sentence(
                    current_word.get('word_original', ''), 
                    current_word.get('language_translated', 'en')
                )
                
                with st.expander("Example in Context", expanded=True):
                    st.markdown(f"**English:** {example['english']}")
                    if example.get('translated'):
                        st.markdown(f"**{self._get_language_name(language_code)}:** {example['translated']}")
                        example_audio = self.text_to_speech(example['translated'], language_code)
                        if example_audio:
                            st.markdown(self.get_audio_html(example_audio), unsafe_allow_html=True)
                
                # Self-assessment
                st.markdown("### Self-Assessment")
                st.markdown("How well did you pronounce the word?")
                
                rating = st.select_slider(
                    "Rate your pronunciation:",
                    options=["Poor", "Fair", "Good", "Very Good", "Excellent"],
                    value="Good"
                )
                
                # Next word button
                if st.button("Next Word", type="primary"):
                    # Save rating if not already in session state
                    if 'practice_scores' not in st.session_state:
                        st.session_state.practice_scores = []
                    
                    # Convert rating to numeric score
                    score_map = {
                        "Poor": 1, 
                        "Fair": 2, 
                        "Good": 3, 
                        "Very Good": 4, 
                        "Excellent": 5
                    }
                    st.session_state.practice_scores.append(score_map.get(rating, 3))
                    
                    # Move to next word
                    st.session_state.current_practice_index += 1
                    st.rerun()
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
                    st.rerun()
        except Exception as e:
            st.error(f"An error occurred in the practice session: {str(e)}")
            st.info("This may be due to missing functionality in Streamlit Cloud. Try using a smaller number of practice words.")
            
            # Add a reset button in case of errors
            if st.button("Reset Practice"):
                if 'practice_words' in st.session_state:
                    del st.session_state.practice_words
                if 'current_practice_index' in st.session_state:
                    del st.session_state.current_practice_index
                if 'practice_scores' in st.session_state:
                    del st.session_state.practice_scores
                st.rerun()
    
    def _get_example_sentence(self, word, language_code):
        """Get an example sentence for the word."""
        if self.translate_text:
            from random import choice
            
            # Simple English templates
            templates = [
                f"The {word} is on the table.",
                f"I like this {word} very much.",
                f"Can you see the {word}?",
                f"This {word} is very useful.",
                f"I need a new {word}."
            ]
            
            # Select a random template
            example = choice(templates)
            
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
    
    def _get_language_name(self, language_code):
        """Convert language code to language name."""
        language_names = {
            "es": "Spanish",
            "fr": "French",
            "de": "German",
            "it": "Italian",
            "pt": "Portuguese",
            "ru": "Russian",
            "ja": "Japanese",
            "zh-CN": "Chinese"
        }
        return language_names.get(language_code, language_code)
    
    def _show_pronunciation_tips(self, word):
        """
        Show pronunciation tips for a word.
        
        Args:
            word: Word dictionary
        """
        # Get language code
        language_code = word.get('language_translated', 'en')
        
        # Get the word to show tips for
        translated_word = word.get('word_translated', '')
        
        # Get difficult sounds for this language
        language_sounds = self.difficult_sounds.get(language_code, {})
        
        # Find sounds in this word
        tips = []
        for sound, data in language_sounds.items():
            if sound in translated_word.lower():
                tips.append(f"**'{sound}'** sounds like **'{data['sound']}'**")
        
        # Display tips
        if tips:
            st.markdown("**Pronunciation tips:**")
            for tip in tips:
                st.markdown(f"- {tip}")
        else:
            st.markdown("*No specific pronunciation tips for this word.*")
    
    def _show_practice_results(self):
        """Show results of the practice session."""
        st.subheader("🎉 Practice Session Completed!")
        
        # Get scores
        scores = st.session_state.practice_scores if 'practice_scores' in st.session_state else []
        
        if not scores:
            st.info("No self-assessments were recorded.")
            return
        
        # Calculate stats
        avg_score = sum(scores) / len(scores)
        max_score = 5  # Max possible score
        
        # Display stats
        st.markdown(f"**Average self-rating: {avg_score:.1f}/5**")
        
        # Progress bar for overall performance
        st.progress(avg_score / max_score)
        
        # Feedback based on average score
        if avg_score >= 4.5:
            st.success("Outstanding work! You feel confident in your pronunciation.")
        elif avg_score >= 3.5:
            st.success("Great job! Your pronunciation practice is going well.")
        elif avg_score >= 2.5:
            st.info("Good effort! Continue practicing to build confidence.")
        else:
            st.warning("Keep practicing! Regular practice will improve your pronunciation.")
        
        # Suggestions for next steps
        st.markdown("""
        ### Next Steps
        
        1. **Listen carefully** to native speakers
        2. **Practice daily** for best results
        3. **Focus on difficult sounds** specific to this language
        4. **Use online pronunciation guides** for additional help
        """)

class AudioProcessor:
    """Class for processing audio frames from WebRTC."""
    
    @staticmethod
    def process_frames(frames):
        """
        Process audio frames from WebRTC to create a WAV file.
        
        Args:
            frames: Audio frames from WebRTC
            
        Returns:
            WAV data as bytes
        """
        # Check if we have any frames
        if not frames:
            return None
        
        # Get the audio data from the frames
        audio_data = b"".join([frame.to_ndarray().tobytes() for frame in frames])
        
        # Create a WAV file in memory
        wav_io = io.BytesIO()
        
        # Get audio parameters from the first frame
        sample_rate = frames[0].sample_rate
        channels = frames[0].layout.channels
        
        # Create a WAV file
        with wave.open(wav_io, "wb") as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data)
        
        # Get the WAV data
        wav_data = wav_io.getvalue()
        return wav_data