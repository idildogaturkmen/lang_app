"""
Direct WebRTC implementation for pronunciation practice
This module uses streamlit-webrtc for audio recording on Streamlit Cloud
"""

import streamlit as st
import time
import tempfile
import io
import os
import wave
import re
from datetime import datetime

# Try importing WebRTC
try:
    from streamlit_webrtc import (
        webrtc_streamer,
        WebRtcMode,
        ClientSettings,
        VideoHTMLAttributes,
    )
    import av  # Required for WebRTC
    HAS_WEBRTC = True
except ImportError:
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
    # Print debug information
    st.session_state.pronunciation_debug = {
        "has_webrtc": HAS_WEBRTC,
        "has_sr": HAS_SR,
        "has_levenshtein": HAS_LEVENSHTEIN
    }
    
    # Create pronunciation practice instance
    return PronunciationPractice(
        text_to_speech_func,
        get_audio_html_func,
        translate_text_func
    )

class PronunciationPractice:
    """
    Class for pronunciation practice with WebRTC audio recording.
    """
    
    def __init__(self, text_to_speech_func, get_audio_html_func, translate_text_func):
        """
        Initialize pronunciation practice.
        
        Args:
            text_to_speech_func: Function for text-to-speech conversion
            get_audio_html_func: Function to get audio HTML
            translate_text_func: Function to translate text
        """
        self.text_to_speech = text_to_speech_func
        self.get_audio_html = get_audio_html_func
        self.translate_text = translate_text_func
        self.difficult_sounds = DIFFICULT_SOUNDS
        
        # Initialize WebRTC
        if HAS_WEBRTC:
            self.client_settings = ClientSettings(
                rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                media_stream_constraints={"video": False, "audio": True},
            )
        
        # Initialize speech recognition
        if HAS_SR:
            self.recognizer = sr.Recognizer()
    
    def render_practice_ui(self, word):
        """
        Render pronunciation practice UI for a word.
        
        Args:
            word: Word dictionary
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
            
            # Check if WebRTC is available
            if HAS_WEBRTC:
                # Generate a unique key for this recorder
                webrtc_key = f"webrtc_{word['id']}_{int(time.time())}"
                
                st.markdown("### Record Your Pronunciation")
                st.markdown("Click the START button below to begin recording. Say the word and then click STOP.")
                
                # Create WebRTC streamer
                webrtc_ctx = webrtc_streamer(
                    key=webrtc_key,
                    mode=WebRtcMode.SENDONLY,
                    client_settings=self.client_settings,
                    video_frame_callback=None,
                    audio_frame_callback=None,
                    async_processing=False,
                )
                
                # Check if recording has finished
                if webrtc_ctx.audio_receiver and not webrtc_ctx.state.playing:
                    # Get audio frames
                    audio_frames = webrtc_ctx.audio_receiver.get_frames()
                    
                    if audio_frames and len(audio_frames) > 0:
                        # Process audio frames
                        audio_buffer = b""
                        for frame in audio_frames:
                            # Convert audio frame to bytes
                            frame_data = frame.to_ndarray().tobytes()
                            audio_buffer += frame_data
                            
                        # Create WAV file
                        wav_data = create_wav_from_audio_buffer(
                            audio_buffer, 
                            sample_rate=frame.sample_rate, 
                            channels=frame.layout.channels
                        )
                        
                        if wav_data:
                            # Show audio playback
                            st.audio(wav_data)
                            
                            # Save file to temp location
                            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                                temp_file.write(wav_data)
                                temp_path = temp_file.name
                            
                            # Analyze pronunciation
                            if st.button("Analyze Pronunciation"):
                                self._analyze_pronunciation(
                                    temp_path, 
                                    translated_word, 
                                    language_code
                                )
            else:
                # Show message if WebRTC is not available
                st.warning("Microphone recording requires the streamlit-webrtc package.")
                st.info("Please add streamlit-webrtc to your requirements.txt file.")
                
                # Show debug info
                if "pronunciation_debug" in st.session_state:
                    st.markdown("### Debug Information")
                    st.json(st.session_state.pronunciation_debug)
    
    def render_practice_session(self, vocabulary, language_code):
        """
        Render pronunciation practice session.
        
        Args:
            vocabulary: List of vocabulary words
            language_code: Language code
        """
        st.markdown("## Pronunciation Practice")
        
        # Show debug info
        if "pronunciation_debug" in st.session_state:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"WebRTC: {'✅' if st.session_state.pronunciation_debug['has_webrtc'] else '❌'}")
            with col2:
                st.write(f"Speech Recog: {'✅' if st.session_state.pronunciation_debug['has_sr'] else '❌'}")
            with col3:
                st.write(f"Levenshtein: {'✅' if st.session_state.pronunciation_debug['has_levenshtein'] else '❌'}")
        
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
            
            # Word info
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"### {current_word.get('word_original', '')}")
                st.markdown("**English**")
            with col2:
                st.markdown(f"### {current_word.get('word_translated', '')}")
                st.markdown(f"**{LANGUAGE_NAMES.get(language_code, language_code)}**")
            
            # Listen to pronunciation
            audio_bytes = self.text_to_speech(current_word.get('word_translated', ''), language_code)
            if audio_bytes:
                st.markdown("**Listen to correct pronunciation:**")
                st.markdown(self.get_audio_html(audio_bytes), unsafe_allow_html=True)
            
            # Pronunciation guide
            with st.expander("Pronunciation Guide", expanded=True):
                self._show_pronunciation_tips(current_word)
            
            # WebRTC recording
            if HAS_WEBRTC:
                st.markdown("### Record Your Pronunciation")
                
                # Generate a unique key for this recorder
                webrtc_key = f"webrtc_session_{current_index}_{int(time.time())}"
                
                # Create WebRTC streamer
                webrtc_ctx = webrtc_streamer(
                    key=webrtc_key,
                    mode=WebRtcMode.SENDONLY,
                    client_settings=self.client_settings,
                    video_frame_callback=None,
                    audio_frame_callback=None,
                    async_processing=False,
                )
                
                # Check if recording has finished
                if webrtc_ctx.audio_receiver and not webrtc_ctx.state.playing:
                    # Get audio frames
                    audio_frames = webrtc_ctx.audio_receiver.get_frames()
                    
                    if audio_frames and len(audio_frames) > 0:
                        # Process audio frames
                        audio_buffer = b""
                        for frame in audio_frames:
                            # Convert audio frame to bytes
                            frame_data = frame.to_ndarray().tobytes()
                            audio_buffer += frame_data
                            
                        # Create WAV file
                        wav_data = create_wav_from_audio_buffer(
                            audio_buffer, 
                            sample_rate=frame.sample_rate, 
                            channels=frame.layout.channels
                        )
                        
                        if wav_data:
                            # Show audio playback
                            st.audio(wav_data)
                            
                            # Save file to temp location
                            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                                temp_file.write(wav_data)
                                temp_path = temp_file.name
                            
                            # Analyze pronunciation
                            score = self._analyze_pronunciation(
                                temp_path, 
                                current_word.get('word_translated', ''), 
                                language_code
                            )
                            
                            # Store score
                            if score is not None:
                                if 'practice_scores' not in st.session_state:
                                    st.session_state.practice_scores = []
                                st.session_state.practice_scores.append(score)
                            
                            # Next button
                            if st.button("Next Word", type="primary"):
                                st.session_state.current_practice_index += 1
                                st.rerun()
            else:
                # Fallback to self-assessment if WebRTC is not available
                st.warning("Real-time pronunciation assessment is not available.")
                st.info("Please add streamlit-webrtc to your requirements.txt file.")
                
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
                    # Convert rating to score
                    score_map = {
                        "Poor": 20, 
                        "Fair": 40, 
                        "Good": 60, 
                        "Very Good": 80, 
                        "Excellent": 95
                    }
                    score = score_map.get(rating, 60)
                    
                    # Store score
                    if 'practice_scores' not in st.session_state:
                        st.session_state.practice_scores = []
                    st.session_state.practice_scores.append(score)
                    
                    # Next word
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
                st.rerun()
    
    def _analyze_pronunciation(self, audio_file, target_word, language_code):
        """
        Analyze pronunciation.
        
        Args:
            audio_file: Audio file path
            target_word: Target word
            language_code: Language code
            
        Returns:
            Similarity score (0-100%)
        """
        if not HAS_SR:
            st.warning("Speech recognition is not available.")
            return None
        
        try:
            # Recognize speech
            with sr.AudioFile(audio_file) as source:
                audio_data = self.recognizer.record(source)
            
            # Try to recognize speech
            try:
                # Get language code for recognition
                sr_language = RECOGNITION_LANGUAGES.get(language_code, "en-US")
                recognized_text = self.recognizer.recognize_google(audio_data, language=sr_language)
            except:
                # Try without language specification
                try:
                    recognized_text = self.recognizer.recognize_google(audio_data)
                except Exception as e:
                    st.error(f"Could not recognize speech: {str(e)}")
                    return 30  # Low score for failed recognition
            
            # Calculate similarity
            if HAS_LEVENSHTEIN:
                # Clean up text
                recognized_text = recognized_text.lower().strip()
                target_word = target_word.lower().strip()
                
                # Calculate Levenshtein distance
                distance = Levenshtein.distance(recognized_text, target_word)
                
                # Calculate similarity as percentage
                max_len = max(len(recognized_text), len(target_word))
                if max_len == 0:
                    similarity = 100
                else:
                    similarity = 100 * (1 - distance / max_len)
            else:
                # Simple match if Levenshtein is not available
                recognized_text = recognized_text.lower().strip()
                target_word = target_word.lower().strip()
                similarity = 100 if recognized_text == target_word else 50
            
            # Show feedback
            self._show_pronunciation_feedback(
                target_word, recognized_text, similarity, language_code
            )
            
            return similarity
            
        except Exception as e:
            st.error(f"Error analyzing pronunciation: {str(e)}")
            return 40  # Below average score on error
    
    def _show_pronunciation_feedback(self, target, recognized, similarity, language_code):
        """
        Show pronunciation feedback.
        
        Args:
            target: Target word
            recognized: Recognized word
            similarity: Similarity score
            language_code: Language code
        """
        st.subheader("Pronunciation Feedback")
        
        # What was heard vs expected
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**You said:**")
            st.markdown(f"### {recognized}")
        with col2:
            st.markdown("**Correct pronunciation:**")
            st.markdown(f"### {target}")
        
        # Similarity score
        st.markdown(f"**Accuracy: {similarity:.0f}%**")
        st.progress(similarity / 100.0)
        
        # Feedback based on score
        if similarity >= 90:
            st.success("Excellent pronunciation!")
        elif similarity >= 70:
            st.info("Good pronunciation!")
        elif similarity >= 50:
            st.warning("Fair pronunciation. Keep practicing!")
        else:
            st.error("Needs improvement. Listen to the example again.")
        
        # Pronunciation tips
        if similarity < 90:
            st.markdown("### Tips for Improvement")
            
            # Find problematic sounds
            problem_sounds = []
            for sound, data in self.difficult_sounds.get(language_code, {}).items():
                if sound in target.lower():
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
    
    def _show_pronunciation_tips(self, word):
        """
        Show pronunciation tips for a word.
        
        Args:
            word: Word dictionary
        """
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
    
    def _get_example_sentence(self, word, language_code):
        """
        Get example sentence.
        
        Args:
            word: Word
            language_code: Language code
            
        Returns:
            Dictionary with English and translated examples
        """
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

def create_wav_from_audio_buffer(audio_buffer, sample_rate=48000, channels=1):
    """
    Create WAV data from audio buffer.
    
    Args:
        audio_buffer: Audio buffer
        sample_rate: Sample rate
        channels: Number of channels
        
    Returns:
        WAV data
    """
    try:
        # Create in-memory WAV file
        wav_io = io.BytesIO()
        
        # Create WAV file
        with wave.open(wav_io, "wb") as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_buffer)
        
        # Get WAV data
        wav_data = wav_io.getvalue()
        return wav_data
    except Exception as e:
        st.error(f"Error creating WAV file: {str(e)}")
        return None