"""
Pronunciation Practice Module for Vocam Language Learning App

This module provides voice recognition capabilities to let users practice
pronunciation and receive feedback on their spoken language skills.
"""

import streamlit as st
import base64
import time
import numpy as np
from io import BytesIO
import re
import tempfile
import os
from typing import Tuple, Dict, List, Optional, Union
import random

# Try importing the required libraries with graceful fallbacks
try:
    import speech_recognition as sr
    has_speech_recognition = True
except ImportError:
    has_speech_recognition = False
    
try:
    import pydub
    from pydub import AudioSegment
    has_pydub = True
except ImportError:
    has_pydub = False

def _default_text_to_speech(text, lang):
    """Fallback text-to-speech function."""
    try:
        from gtts import gTTS
        tts = gTTS(text=text, lang=lang, slow=False)
        mp3_fp = BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        audio_bytes = mp3_fp.read()
        return audio_bytes
    except Exception as e:
        print(f"Text-to-speech error: {e}")
        return None

def _default_get_audio_html(audio_bytes):
    """Fallback function to generate HTML for audio playback."""
    if not audio_bytes:
        return "<p>Audio not available</p>"
    audio_base64 = base64.b64encode(audio_bytes).decode()
    audio_tag = f'<audio src="data:audio/mp3;base64,{audio_base64}" controls></audio>'
    return audio_tag

def _default_translate_text(text, target_language):
    """Fallback translation function."""
    try:
        # Try to use Google Translate API
        from google.cloud import translate_v2 as translate
        translate_client = translate.Client()
        result = translate_client.translate(text, target_language=target_language)
        return result["translatedText"]
    except:
        try:
            # Try to use deep_translator
            from deep_translator import GoogleTranslator
            translator = GoogleTranslator(source='auto', target=target_language)
            return translator.translate(text)
        except:
            # Return the original text if translation fails
            return f"[Translation to {target_language} not available]"
        

class PronunciationPractice:
    """Handles voice recognition and pronunciation feedback."""
    
    def __init__(self, 
             text_to_speech_func=None, 
             get_audio_html_func=None,
             translate_text_func=None):
        """
        Initialize the pronunciation practice module.
        
        Args:
            text_to_speech_func: Function that converts text to speech
            get_audio_html_func: Function that generates HTML for audio playback
            translate_text_func: Function that translates text between languages
        """
        # Use provided functions or fallbacks
        self.text_to_speech = text_to_speech_func or _default_text_to_speech
        self.get_audio_html = get_audio_html_func or _default_get_audio_html
        self.translate_text = translate_text_func or _default_translate_text
            
    def is_available(self) -> bool:
        """Check if voice recognition is available."""
        return has_speech_recognition and has_pydub
    
    def get_requirements(self) -> List[str]:
        """Get a list of required packages that are missing."""
        missing = []
        if not has_speech_recognition:
            missing.append("SpeechRecognition (pip install SpeechRecognition)")
        if not has_pydub:
            missing.append("pydub (pip install pydub)")
        if not has_speech_recognition or not has_pydub:
            missing.append("PyAudio (pip install PyAudio)")
        return missing
    
    def render_practice_ui(self, word: Dict, vocabulary: List[Dict] = None) -> None:
        """
        Render the pronunciation practice UI for a given word.
        
        Args:
            word: Dictionary containing word information
            vocabulary: Optional list of vocabulary words for related practice
        """
        if not self.is_available():
            st.warning("Pronunciation practice requires additional packages.")
            st.info("Please install the following packages to enable pronunciation practice:")
            for pkg in self.get_requirements():
                st.code(f"pip install {pkg}")
            return
        
        with st.expander("🎤 Practice Pronunciation", expanded=True):
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("### Listen to the correct pronunciation")
                if word.get('word_translated') and self.text_to_speech and self.get_audio_html:
                    audio_bytes = self.text_to_speech(word['word_translated'], word['language_translated'])
                    if audio_bytes:
                        st.markdown(self.get_audio_html(audio_bytes), unsafe_allow_html=True)
                        st.markdown("👆 Listen carefully and try to match the pronunciation")
                
                # Show pronunciation tips
                st.markdown("### Pronunciation Tips")
                self._show_pronunciation_tips(word)
            
            with col2:
                st.markdown("### Your Pronunciation")
                
                # Add record button for voice input
                if st.button("🎙️ Record Pronunciation"):
                    with st.spinner("Listening... Speak now"):
                        audio_data, error = self._record_audio()
                        
                        if error:
                            st.error(f"Error recording audio: {error}")
                        elif audio_data:
                            # Save to session state to preserve between renders
                            st.session_state.last_recorded_audio = audio_data
                            st.rerun()
                
                # If we have recorded audio, display and analyze it
                if hasattr(st.session_state, 'last_recorded_audio') and st.session_state.last_recorded_audio:
                    # Convert the audio data to a format that can be played in the browser
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                        tmp_file.write(st.session_state.last_recorded_audio)
                        tmp_path = tmp_file.name
                    
                    # Display audio player for recorded audio
                    audio_bytes = open(tmp_path, 'rb').read()
                    st.audio(audio_bytes, format="audio/wav")
                    
                    # Now try to recognize the speech
                    with st.spinner("Analyzing your pronunciation..."):
                        recognized_text, confidence = self._recognize_speech(
                            st.session_state.last_recorded_audio,
                            language_code=self.language_codes.get(word['language_translated'], 'en-US')
                        )
                        
                        # Calculate similarity score
                        similarity_score = self._calculate_similarity(recognized_text, word['word_translated'])
                        
                        # Show results
                        self._show_pronunciation_results(
                            word['word_translated'], recognized_text, similarity_score, confidence
                        )
                    
                    # Clean up temp file
                    try:
                        os.unlink(tmp_path)
                    except:
                        pass
                    
                    # Option to clear recording and try again
                    if st.button("Clear and Try Again"):
                        if hasattr(st.session_state, 'last_recorded_audio'):
                            del st.session_state.last_recorded_audio
                        st.rerun()
    
    def render_practice_session(self, vocabulary: List[Dict], language_code: str) -> None:
        """
        Render a full pronunciation practice session with multiple words.
        
        Args:
            vocabulary: List of vocabulary words
            language_code: Language code for practice
        """
        if not self.is_available():
            st.warning("Pronunciation practice requires additional packages.")
            st.info("Please install the following packages to enable pronunciation practice:")
            for pkg in self.get_requirements():
                st.code(f"pip install {pkg}")
            return
        
        # Get words for the selected language
        filtered_vocab = [w for w in vocabulary if w.get('language_translated') == language_code]
        
        if not filtered_vocab:
            st.warning(f"No vocabulary words found for this language. Add words in Camera Mode first.")
            return
        
        # Initialize session state for practice session
        if 'practice_words' not in st.session_state:
            # Select 5 random words for practice
            practice_size = min(5, len(filtered_vocab))
            st.session_state.practice_words = random.sample(filtered_vocab, practice_size)
            st.session_state.current_practice_index = 0
            st.session_state.practice_scores = []
        
        # Show progress
        current_index = st.session_state.current_practice_index
        total_words = len(st.session_state.practice_words)
        st.progress((current_index) / total_words)
        st.markdown(f"**Word {current_index + 1} of {total_words}**")
        
        # Get current word
        if current_index < total_words:
            current_word = st.session_state.practice_words[current_index]
            
            # Display word information
            st.markdown(f"## {current_word['word_translated']}")
            st.markdown(f"*Means: {current_word['word_original']}*")
            
            # Show practice UI for this word
            self.render_practice_ui(current_word)
            
            # Navigation buttons
            cols = st.columns([1, 1, 1])
            
            with cols[0]:
                if current_index > 0:
                    if st.button("⬅️ Previous Word"):
                        st.session_state.current_practice_index -= 1
                        if hasattr(st.session_state, 'last_recorded_audio'):
                            del st.session_state.last_recorded_audio
                        st.rerun()
            
            with cols[2]:
                if current_index < total_words - 1:
                    if st.button("Next Word ➡️"):
                        st.session_state.current_practice_index += 1
                        if hasattr(st.session_state, 'last_recorded_audio'):
                            del st.session_state.last_recorded_audio
                        st.rerun()
                else:
                    if st.button("Finish Practice 🎉"):
                        st.session_state.practice_complete = True
                        st.rerun()
        
        # Show practice results
        if hasattr(st.session_state, 'practice_complete') and st.session_state.practice_complete:
            st.success("Practice session complete!")
            
            # Show summary of performance
            if st.session_state.practice_scores:
                avg_score = sum(st.session_state.practice_scores) / len(st.session_state.practice_scores)
                st.markdown(f"### Your average pronunciation score: {avg_score:.1f}%")
                
                # Create a bar chart of scores
                score_data = pd.DataFrame({
                    'Word': [w['word_translated'] for w in st.session_state.practice_words[:len(st.session_state.practice_scores)]],
                    'Score': st.session_state.practice_scores
                })
                
                st.bar_chart(score_data.set_index('Word'))
            
            # Option to start a new practice session
            if st.button("Start New Practice Session"):
                # Reset practice session state
                if 'practice_words' in st.session_state:
                    del st.session_state.practice_words
                if 'current_practice_index' in st.session_state:
                    del st.session_state.current_practice_index
                if 'practice_scores' in st.session_state:
                    del st.session_state.practice_scores
                if 'practice_complete' in st.session_state:
                    del st.session_state.practice_complete
                if 'last_recorded_audio' in st.session_state:
                    del st.session_state.last_recorded_audio
                st.rerun()
    
    def _record_audio(self, duration=5) -> Tuple[bytes, Optional[str]]:
        """
        Record audio from the microphone.
        
        Args:
            duration: Recording duration in seconds
            
        Returns:
            Tuple of (audio_data, error_message)
        """
        if not has_speech_recognition:
            return None, "SpeechRecognition library not available"
        
        try:
            with sr.Microphone() as source:
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source)
                
                # Record audio
                audio = self.recognizer.record(source, duration=duration)
                
                # Convert to WAV format
                wav_data = audio.get_wav_data()
                
                return wav_data, None
        except Exception as e:
            return None, str(e)
    
    def _recognize_speech(self, audio_data, language_code='en-US') -> Tuple[str, float]:
        """
        Recognize speech from audio data.
        
        Args:
            audio_data: Audio data to recognize
            language_code: Language code for recognition
            
        Returns:
            Tuple of (recognized_text, confidence)
        """
        if not has_speech_recognition:
            return "", 0.0
        
        try:
            # Convert audio bytes to AudioData object
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(audio_data)
                tmp_path = tmp_file.name
            
            with sr.AudioFile(tmp_path) as source:
                audio = self.recognizer.record(source)
            
            # Use Google's speech recognition
            result = self.recognizer.recognize_google(
                audio, 
                language=language_code,
                show_all=True
            )
            
            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except:
                pass
            
            # Extract text and confidence
            if result and isinstance(result, dict) and 'alternative' in result:
                top_result = result['alternative'][0]
                text = top_result.get('transcript', '').lower()
                confidence = top_result.get('confidence', 0.0)
                return text, confidence
            elif result and isinstance(result, list) and len(result) > 0:
                return result[0], 0.5  # Default confidence if not provided
            else:
                return "", 0.0
        except Exception as e:
            print(f"Recognition error: {e}")
            return "", 0.0
    
    def _calculate_similarity(self, recognized_text: str, target_text: str) -> float:
        """
        Calculate similarity between recognized text and target text.
        
        Args:
            recognized_text: Text recognized from speech
            target_text: Expected text
            
        Returns:
            Similarity score between 0 and 100
        """
        # Normalize both texts
        recognized = recognized_text.lower().strip()
        target = target_text.lower().strip()
        
        if not recognized or not target:
            return 0.0
        
        # Direct match check
        if recognized == target:
            return 100.0
        
        # Check if target appears in recognized
        if target in recognized:
            return 90.0
        
        # Check if recognized is in target
        if recognized in target:
            return 80.0
        
        # Use Levenshtein distance for partial match scoring
        try:
            import Levenshtein
            distance = Levenshtein.distance(recognized, target)
            max_len = max(len(recognized), len(target))
            if max_len == 0:
                return 0.0
            similarity = (1 - (distance / max_len)) * 100
            return max(0, similarity)
        except ImportError:
            # Fallback to simple character matching if Levenshtein not available
            # Count matching characters in sequence
            matches = 0
            for r, t in zip(recognized, target):
                if r == t:
                    matches += 1
            
            similarity = (matches / len(target)) * 100
            return similarity
    
    def _show_pronunciation_tips(self, word: Dict) -> None:
        """
        Show pronunciation tips for a word.
        
        Args:
            word: Word dictionary with translation and language info
        """
        language_code = word.get('language_translated', 'en')
        word_text = word.get('word_translated', '')
        
        # Get language-specific difficult sounds
        language_sounds = self.difficult_sounds.get(language_code, {})
        
        # Check for difficult sounds in the word
        tips = []
        for sound, tip in language_sounds.items():
            if sound in word_text.lower():
                tips.append(f"- The sound '{sound}' is pronounced as {tip}")
        
        # Show general pronunciation pattern
        if language_code == 'es':
            tips.append("- Spanish is pronounced as it's written, with consistent vowel sounds")
        elif language_code == 'fr':
            tips.append("- French often has silent letters at the end of words")
        elif language_code == 'de':
            tips.append("- German has strong consonants and consistent vowel pronunciation")
        elif language_code == 'it':
            tips.append("- Italian has melodic pronunciation with clear vowels")
        
        # Display tips
        if tips:
            for tip in tips:
                st.markdown(tip)
        else:
            st.markdown("- Focus on clear pronunciation and matching the audio example")
        
        # Add tip for listening and repeating
        st.markdown("- Listen to the audio multiple times and try to mimic it exactly")
    
    def _show_pronunciation_results(
        self, 
        expected_text: str, 
        recognized_text: str, 
        similarity_score: float, 
        confidence: float
    ) -> None:
        """
        Show pronunciation analysis results.
        
        Args:
            expected_text: What should have been said
            recognized_text: What was actually recognized
            similarity_score: How similar they are (0-100)
            confidence: Speech recognition confidence
        """
        st.markdown("### Pronunciation Analysis")
        
        # Display recognized text
        st.markdown(f"**Recognized:** {recognized_text or 'No speech detected'}")
        
        # Display score with color coding
        if similarity_score >= 80:
            st.success(f"Pronunciation Score: {similarity_score:.1f}%")
            feedback = "Excellent pronunciation! Keep it up!"
        elif similarity_score >= 60:
            st.info(f"Pronunciation Score: {similarity_score:.1f}%")
            feedback = "Good pronunciation. Try to match the audio more closely."
        elif similarity_score >= 40:
            st.warning(f"Pronunciation Score: {similarity_score:.1f}%")
            feedback = "Your pronunciation needs some work. Listen carefully to the example and try again."
        else:
            st.error(f"Pronunciation Score: {similarity_score:.1f}%")
            feedback = "Try again! Focus on matching the sounds in the audio example."
        
        # Show feedback
        st.markdown(f"**Feedback:** {feedback}")
        
        # Store score in session state for session-level tracking
        if hasattr(st.session_state, 'practice_scores'):
            st.session_state.practice_scores.append(similarity_score)
        
        # Show specific differences if available
        if similarity_score < 100 and recognized_text and expected_text:
            st.markdown("### Pronunciation Tips")
            
            if recognized_text in expected_text:
                st.markdown("- Your pronunciation is close! Try to pronounce the full word clearly.")
            elif expected_text in recognized_text:
                st.markdown("- You added extra sounds. Try to pronounce just the word itself.")
            else:
                # Highlight specific sounds that might be wrong
                different_chars = set(expected_text).symmetric_difference(set(recognized_text))
                if different_chars:
                    sound_list = ", ".join(f"'{c}'" for c in different_chars)
                    st.markdown(f"- Focus on the sounds: {sound_list}")
                
                # Check for common issues
                if len(recognized_text) < len(expected_text):
                    st.markdown("- You may be missing some sounds. Pronounce each syllable clearly.")
                else:
                    st.markdown("- Try again, focusing on matching the rhythm of the example audio.")
        
        # If excellent, offer to move to next word
        if similarity_score >= 80 and hasattr(st.session_state, 'practice_words'):
            if st.session_state.current_practice_index < len(st.session_state.practice_words) - 1:
                if st.button("Continue to Next Word ➡️"):
                    st.session_state.current_practice_index += 1
                    if hasattr(st.session_state, 'last_recorded_audio'):
                        del st.session_state.last_recorded_audio
                    st.rerun()


# Import pandas for visualization in the practice session
import pandas as pd

# Function to create the practice module with dependencies injected
def create_pronunciation_practice(text_to_speech_func, get_audio_html_func, translate_text_func=None):
    """
    Create a pronunciation practice module with the required dependencies.
    
    Args:
        text_to_speech_func: Function that converts text to speech
        get_audio_html_func: Function that generates HTML for audio playback
        translate_text_func: Function that translates text
        
    Returns:
        PronunciationPractice instance
    """
    return PronunciationPractice(
        text_to_speech_func=text_to_speech_func,
        get_audio_html_func=get_audio_html_func,
        translate_text_func=translate_text_func
    )