"""
Simplified WebRTC implementation for pronunciation practice
This module focuses on making real-time recording work reliably
"""

import streamlit as st
import time
import tempfile
import io
import os
import wave
import re
from datetime import datetime

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

class SimplePronunciationPractice:
    """
    Simplified implementation of pronunciation practice using self-assessment
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
            
            # Simple self-assessment version
            st.markdown("### Record Your Pronunciation")
            st.markdown("Record yourself saying the word using your device's voice recorder app, then rate your pronunciation:")
            
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
            
            # Self-assessment without audio recording
            st.markdown("### Practice Your Pronunciation")
            st.markdown("""
            1. Listen to the correct pronunciation above
            2. Practice saying the word aloud several times
            3. Rate how well you pronounced the word
            """)
            
            # Simple self-assessment
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
            score = score_map.get(rating, 60)
            
            # Show feedback
            self._show_simple_feedback(
                current_word.get('word_translated', ''), 
                language_code, 
                score
            )
            
            # Submit button
            if st.button("Submit and Continue", type="primary"):
                # Store score
                if 'practice_scores' not in st.session_state:
                    st.session_state.practice_scores = []
                st.session_state.practice_scores.append(score)
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
        """Show simplified pronunciation feedback"""
        st.markdown("### Pronunciation Feedback")
        
        # Display score
        st.markdown(f"**Pronunciation accuracy: {similarity_score}%**")
        st.progress(similarity_score / 100.0)
        
        # Feedback based on score
        if similarity_score >= 90:
            st.success("Excellent pronunciation!")
        elif similarity_score >= 70:
            st.info("Good pronunciation!")
        elif similarity_score >= 50:
            st.warning("Fair pronunciation. Keep practicing!")
        else:
            st.error("Needs improvement. Listen to the example again.")
        
        # Pronunciation tips
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