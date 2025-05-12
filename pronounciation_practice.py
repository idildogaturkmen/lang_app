"""
Pronunciation Practice Module for Vocam Language Learning App
This module provides speech recognition and pronunciation feedback functionality.
"""

import time
import os
import re
from io import BytesIO
import numpy as np
import streamlit as st
import tempfile

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
        # Try to import required libraries
        import speech_recognition as sr
        import Levenshtein
        from pydub import AudioSegment
        
        # Create and return the practice module
        return PronunciationPractice(
            text_to_speech_func,
            get_audio_html_func,
            translate_text_func,
            sr,
            Levenshtein,
            AudioSegment
        )
    except ImportError as e:
        # Create a dummy practice module that shows installation instructions
        return DummyPronunciationPractice(
            text_to_speech_func,
            get_audio_html_func,
            translate_text_func
        )

class DummyPronunciationPractice:
    """
    Dummy class that provides instructions for installing required packages
    when actual pronunciation practice isn't available.
    """
    
    def __init__(self, text_to_speech_func, get_audio_html_func, translate_text_func):
        self.text_to_speech = text_to_speech_func
        self.get_audio_html = get_audio_html_func
        self.translate_text = translate_text_func
        # Initialize difficult_sounds attribute
        self.difficult_sounds = DIFFICULT_SOUNDS
    
    def render_practice_ui(self, word):
        """Render a message about installing required packages."""
        with st.expander("🎤 Practice Pronunciation"):
            st.warning("Pronunciation practice requires additional packages.")
            st.info("To enable pronunciation practice, install the following packages:")
            st.code("sudo apt-get install portaudio19-dev python3-pyaudio")
            st.code("pip install SpeechRecognition pydub PyAudio python-Levenshtein")
            
            st.markdown("After installing, restart the application to use pronunciation practice.")
    
    def render_practice_session(self, vocabulary, language_code):
        """Render a message about installing required packages for the practice session."""
        st.warning("Pronunciation practice requires additional packages.")
        st.info("To enable pronunciation practice, install the following packages:")
        
        st.markdown("### System Dependencies:")
        st.code("sudo apt-get install portaudio19-dev python3-pyaudio")
        
        st.markdown("### Python Packages:")
        st.code("pip install SpeechRecognition pydub PyAudio python-Levenshtein")
        
        st.markdown("After installing, restart the application to use pronunciation practice.")
        
        # Show sample of what it will look like
        st.markdown("### Sample Pronunciation Feature")
        st.image("https://i.ibb.co/GTxfJsQ/pronunciation-practice-sample.png", 
                caption="Sample pronunciation practice interface")

class PronunciationPractice:
    """
    Main class for pronunciation practice functionality.
    Provides speech recognition and feedback for language learning.
    """
    
    def __init__(self, text_to_speech_func, get_audio_html_func, translate_text_func, 
                 speech_recognition, levenshtein, audio_segment):
        # Store dependencies
        self.text_to_speech = text_to_speech_func
        self.get_audio_html = get_audio_html_func
        self.translate_text = translate_text_func
        self.sr = speech_recognition
        self.Levenshtein = levenshtein
        self.AudioSegment = audio_segment
        
        # Create a speech recognizer
        self.recognizer = self.sr.Recognizer()
        
        # Initialize difficult_sounds attribute 
        self.difficult_sounds = DIFFICULT_SOUNDS
        
        # Set language-specific properties
        self._init_language_settings()
    
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
            
            # Record pronunciation button
            if st.button("Record Pronunciation", key=f"record_{word['id']}"):
                # Record and analyze
                self._record_and_analyze(translated_word, language_code)
    
    def render_practice_session(self, vocabulary, language_code):
        """
        Render a full pronunciation practice session with multiple words.
        
        Args:
            vocabulary: List of vocabulary words
            language_code: Language code for the practice session
        """
        # If there's no practice session data, show a message
        if 'practice_words' not in st.session_state:
            st.info("Click 'Start Practice Session' to begin pronunciation practice.")
            return
        
        # Display current practice word
        current_index = st.session_state.current_practice_index
        if current_index < len(st.session_state.practice_words):
            current_word = st.session_state.practice_words[current_index]
            
            # Progress bar
            progress = current_index / len(st.session_state.practice_words)
            st.progress(progress)
            st.subheader(f"Word {current_index + 1} of {len(st.session_state.practice_words)}")
            
            # Display the practice UI for this word
            self.render_practice_ui(current_word)
            
            # Show stats if available
            if 'practice_scores' in st.session_state and len(st.session_state.practice_scores) > 0:
                scores = st.session_state.practice_scores
                avg_score = sum(scores) / len(scores)
                st.info(f"Current accuracy: {avg_score:.0f}%")
            
            # Next word button
            if st.button("Next Word", type="primary"):
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
    
    def _record_and_analyze(self, target_word, language_code):
        """
        Record user's pronunciation and analyze it.
        
        Args:
            target_word: The word the user should pronounce
            language_code: Language code for speech recognition
        """
        # Create a placeholder for the recording UI
        recording_placeholder = st.empty()
        
        try:
            with recording_placeholder.container():
                st.info("🎤 Recording... Speak the word clearly")
                
                # Record audio
                audio_data = self._record_audio(5)  # 5 seconds recording
                
                if audio_data:
                    # Display success message
                    st.success("Recording complete!")
                    
                    # Play back recording
                    st.audio(audio_data, format="audio/wav")
                    
                    # Convert audio data to the format needed for recognition
                    # Use a temporary file for the audio data
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as f:
                        f.write(audio_data)
                        temp_filename = f.name
                    
                    # Process the audio for speech recognition
                    with self.sr.AudioFile(temp_filename) as source:
                        audio = self.recognizer.record(source)
                    
                    # Perform speech recognition
                    recognition_lang = self.recognition_languages.get(language_code, "en-US")
                    
                    try:
                        recognized_text = self.recognizer.recognize_google(
                            audio, 
                            language=recognition_lang
                        )
                        
                        # Clean up temp file
                        try:
                            os.unlink(temp_filename)
                        except:
                            pass
                        
                        # Show recognition result
                        recognized_text = recognized_text.lower().strip()
                        target_word = target_word.lower().strip()
                        
                        # Calculate similarity score
                        similarity = self._calculate_similarity(recognized_text, target_word)
                        
                        # Store score
                        if 'practice_scores' not in st.session_state:
                            st.session_state.practice_scores = []
                        st.session_state.practice_scores.append(similarity)
                        
                        # Clear the recording UI
                        recording_placeholder.empty()
                        
                        # Display result with feedback
                        self._show_pronunciation_feedback(
                            target_word, recognized_text, similarity, language_code
                        )
                    
                    except Exception as e:
                        # Clear the recording UI
                        recording_placeholder.empty()
                        st.error(f"Could not recognize speech: {str(e)}")
                        
                        # If we got an error, it might be because the word wasn't spoken
                        # or the recognition failed. Either way, provide guidance.
                        st.info("Try again! Make sure to speak clearly and closely to the microphone.")
                else:
                    # Clear the recording UI
                    recording_placeholder.empty()
                    st.error("No audio data captured. Please check your microphone settings.")
        
        except Exception as e:
            # Clear the recording UI
            recording_placeholder.empty()
            st.error(f"Error recording audio: {str(e)}")
            
            # Provide fallback instructions
            st.info("""
            If you're having trouble with the microphone:
            1. Make sure your browser has permission to access the microphone
            2. Try using a headset or external microphone
            3. Speak clearly and closely to the microphone
            """)
    
    def _record_audio(self, duration=5):
        """
        Record audio from the microphone.
        
        Args:
            duration: Recording duration in seconds
            
        Returns:
            Audio data as bytes
        """
        # Create a recognizer and microphone instance
        r = self.sr.Recognizer()
        mic = self.sr.Microphone()
        
        # Record audio
        with mic as source:
            # Adjust for ambient noise
            r.adjust_for_ambient_noise(source)
            audio = r.listen(source, timeout=duration, phrase_time_limit=duration)
        
        # Convert audio to WAV format
        wav_data = audio.get_wav_data()
        return wav_data
    
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
            if similarity >= 90:
                st.success(f"Excellent pronunciation! {similarity:.0f}% accuracy")
            elif similarity >= 70:
                st.info(f"Good pronunciation! {similarity:.0f}% accuracy")
            elif similarity >= 50:
                st.warning(f"Fair pronunciation. {similarity:.0f}% accuracy")
            else:
                st.error(f"Needs improvement. {similarity:.0f}% accuracy")
            
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
            if sound in target.lower() and (sound not in recognized.lower() or similarity < 70):
                problematic.append(sound)
        
        # If we didn't find specific problems, check for other common issues
        if not problematic and similarity < 50:
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
    
    def _show_practice_results(self):
        """Show results of the practice session."""
        st.subheader("🎉 Practice Session Completed!")
        
        # Get scores
        scores = st.session_state.practice_scores
        
        if not scores:
            st.info("No pronunciation attempts were recorded.")
            return
        
        # Calculate stats
        avg_score = sum(scores) / len(scores)
        min_score = min(scores)
        max_score = max(scores)
        
        # Display stats
        st.markdown(f"**Average accuracy: {avg_score:.0f}%**")
        st.markdown(f"Best pronunciation: {max_score:.0f}%")
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