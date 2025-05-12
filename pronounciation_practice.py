"""
Simplified Pronunciation Practice Module for Vocam Language Learning App
This version works on Streamlit Cloud by focusing on pronunciation tips and examples
without requiring microphone access.
"""

import streamlit as st

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
        # Create the simplified practice module
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
            def render_practice_ui(self, word):
                with st.expander("🎤 Pronunciation Practice"):
                    st.warning("Pronunciation practice is currently unavailable.")
                    st.info("This feature is not supported in the current environment.")
            
            def render_practice_session(self, vocabulary, language_code):
                st.warning("Pronunciation practice is currently unavailable.")
                st.info("This feature is not supported in the current environment.")
        
        return FallbackPractice()

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
                    # Use standard library random instead of numpy
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