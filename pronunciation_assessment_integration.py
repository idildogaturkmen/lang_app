"""
Integration script for pronunciation assessment in main.py

This module enhances the pronunciation practice component by
adding AI-based assessment functionality to evaluate user recordings
"""

import streamlit as st
import sys
import os
import importlib.util
from pathlib import Path

def initialize_pronunciation_assessment():
    """
    Initialize the pronunciation assessment module
    and integrate it with the existing pronunciation practice
    
    This should be called in main.py after initializing the pronunciation_practice module
    """
    # Try to import the pronunciation assessment module
    try:
        # First check if the file exists in the current directory
        if os.path.exists("pronunciation_assessment.py"):
            # Import the module dynamically
            spec = importlib.util.spec_from_file_location(
                "pronunciation_assessment", 
                "pronunciation_assessment.py"
            )
            pronunciation_assessment = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(pronunciation_assessment)
            
            # Setup the assessment capabilities
            result = pronunciation_assessment.setup_pronunciation_assessment()
            
            if result:
                print("Successfully enhanced pronunciation practice with AI assessment")
                return True
            else:
                print("Failed to enhance pronunciation practice - module not initialized yet")
                return False
        else:
            print("No pronunciation_assessment.py file found - creating it")
            _create_assessment_module()
            return False
    except Exception as e:
        print(f"Error initializing pronunciation assessment: {e}")
        return False

def _create_assessment_module():
    """
    Create the pronunciation_assessment.py file if it doesn't exist
    """
    # Define the module content
    module_content = """
# pronunciation_assessment.py

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import io
import os
import wave

class PronunciationAssessor:
    \"\"\"Advanced pronunciation assessment using speech recognition and phonetic analysis\"\"\"
    
    def __init__(self):
        \"\"\"Initialize the pronunciation assessor with language-specific settings\"\"\"
        # Initialize phoneme maps for different languages
        self.phoneme_maps = {
            "es": self._init_spanish_phonemes(),
            "fr": self._init_french_phonemes(),
            "de": self._init_german_phonemes(),
            "it": self._init_italian_phonemes(),
        }
        
        # Common error patterns by language
        self.error_patterns = {
            "es": {
                'j': {'english_error': 'j as in jump', 'correct': 'h as in hat'},
                'll': {'english_error': 'l as in light', 'correct': 'y as in yes'},
                'ñ': {'english_error': 'n as in no', 'correct': 'ny as in canyon'},
                'rr': {'english_error': 'english r', 'correct': 'rolled r'},
            },
            "fr": {
                'r': {'english_error': 'english r', 'correct': 'guttural r'},
                'u': {'english_error': 'oo as in moon', 'correct': 'ü with rounded lips'},
                'eu': {'english_error': 'u as in up', 'correct': 'ö as in "bird"'},
                'ou': {'english_error': 'ow as in how', 'correct': 'oo as in moon'},
            },
            "de": {
                'ch': {'english_error': 'ch as in chair', 'correct': 'soft h after e/i, harsh h after a/o/u'},
                'ü': {'english_error': 'u as in up', 'correct': 'ee with rounded lips'},
                'ö': {'english_error': 'o as in hot', 'correct': 'e with rounded lips'},
                'ei': {'english_error': 'ey as in they', 'correct': 'eye as in my'},
            },
            "it": {
                'gli': {'english_error': 'gl as in glitter', 'correct': 'lli as in million'},
                'gn': {'english_error': 'gn as in gnat', 'correct': 'ny as in canyon'},
                'c+e/i': {'english_error': 'k as in cat', 'correct': 'ch as in chat'},
                'zz': {'english_error': 'z as in zone', 'correct': 'ts as in bits'},
            }
        }
        
    def _init_spanish_phonemes(self):
        \"\"\"Initialize Spanish phoneme mapping\"\"\"
        return {
            'a': 'ah', 'e': 'eh', 'i': 'ee', 'o': 'oh', 'u': 'oo',
            'j': 'h', 'll': 'y', 'ñ': 'ny', 'rr': 'rr', 'r': 'r',
            'c+e/i': 's', 'c+a/o/u': 'k', 'z': 's', 'g+e/i': 'h'
        }
        
    def _init_french_phonemes(self):
        \"\"\"Initialize French phoneme mapping\"\"\"
        return {
            'a': 'ah', 'e': 'uh', 'i': 'ee', 'o': 'oh', 'u': 'ü',
            'ai': 'eh', 'oi': 'wa', 'au': 'oh', 'eau': 'oh', 'eu': 'uh',
            'ou': 'oo', 'in': 'an~', 'ain': 'an~', 'ein': 'an~',
            'on': 'on~', 'un': 'un~', 'r': 'R', 'gn': 'ny'
        }
        
    def _init_german_phonemes(self):
        \"\"\"Initialize German phoneme mapping\"\"\"
        return {
            'a': 'ah', 'e': 'eh', 'i': 'ee', 'o': 'oh', 'u': 'oo',
            'ä': 'eh', 'ö': 'eu', 'ü': 'ü', 'ei': 'eye', 'ie': 'ee',
            'eu': 'oy', 'äu': 'oy', 'ch+e/i': 'sh', 'ch+a/o/u': 'kh',
            'sch': 'sh', 'st': 'sht', 'sp': 'shp', 'v': 'f', 'w': 'v',
            'z': 'ts', 'j': 'y'
        }
        
    def _init_italian_phonemes(self):
        \"\"\"Initialize Italian phoneme mapping\"\"\"
        return {
            'a': 'ah', 'e': 'eh', 'i': 'ee', 'o': 'oh', 'u': 'oo',
            'c+e/i': 'ch', 'c+a/o/u': 'k', 'ch': 'k', 'gh': 'g',
            'gli': 'ly', 'gn': 'ny', 'sc+e/i': 'sh', 'z': 'ts/dz'
        }
        
    def analyze_pronunciation(self, audio_data, target_word, recognized_text, language_code):
        \"\"\"
        Perform comprehensive pronunciation analysis
        
        Args:
            audio_data: Binary audio data
            target_word: The word the user should be pronouncing
            recognized_text: Text recognized from speech recognition
            language_code: Language code (e.g., 'es', 'fr')
            
        Returns:
            Dictionary with analysis results
        \"\"\"
        results = {}
        
        # Basic text comparison
        results.update(self._analyze_text_similarity(target_word, recognized_text))
        
        # Phonetic analysis
        results.update(self._analyze_phonetic_patterns(target_word, recognized_text, language_code))
        
        # Generate specific error feedback
        results['errors'] = self._identify_specific_errors(target_word, recognized_text, language_code)
        
        # Generate overall feedback and score
        results['overall_score'] = self._calculate_overall_score(results)
        results['feedback'] = self._generate_feedback(results, language_code)
        
        return results
    
    def _analyze_text_similarity(self, target_word, recognized_text):
        \"\"\"
        Analyze the text similarity between target and recognized text
        
        Args:
            target_word: Target word
            recognized_text: Recognized text from speech recognition
            
        Returns:
            Dictionary with text similarity metrics
        \"\"\"
        try:
            import Levenshtein
            has_levenshtein = True
        except ImportError:
            has_levenshtein = False
        
        # Simple string normalization
        target_norm = target_word.lower().strip()
        
        # If recognized text is empty, return zeros
        if not recognized_text:
            return {
                'exact_match': False,
                'text_similarity': 0,
                'recognized_text': '',
                'target_word': target_norm
            }
        
        # Normalize recognized text - take the first word if multiple words
        recognized_norm = recognized_text.lower().strip().split()[0]
        
        # Check for exact match
        exact_match = (target_norm == recognized_norm)
        
        # Calculate similarity
        if has_levenshtein:
            # Use Levenshtein distance
            distance = Levenshtein.distance(target_norm, recognized_norm)
            max_len = max(len(target_norm), len(recognized_norm))
            
            # Convert to similarity percentage
            similarity = 100 - (distance / max_len * 100) if max_len > 0 else 0
        else:
            # Fallback simple character overlap method
            common_chars = set(target_norm).intersection(set(recognized_norm))
            total_chars = set(target_norm).union(set(recognized_norm))
            similarity = len(common_chars) / len(total_chars) * 100 if total_chars else 0
        
        # Return results
        return {
            'exact_match': exact_match,
            'text_similarity': similarity,
            'recognized_text': recognized_norm,
            'target_word': target_norm
        }
    
    def _analyze_phonetic_patterns(self, target_word, recognized_text, language_code):
        \"\"\"
        Analyze phonetic patterns in the target and recognized words
        
        Args:
            target_word: Target word
            recognized_text: Recognized text from speech recognition
            language_code: Language code
            
        Returns:
            Dictionary with phonetic analysis results
        \"\"\"
        # If recognized text is empty, return defaults
        if not recognized_text:
            return {
                'phonetic_similarity': 0,
                'phoneme_matches': [],
                'phoneme_errors': []
            }
        
        # Get the phoneme map for this language
        phoneme_map = self.phoneme_maps.get(language_code, {})
        
        # Simplified phonetic analysis (for detailed analysis, use a phonetics library)
        # For each character, check if it exists in both words at similar positions
        target = target_word.lower()
        recognized = recognized_text.lower()
        
        matches = []
        errors = []
        
        # Compare character by character with position tolerance
        max_pos_diff = 2  # Allow characters to be off by 2 positions
        
        for i, char in enumerate(target):
            found = False
            # Check if this character exists in the recognized text within tolerance
            for j in range(max(0, i - max_pos_diff), min(len(recognized), i + max_pos_diff + 1)):
                if j < len(recognized) and recognized[j] == char:
                    matches.append({
                        'char': char,
                        'target_pos': i,
                        'recognized_pos': j,
                        'position_diff': abs(i - j)
                    })
                    found = True
                    break
            
            if not found:
                # Check for known phonetic substitutions
                error = {
                    'char': char,
                    'target_pos': i,
                    'error_type': 'missing'
                }
                
                # Check if this is a common mistake in this language
                for pattern, info in self.error_patterns.get(language_code, {}).items():
                    if pattern in target[max(0, i-len(pattern)+1):i+1]:
                        error['error_type'] = 'phonetic'
                        error['expected'] = info['correct']
                        error['common_error'] = info['english_error']
                        break
                
                errors.append(error)
        
        # Check for extra characters in recognized text
        for i, char in enumerate(recognized):
            if not any(m['recognized_pos'] == i for m in matches):
                errors.append({
                    'char': char,
                    'recognized_pos': i,
                    'error_type': 'extra'
                })
        
        # Calculate phonetic similarity score
        total_chars = len(target)
        matched_chars = len(matches)
        phonetic_similarity = (matched_chars / total_chars * 100) if total_chars > 0 else 0
        
        # Return results
        return {
            'phonetic_similarity': phonetic_similarity,
            'phoneme_matches': matches,
            'phoneme_errors': errors
        }
    
    def _identify_specific_errors(self, target_word, recognized_text, language_code):
        \"\"\"Identify specific pronunciation errors\"\"\"
        errors = []
        
        # If no recognized text, can't identify specific errors
        if not recognized_text:
            return [{'type': 'no_speech', 'message': 'No speech was detected or recognized'}]
        
        # Get error patterns for this language
        language_errors = self.error_patterns.get(language_code, {})
        
        # Check for language-specific errors
        target = target_word.lower()
        recognized = recognized_text.lower()
        
        # Check for each error pattern
        for pattern, info in language_errors.items():
            if pattern in target and pattern not in recognized:
                errors.append({
                    'type': 'phonetic',
                    'pattern': pattern,
                    'expected': info['correct'],
                    'likely_pronounced': info['english_error'],
                    'message': f"The '{pattern}' sound should be pronounced as {info['correct']}, not as {info['english_error']}"
                })
        
        # Check for missing or added sounds
        target_chars = set(target)
        recognized_chars = set(recognized)
        
        missing_chars = target_chars - recognized_chars
        for char in missing_chars:
            if char.isalpha():  # Only consider alphabetic characters
                errors.append({
                    'type': 'missing',
                    'char': char,
                    'message': f"The '{char}' sound was missed in your pronunciation"
                })
        
        extra_chars = recognized_chars - target_chars
        for char in extra_chars:
            if char.isalpha():  # Only consider alphabetic characters
                errors.append({
                    'type': 'extra',
                    'char': char,
                    'message': f"You added an extra '{char}' sound that isn't in the original word"
                })
        
        # If no specific errors found but not a perfect match
        if not errors and target != recognized:
            errors.append({
                'type': 'general',
                'message': f"Your pronunciation differed from the expected. Try listening to the correct pronunciation again."
            })
        
        # If perfect match
        if target == recognized and not errors:
            errors.append({
                'type': 'perfect',
                'message': "Perfect pronunciation! The word was recognized exactly."
            })
        
        return errors
    
    def _calculate_overall_score(self, results):
        \"\"\"Calculate overall pronunciation score from component scores\"\"\"
        # Component weights
        weights = {
            'text_similarity': 0.7,      # Text recognition has highest weight
            'phonetic_similarity': 0.3,  # Phonetic similarity is also important
        }
        
        # Calculate weighted score
        total_score = 0
        total_weight = 0
        
        for metric, weight in weights.items():
            if metric in results:
                total_score += results[metric] * weight
                total_weight += weight
        
        # Normalize score - default to 60 if no metrics
        final_score = total_score / total_weight if total_weight > 0 else 60
        
        # Round to nearest integer
        return round(final_score)
    
    def _generate_feedback(self, results, language_code):
        \"\"\"Generate comprehensive feedback based on analysis results\"\"\"
        overall_score = results.get('overall_score', 0)
        errors = results.get('errors', [])
        
        # Generate feedback messages based on score ranges
        if overall_score >= 90:
            general_feedback = "Excellent pronunciation! You sound like a native speaker."
        elif overall_score >= 80:
            general_feedback = "Very good pronunciation! Just a few minor adjustments needed."
        elif overall_score >= 70:
            general_feedback = "Good pronunciation! Keep practicing to sound more natural."
        elif overall_score >= 50:
            general_feedback = "Fair pronunciation. Focus on the specific sounds highlighted below."
        else:
            general_feedback = "Needs improvement. Let's break down the word and practice one sound at a time."
        
        # Add specific error feedback
        specific_feedback = []
        
        for error in errors:
            if error['type'] != 'perfect' and error['type'] != 'general':
                specific_feedback.append(error['message'])
        
        # Add language-specific tips
        language_tips = {
            "es": "Spanish requires clear vowels and softer consonants than English. The 'r' sound is particularly important.",
            "fr": "French pronunciation focuses on nasal sounds and the unique 'r' sound from the back of the throat.",
            "de": "German requires precise consonants and attention to umlauts (ä, ö, ü) which change the vowel sound.",
            "it": "Italian has clear, distinct vowels and double consonants that should be pronounced longer."
        }
        
        language_tip = language_tips.get(language_code, "Pay attention to the unique sounds of this language.")
        
        # Combine feedback
        combined_feedback = [general_feedback]
        
        if specific_feedback:
            combined_feedback.extend(specific_feedback)
        
        combined_feedback.append(language_tip)
        
        # Add improvement suggestion
        if overall_score < 70:
            combined_feedback.append("Try speaking more slowly and emphasizing each syllable separately.")
        
        return combined_feedback
    
    def visualize_comparison(self, target_word, recognized_text):
        \"\"\"
        Create a visual comparison between target and recognized words
        
        Args:
            target_word: Target word
            recognized_text: Recognized text
            
        Returns:
            Matplotlib figure object
        \"\"\"
        if not recognized_text:
            # Create a simple figure showing target only
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.text(0.5, 0.5, "No speech detected", ha='center', va='center', fontsize=16)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.tight_layout()
            return fig
        
        # Normalize text
        target = target_word.lower()
        recognized = recognized_text.lower()
        
        # Create a visual comparison
        fig, ax = plt.subplots(figsize=(10, 3))
        
        # Draw target word
        target_x = np.linspace(0.1, 0.9, len(target))
        ax.text(0.05, 0.8, "Target:", fontsize=12, ha='right')
        
        for i, char in enumerate(target):
            # Check if this character is in recognized text
            if char in recognized:
                color = 'green'
            else:
                color = 'red'
            ax.text(target_x[i], 0.8, char, fontsize=16, ha='center', color=color)
        
        # Draw recognized word
        recognized_x = np.linspace(0.1, 0.9, len(recognized))
        ax.text(0.05, 0.4, "You said:", fontsize=12, ha='right')
        
        for i, char in enumerate(recognized):
            # Check if this character is in target text
            if char in target:
                color = 'green'
            else:
                color = 'blue'
            ax.text(recognized_x[i], 0.4, char, fontsize=16, ha='center', color=color)
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Correct'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Missing'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Extra')
        ]
        ax.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, 0.15))
        
        # Remove axes
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        return fig

# Integration with SimplePronunciationPractice class
def integrate_with_pronunciation_practice(practice_instance):
    \"\"\"
    Integrate the PronunciationAssessor with SimplePronunciationPractice
    
    Args:
        practice_instance: Instance of SimplePronunciationPractice class
        
    Returns:
        Updated instance with enhanced assessment capabilities
    \"\"\"
    # Create an assessor instance
    assessor = PronunciationAssessor()
    
    # Add the assessor to the practice instance
    practice_instance.assessor = assessor
    
    # Override _evaluate_pronunciation method
    def enhanced_evaluate_pronunciation(self, audio_data, target_word, language_code):
        \"\"\"Enhanced pronunciation evaluation with detailed analysis and feedback\"\"\"
        # Show evaluation status
        status = st.empty()
        status.info("Analyzing your pronunciation in detail... Please wait.")
        
        # Use speech recognition to get recognized text
        recognized_text = self._recognize_speech(audio_data, language_code)
        
        # Perform comprehensive assessment
        results = self.assessor.analyze_pronunciation(
            audio_data=audio_data,
            target_word=target_word,
            recognized_text=recognized_text,
            language_code=language_code
        )
        
        # Store results in session state for later use
        st.session_state.last_pronunciation_results = results
        
        # Clear status message
        status.empty()
        
        # Return the overall score
        return results.get('overall_score', 60)
    
    # Override _show_simple_feedback method
    def enhanced_feedback_display(self, target_word, language_code, similarity_score):
        \"\"\"Show comprehensive pronunciation feedback with visual indicators\"\"\"
        # Get the results from session state
        results = getattr(st.session_state, 'last_pronunciation_results', {})
        recognized_text = results.get('recognized_text', '')
        
        st.markdown("### Pronunciation Feedback")
        
        # Create columns for scores
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Overall Score**")
            overall_score = results.get('overall_score', similarity_score)
            score_color = self._get_score_color(overall_score)
            st.markdown(f\"\"\"
            <div style="text-align: center; font-size: 24px; font-weight: bold; 
                        color: {score_color};">
                {overall_score}%
            </div>
            \"\"\", unsafe_allow_html=True)
            
        with col2:
            st.markdown("**Word Match**")
            text_score = results.get('text_similarity', 0)
            text_color = self._get_score_color(text_score)
            st.markdown(f\"\"\"
            <div style="text-align: center; font-size: 24px; font-weight: bold; 
                        color: {text_color};">
                {text_score:.0f}%
            </div>
            \"\"\", unsafe_allow_html=True)
        
        # Display progress bar for overall score
        st.progress(overall_score / 100.0)
        
        # Display recognized text if available
        if recognized_text:
            st.markdown("### What We Heard")
            
            # Show comparison
            comparison_col1, comparison_col2 = st.columns(2)
            with comparison_col1:
                st.markdown(f"**You said:**")
                st.markdown(f"<div style='padding: 10px; background-color: #f0f2f6; border-radius: 5px;'>{recognized_text}</div>", unsafe_allow_html=True)
            
            with comparison_col2:
                st.markdown(f"**Target word:**")
                st.markdown(f"<div style='padding: 10px; background-color: #e1f5fe; border-radius: 5px;'>{target_word}</div>", unsafe_allow_html=True)
        
        # Add visual comparison of sounds
        st.markdown("### Sound Comparison")
        
        # Create visual comparison
        if hasattr(self, 'assessor'):
            fig = self.assessor.visualize_comparison(target_word, recognized_text)
            st.pyplot(fig)
        
        # Specific feedback based on results
        st.markdown("### Feedback & Tips")
        
        if overall_score >= 90:
            st.success("✅ Excellent pronunciation! You sound very natural.")
        elif overall_score >= 75:
            st.info("👍 Good pronunciation! Just a few small adjustments needed.")
        elif overall_score >= 60:
            st.warning("🔄 Fair pronunciation. Keep practicing the specific sounds below.")
        else:
            st.error("⚠️ Needs improvement. Focus on the core sounds highlighted below.")
        
        # Show specific error feedback
        feedback_messages = results.get('feedback', [])
        
        for i, message in enumerate(feedback_messages):
            if i == 0:  # Skip general message which we've already covered with emojis
                continue
            st.markdown(f"- {message}")
        
        # Add pronunciation practice suggestions
        st.markdown("### How to Improve")
        st.markdown(\"\"\"
        1. **Listen and repeat** - Play the correct pronunciation and repeat multiple times
        2. **Break it down** - Practice difficult sounds individually before combining them
        3. **Record yourself** - Recording and comparing helps identify specific differences
        4. **Slow down** - Speaking too quickly often leads to missed or incorrect sounds
        5. **Watch mouth movements** - Observe how native speakers form sounds with their mouth
        \"\"\")
    
    # Helper method for color coding
    def _get_score_color(score):
        \"\"\"Get color based on score\"\"\"
        if score >= 90:
            return "#4CAF50"  # Green
        elif score >= 75:
            return "#8BC34A"  # Light Green
        elif score >= 60:
            return "#FFC107"  # Amber
        elif score >= 40:
            return "#FF9800"  # Orange
        else:
            return "#F44336"  # Red
    
    # Attach the new methods to the practice instance
    import types
    practice_instance._evaluate_pronunciation = types.MethodType(enhanced_evaluate_pronunciation, practice_instance)
    practice_instance._show_simple_feedback = types.MethodType(enhanced_feedback_display, practice_instance)
    practice_instance._get_score_color = staticmethod(_get_score_color)
    
    return practice_instance

def setup_pronunciation_assessment():
    \"\"\"
    Setup function to be called from main.py
    
    This will enhance the pronunciation practice module with AI assessment
    \"\"\"
    # Check if required packages are installed
    missing_packages = []
    
    try:
        import matplotlib
    except ImportError:
        missing_packages.append("matplotlib")
    
    try:
        import numpy
    except ImportError:
        missing_packages.append("numpy")
    
    try:
        import Levenshtein
    except ImportError:
        missing_packages.append("python-Levenshtein")
    
    # If packages are missing, inform the user
    if missing_packages:
        st.warning(f"For full AI pronunciation assessment, please install: {', '.join(missing_packages)}")
        st.markdown(f"```pip install {' '.join(missing_packages)}```")
    
    # Check if we're in the pronunciation practice module
    if 'pronunciation_practice' in st.session_state:
        # Enhance the existing instance
        st.session_state.pronunciation_practice = integrate_with_pronunciation_practice(
            st.session_state.pronunciation_practice
        )
        return True
    
    return False
"""
    
    # Try to write the file
    try:
        with open("pronunciation_assessment.py", "w") as f:
            f.write(module_content)
        print("Created pronunciation_assessment.py file")
        return True
    except Exception as e:
        print(f"Error creating file: {e}")
        return False

def add_to_main_py():
    """
    Add initialization code to main.py to enable pronunciation assessment
    
    This should be called after the file is created to update main.py
    """
    try:
        # Read the current main.py content
        with open("main.py", "r") as f:
            main_content = f.read()
        
        # Check if we need to add the import
        if "from pronunciation_assessment_integration import initialize_pronunciation_assessment" not in main_content:
            # Find a good place to add the import - after other imports
            import_section_end = main_content.find("# First, display Python version")
            if import_section_end > 0:
                # Add our import before the first comment
                new_content = main_content[:import_section_end] + \
                              "from pronunciation_assessment_integration import initialize_pronunciation_assessment\n\n" + \
                              main_content[import_section_end:]
                
                # Write back the updated content
                with open("main.py", "w") as f:
                    f.write(new_content)
                print("Added import to main.py")
                
                # Now add the initialization call in the pronunciation practice section
                # This is harder to automate - user might need to add this manually
                return True
            else:
                print("Couldn't find a good place to add the import")
                return False
        else:
            print("Import already exists in main.py")
            return True
    except Exception as e:
        print(f"Error updating main.py: {e}")
        return False