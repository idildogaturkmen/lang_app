"""
Enhanced Pronunciation Assessment Module for Vocam

This module adds AI-based pronunciation assessment to the 
SimplePronunciationPractice class in pronunciation_practice.py
"""

import streamlit as st
import numpy as np
import io
import wave
import tempfile
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import librosa
import os

class PronunciationAssessor:
    """Advanced pronunciation assessment using speech recognition and phonetic analysis"""
    
    def __init__(self):
        """Initialize the pronunciation assessor with language-specific settings"""
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
        """Initialize Spanish phoneme mapping"""
        return {
            'a': 'ah', 'e': 'eh', 'i': 'ee', 'o': 'oh', 'u': 'oo',
            'j': 'h', 'll': 'y', 'ñ': 'ny', 'rr': 'rr', 'r': 'r',
            'c+e/i': 's', 'c+a/o/u': 'k', 'z': 's', 'g+e/i': 'h'
        }
        
    def _init_french_phonemes(self):
        """Initialize French phoneme mapping"""
        return {
            'a': 'ah', 'e': 'uh', 'i': 'ee', 'o': 'oh', 'u': 'ü',
            'ai': 'eh', 'oi': 'wa', 'au': 'oh', 'eau': 'oh', 'eu': 'uh',
            'ou': 'oo', 'in': 'an~', 'ain': 'an~', 'ein': 'an~',
            'on': 'on~', 'un': 'un~', 'r': 'R', 'gn': 'ny'
        }
        
    def _init_german_phonemes(self):
        """Initialize German phoneme mapping"""
        return {
            'a': 'ah', 'e': 'eh', 'i': 'ee', 'o': 'oh', 'u': 'oo',
            'ä': 'eh', 'ö': 'eu', 'ü': 'ü', 'ei': 'eye', 'ie': 'ee',
            'eu': 'oy', 'äu': 'oy', 'ch+e/i': 'sh', 'ch+a/o/u': 'kh',
            'sch': 'sh', 'st': 'sht', 'sp': 'shp', 'v': 'f', 'w': 'v',
            'z': 'ts', 'j': 'y'
        }
        
    def _init_italian_phonemes(self):
        """Initialize Italian phoneme mapping"""
        return {
            'a': 'ah', 'e': 'eh', 'i': 'ee', 'o': 'oh', 'u': 'oo',
            'c+e/i': 'ch', 'c+a/o/u': 'k', 'ch': 'k', 'gh': 'g',
            'gli': 'ly', 'gn': 'ny', 'sc+e/i': 'sh', 'z': 'ts/dz'
        }
        
    def analyze_pronunciation(self, audio_data, target_word, recognized_text, language_code):
        """
        Perform comprehensive pronunciation analysis
        
        Args:
            audio_data: Binary audio data
            target_word: The word the user should be pronouncing
            recognized_text: Text recognized from speech recognition
            language_code: Language code (e.g., 'es', 'fr')
            
        Returns:
            Dictionary with analysis results
        """
        results = {}
        
        # Basic text comparison
        results.update(self._analyze_text_similarity(target_word, recognized_text))
        
        # Phonetic analysis
        results.update(self._analyze_phonetic_patterns(target_word, recognized_text, language_code))
        
        # Audio signal analysis if we have audio data
        if audio_data:
            results.update(self._analyze_audio_features(audio_data, target_word, language_code))
        
        # Generate specific error feedback
        results['errors'] = self._identify_specific_errors(target_word, recognized_text, language_code)
        
        # Generate overall feedback and score
        results['overall_score'] = self._calculate_overall_score(results)
        results['feedback'] = self._generate_feedback(results, language_code)
        
        return results
    
    def _analyze_text_similarity(self, target_word, recognized_text):
        """
        Analyze the text similarity between target and recognized text
        
        Args:
            target_word: Target word
            recognized_text: Recognized text from speech recognition
            
        Returns:
            Dictionary with text similarity metrics
        """
        import Levenshtein
        
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
        
        # Calculate Levenshtein distance
        distance = Levenshtein.distance(target_norm, recognized_norm)
        max_len = max(len(target_norm), len(recognized_norm))
        
        # Convert to similarity percentage
        similarity = 100 - (distance / max_len * 100) if max_len > 0 else 0
        
        # Return results
        return {
            'exact_match': exact_match,
            'text_similarity': similarity,
            'recognized_text': recognized_norm,
            'target_word': target_norm
        }
    
    def _analyze_phonetic_patterns(self, target_word, recognized_text, language_code):
        """
        Analyze phonetic patterns in the target and recognized words
        
        Args:
            target_word: Target word
            recognized_text: Recognized text from speech recognition
            language_code: Language code
            
        Returns:
            Dictionary with phonetic analysis results
        """
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
    
    def _analyze_audio_features(self, audio_data, target_word, language_code):
        """
        Analyze audio features for pronunciation assessment
        
        Args:
            audio_data: Binary audio data
            target_word: Target word
            language_code: Language code
            
        Returns:
            Dictionary with audio analysis results
        """
        try:
            # Use librosa for audio analysis if available
            try:
                import librosa
                has_librosa = True
            except ImportError:
                has_librosa = False
            
            # If librosa is not available, return basic estimates
            if not has_librosa:
                return {
                    'rhythm_score': 70,  # Default score
                    'intonation_score': 70,  # Default score
                    'fluency_score': 70   # Default score
                }
            
            # Convert audio data to numpy array for analysis
            audio_array = self._audio_to_array(audio_data)
            
            if audio_array is None:
                return {
                    'rhythm_score': 60,
                    'intonation_score': 60,
                    'fluency_score': 60
                }
            
            # Get audio features
            # Sample rate often 44100 Hz for web audio
            sr = 44100
            
            # Extract audio features
            # 1. Volume envelope
            volume_env = np.abs(audio_array)
            rms_energy = librosa.feature.rms(y=audio_array)[0]
            
            # 2. Zero crossing rate (consonant detection)
            zcr = librosa.feature.zero_crossing_rate(audio_array)[0]
            
            # 3. Spectral centroid (brightness/sharpness of sound)
            spectral_centroid = librosa.feature.spectral_centroid(y=audio_array, sr=sr)[0]
            
            # Analyze rhythm (based on energy fluctuations)
            rhythm_score = self._calculate_rhythm_score(rms_energy)
            
            # Analyze intonation (based on pitch variations)
            intonation_score = self._calculate_intonation_score(spectral_centroid)
            
            # Analyze fluency (based on continuity and pauses)
            fluency_score = self._calculate_fluency_score(rms_energy, zcr)
            
            # Return results
            return {
                'rhythm_score': rhythm_score,
                'intonation_score': intonation_score,
                'fluency_score': fluency_score,
                'audio_features': {
                    'rms_energy': rms_energy.mean(),
                    'zcr': zcr.mean(),
                    'spectral_centroid': spectral_centroid.mean()
                }
            }
        except Exception as e:
            print(f"Error in audio analysis: {e}")
            # Return default values on error
            return {
                'rhythm_score': 65,
                'intonation_score': 65,
                'fluency_score': 65,
                'analysis_error': str(e)
            }
    
    def _audio_to_array(self, audio_data):
        """Convert audio bytes to numpy array"""
        try:
            # Create a temporary file to save the audio data
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_filename = temp_file.name
                temp_file.write(audio_data)
            
            # Use librosa to load the audio file
            import librosa
            audio_array, sr = librosa.load(temp_filename, sr=None)
            
            # Remove the temporary file
            os.unlink(temp_filename)
            
            return audio_array
        except Exception as e:
            print(f"Error converting audio to array: {e}")
            
            # Alternative approach using wave module
            try:
                with io.BytesIO(audio_data) as audio_io:
                    with wave.open(audio_io, 'rb') as wav_file:
                        # Get audio parameters
                        n_channels = wav_file.getnchannels()
                        sample_width = wav_file.getsampwidth()
                        framerate = wav_file.getframerate()
                        n_frames = wav_file.getnframes()
                        
                        # Read the audio frames
                        frames = wav_file.readframes(n_frames)
                
                # Convert to numpy array (assuming 16-bit audio)
                import numpy as np
                audio_array = np.frombuffer(frames, dtype=np.int16)
                
                # Convert to float in range [-1, 1]
                audio_array = audio_array.astype(np.float32) / 32768.0
                
                # If stereo, convert to mono by averaging channels
                if n_channels == 2:
                    audio_array = audio_array.reshape(-1, 2).mean(axis=1)
                
                return audio_array
            except Exception as e2:
                print(f"Alternative audio conversion also failed: {e2}")
                return None
    
    def _calculate_rhythm_score(self, rms_energy, threshold=0.1):
        """Calculate rhythm score based on energy envelope"""
        try:
            # Calculate energy fluctuations
            energy_diff = np.diff(rms_energy)
            
            # Count significant energy changes
            significant_changes = np.sum(np.abs(energy_diff) > threshold)
            
            # Normalize to a score between 0-100
            expected_changes = len(rms_energy) * 0.1  # Expect changes in about 10% of frames
            change_ratio = significant_changes / expected_changes if expected_changes > 0 else 0
            
            # Calculate rhythm score - closer to 1.0 ratio is better
            if change_ratio > 2.0:
                # Too many fluctuations - not smooth enough
                rhythm_score = 100 - min(100, (change_ratio - 2.0) * 50)
            elif change_ratio < 0.5:
                # Too few fluctuations - too monotone
                rhythm_score = 100 - min(100, (0.5 - change_ratio) * 100)
            else:
                # Good range
                rhythm_score = 100 - min(100, abs(1.0 - change_ratio) * 50)
            
            return max(0, min(100, rhythm_score))
        except Exception as e:
            print(f"Error calculating rhythm score: {e}")
            return 70  # Default score on error
    
    def _calculate_intonation_score(self, spectral_centroid):
        """Calculate intonation score based on spectral centroid variations"""
        try:
            # Calculate centroid variations
            centroid_diff = np.diff(spectral_centroid)
            
            # Analyze variation - some variation is good, too much is bad
            variation = np.std(centroid_diff)
            
            # Normalize to a score
            # Ideal variation should be moderate - not too flat, not too varied
            if variation < 50:
                # Too monotone
                intonation_score = 50 + variation
            elif variation > 500:
                # Too varied/unstable
                intonation_score = 100 - min(50, (variation - 500) / 20)
            else:
                # Good range
                intonation_score = 75 + (250 - abs(variation - 250)) / 10
            
            return max(0, min(100, intonation_score))
        except Exception as e:
            print(f"Error calculating intonation score: {e}")
            return 70  # Default score on error
    
    def _calculate_fluency_score(self, rms_energy, zcr, energy_threshold=0.05):
        """Calculate fluency score based on continuity and pauses"""
        try:
            # Detect regions of silence (pauses)
            is_silent = rms_energy < energy_threshold
            
            # Count silence frames and transitions
            silent_frames = np.sum(is_silent)
            transitions = np.sum(np.abs(np.diff(is_silent.astype(int))))
            
            # Calculate scores
            silence_ratio = silent_frames / len(rms_energy)
            transition_rate = transitions / len(rms_energy)
            
            # Ideal: few silent frames and few transitions (smooth speech)
            if silence_ratio > 0.4:
                # Too much silence
                silence_score = 100 - min(100, (silence_ratio - 0.4) * 200)
            else:
                silence_score = 100 - min(100, silence_ratio * 100)
            
            # Some transitions are normal, too many indicate stuttering
            if transition_rate > 0.2:
                # Too many transitions/stutters
                transition_score = 100 - min(100, (transition_rate - 0.2) * 300)
            else:
                transition_score = 100 - min(100, transition_rate * 200)
            
            # Combine scores
            fluency_score = (silence_score + transition_score) / 2
            
            return max(0, min(100, fluency_score))
        except Exception as e:
            print(f"Error calculating fluency score: {e}")
            return 70  # Default score on error
    
    def _identify_specific_errors(self, target_word, recognized_text, language_code):
        """Identify specific pronunciation errors"""
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
        """Calculate overall pronunciation score from component scores"""
        # Component weights
        weights = {
            'text_similarity': 0.5,    # Text recognition has highest weight
            'phonetic_similarity': 0.3, # Phonetic similarity is also important
            'rhythm_score': 0.1,       # Rhythm matters less but still important
            'intonation_score': 0.05,  # Intonation matters less
            'fluency_score': 0.05      # Fluency matters less
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
        """Generate comprehensive feedback based on analysis results"""
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
        """
        Create a visual comparison between target and recognized words
        
        Args:
            target_word: Target word
            recognized_text: Recognized text
            
        Returns:
            Matplotlib figure object
        """
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
        
        # Determine character-by-character similarity
        target_chars = list(target)
        recognized_chars = list(recognized)
        
        # Calculate Levenshtein matrix for visualization
        # This helps show insertions, deletions, and substitutions
        import numpy as np
        rows = len(recognized_chars) + 1
        cols = len(target_chars) + 1
        
        # Initialize Levenshtein distance matrix
        matrix = np.zeros((rows, cols), dtype=int)
        
        # Fill first row and column
        for i in range(rows):
            matrix[i, 0] = i
        for j in range(cols):
            matrix[0, j] = j
        
        # Fill the matrix
        for i in range(1, rows):
            for j in range(1, cols):
                if recognized_chars[i-1] == target_chars[j-1]:
                    matrix[i, j] = matrix[i-1, j-1]  # Match
                else:
                    # Cost of operations (insertion, deletion, substitution)
                    matrix[i, j] = min(
                        matrix[i-1, j] + 1,    # Deletion
                        matrix[i, j-1] + 1,    # Insertion
                        matrix[i-1, j-1] + 1   # Substitution
                    )
        
        # Trace back to find operations
        operations = []
        i, j = rows-1, cols-1
        
        while i > 0 or j > 0:
            if i > 0 and j > 0 and recognized_chars[i-1] == target_chars[j-1]:
                operations.append(('match', i-1, j-1))
                i -= 1
                j -= 1
            elif i > 0 and (j == 0 or matrix[i-1, j] + 1 == matrix[i, j]):
                operations.append(('deletion', i-1, None))
                i -= 1
            elif j > 0 and (i == 0 or matrix[i, j-1] + 1 == matrix[i, j]):
                operations.append(('insertion', None, j-1))
                j -= 1
            else:
                operations.append(('substitution', i-1, j-1))
                i -= 1
                j -= 1
        
        # Reverse operations to get them in the right order
        operations.reverse()
        
        # Display characters with color coding
        target_x = np.linspace(0.1, 0.9, len(target_chars))
        recognized_x = np.linspace(0.1, 0.9, len(recognized_chars))
        
        # Draw target characters
        ax.text(0.05, 0.8, "Target:", fontsize=12, ha='right')
        for i, char in enumerate(target_chars):
            color = 'green'  # Default color
            
            # Check if this character has a match, substitution, or insertion
            for op, rec_idx, tgt_idx in operations:
                if tgt_idx == i:
                    if op == 'match':
                        color = 'green'
                    elif op == 'substitution':
                        color = 'red'
                    elif op == 'insertion':
                        color = 'orange'
                    break
            
            ax.text(target_x[i], 0.8, char, fontsize=16, ha='center', color=color)
        
        # Draw recognized characters
        ax.text(0.05, 0.4, "You said:", fontsize=12, ha='right')
        for i, char in enumerate(recognized_chars):
            color = 'green'  # Default color
            
            # Check if this character has a match, substitution, or deletion
            for op, rec_idx, tgt_idx in operations:
                if rec_idx == i:
                    if op == 'match':
                        color = 'green'
                    elif op == 'substitution':
                        color = 'red'
                    elif op == 'deletion':
                        color = 'blue'
                    break
            
            ax.text(recognized_x[i], 0.4, char, fontsize=16, ha='center', color=color)
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Correct'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Different'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='Extra in target'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Extra in yours')
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
    """
    Integrate the PronunciationAssessor with SimplePronunciationPractice
    
    Args:
        practice_instance: Instance of SimplePronunciationPractice class
        
    Returns:
        Updated instance with enhanced assessment capabilities
    """
    # Create an assessor instance
    assessor = PronunciationAssessor()
    
    # Add the assessor to the practice instance
    practice_instance.assessor = assessor
    
    # Override _evaluate_pronunciation method
    def enhanced_evaluate_pronunciation(self, audio_data, target_word, language_code):
        """Enhanced pronunciation evaluation with detailed analysis and feedback"""
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
        """Show comprehensive pronunciation feedback with visual indicators"""
        # Get the results from session state
        results = getattr(st.session_state, 'last_pronunciation_results', {})
        recognized_text = results.get('recognized_text', '')
        
        st.markdown("### Pronunciation Feedback")
        
        # Create columns for scores
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("**Overall Score**")
            overall_score = results.get('overall_score', similarity_score)
            score_color = self._get_score_color(overall_score)
            st.markdown(f"""
            <div style="text-align: center; font-size: 24px; font-weight: bold; 
                        color: {score_color};">
                {overall_score}%
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("**Word Match**")
            text_score = results.get('text_similarity', 0)
            text_color = self._get_score_color(text_score)
            st.markdown(f"""
            <div style="text-align: center; font-size: 24px; font-weight: bold; 
                        color: {text_color};">
                {text_score:.0f}%
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("**Rhythm**")
            rhythm_score = results.get('rhythm_score', 70)
            rhythm_color = self._get_score_color(rhythm_score)
            st.markdown(f"""
            <div style="text-align: center; font-size: 24px; font-weight: bold; 
                        color: {rhythm_color};">
                {rhythm_score:.0f}%
            </div>
            """, unsafe_allow_html=True)
            
        with col4:
            st.markdown("**Intonation**")
            intonation_score = results.get('intonation_score', 70)
            intonation_color = self._get_score_color(intonation_score)
            st.markdown(f"""
            <div style="text-align: center; font-size: 24px; font-weight: bold; 
                        color: {intonation_color};">
                {intonation_score:.0f}%
            </div>
            """, unsafe_allow_html=True)
        
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
        
        # Show improvement suggestion
        error_types = [e.get('type') for e in results.get('errors', [])]
        
        if 'perfect' in error_types:
            improvement_tip = "Great job! Keep practicing with more complex words."
        elif 'phonetic' in error_types:
            improvement_tip = "Focus on the specific sounds highlighted in red. Listen carefully to the native pronunciation and try to mimic the exact sounds."
        elif 'missing' in error_types:
            improvement_tip = "Make sure to pronounce all the sounds in the word. Try speaking more slowly to articulate each sound."
        elif 'extra' in error_types:
            improvement_tip = "You're adding extra sounds. Try to be more precise with your pronunciation."
        else:
            improvement_tip = "Listen to the native pronunciation again and try to match the rhythm and flow."
        
        st.markdown(f"""
        <div style="background-color: #e8f5e9; padding: 15px; border-radius: 5px; margin-top: 15px;">
            <h4 style="margin-top: 0;">💡 Practice Tip</h4>
            <p>{improvement_tip}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Add pronunciation practice suggestions
        st.markdown("### How to Improve")
        st.markdown("""
        1. **Listen and repeat** - Play the correct pronunciation and repeat multiple times
        2. **Break it down** - Practice difficult sounds individually before combining them
        3. **Record yourself** - Recording and comparing helps identify specific differences
        4. **Slow down** - Speaking too quickly often leads to missed or incorrect sounds
        5. **Watch mouth movements** - Observe how native speakers form sounds with their mouth
        """)
    
    # Helper method for color coding
    def _get_score_color(score):
        """Get color based on score"""
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
    """
    Setup function to be called from main.py
    
    This will enhance the pronunciation practice module with AI assessment
    """
    # Check if required packages are installed
    missing_packages = []
    
    try:
        import librosa
    except ImportError:
        missing_packages.append("librosa")
    
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