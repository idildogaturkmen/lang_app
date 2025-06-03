# pronunciation_practice.py - Comprehensive Pronunciation Practice with AI Feedback
"""
Enhanced Pronunciation Practice Module with Real-time AI Feedback

Features:
- Real-time audio recording and analysis
- AI-powered pronunciation assessment
- Visual feedback with spectrograms
- Phonetic analysis and comparison
- Multi-language support with language-specific error detection
- Progressive feedback during and after recording
"""

import streamlit as st
import time
import tempfile
import io
import os
import wave
import re
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import base64
import requests
import json
import threading
import queue
from collections import deque

# Core imports with fallbacks
try:
    from streamlit_webrtc import (
        webrtc_streamer, WebRtcMode, ClientSettings, RTCConfiguration, MediaStreamConstraints
    )
    import av
    HAS_WEBRTC = True
except ImportError:
    HAS_WEBRTC = False

try:
    import speech_recognition as sr
    HAS_SR = True
except ImportError:
    HAS_SR = False

try:
    import Levenshtein
    HAS_LEVENSHTEIN = True
except ImportError:
    HAS_LEVENSHTEIN = False

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

try:
    from custom_audio_recorder import audio_recorder
    HAS_CUSTOM_RECORDER = True
except ImportError:
    HAS_CUSTOM_RECORDER = False

# Language configurations
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

LANGUAGE_NAMES = {
    "es": "Spanish", "fr": "French", "de": "German", "it": "Italian",
    "pt": "Portuguese", "ru": "Russian", "ja": "Japanese", "zh-CN": "Chinese"
}

RECOGNITION_LANGUAGES = {
    "es": "es-ES", "fr": "fr-FR", "de": "de-DE", "it": "it-IT",
    "pt": "pt-BR", "ru": "ru-RU", "ja": "ja-JP", "zh-CN": "zh-CN", "en": "en-US"
}

class RealTimeAudioAnalyzer:
    """Real-time audio analysis for live pronunciation feedback"""
    
    def __init__(self, target_word, language_code):
        self.target_word = target_word
        self.language_code = language_code
        self.audio_buffer = deque(maxlen=50)  # Keep last 50 audio frames
        self.analysis_queue = queue.Queue()
        self.feedback_queue = queue.Queue()
        self.is_analyzing = False
        
    def add_audio_frame(self, audio_frame):
        """Add new audio frame for real-time analysis"""
        try:
            # Convert audio frame to numpy array
            audio_data = audio_frame.to_ndarray()
            self.audio_buffer.append(audio_data)
            
            # Trigger analysis if we have enough data
            if len(self.audio_buffer) >= 10 and not self.is_analyzing:
                self._analyze_current_buffer()
                
        except Exception as e:
            print(f"Error processing audio frame: {e}")
    
    def _analyze_current_buffer(self):
        """Analyze current audio buffer for real-time feedback"""
        if not self.audio_buffer:
            return
            
        try:
            # Combine recent audio frames
            combined_audio = np.concatenate(list(self.audio_buffer))
            
            # Basic audio metrics
            volume = np.sqrt(np.mean(combined_audio**2)) * 100
            
            # Zero crossing rate (consonant detection)
            zero_crossings = np.sum(np.diff(np.sign(combined_audio)) != 0)
            clarity = min(100, zero_crossings / len(combined_audio) * 1000)
            
            # Spectral analysis for pitch
            if HAS_LIBROSA:
                try:
                    # Estimate pitch
                    pitches, magnitudes = librosa.piptrack(y=combined_audio, sr=48000)
                    pitch_values = pitches[magnitudes > np.max(magnitudes) * 0.1]
                    avg_pitch = np.mean(pitch_values) if len(pitch_values) > 0 else 0
                    pitch_accuracy = min(100, max(0, 100 - abs(avg_pitch - 200) / 2))
                except:
                    pitch_accuracy = 70
            else:
                pitch_accuracy = 70
            
            # Generate real-time feedback
            feedback = self._generate_realtime_feedback(volume, clarity, pitch_accuracy)
            
            # Store metrics in session state
            st.session_state.realtime_metrics = {
                'volume': volume,
                'clarity': clarity,
                'pitchAccuracy': pitch_accuracy,
                'feedback': feedback
            }
            
        except Exception as e:
            print(f"Error in real-time analysis: {e}")
    
    def _generate_realtime_feedback(self, volume, clarity, pitch_accuracy):
        """Generate real-time feedback message"""
        if volume < 20:
            return "🔇 Speak louder - I can barely hear you"
        elif volume > 80:
            return "🔊 Too loud - speak a bit softer"
        elif clarity < 40:
            return "🗣️ Speak more clearly - articulate each sound"
        elif pitch_accuracy < 50:
            return "🎵 Adjust your tone - try to match the target pronunciation"
        elif volume >= 40 and clarity >= 60 and pitch_accuracy >= 60:
            return "✅ Great! Your pronunciation sounds good"
        else:
            return "👍 Good - keep speaking clearly"

class PronunciationAssessor:
    """Advanced AI-powered pronunciation assessment"""
    
    def __init__(self):
        self.phoneme_maps = {
            "es": self._init_spanish_phonemes(),
            "fr": self._init_french_phonemes(),
            "de": self._init_german_phonemes(),
            "it": self._init_italian_phonemes(),
        }
        
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
        return {
            'a': 'ah', 'e': 'eh', 'i': 'ee', 'o': 'oh', 'u': 'oo',
            'j': 'h', 'll': 'y', 'ñ': 'ny', 'rr': 'rr', 'r': 'r'
        }
    
    def _init_french_phonemes(self):
        return {
            'a': 'ah', 'e': 'uh', 'i': 'ee', 'o': 'oh', 'u': 'ü',
            'ai': 'eh', 'oi': 'wa', 'au': 'oh', 'r': 'R', 'gn': 'ny'
        }
    
    def _init_german_phonemes(self):
        return {
            'a': 'ah', 'e': 'eh', 'i': 'ee', 'o': 'oh', 'u': 'oo',
            'ä': 'eh', 'ö': 'eu', 'ü': 'ü', 'ei': 'eye', 'ie': 'ee'
        }
    
    def _init_italian_phonemes(self):
        return {
            'a': 'ah', 'e': 'eh', 'i': 'ee', 'o': 'oh', 'u': 'oo',
            'c+e/i': 'ch', 'gli': 'ly', 'gn': 'ny', 'sc+e/i': 'sh'
        }
    
    def analyze_pronunciation(self, audio_data, target_word, recognized_text, language_code):
        """Comprehensive AI-powered pronunciation analysis"""
        results = {}
        
        # Text similarity analysis
        results.update(self._analyze_text_similarity(target_word, recognized_text))
        
        # Phonetic pattern analysis
        results.update(self._analyze_phonetic_patterns(target_word, recognized_text, language_code))
        
        # Audio feature analysis
        if audio_data and HAS_LIBROSA:
            results.update(self._analyze_audio_features(audio_data))
        
        # Generate AI feedback
        results['errors'] = self._identify_specific_errors(target_word, recognized_text, language_code)
        results['overall_score'] = self._calculate_overall_score(results)
        results['feedback'] = self._generate_ai_feedback(results, language_code)
        results['improvement_suggestions'] = self._generate_improvement_suggestions(results, language_code)
        
        return results
    
    def _analyze_text_similarity(self, target_word, recognized_text):
        """Analyze text similarity using Levenshtein distance"""
        target_norm = target_word.lower().strip()
        
        if not recognized_text:
            return {'exact_match': False, 'text_similarity': 0, 'recognized_text': '', 'target_word': target_norm}
        
        recognized_norm = recognized_text.lower().strip().split()[0]
        exact_match = (target_norm == recognized_norm)
        
        if HAS_LEVENSHTEIN:
            distance = Levenshtein.distance(target_norm, recognized_norm)
            max_len = max(len(target_norm), len(recognized_norm))
            similarity = 100 - (distance / max_len * 100) if max_len > 0 else 0
        else:
            # Fallback similarity calculation
            common_chars = set(target_norm).intersection(set(recognized_norm))
            total_chars = set(target_norm).union(set(recognized_norm))
            similarity = len(common_chars) / len(total_chars) * 100 if total_chars else 0
        
        return {
            'exact_match': exact_match,
            'text_similarity': similarity,
            'recognized_text': recognized_norm,
            'target_word': target_norm
        }
    
    def _analyze_phonetic_patterns(self, target_word, recognized_text, language_code):
        """Analyze phonetic patterns and common mispronunciations"""
        if not recognized_text:
            return {'phonetic_similarity': 0, 'phoneme_matches': [], 'phoneme_errors': []}
        
        target = target_word.lower()
        recognized = recognized_text.lower()
        
        matches = []
        errors = []
        
        # Character-by-character comparison with position tolerance
        for i, char in enumerate(target):
            found = False
            for j in range(max(0, i-2), min(len(recognized), i+3)):
                if j < len(recognized) and recognized[j] == char:
                    matches.append({
                        'char': char, 'target_pos': i, 'recognized_pos': j,
                        'position_diff': abs(i - j)
                    })
                    found = True
                    break
            
            if not found:
                error = {'char': char, 'target_pos': i, 'error_type': 'missing'}
                # Check for known phonetic substitutions
                for pattern, info in self.error_patterns.get(language_code, {}).items():
                    if pattern in target[max(0, i-len(pattern)+1):i+1]:
                        error.update({
                            'error_type': 'phonetic',
                            'expected': info['correct'],
                            'common_error': info['english_error']
                        })
                        break
                errors.append(error)
        
        phonetic_similarity = (len(matches) / len(target) * 100) if target else 0
        
        return {
            'phonetic_similarity': phonetic_similarity,
            'phoneme_matches': matches,
            'phoneme_errors': errors
        }
    
    def _analyze_audio_features(self, audio_data):
        """Analyze audio features using librosa"""
        try:
            # Convert audio to numpy array
            audio_array = self._audio_to_array(audio_data)
            if audio_array is None:
                return {'rhythm_score': 70, 'intonation_score': 70, 'fluency_score': 70}
            
            sr = 44100
            
            # Extract features
            rms_energy = librosa.feature.rms(y=audio_array)[0]
            zcr = librosa.feature.zero_crossing_rate(audio_array)[0]
            spectral_centroid = librosa.feature.spectral_centroid(y=audio_array, sr=sr)[0]
            
            # Calculate scores
            rhythm_score = self._calculate_rhythm_score(rms_energy)
            intonation_score = self._calculate_intonation_score(spectral_centroid)
            fluency_score = self._calculate_fluency_score(rms_energy, zcr)
            
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
            return {'rhythm_score': 65, 'intonation_score': 65, 'fluency_score': 65}
    
    def _audio_to_array(self, audio_data):
        """Convert audio bytes to numpy array"""
        try:
            if HAS_LIBROSA:
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    temp_file.write(audio_data)
                    temp_filename = temp_file.name
                
                audio_array, sr = librosa.load(temp_filename, sr=None)
                os.unlink(temp_filename)
                return audio_array
            else:
                # Fallback using wave module
                with io.BytesIO(audio_data) as audio_io:
                    with wave.open(audio_io, 'rb') as wav_file:
                        frames = wav_file.readframes(wav_file.getnframes())
                        audio_array = np.frombuffer(frames, dtype=np.int16)
                        return audio_array.astype(np.float32) / 32768.0
        except Exception as e:
            print(f"Error converting audio: {e}")
            return None
    
    def _calculate_rhythm_score(self, rms_energy, threshold=0.1):
        """Calculate rhythm score based on energy envelope"""
        try:
            energy_diff = np.diff(rms_energy)
            significant_changes = np.sum(np.abs(energy_diff) > threshold)
            expected_changes = len(rms_energy) * 0.1
            change_ratio = significant_changes / expected_changes if expected_changes > 0 else 0
            
            if change_ratio > 2.0:
                rhythm_score = 100 - min(100, (change_ratio - 2.0) * 50)
            elif change_ratio < 0.5:
                rhythm_score = 100 - min(100, (0.5 - change_ratio) * 100)
            else:
                rhythm_score = 100 - min(100, abs(1.0 - change_ratio) * 50)
            
            return max(0, min(100, rhythm_score))
        except:
            return 70
    
    def _calculate_intonation_score(self, spectral_centroid):
        """Calculate intonation score based on spectral centroid variations"""
        try:
            variation = np.std(np.diff(spectral_centroid))
            
            if variation < 50:
                intonation_score = 50 + variation
            elif variation > 500:
                intonation_score = 100 - min(50, (variation - 500) / 20)
            else:
                intonation_score = 75 + (250 - abs(variation - 250)) / 10
            
            return max(0, min(100, intonation_score))
        except:
            return 70
    
    def _calculate_fluency_score(self, rms_energy, zcr, energy_threshold=0.05):
        """Calculate fluency score based on continuity and pauses"""
        try:
            is_silent = rms_energy < energy_threshold
            silent_frames = np.sum(is_silent)
            transitions = np.sum(np.abs(np.diff(is_silent.astype(int))))
            
            silence_ratio = silent_frames / len(rms_energy)
            transition_rate = transitions / len(rms_energy)
            
            if silence_ratio > 0.4:
                silence_score = 100 - min(100, (silence_ratio - 0.4) * 200)
            else:
                silence_score = 100 - min(100, silence_ratio * 100)
            
            if transition_rate > 0.2:
                transition_score = 100 - min(100, (transition_rate - 0.2) * 300)
            else:
                transition_score = 100 - min(100, transition_rate * 200)
            
            return (silence_score + transition_score) / 2
        except:
            return 70
    
    def _identify_specific_errors(self, target_word, recognized_text, language_code):
        """Identify specific pronunciation errors using AI analysis"""
        errors = []
        
        if not recognized_text:
            return [{'type': 'no_speech', 'message': 'No speech was detected'}]
        
        target = target_word.lower()
        recognized = recognized_text.lower()
        
        # Language-specific error detection
        language_errors = self.error_patterns.get(language_code, {})
        for pattern, info in language_errors.items():
            if pattern in target and pattern not in recognized:
                errors.append({
                    'type': 'phonetic',
                    'pattern': pattern,
                    'expected': info['correct'],
                    'likely_pronounced': info['english_error'],
                    'message': f"The '{pattern}' should sound like {info['correct']}, not {info['english_error']}"
                })
        
        # Character-level analysis
        target_chars = set(target)
        recognized_chars = set(recognized)
        
        for char in target_chars - recognized_chars:
            if char.isalpha():
                errors.append({
                    'type': 'missing',
                    'char': char,
                    'message': f"You missed the '{char}' sound"
                })
        
        for char in recognized_chars - target_chars:
            if char.isalpha():
                errors.append({
                    'type': 'extra',
                    'char': char,
                    'message': f"You added an extra '{char}' sound"
                })
        
        if target == recognized and not errors:
            errors.append({'type': 'perfect', 'message': "Perfect pronunciation!"})
        elif not errors:
            errors.append({'type': 'general', 'message': "Your pronunciation differed slightly from the target"})
        
        return errors
    
    def _calculate_overall_score(self, results):
        """Calculate AI-weighted overall pronunciation score"""
        weights = {
            'text_similarity': 0.4,
            'phonetic_similarity': 0.3,
            'rhythm_score': 0.1,
            'intonation_score': 0.1,
            'fluency_score': 0.1
        }
        
        total_score = 0
        total_weight = 0
        
        for metric, weight in weights.items():
            if metric in results:
                total_score += results[metric] * weight
                total_weight += weight
        
        return round(total_score / total_weight) if total_weight > 0 else 60
    
    def _generate_ai_feedback(self, results, language_code):
        """Generate AI-powered feedback messages"""
        score = results.get('overall_score', 0)
        
        if score >= 90:
            general = "🌟 Outstanding! Your pronunciation is nearly perfect."
        elif score >= 80:
            general = "🎉 Excellent pronunciation! Just minor fine-tuning needed."
        elif score >= 70:
            general = "👍 Good job! Your pronunciation is quite clear."
        elif score >= 50:
            general = "📚 Getting there! Focus on the specific sounds highlighted."
        else:
            general = "🎯 Let's practice! Break down the word into individual sounds."
        
        return [general]
    
    def _generate_improvement_suggestions(self, results, language_code):
        """Generate AI-powered improvement suggestions"""
        suggestions = []
        
        errors = results.get('errors', [])
        rhythm_score = results.get('rhythm_score', 70)
        intonation_score = results.get('intonation_score', 70)
        
        # Specific error-based suggestions
        for error in errors:
            if error['type'] == 'phonetic':
                suggestions.append(f"Practice the {error['pattern']} sound: {error['expected']}")
            elif error['type'] == 'missing':
                suggestions.append(f"Make sure to pronounce the '{error['char']}' sound clearly")
        
        # Audio-based suggestions
        if rhythm_score < 60:
            suggestions.append("Work on your rhythm - try clapping while saying the word")
        if intonation_score < 60:
            suggestions.append("Practice the melody of the word - listen to native speakers")
        
        # General suggestions
        suggestions.extend([
            "Record yourself and compare with native pronunciation",
            "Practice in front of a mirror to observe mouth movements",
            "Break the word into syllables and practice each part separately"
        ])
        
        return suggestions[:3]  # Return top 3 suggestions

class ComprehensivePronunciationPractice:
    """Main class combining all pronunciation practice features with real-time AI feedback"""
    
    def __init__(self, text_to_speech_func, get_audio_html_func, translate_text_func, get_example_sentence_func=None):
        self.text_to_speech = text_to_speech_func
        self.get_audio_html = get_audio_html_func
        self.translate_text = translate_text_func
        self.get_example_sentence_func = get_example_sentence_func
        
        # Initialize AI components
        self.assessor = PronunciationAssessor()
        self.realtime_analyzer = None
        
        # Initialize speech recognition
        if HAS_SR:
            self.recognizer = sr.Recognizer()
            self.recognizer.energy_threshold = 300
        
        # Check for custom recorder
        self.has_custom_recorder = HAS_CUSTOM_RECORDER
    
    def render_practice_ui(self, word):
        """Render the complete pronunciation practice UI for a word"""
        # Word information
        original_word = word.get('word_original', '')
        translated_word = word.get('word_translated', '')
        language_code = word.get('language_translated', 'en')
        
        # Initialize real-time analyzer
        self.realtime_analyzer = RealTimeAudioAnalyzer(translated_word, language_code)
        
        st.subheader(f"Practice: {translated_word}")
        
        # Show example sentence
        self._show_example_sentence(original_word, language_code)
        
        # Play correct pronunciation
        st.markdown("**🔊 Listen to correct pronunciation:**")
        audio_bytes = self.text_to_speech(translated_word, language_code)
        if audio_bytes:
            st.markdown(self.get_audio_html(audio_bytes), unsafe_allow_html=True)
        
        # Show pronunciation tips
        self._show_pronunciation_tips(word)
        
        # Real-time feedback section
        st.markdown("### 🎙️ Record Your Pronunciation")
        
        # Real-time metrics container
        self._show_realtime_metrics()
        
        # Recording interface with real-time feedback
        audio_recorded = self._render_recording_interface(translated_word, language_code)
        
        # AI analysis and feedback - IMPROVED DETECTION
        if (audio_recorded or 
            (hasattr(st.session_state, 'audio_data') and st.session_state.audio_data is not None) or
            (hasattr(st.session_state, 'audio_data_received') and st.session_state.audio_data_received)):
            
            # Check if we already analyzed this recording
            if 'last_pronunciation_results' not in st.session_state:
                self._render_ai_feedback(translated_word, language_code)
            else:
                # Show existing results
                st.markdown("---")
                results = st.session_state.last_pronunciation_results
                results['language_code'] = language_code  # Ensure language code is set
                self._display_ai_feedback(translated_word, results)
                
                # Add action buttons
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("📚 Save to Vocabulary", key=f"save_result_{translated_word}"):
                        st.session_state.save_pronunciation_word = {
                            'original': translated_word,
                            'translated': translated_word,
                            'language': language_code,
                            'score': results.get('overall_score', 0),
                            'recognized': results.get('recognized_text', '')
                        }
                        st.success("✅ Word will be saved to vocabulary!")
                        st.rerun()
                
                with col2:
                    if st.button("🔄 Record Again", key=f"retry_result_{translated_word}"):
                        self._clear_audio_and_retry()

    def _show_example_sentence(self, word, language_code):
        """Show example sentence with translation"""
        example = self._get_example_sentence(word, language_code)
        
        with st.expander("📖 Example in Context", expanded=False):
            st.markdown(f"**English:** {example['english']}")
            
            if example['translated']:
                lang_name = LANGUAGE_NAMES.get(language_code, language_code)
                st.markdown(f"**{lang_name}:** {example['translated']}")
                
                # Play example audio
                example_audio = self.text_to_speech(example['translated'], language_code)
                if example_audio:
                    st.markdown(self.get_audio_html(example_audio), unsafe_allow_html=True)
    
    def _show_pronunciation_tips(self, word):
        """Show language-specific pronunciation tips"""
        language_code = word.get('language_translated', 'en')
        translated_word = word.get('word_translated', '')
        
        language_sounds = DIFFICULT_SOUNDS.get(language_code, {})
        tips = []
        
        for sound, data in language_sounds.items():
            if sound in translated_word.lower():
                tips.append(f"**'{sound}'** sounds like **'{data['sound']}'** ({data['example']})")
        
        if tips:
            with st.expander("💡 Pronunciation Tips", expanded=False):
                for tip in tips:
                    st.markdown(f"- {tip}")
    
    def _show_realtime_metrics(self):
        """Display real-time pronunciation metrics"""
        if 'realtime_metrics' in st.session_state:
            metrics = st.session_state.realtime_metrics
            
            # Create metrics display
            col1, col2, col3 = st.columns(3)
            
            with col1:
                volume = metrics.get('volume', 0)
                st.metric("Volume", f"{int(volume)}%")
                color = "#4CAF50" if volume >= 40 else "#FFC107" if volume >= 20 else "#F44336"
                st.markdown(f"""
                <div style="width: 100%; background-color: #e0e0e0; border-radius: 4px; height: 8px;">
                    <div style="width: {volume}%; background-color: {color}; height: 8px; border-radius: 4px;"></div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                clarity = metrics.get('clarity', 0)
                st.metric("Clarity", f"{int(clarity)}%")
                color = "#4CAF50" if clarity >= 70 else "#FFC107" if clarity >= 40 else "#F44336"
                st.markdown(f"""
                <div style="width: 100%; background-color: #e0e0e0; border-radius: 4px; height: 8px;">
                    <div style="width: {clarity}%; background-color: {color}; height: 8px; border-radius: 4px;"></div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                pitch = metrics.get('pitchAccuracy', 0)
                st.metric("Pitch", f"{int(pitch)}%")
                color = "#4CAF50" if pitch >= 70 else "#FFC107" if pitch >= 50 else "#F44336"
                st.markdown(f"""
                <div style="width: 100%; background-color: #e0e0e0; border-radius: 4px; height: 8px;">
                    <div style="width: {pitch}%; background-color: {color}; height: 8px; border-radius: 4px;"></div>
                </div>
                """, unsafe_allow_html=True)
            
            # Display feedback message
            feedback = metrics.get('feedback', 'Ready to practice!')
            st.info(feedback)
    
    def _render_recording_interface(self, target_word, language_code):
        """Render the recording interface with real-time feedback"""
        st.markdown("🎙️ **Record yourself saying:** " + f"**{target_word}**")
        
        # Add recording tips
        with st.expander("💡 Recording Tips", expanded=False):
            st.markdown("""
            - **Speak clearly** and at normal volume
            - **Use a quiet environment** to reduce background noise
            - **Speak directly** into your microphone
            - **Take your time** - don't rush the pronunciation
            - **Listen to the correct pronunciation** first if needed
            """)
        
        if self.has_custom_recorder:
            return self._render_custom_recorder()
        elif HAS_WEBRTC:
            return self._render_webrtc_recorder(target_word, language_code)
        else:
            return self._render_upload_recorder()
    
    def _render_custom_recorder(self):
        """Render custom JavaScript recorder with real-time feedback"""
        try:
            from custom_audio_recorder import audio_recorder
            audio_bytes = audio_recorder()
            
            if audio_bytes:
                st.session_state.audio_data = audio_bytes
                st.session_state.audio_data_received = True
                
                st.success("✅ Recording captured successfully!")
                st.audio(audio_bytes)
                return True
            return False
        except Exception as e:
            st.error(f"Custom recorder error: {e}")
            return self._render_upload_recorder()
    
    def _render_webrtc_recorder(self, target_word, language_code):
        """Render WebRTC recorder with real-time analysis"""
        if 'audio_frames' not in st.session_state:
            st.session_state.audio_frames = []
        
        def audio_frame_callback(frame):
            try:
                # Add frame to buffer for real-time analysis
                if self.realtime_analyzer:
                    self.realtime_analyzer.add_audio_frame(frame)
                
                # Store frame for final analysis
                sound = frame.to_ndarray()
                st.session_state.audio_frames.append(sound)
            except Exception as e:
                print(f"Frame callback error: {e}")
            return frame
        
        # WebRTC streamer
        webrtc_ctx = webrtc_streamer(
            key="ai-pronunciation-recorder",
            mode=WebRtcMode.SENDONLY,
            audio_frame_callback=audio_frame_callback,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": False, "audio": True},
        )
        
        # Recording status
        if webrtc_ctx.state.playing:
            st.info("🎙️ Recording... Speak clearly!")
        else:
            if st.session_state.audio_frames:
                st.success("✅ Recording complete!")
                
                if st.button("🔬 Analyze Pronunciation", type="primary"):
                    with st.spinner("Processing with AI..."):
                        # Combine audio frames
                        combined_audio = np.concatenate(st.session_state.audio_frames, axis=0)
                        
                        # Convert to WAV
                        byte_io = io.BytesIO()
                        with wave.open(byte_io, 'wb') as wf:
                            wf.setnchannels(1)
                            wf.setsampwidth(2)
                            wf.setframerate(48000)
                            wf.writeframes(combined_audio.tobytes())
                        
                        byte_io.seek(0)
                        audio_bytes = byte_io.read()
                        
                        st.session_state.audio_data = audio_bytes
                        st.session_state.audio_data_received = True
                        st.session_state.audio_frames = []
                        
                        st.rerun()
            else:
                st.info("Click START above to begin recording")
        
        return 'audio_data' in st.session_state and st.session_state.audio_data
    
    def _render_upload_recorder(self):
        """Render file upload recorder as fallback"""
        st.markdown("**📁 Upload Recording**")
        st.markdown("Record using your device and upload the audio file:")
        
        uploaded_file = st.file_uploader(
            "Upload pronunciation recording", 
            type=["wav", "mp3", "ogg", "m4a"]
        )
        
        if uploaded_file:
            audio_bytes = uploaded_file.read()
            st.session_state.audio_data = audio_bytes
            st.session_state.audio_data_received = True
            
            st.success("✅ Audio uploaded successfully!")
            st.audio(audio_bytes)
            return True
        
        return False
    
    def _render_ai_feedback(self, target_word, language_code):
        """Render comprehensive AI-powered feedback"""
        
        # Add a visual separator
        st.markdown("---")
        st.markdown("### 🤖 AI Analysis in Progress...")
        
        # Create a progress indicator
        with st.spinner("🤖 AI is analyzing your pronunciation..."):
            try:
                # Speech recognition
                recognized_text = self._recognize_speech(st.session_state.audio_data, language_code)
                
                # AI analysis
                results = self.assessor.analyze_pronunciation(
                    st.session_state.audio_data,
                    target_word,
                    recognized_text,
                    language_code
                )
                
                # Store results
                st.session_state.last_pronunciation_results = results
                
                # Add language code to results for saving
                results['language_code'] = language_code
                
            except Exception as e:
                st.error(f"Error during AI analysis: {str(e)}")
                # Provide fallback results
                results = {
                    'overall_score': 50,
                    'text_similarity': 30,
                    'rhythm_score': 60,
                    'intonation_score': 60,
                    'fluency_score': 60,
                    'recognized_text': '',
                    'feedback': ['Unable to complete full analysis. Please try again.'],
                    'errors': [{'type': 'general', 'message': 'Analysis error occurred'}],
                    'improvement_suggestions': ['Try speaking more clearly', 'Ensure good microphone quality'],
                    'language_code': language_code
                }
        
        # Clear the spinner and show results
        st.empty()
        
        # Display comprehensive feedback
        self._display_ai_feedback(target_word, results)
        
        # Add auto-progression for session modes
        if hasattr(st.session_state, 'practice_session_words') and hasattr(st.session_state, 'current_session_index'):
            # This is a session mode - add progression controls
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📚 Save & Continue", key=f"save_continue_{target_word}", type="primary"):
                    # Save word and progress
                    self._save_and_progress(target_word, results)
            
            with col2:
                if st.button("⏭️ Skip Word", key=f"skip_{target_word}"):
                    self._progress_to_next_word()
            
            with col3:
                if st.button("🔄 Try Again", key=f"retry_{target_word}"):
                    self._clear_audio_and_retry()

    def _save_and_progress(self, target_word, results):
        """Save word to vocabulary and progress to next word"""
        # Save to vocabulary
        st.session_state.save_pronunciation_word = {
            'original': target_word,
            'translated': target_word,
            'language': results.get('language_code', 'en'),
            'score': results.get('overall_score', 0),
            'recognized': results.get('recognized_text', '')
        }
        
        # Update session stats immediately
        st.session_state.words_studied += 1
        if results.get('overall_score', 0) >= 70:  # Consider it "learned" if score is good
            st.session_state.words_learned += 1
        
        # Progress to next word
        self._progress_to_next_word()

    def _progress_to_next_word(self):
        """Progress to the next word in the session"""
        if hasattr(st.session_state, 'current_session_index'):
            st.session_state.current_session_index += 1
        
        # Clear audio data for next recording
        self._clear_audio_and_retry()

    def _clear_audio_and_retry(self):
        """Clear audio data to allow new recording"""
        if 'audio_data' in st.session_state:
            st.session_state.audio_data = None
        if 'audio_data_received' in st.session_state:
            st.session_state.audio_data_received = False
        if 'last_pronunciation_results' in st.session_state:
            del st.session_state.last_pronunciation_results
        st.rerun()
        
    def _recognize_speech(self, audio_data, language_code):
        """Recognize speech from audio data"""
        if not HAS_SR:
            return ""
        
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as audio_file:
                audio_file.write(audio_data)
                audio_file.close()
                
                with sr.AudioFile(audio_file.name) as source:
                    # Adjust for ambient noise
                    self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    audio = self.recognizer.record(source)
                    
                    rec_lang = RECOGNITION_LANGUAGES.get(language_code, "en-US")
                    
                    # Try multiple recognition services for better results
                    recognized_text = ""
                    
                    # First try: Google Speech Recognition (free)
                    try:
                        recognized_text = self.recognizer.recognize_google(audio, language=rec_lang)
                        print(f"✅ Google recognized: '{recognized_text}'")
                        return recognized_text.lower()
                    except (sr.UnknownValueError, sr.RequestError) as e:
                        print(f"Google recognition failed: {e}")
                    
                    # Fallback: Try with different language settings
                    try:
                        recognized_text = self.recognizer.recognize_google(audio, language="en-US")
                        print(f"✅ Fallback recognized: '{recognized_text}'")
                        return recognized_text.lower()
                    except:
                        pass
                        
                    return ""
                    
        except Exception as e:
            print(f"Speech recognition error: {e}")
            return ""
        finally:
            try:
                os.unlink(audio_file.name)
            except:
                pass
    
    def _get_example_sentence(self, word, language_code):
        """Get example sentence using external function or fallback"""
        if self.get_example_sentence_func:
            return self.get_example_sentence_func(word, language_code)
        else:
            # Simple fallback
            return {
                "english": f"I like this {word} very much.",
                "translated": "",
                "source": "fallback_template"
            }
    
    def _get_score_color(self, score):
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
    
    def _show_pronunciation_history(self, word_text):
        """Show pronunciation history and progress"""
        if 'pronunciation_history' not in st.session_state:
            st.session_state.pronunciation_history = {}
        
        word_key = word_text.lower()
        if word_key not in st.session_state.pronunciation_history:
            st.session_state.pronunciation_history[word_key] = []
        
        # Add current score to history
        if 'last_pronunciation_results' in st.session_state:
            results = st.session_state.last_pronunciation_results
            score = results.get('overall_score', 0)
            timestamp = datetime.now().strftime("%H:%M")
            
            st.session_state.pronunciation_history[word_key].append({
                'timestamp': timestamp,
                'score': score
            })
            
            # Keep only last 10 attempts
            if len(st.session_state.pronunciation_history[word_key]) > 10:
                st.session_state.pronunciation_history[word_key].pop(0)
        
        # Display history
        history = st.session_state.pronunciation_history[word_key]
        if len(history) > 1:
            with st.expander("📈 Your Progress History"):
                scores = [item['score'] for item in history]
                attempts = list(range(1, len(scores) + 1))
                
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(attempts, scores, marker='o', linestyle='-', color='#1976D2')
                ax.set_xlabel('Attempt')
                ax.set_ylabel('Score (%)')
                ax.set_title('Pronunciation Progress')
                ax.set_ylim(0, 100)
                ax.grid(True, alpha=0.3)
                
                st.pyplot(fig)
                
                # Progress message
                if len(scores) >= 2:
                    improvement = scores[-1] - scores[0]
                    if improvement > 0:
                        st.success(f"🎉 You've improved by {improvement:.0f} points!")
                    elif improvement == 0:
                        st.info("💪 Keep practicing to improve further!")
                    else:
                        st.warning("📚 Try focusing on the specific tips above.")

# Main creation function
def create_pronunciation_practice(text_to_speech_func, get_audio_html_func, translate_text_func, get_example_sentence_func=None):
    """
    Create the comprehensive pronunciation practice module
    
    Args:
        text_to_speech_func: Function for text-to-speech conversion
        get_audio_html_func: Function to get audio HTML
        translate_text_func: Function to translate text
        get_example_sentence_func: Function to get example sentences
    
    Returns:
        ComprehensivePronunciationPractice instance with AI capabilities
    """
    return ComprehensivePronunciationPractice(
        text_to_speech_func,
        get_audio_html_func,
        translate_text_func,
        get_example_sentence_func
    )