"""
Enhanced Quiz System for Vocam Language Learning App

This module implements a robust, user-friendly quiz system with multiple question types,
detailed feedback, and an improved learning experience.
"""

import streamlit as st
import os
import random
import io
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from collections import defaultdict

class QuizSystem:
    def __init__(self, db_functions, text_to_speech, get_audio_html, get_example_sentence, get_pronunciation_guide):
        """Initialize the quiz system with required dependencies.
        
        Args:
            db_functions: Dictionary containing database functions:
                - get_all_vocabulary_direct: Function to get vocabulary
                - update_word_progress_direct: Function to update word progress
            text_to_speech: Function to convert text to speech
            get_audio_html: Function to generate HTML for audio
            get_example_sentence: Function to get example sentences
            get_pronunciation_guide: Function to get pronunciation guide
        """
        self.get_all_vocabulary_direct = db_functions['get_all_vocabulary_direct']
        self.update_word_progress_direct = db_functions['update_word_progress_direct']
        self.text_to_speech = text_to_speech
        self.get_audio_html = get_audio_html
        self.get_example_sentence = get_example_sentence
        self.get_pronunciation_guide = get_pronunciation_guide
        
        # Define question types
        self.QUESTION_TYPES = [
            "translation_to_target",     # English → Target language
            "translation_to_english",    # Target language → English
            "image_recognition",         # Show image, select correct word
            "category_match",            # Match word to correct category
            "sentence_completion",       # Fill in blank in a sentence
            "multiple_choice_category",  # Choose words from same category
            "audio_recognition"          # Hear word, select correct option
        ]
    
    def get_language_name_from_code(self, language_code, languages):
        """Get the language name from its code."""
        for name, code in languages.items():
            if code == language_code:
                return name
        return language_code
    
    def get_possible_question_types(self, word, vocabulary):
        """Determine which question types are possible for the current word."""
        possible_types = []
        
        # All words can have basic translation questions
        possible_types.append("translation_to_target")
        possible_types.append("translation_to_english")
        
        # Image recognition requires an image
        if word.get('image_path') and os.path.exists(word.get('image_path', '')):
            possible_types.append("image_recognition")
        
        # Category matching requires a category
        if word.get('category') and word['category'] != "other" and word['category'] != "manual":
            possible_types.append("category_match")
            
            # Find if there are other words in the same category
            same_category_words = [w for w in vocabulary 
                                if w.get('category') == word.get('category') and w['id'] != word['id']]
            if len(same_category_words) >= 2:  # Need at least 3 total including our word
                possible_types.append("multiple_choice_category")
        
        # Sentence completion is always possible
        possible_types.append("sentence_completion")
        
        # Audio recognition is always possible
        possible_types.append("audio_recognition")
        
        return possible_types
    
    def setup_new_question(self, vocabulary, languages, current_question_num=0, total_questions=5):
        """Set up a new quiz question with enhanced functionality."""
        if not vocabulary or len(vocabulary) < 4:
            return False
        
        # Store total questions for progress tracking
        st.session_state.total_quiz_questions = total_questions
        st.session_state.current_question_num = current_question_num + 1
        
        # Select a random word as the question
        st.session_state.current_quiz_word = random.choice(vocabulary)
        word = st.session_state.current_quiz_word
        
        # Determine what question types are possible with this word
        possible_types = self.get_possible_question_types(word, vocabulary)
        
        # Choose a question type based on question number to ensure variety
        # For first few questions, use simpler types to ease the user in
        if current_question_num == 0:
            question_type = "translation_to_target"  # Start with basic translation
        elif current_question_num == 1 and "translation_to_english" in possible_types:
            question_type = "translation_to_english"  # Second question type
        else:
            # For subsequent questions, choose randomly from possible types
            question_type = random.choice(possible_types)
        
        # Store the question type
        st.session_state.current_question_type = question_type
        
        # Create options based on question type
        if question_type == "translation_to_target":
            # Translation question: English → Target language
            self.setup_translation_question(word, vocabulary, languages, to_english=False)
        
        elif question_type == "translation_to_english":
            # Reverse translation: Target language → English
            self.setup_translation_question(word, vocabulary, languages, to_english=True)
        
        elif question_type == "image_recognition":
            # Image recognition: Show image, select correct word
            self.setup_image_recognition_question(word, vocabulary, languages)
        
        elif question_type == "category_match":
            # Category matching: Match word to correct category
            self.setup_category_match_question(word, vocabulary, languages)
        
        elif question_type == "sentence_completion":
            # Sentence completion: Fill in blank in sentence
            self.setup_sentence_completion_question(word, vocabulary, languages)
        
        elif question_type == "multiple_choice_category":
            # Choose words from same category
            self.setup_multiple_choice_category_question(word, vocabulary, languages)
        
        elif question_type == "audio_recognition":
            # Audio recognition: Hear word, select correct option
            self.setup_audio_recognition_question(word, vocabulary, languages)
        
        # Set up question elements
        st.session_state.answered = False
        st.session_state.feedback_message = ""
        st.session_state.selected_option = None
        
        return True
    
    def setup_translation_question(self, word, vocabulary, languages, to_english=False):
        """Set up a basic translation question."""
        if to_english:
            # Target language → English
            st.session_state.question_text = f"What is the English translation of '{word['word_translated']}'?"
            
            # Create options (3 wrong + 1 correct)
            options = [{'id': word['id'], 'word': word['word_original']}]
            
            # Find wrong options with different original words
            remaining_vocab = [w for w in vocabulary if w['id'] != word['id']]
            while len(options) < 4 and remaining_vocab:
                wrong_option = random.choice(remaining_vocab)
                remaining_vocab.remove(wrong_option)
                
                if not any(o['word'] == wrong_option['word_original'] for o in options):
                    options.append({
                        'id': wrong_option['id'], 
                        'word': wrong_option['word_original']
                    })
        else:
            # English → Target language
            st.session_state.question_text = f"What is the {self.get_language_name_from_code(word['language_translated'], languages)} translation of '{word['word_original']}'?"
            
            # Create options (3 wrong + 1 correct)
            options = [{'id': word['id'], 'word': word['word_translated']}]
            
            # Find wrong options (preferably in the same language)
            same_lang_vocab = [w for w in vocabulary 
                            if w['id'] != word['id'] and w['language_translated'] == word['language_translated']]
            
            # If not enough words in the same language, use words from any language
            if len(same_lang_vocab) < 3:
                remaining_vocab = [w for w in vocabulary if w['id'] != word['id']]
            else:
                remaining_vocab = same_lang_vocab
                
            while len(options) < 4 and remaining_vocab:
                wrong_option = random.choice(remaining_vocab)
                remaining_vocab.remove(wrong_option)
                
                if not any(o['word'] == wrong_option['word_translated'] for o in options):
                    options.append({
                        'id': wrong_option['id'], 
                        'word': wrong_option['word_translated']
                    })
        
        # Shuffle options
        random.shuffle(options)
        st.session_state.quiz_options = options
    
    def setup_image_recognition_question(self, word, vocabulary, languages):
        """Set up a question that shows an image and asks for the correct word."""
        st.session_state.question_text = f"What is this object called in {self.get_language_name_from_code(word['language_translated'], languages)}?"
        
        # Create options (3 wrong + 1 correct)
        options = [{'id': word['id'], 'word': word['word_translated']}]
        
        # Find wrong options in the same language
        same_lang_vocab = [w for w in vocabulary 
                        if w['id'] != word['id'] and w['language_translated'] == word['language_translated']]
        
        # If not enough words in the same language, use words from any language
        if len(same_lang_vocab) < 3:
            remaining_vocab = [w for w in vocabulary if w['id'] != word['id']]
        else:
            remaining_vocab = same_lang_vocab
            
        while len(options) < 4 and remaining_vocab:
            wrong_option = random.choice(remaining_vocab)
            remaining_vocab.remove(wrong_option)
            
            if not any(o['word'] == wrong_option['word_translated'] for o in options):
                options.append({
                    'id': wrong_option['id'], 
                    'word': wrong_option['word_translated']
                })
        
        # Shuffle options
        random.shuffle(options)
        st.session_state.quiz_options = options
    
    def setup_category_match_question(self, word, vocabulary, languages):
        """Set up a question to match a word to its category."""
        st.session_state.question_text = f"Which category does the word '{word['word_original']}' ({word['word_translated']}) belong to?"
        
        # Correct category
        correct_category = word.get('category', 'other')
        
        # Get all possible categories from vocabulary
        all_categories = set(w.get('category', 'other') for w in vocabulary 
                            if w.get('category') and w.get('category') != 'manual')
        
        # Remove the correct category
        if correct_category in all_categories:
            all_categories.remove(correct_category)
        
        # Create options (3 wrong + 1 correct)
        options = [{'id': word['id'], 'word': correct_category}]
        
        # Add wrong categories
        all_categories = list(all_categories)
        while len(options) < 4 and all_categories:
            wrong_category = random.choice(all_categories)
            all_categories.remove(wrong_category)
            
            options.append({
                'id': f"wrong_{len(options)}", 
                'word': wrong_category
            })
        
        # If we don't have enough categories, add some generic ones
        generic_categories = ["clothing", "colors", "emotions", "weather", "family", "places"]
        while len(options) < 4:
            for cat in generic_categories:
                if cat not in [o['word'] for o in options]:
                    options.append({
                        'id': f"wrong_{len(options)}", 
                        'word': cat
                    })
                    break
        
        # Shuffle options
        random.shuffle(options)
        st.session_state.quiz_options = options
    
    def setup_sentence_completion_question(self, word, vocabulary, languages):
        """Set up a sentence completion question."""
        # Create a sentence with the word
        if word.get('word_original'):
            # Get an example sentence
            example = self.get_example_sentence(word['word_original'], word['language_translated'])
            
            if example and example.get('translated'):
                # Use the translated example
                full_sentence = example['translated']
                
                # Replace the word with a blank
                blank_sentence = full_sentence.replace(word['word_translated'], "______")
                
                if blank_sentence != full_sentence:  # Only if replacement worked
                    st.session_state.question_text = "Complete this sentence: " + blank_sentence
                    
                    # Create options (3 wrong + 1 correct)
                    options = [{'id': word['id'], 'word': word['word_translated']}]
                    
                    # Find wrong options in the same language
                    same_lang_vocab = [w for w in vocabulary 
                                    if w['id'] != word['id'] and w['language_translated'] == word['language_translated']]
                    
                    # If not enough words, use words from any language
                    if len(same_lang_vocab) < 3:
                        remaining_vocab = [w for w in vocabulary if w['id'] != word['id']]
                    else:
                        remaining_vocab = same_lang_vocab
                        
                    while len(options) < 4 and remaining_vocab:
                        wrong_option = random.choice(remaining_vocab)
                        remaining_vocab.remove(wrong_option)
                        
                        if not any(o['word'] == wrong_option['word_translated'] for o in options):
                            options.append({
                                'id': wrong_option['id'], 
                                'word': wrong_option['word_translated']
                            })
                    
                    # Shuffle options
                    random.shuffle(options)
                    st.session_state.quiz_options = options
                    return
        
        # Fallback to basic translation if sentence creation failed
        self.setup_translation_question(word, vocabulary, languages, to_english=False)
    
    def setup_multiple_choice_category_question(self, word, vocabulary, languages):
        """Set up a question to identify words in the same category."""
        category = word.get('category', 'other')
        
        # Get all words in the same category
        same_category_words = [w for w in vocabulary 
                            if w.get('category') == category and w['id'] != word['id']]
        
        if len(same_category_words) < 2:
            # Not enough words in this category, fall back to translation
            self.setup_translation_question(word, vocabulary, languages, to_english=False)
            return
        
        # Select a random word from the same category
        selected_cat_word = random.choice(same_category_words)
        same_category_words.remove(selected_cat_word)
        
        st.session_state.question_text = f"Which word belongs to the same category as '{word['word_original']}' and '{selected_cat_word['word_original']}'?"
        
        # Create options with 1 correct (same category) and 3 wrong (different categories)
        if same_category_words:
            correct_option = random.choice(same_category_words)
            options = [{'id': correct_option['id'], 'word': correct_option['word_original']}]
        else:
            # No more words in the same category, fall back to translation
            self.setup_translation_question(word, vocabulary, languages, to_english=False)
            return
        
        # Get words from different categories
        diff_category_words = [w for w in vocabulary 
                            if w.get('category') != category and w.get('category') and w.get('category') != 'manual']
        
        # Add wrong options
        while len(options) < 4 and diff_category_words:
            wrong_option = random.choice(diff_category_words)
            diff_category_words.remove(wrong_option)
            
            if not any(o['word'] == wrong_option['word_original'] for o in options):
                options.append({
                    'id': wrong_option['id'], 
                    'word': wrong_option['word_original']
                })
        
        # Shuffle options
        random.shuffle(options)
        st.session_state.quiz_options = options
    
    def setup_audio_recognition_question(self, word, vocabulary, languages):
        """Set up an audio recognition question."""
        st.session_state.question_text = "Listen to the pronunciation and select the correct word:"
        
        # Generate audio for the word
        audio_bytes = self.text_to_speech(word['word_translated'], word['language_translated'])
        st.session_state.question_audio = audio_bytes
        
        # Create options (3 wrong + 1 correct)
        options = [{'id': word['id'], 'word': word['word_translated']}]
        
        # Find wrong options in the same language
        same_lang_vocab = [w for w in vocabulary 
                        if w['id'] != word['id'] and w['language_translated'] == word['language_translated']]
        
        # If not enough words in the same language, use words from any language
        if len(same_lang_vocab) < 3:
            remaining_vocab = [w for w in vocabulary if w['id'] != word['id']]
        else:
            remaining_vocab = same_lang_vocab
            
        while len(options) < 4 and remaining_vocab:
            wrong_option = random.choice(remaining_vocab)
            remaining_vocab.remove(wrong_option)
            
            if not any(o['word'] == wrong_option['word_translated'] for o in options):
                options.append({
                    'id': wrong_option['id'], 
                    'word': wrong_option['word_translated']
                })
        
        # Shuffle options
        random.shuffle(options)
        st.session_state.quiz_options = options
    
    def check_answer(self, selected_index, languages, gamification=None):
        """Check if selected quiz answer is correct with enhanced feedback."""
        if st.session_state.answered:
            return
        
        selected_option = st.session_state.quiz_options[selected_index]
        st.session_state.selected_option = selected_option
        
        word = st.session_state.current_quiz_word
        question_type = st.session_state.current_question_type
        
        # Determine if answer is correct based on question type
        is_correct = False
        
        if question_type in ["translation_to_target", "translation_to_english", "image_recognition", "audio_recognition"]:
            # For translation and recognition questions, check if IDs match
            is_correct = selected_option['id'] == word['id']
        
        elif question_type == "category_match":
            # For category matching, check if selected category matches word category
            is_correct = selected_option['word'] == word.get('category', 'other')
        
        elif question_type == "sentence_completion":
            # For sentence completion, check if selected word matches target word
            is_correct = selected_option['word'] == word['word_translated']
        
        elif question_type == "multiple_choice_category":
            # For multiple choice category, check if the selected word has the same category
            # Get the vocab item for the selected word
            selected_word = next((w for w in st.session_state.vocabulary if w['id'] == selected_option['id']), None)
            is_correct = selected_word and selected_word.get('category') == word.get('category')
        
        # Update database
        self.update_word_progress_direct(word['id'], is_correct)
        
        # Update session stats
        st.session_state.words_studied += 1
        if is_correct:
            st.session_state.words_learned += 1
            st.session_state.quiz_score += 1
        
        st.session_state.quiz_total += 1
        st.session_state.answered = True
        
        # Create detailed feedback based on question type
        if is_correct:
            feedback = "Correct! "
        else:
            feedback = "Not quite. "
        
        if question_type == "translation_to_target":
            if is_correct:
                feedback += f"'{word['word_original']}' translates to '{word['word_translated']}' in {self.get_language_name_from_code(word['language_translated'], languages)}."
            else:
                correct_option = next((o for o in st.session_state.quiz_options if o['id'] == word['id']), None)
                if correct_option:
                    feedback += f"You selected '{selected_option['word']}', but the correct translation of '{word['word_original']}' is '{correct_option['word']}'."
                else:
                    feedback += f"The correct translation of '{word['word_original']}' is '{word['word_translated']}'."
        
        elif question_type == "translation_to_english":
            if is_correct:
                feedback += f"'{word['word_translated']}' in {self.get_language_name_from_code(word['language_translated'], languages)} translates to '{word['word_original']}' in English."
            else:
                correct_option = next((o for o in st.session_state.quiz_options if o['id'] == word['id']), None)
                if correct_option:
                    feedback += f"You selected '{selected_option['word']}', but the correct English translation of '{word['word_translated']}' is '{correct_option['word']}'."
                else:
                    feedback += f"The correct English translation of '{word['word_translated']}' is '{word['word_original']}'."
        
        elif question_type == "image_recognition":
            if is_correct:
                feedback += f"The image shows '{word['word_original']}' which is '{word['word_translated']}' in {self.get_language_name_from_code(word['language_translated'], languages)}."
            else:
                correct_option = next((o for o in st.session_state.quiz_options if o['id'] == word['id']), None)
                if correct_option:
                    feedback += f"You selected '{selected_option['word']}', but the image shows '{word['word_original']}' which is '{correct_option['word']}' in {self.get_language_name_from_code(word['language_translated'], languages)}."
                else:
                    feedback += f"The image shows '{word['word_original']}' which is '{word['word_translated']}' in {self.get_language_name_from_code(word['language_translated'], languages)}."
        
        elif question_type == "category_match":
            if is_correct:
                feedback += f"'{word['word_original']}' ({word['word_translated']}) belongs to the category '{word['category']}'."
            else:
                feedback += f"You selected '{selected_option['word']}', but '{word['word_original']}' ({word['word_translated']}) belongs to the category '{word['category']}'."
        
        elif question_type == "sentence_completion":
            if is_correct:
                feedback += f"The correct word to complete the sentence is '{word['word_translated']}'."
            else:
                feedback += f"You selected '{selected_option['word']}', but the correct word to complete the sentence is '{word['word_translated']}'."
        
        elif question_type == "multiple_choice_category":
            if is_correct:
                feedback += f"Correct! '{selected_option['word']}' belongs to the same category ('{word['category']}') as '{word['word_original']}'."
            else:
                selected_word = next((w for w in st.session_state.vocabulary if w['id'] == selected_option['id']), None)
                if selected_word:
                    feedback += f"You selected '{selected_option['word']}' (category: '{selected_word.get('category', 'unknown')}'), but it's not in the '{word['category']}' category."
                else:
                    feedback += f"The words are in the '{word['category']}' category."
        
        elif question_type == "audio_recognition":
            if is_correct:
                feedback += f"The audio was saying '{word['word_translated']}', which means '{word['word_original']}' in English."
            else:
                correct_option = next((o for o in st.session_state.quiz_options if o['id'] == word['id']), None)
                if correct_option:
                    feedback += f"You selected '{selected_option['word']}', but the audio was saying '{correct_option['word']}', which means '{word['word_original']}' in English."
                else:
                    feedback += f"The audio was saying '{word['word_translated']}', which means '{word['word_original']}' in English."
        
        # Store feedback
        st.session_state.feedback_message = feedback
        
        # Check if any challenges are completed - with error handling
        if gamification:
            try:
                gamification.check_challenge_progress(
                    quiz_score=st.session_state.quiz_score,
                    quiz_total=st.session_state.quiz_total
                )
                
                # Check for quiz-related achievements
                if st.session_state.quiz_total >= 5:  # Only check if quiz is substantial
                    gamification.check_achievements(
                        "quiz_completed",
                        score=st.session_state.quiz_score,
                        total=st.session_state.quiz_total
                    )
            except Exception as e:
                print(f"Gamification error in check_answer: {e}")
        
        return is_correct
    
    def display_quiz_question(self, languages, manage_session=None):
        """Display the current quiz question with enhanced UI."""
        word = st.session_state.current_quiz_word
        question_type = st.session_state.current_question_type
        
        # Display progress
        current_q = st.session_state.current_question_num
        total_q = st.session_state.total_quiz_questions
        progress = current_q / total_q
        
        st.progress(progress)
        st.markdown(f"**Question {current_q}/{total_q}**")
        
        # Display question text
        st.markdown(f"## {st.session_state.question_text}")
        
        # Display additional elements based on question type
        if question_type == "image_recognition" and word['image_path'] and os.path.exists(word['image_path']):
            try:
                image = Image.open(word['image_path'])
                st.image(image, caption="Identify this object", width=300)
            except Exception as e:
                st.error(f"Error displaying image: {e}")
        
        elif question_type == "audio_recognition" and hasattr(st.session_state, 'question_audio'):
            try:
                st.markdown("🔊 **Listen to the word:**")
                st.markdown(self.get_audio_html(st.session_state.question_audio), unsafe_allow_html=True)
                st.markdown("*Click the play button to hear the word*")
            except Exception as e:
                st.error(f"Error playing audio: {e}")
        
        # Create answer buttons with custom styling
        cols = st.columns(len(st.session_state.quiz_options))
        for i, option in enumerate(st.session_state.quiz_options):
            with cols[i]:
                # Format the display text based on question type
                if question_type == "category_match" or question_type == "multiple_choice_category":
                    display_text = option['word'].title()  # Capitalize category names
                else:
                    display_text = option['word']
                
                # Determine button appearance based on answer status
                if st.session_state.answered:
                    is_selected = i == st.session_state.quiz_options.index(st.session_state.selected_option)
                    is_correct_option = (
                        (question_type in ["translation_to_target", "translation_to_english", "image_recognition", "audio_recognition"] 
                        and option['id'] == word['id']) or
                        (question_type == "category_match" and option['word'] == word.get('category', 'other')) or
                        (question_type == "sentence_completion" and option['word'] == word['word_translated']) or
                        (question_type == "multiple_choice_category" and 
                        next((w for w in st.session_state.vocabulary if w['id'] == option['id']), {}).get('category') == word.get('category'))
                    )
                    
                    if is_correct_option:
                        if is_selected:
                            # Correct and selected - success with checkmark
                            st.success(f"✓ {display_text}")
                        else:
                            # Correct but not selected - just success
                            st.success(display_text)
                    else:
                        if is_selected:
                            # Wrong and selected - error with X
                            st.error(f"✗ {display_text}")
                        else:
                            # Wrong and not selected - neutral
                            st.button(display_text, key=f"option_{i}", disabled=True)
                else:
                    # Not answered yet - clickable button
                    if st.button(display_text, key=f"option_{i}"):
                        is_correct = self.check_answer(i, languages, st.session_state.get('gamification', None))
                        # Force rerun to update UI
                        st.rerun()
        
        # Display feedback after answering
        if st.session_state.answered:
            # Display feedback message
            if st.session_state.feedback_message:
                feedback_type = "success" if st.session_state.quiz_score > st.session_state.quiz_total - 1 else "error"
                if feedback_type == "success":
                    st.success(st.session_state.feedback_message)
                else:
                    st.error(st.session_state.feedback_message)
            
            # Pronunciation section
            if question_type != "audio_recognition":  # Don't show if already played
                st.markdown("### Pronunciation:")
                audio_bytes = self.text_to_speech(word['word_translated'], word['language_translated'])
                if audio_bytes:
                    st.markdown(self.get_audio_html(audio_bytes), unsafe_allow_html=True)
            
            # Add pronunciation tips
            pronunciation_tips = self.get_pronunciation_guide(word['word_translated'], word['language_translated'])
            if pronunciation_tips:
                st.markdown("**Tips:**")
                for tip in pronunciation_tips:
                    st.markdown(f"- {tip}")
            
            # Display an example if not already shown
            if question_type != "sentence_completion":
                example = self.get_example_sentence(word['word_original'], word['language_translated'])
                if example and example.get('translated'):
                    st.markdown("**Example usage:**")
                    st.markdown(f"**English:** {example['english']}")
                    st.markdown(f"**{self.get_language_name_from_code(word['language_translated'], languages)}:** {example['translated']}")
            
            # Next question or finish quiz button
            if st.session_state.current_question_num < st.session_state.total_quiz_questions:
                if st.button("Next Question", type="primary"):
                    # Setup next question
                    self.setup_new_question(
                        st.session_state.vocabulary, 
                        languages,
                        current_question_num=st.session_state.current_question_num,
                        total_questions=st.session_state.total_quiz_questions
                    )
                    st.rerun()
            else:
                # Finish quiz button
                if st.button("Finish Quiz", type="primary"):
                    st.session_state.current_quiz_word = None
                    st.session_state.quiz_options = []
                    st.session_state.quiz_completed = True
                    # End session
                    if st.session_state.session_id and manage_session:
                        manage_session("end")
                    st.rerun()
    
    def display_quiz_results(self):
        """Display enhanced quiz results."""
        st.success(f"Quiz completed! Your score: {st.session_state.quiz_score}/{st.session_state.quiz_total}")
        
        # Calculate and display accuracy
        accuracy = (st.session_state.quiz_score / st.session_state.quiz_total) * 100
        
        # Create a gauge-like visualization
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            # Create a pyplot figure for the gauge
            fig, ax = plt.subplots(figsize=(8, 4))
            
            # Define colors based on accuracy
            if accuracy >= 90:
                color = '#2ecc71'  # Green
            elif accuracy >= 70:
                color = '#f39c12'  # Orange
            else:
                color = '#e74c3c'  # Red
            
            # Create a simple gauge
            ax.barh(0, accuracy, height=0.5, color=color)
            ax.barh(0, 100, height=0.5, color='#ecf0f1', alpha=0.3)
            
            # Add the percentage text
            ax.text(accuracy/2, 0, f"{accuracy:.1f}%", ha='center', va='center', fontsize=20, color='white', fontweight='bold')
            
            # Remove axes and set limits
            ax.set_xlim(0, 100)
            ax.set_ylim(-0.5, 0.5)
            ax.axis('off')
            
            # Add title
            ax.set_title('Quiz Accuracy', fontsize=16, pad=20)
            
            # Display the figure
            st.pyplot(fig)
        
        # Display feedback based on score
        if accuracy >= 90:
            st.balloons()
            st.markdown("### 🎖️ Excellent job! You're mastering these words!")
            st.markdown("Your fluency is improving rapidly! Keep up the great work.")
        elif accuracy >= 70:
            st.markdown("### 👍 Good work! Keep practicing to improve.")
            st.markdown("You're making solid progress. A bit more practice will help these words stick.")
        else:
            st.markdown("### 📚 Keep practicing! You're on the right track.")
            st.markdown("Learning a language takes time. Try reviewing these words again tomorrow.")
        
        # Show learning suggestions
        st.subheader("Learning Tips")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Based on your quiz results:**")
            if accuracy < 70:
                st.markdown("- Try shorter, more frequent practice sessions")
                st.markdown("- Focus on one category of words at a time")
                st.markdown("- Use the audio pronunciation feature more often")
            else:
                st.markdown("- Challenge yourself with sentence completion")
                st.markdown("- Try learning related words in groups")
                st.markdown("- Practice your pronunciation by speaking aloud")
        
        with col2:
            st.markdown("**Next steps:**")
            st.markdown("- Review any words you missed")
            st.markdown("- Try a different quiz format tomorrow")
            st.markdown("- Add more words to your vocabulary")
        
        # Add a share/save results option
        st.markdown("---")
        st.markdown("**Save your progress:**")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Practice these words again"):
                # Start a new quiz with the same settings
                st.session_state.quiz_score = 0
                st.session_state.quiz_total = 0
                st.session_state.quiz_completed = False
                st.rerun()
        
        with col2:
            if st.button("Try a different language"):
                # Reset quiz state completely
                st.session_state.current_quiz_word = None
                st.session_state.quiz_options = []
                st.session_state.quiz_score = 0
                st.session_state.quiz_total = 0
                st.session_state.quiz_completed = False
                st.rerun()
    
    def start_new_quiz(self, vocabulary, languages, num_questions=5, manage_session=None):
        """Start a new quiz with enhanced functionality."""
        # Reset quiz state
        st.session_state.quiz_score = 0
        st.session_state.quiz_total = 0
        st.session_state.answered = False
        st.session_state.quiz_completed = False
        st.session_state.vocabulary = vocabulary  # Store for later use
        st.session_state.feedback_message = ""
        
        if not vocabulary or len(vocabulary) < 4:
            st.warning("Not enough vocabulary words for a quiz (need at least 4).")
            return False
        
        # Start a new session if needed
        if not st.session_state.session_id and manage_session:
            st.session_state.session_id = manage_session("start")
            st.session_state.words_studied = 0
            st.session_state.words_learned = 0
        
        # Set up first question
        self.setup_new_question(vocabulary, languages, 0, num_questions)
        return True