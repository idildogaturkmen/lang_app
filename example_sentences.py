"""
Example Sentences Generator
--------------------------
This module provides functions to generate natural example sentences
using multiple dictionary APIs with fallback options.
"""

import requests
import os
import json
import numpy as np
import time
from functools import lru_cache

class ExampleSentenceGenerator:
    def __init__(self, translate_func=None):
        """
        Initialize the example sentence generator.
        
        Args:
            translate_func: A function that takes (text, target_language) and returns translated text
        """
        self.translate_func = translate_func
        self.setup_cache_dir()
        
    def setup_cache_dir(self):
        """Create cache directory if it doesn't exist."""
        self.cache_dir = "sentence_cache"
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def get_example_sentence(self, word, target_language, category=None):
        """
        Get an example sentence for a word with translation.
        
        Args:
            word: The word to get an example for
            target_language: The language code to translate to
            category: Optional category hint for better templates
            
        Returns:
            dict: {"english": "Example sentence.", "translated": "Translated example.", "source": "api_name"}
        """
        try:
            # Clean the word (remove any trailing punctuation, etc.)
            word = word.strip().lower()
            
            # First check cache
            cached_example = self._get_cached_example(word, target_language)
            if cached_example:
                return cached_example
            
            # Try API methods in sequence
            methods = [
                self._get_free_dictionary_example,
                self._get_wordnik_example,
                self._get_owlbot_example,
                self._get_template_example
            ]
            
            for method in methods:
                example = method(word, target_language, category)
                if example and example["english"]:
                    # Store in cache if it's from an API (not template)
                    if "template" not in example.get("source", ""):
                        self._cache_example(word, target_language, example)
                    return example
            
            # If all methods fail, use a simple template
            return {
                "english": f"This is a {word}.",
                "translated": self._translate(f"This is a {word}.", target_language),
                "source": "simple_template"
            }
            
        except Exception as e:
            print(f"Error getting example sentence: {e}")
            # Ultimate fallback
            return {
                "english": f"The {word}.",
                "translated": "",
                "source": "error_fallback"
            }
    
    def _translate(self, text, target_language):
        """Translate text using the provided translation function."""
        if not text:
            return ""
            
        if self.translate_func:
            try:
                return self.translate_func(text, target_language)
            except Exception as e:
                print(f"Translation error: {e}")
                return ""
        else:
            return f"[Translation to {target_language}]"
    
    @lru_cache(maxsize=300)
    def _get_cached_example(self, word, target_language):
        """Get cached example if available."""
        cache_file = f"{self.cache_dir}/{word}_{target_language}.json"
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Cache read error: {e}")
        return None
    
    def _cache_example(self, word, target_language, example):
        """Cache an example for future use."""
        cache_file = f"{self.cache_dir}/{word}_{target_language}.json"
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(example, f, ensure_ascii=False)
        except Exception as e:
            print(f"Cache write error: {e}")
    
    def _get_free_dictionary_example(self, word, target_language, category=None):
        """Get example from Free Dictionary API."""
        try:
            url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}"
            
            response = requests.get(url, timeout=5)
            if response.status_code != 200:
                return None
                
            data = response.json()
            
            # Extract example sentences
            examples = []
            if isinstance(data, list) and len(data) > 0:
                for meaning in data[0].get('meanings', []):
                    for definition in meaning.get('definitions', []):
                        if 'example' in definition and definition['example']:
                            examples.append(definition['example'])
            
            if examples:
                # Select example with appropriate length
                good_examples = [ex for ex in examples if 4 <= len(ex.split()) <= 12]
                if good_examples:
                    english_example = np.random.choice(good_examples)
                else:
                    english_example = np.random.choice(examples)
                
                # Translate to target language
                translated_example = self._translate(english_example, target_language)
                
                return {
                    "english": english_example,
                    "translated": translated_example,
                    "source": "free_dictionary_api"
                }
            
            return None
        except Exception as e:
            print(f"Free Dictionary API error: {e}")
            return None
    
    def _get_wordnik_example(self, word, target_language, category=None):
        """Get example from Wordnik API (no key needed for this endpoint)."""
        try:
            # This is a public endpoint that doesn't require API key
            url = f"https://api.wordnik.com/v4/word.json/{word}/examples"
            
            response = requests.get(url, timeout=5)
            if response.status_code != 200:
                return None
                
            data = response.json()
            
            examples = []
            for example in data.get('examples', []):
                if 'text' in example and example['text']:
                    text = example['text']
                    # Clean up the text
                    text = text.replace('"', '').replace('"', '')
                    examples.append(text)
            
            if examples:
                # Filter for good examples (contains the word, proper length)
                good_examples = [ex for ex in examples 
                               if word.lower() in ex.lower() 
                               and 5 <= len(ex.split()) <= 12]
                
                if good_examples:
                    english_example = np.random.choice(good_examples)
                elif examples:
                    english_example = np.random.choice(examples)
                else:
                    return None
                
                # Translate to target language
                translated_example = self._translate(english_example, target_language)
                
                return {
                    "english": english_example,
                    "translated": translated_example,
                    "source": "wordnik_api"
                }
            
            return None
        except Exception as e:
            print(f"Wordnik API error: {e}")
            return None
    
    def _get_owlbot_example(self, word, target_language, category=None):
        """Get example from Owlbot API (tries the public endpoint)."""
        try:
            # Check if API key is available
            owl_api_key = os.environ.get("OWLBOT_KEY")
            if not owl_api_key:
                try:
                    import streamlit as st
                    owl_api_key = st.secrets.get("owlbot", {}).get("key")
                except:
                    pass
            
            # If we have an API key, use the authenticated endpoint
            if owl_api_key:
                headers = {"Authorization": f"Token {owl_api_key}"}
                url = f"https://owlbot.info/api/v4/dictionary/{word}"
                
                response = requests.get(url, headers=headers, timeout=5)
                if response.status_code != 200:
                    return None
                    
                data = response.json()
                
                examples = []
                for definition in data.get('definitions', []):
                    if 'example' in definition and definition['example']:
                        examples.append(definition['example'])
                
                if examples:
                    english_example = np.random.choice(examples)
                    translated_example = self._translate(english_example, target_language)
                    
                    return {
                        "english": english_example,
                        "translated": translated_example,
                        "source": "owlbot_api"
                    }
            
            return None
        except Exception as e:
            print(f"Owlbot API error: {e}")
            return None
    
    def _get_template_example(self, word, target_language, category=None):
        """Generate an example using templates based on word category."""
        try:
            # Use provided category or try to infer it
            word_category = category or self._infer_category(word)
            
            # Category-specific templates
            templates = {
                # Objects (default)
                "default": [
                    f"I use this {word} every day.",
                    f"The {word} is in the kitchen.",
                    f"Can you pass me that {word}?",
                    f"This {word} is very useful.",
                    f"Where did you find this {word}?"
                ],
                
                # People
                "person": [
                    f"The {word} is talking to John.",
                    f"I need to call the {word} today.",
                    f"My sister is a {word}.",
                    f"We met a friendly {word} yesterday.",
                    f"The {word} helped us find our way."
                ],
                
                # Food
                "food": [
                    f"This {word} tastes amazing.",
                    f"I love eating {word} for lunch.",
                    f"Would you like some {word}?",
                    f"My mother makes delicious {word}.",
                    f"They serve good {word} here."
                ],
                
                # Animals
                "animal": [
                    f"The {word} is running in the park.",
                    f"My neighbor has a {word} as a pet.",
                    f"We saw a {word} at the zoo.",
                    f"The {word} is sleeping under the tree.",
                    f"That {word} is very friendly."
                ],
                
                # Clothing
                "clothing": [
                    f"I'm wearing a new {word} today.",
                    f"This {word} is very comfortable.",
                    f"My brother bought a red {word}.",
                    f"Where did you get that {word}?",
                    f"This {word} is too big for me."
                ],
                
                # Vehicles
                "vehicle": [
                    f"The {word} is parked outside.",
                    f"My dad drives a {word} to work.",
                    f"We rented a {word} for our trip.",
                    f"The {word} is very fast.",
                    f"That blue {word} belongs to my neighbor."
                ],
                
                # Electronics
                "electronics": [
                    f"My new {word} works perfectly.",
                    f"This {word} needs charging.",
                    f"I use my {word} every day.",
                    f"Can you fix this {word}?",
                    f"The {word} is very expensive."
                ]
            }
            
            # Get templates for this category
            category_templates = templates.get(word_category, templates["default"])
            
            # Select a random template
            english_example = np.random.choice(category_templates)
            
            # Translate
            translated_example = self._translate(english_example, target_language)
            
            return {
                "english": english_example,
                "translated": translated_example,
                "source": "template_" + word_category
            }
        except Exception as e:
            print(f"Template error: {e}")
            return None
    
    def _infer_category(self, word):
        """Try to infer word category for better templates."""
        # This is a simple version - you can expand with more categories
        
        # Person detection
        person_words = ["doctor", "teacher", "student", "chef", "driver", "worker", 
                        "artist", "child", "man", "woman", "friend", "parent"]
        
        if word.lower() in person_words:
            return "person"
        
        # Food detection
        food_words = ["apple", "banana", "bread", "cake", "chicken", "coffee", 
                      "pizza", "sandwich", "fruit", "vegetable", "meat", "fish"]
        
        if word.lower() in food_words:
            return "food"
        
        # Animal detection
        animal_words = ["dog", "cat", "bird", "fish", "horse", "cow", "elephant",
                       "lion", "tiger", "bear", "rabbit", "monkey"]
        
        if word.lower() in animal_words:
            return "animal"
        
        # Vehicle detection
        vehicle_words = ["car", "bus", "train", "bike", "bicycle", "motorcycle",
                        "truck", "boat", "ship", "plane", "airplane"]
        
        if word.lower() in vehicle_words:
            return "vehicle"
        
        # Electronics detection
        electronics_words = ["phone", "computer", "laptop", "tablet", "camera",
                           "television", "tv", "radio", "speaker", "headphones"]
        
        if word.lower() in electronics_words:
            return "electronics"
        
        # Clothing detection
        clothing_words = ["shirt", "pants", "dress", "jacket", "coat", "hat",
                         "shoes", "boots", "socks", "gloves", "scarf"]
        
        if word.lower() in clothing_words:
            return "clothing"
        
        # Default category
        return "default"