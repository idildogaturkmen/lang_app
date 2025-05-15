"""
Enhanced Example Sentences Generator
-----------------------------------
Generates natural, contextually appropriate example sentences for vocabulary words
with semantic awareness to avoid awkward or inappropriate examples.
"""

import requests
import os
import json
import numpy as np
import time
import re
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
        self._setup_semantic_knowledge()
        
    def setup_cache_dir(self):
        """Create cache directory if it doesn't exist."""
        self.cache_dir = "sentence_cache"
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def _setup_semantic_knowledge(self):
        """Set up semantic knowledge bases for better sentence generation."""
        # Words representing people/occupations
        self.person_words = set([
            "person", "man", "woman", "boy", "girl", "child", "baby", "adult", "teenager",
            "doctor", "teacher", "student", "chef", "driver", "worker", "engineer", "artist", 
            "musician", "athlete", "player", "friend", "parent", "mother", "father", "brother", 
            "sister", "aunt", "uncle", "cousin", "grandfather", "grandmother", "neighbor",
            "lawyer", "nurse", "dentist", "professor", "scientist", "farmer", "writer"
        ])
        
        # Uncountable nouns
        self.uncountable_nouns = set([
            "water", "air", "money", "food", "rice", "coffee", "tea", "milk", "bread", 
            "sugar", "salt", "oil", "vinegar", "soup", "advice", "information", "news", 
            "furniture", "luggage", "traffic", "weather", "homework", "work", "fun",
            "knowledge", "wisdom", "happiness", "sadness", "anger", "fear", "courage",
            "patience", "intelligence", "research", "education", "health", "safety", "time"
        ])
        
        # Abstract concepts
        self.abstract_concepts = set([
            "love", "peace", "happiness", "freedom", "justice", "beauty", "truth", "knowledge",
            "wisdom", "courage", "hope", "faith", "democracy", "imagination", "creativity",
            "innovation", "progress", "development", "success", "failure", "equality",
            "honor", "respect", "kindness", "patience", "understanding", "friendship", "greed"
        ])
        
        # Plural forms that need special handling
        self.plural_forms = {
            "people": "person", "men": "man", "women": "woman", "children": "child",
            "mice": "mouse", "feet": "foot", "teeth": "tooth", "geese": "goose"
        }
        
        # Common verbs for different categories
        self.category_verbs = {
            "person": ["talking to", "helping", "meeting", "calling", "visiting", "working with"],
            "food": ["eating", "cooking", "tasting", "ordering", "enjoying", "serving"],
            "animal": ["watching", "feeding", "petting", "seeing", "hearing", "photographing"],
            "clothing": ["wearing", "buying", "trying on", "washing", "ironing", "folding"],
            "vehicle": ["driving", "riding", "renting", "parking", "washing", "fixing"],
            "electronics": ["using", "charging", "repairing", "upgrading", "installing", "configuring"],
            "furniture": ["moving", "arranging", "cleaning", "dusting", "polishing", "replacing"],
            "place": ["visiting", "going to", "exploring", "discovering", "leaving", "entering"],
            "tool": ["using", "borrowing", "sharpening", "cleaning", "fixing", "storing"],
            "instrument": ["playing", "practicing", "tuning", "learning", "mastering", "hearing"]
        }
    
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
            # Clean the word
            word = word.strip().lower()
            
            # Check for plural forms
            if word in self.plural_forms:
                is_plural = True
                singular_form = self.plural_forms[word]
            else:
                is_plural = word.endswith('s') and not word.endswith('ss')
                singular_form = word[:-1] if is_plural else word
            
            # First check cache (using both plural and singular forms)
            cached_example = self._get_cached_example(word, target_language)
            if not cached_example and is_plural:
                cached_example = self._get_cached_example(singular_form, target_language)
                
            if cached_example:
                # Adapt cached example for plural/singular if needed
                if is_plural and singular_form in cached_example["english"]:
                    cached_example["english"] = cached_example["english"].replace(singular_form, word)
                    if cached_example["translated"]:
                        cached_example["translated"] = self._translate(cached_example["english"], target_language)
                return cached_example
            
            # Determine semantic category for better examples
            if not category:
                category = self._infer_category(word)
            
            # Try API methods in sequence
            methods = [
                self._get_free_dictionary_example,
                self._get_wordnik_example,
                self._get_owlbot_example
            ]
            
            for method in methods:
                example = method(word, target_language, category)
                if example and example["english"]:
                    # Verify the example doesn't contain awkward or inappropriate content
                    if self._validate_example(example["english"], word, category):
                        # Cache the good example
                        self._cache_example(word, target_language, example)
                        return example
            
            # If APIs don't provide good examples, use semantic templates
            example = self._get_semantic_template(word, target_language, category, is_plural)
            return example
            
        except Exception as e:
            print(f"Error getting example sentence: {e}")
            # Ultimate fallback with safe example
            return self._get_safe_fallback(word, target_language, category)
    
    def _validate_example(self, example, word, category):
        """Check if an example sentence is appropriate and natural."""
        # Lowercase for comparison
        example_lower = example.lower()
        word_lower = word.lower()
        
        # Filter out awkward or inappropriate content
        red_flags = [
            # For people
            (word_lower in self.person_words) and any(x in example_lower for x in [
                "need a new", "is very useful", "on the table", "in the kitchen",
                "i bought", "i sold", "i own", "new " + word_lower
            ]),
            
            # For uncountable nouns
            (word_lower in self.uncountable_nouns) and any(x + " " + word_lower in example_lower for x in [
                "a", "an", "one", "two", "three", "few", "many", "several"
            ]),
            
            # General awkward constructions
            "lorem ipsum" in example_lower,
            len(example.split()) < 3,  # Too short
            len(example.split()) > 20,  # Too long
            example.count(word_lower) > 2,  # Word repeated too many times
            re.search(r'https?://', example_lower),  # Contains URL
            "example" in example_lower and "sentence" in example_lower,  # Meta-example
            example_lower.startswith(("example", "sentence", "definition", "meaning")),
            
            # Grammatical checks
            word_lower in self.uncountable_nouns and re.search(f"an? {word_lower}", example_lower)
        ]
        
        # If any red flags are found, reject the example
        return not any(red_flags)
    
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
        """Get example from Free Dictionary API with semantic validation."""
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
                # Reject examples that don't actually contain the word
                # or contain awkward constructions
                valid_examples = []
                for example in examples:
                    if (word.lower() in example.lower() and 
                        self._validate_example(example, word, category)):
                        valid_examples.append(example)
                
                if valid_examples:
                    # Select example with appropriate length
                    good_examples = [ex for ex in valid_examples if 4 <= len(ex.split()) <= 12]
                    
                    if good_examples:
                        english_example = np.random.choice(good_examples)
                    else:
                        english_example = np.random.choice(valid_examples)
                    
                    # Capitalize first letter if needed
                    if not english_example[0].isupper():
                        english_example = english_example[0].upper() + english_example[1:]
                    
                    # Ensure ending punctuation
                    if not english_example[-1] in '.!?':
                        english_example += '.'
                    
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
        """Get example from Wordnik API with semantic validation."""
        try:
            # Try with API key first if available
            wordnik_api_key = os.environ.get("WORDNIK_API_KEY")
            if wordnik_api_key:
                url = f"https://api.wordnik.com/v4/word.json/{word}/examples?api_key={wordnik_api_key}"
            else:
                # Fallback to public endpoint (may have rate limits)
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
                # Filter for valid examples (contains the word correctly, proper length)
                valid_examples = []
                for example in examples:
                    if (word.lower() in example.lower() and 
                        self._validate_example(example, word, category)):
                        valid_examples.append(example)
                
                if valid_examples:
                    good_examples = [ex for ex in valid_examples 
                                    if 5 <= len(ex.split()) <= 12]
                    
                    if good_examples:
                        english_example = np.random.choice(good_examples)
                    else:
                        english_example = np.random.choice(valid_examples)
                    
                    # Clean up the example
                    english_example = self._clean_example(english_example)
                    
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
        """Get example from Owlbot API with semantic validation."""
        try:
            # Check if API key is available
            owl_api_key = os.environ.get("OWLBOT_KEY")
            if not owl_api_key:
                try:
                    import streamlit as st
                    owl_api_key = st.secrets.get("owlbot", {}).get("key")
                except:
                    return None
            
            if not owl_api_key:
                return None  # No API key available
                
            # If we have an API key, use the authenticated endpoint
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
                # Filter examples
                valid_examples = []
                for example in examples:
                    if self._validate_example(example, word, category):
                        valid_examples.append(example)
                
                if valid_examples:
                    english_example = np.random.choice(valid_examples)
                    english_example = self._clean_example(english_example)
                    
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
    
    def _clean_example(self, example):
        """Clean and format an example sentence."""
        # Remove quotes, extra spaces, etc.
        example = example.strip()
        example = re.sub(r'\s+', ' ', example)
        
        # Capitalize first letter
        if example and not example[0].isupper():
            example = example[0].upper() + example[1:]
        
        # Ensure ending punctuation
        if example and not example[-1] in '.!?':
            example += '.'
        
        return example
    
    def _get_semantic_template(self, word, target_language, category, is_plural=False):
        """Generate an example using semantically appropriate templates."""
        try:
            # Handle people differently to avoid dehumanizing language
            if word in self.person_words or category == "person":
                return self._get_person_example(word, target_language, is_plural)
                
            # Handle uncountable nouns
            elif word in self.uncountable_nouns:
                return self._get_uncountable_example(word, target_language)
                
            # Handle abstract concepts
            elif word in self.abstract_concepts:
                return self._get_abstract_example(word, target_language)
                
            # Get templates for specific categories
            else:
                return self._get_category_example(word, target_language, category, is_plural)
                
        except Exception as e:
            print(f"Template error: {e}")
            return self._get_safe_fallback(word, target_language, category)
    
    def _get_person_example(self, word, target_language, is_plural=False):
        """Generate appropriate examples for people/occupations."""
        # Choose the right article/pronoun for singular/plural
        art = "The" if not is_plural else "The"
        pronoun = "is" if not is_plural else "are"
        
        templates = [
            f"{art} {word} {pronoun} talking on the phone.",
            f"{art} {word} {pronoun} reading a book.",
            f"{art} {word} {pronoun} very friendly.",
            f"We saw a {word} at the store yesterday.",
            f"My neighbor is a {word} who lives downtown.",
            f"She works as a {word} in the city.",
            f"He became a {word} last year.",
            f"They hired a new {word} for the team.",
            f"The {word} helped us with our problem.",
            f"I want to become a {word} someday."
        ]
        
        # Adjust grammar for plural forms
        if is_plural:
            templates = [
                f"{art} {word} are talking on the phone.",
                f"{art} {word} are reading books.",
                f"{art} {word} are very friendly.",
                f"We saw some {word} at the store yesterday.",
                f"My neighbors are {word} who live downtown.",
                f"They work as {word} in the city.",
                f"They became {word} last year.",
                f"They hired new {word} for the team.",
                f"The {word} helped us with our problem.",
                f"I want to work with {word} someday."
            ]
        
        english_example = np.random.choice(templates)
        translated_example = self._translate(english_example, target_language)
        
        return {
            "english": english_example,
            "translated": translated_example,
            "source": "template_person"
        }
    
    def _get_uncountable_example(self, word, target_language):
        """Generate appropriate examples for uncountable nouns."""
        templates = [
            f"I need some {word}.",
            f"There is {word} in the glass.",
            f"Do you have enough {word}?",
            f"We bought some {word} yesterday.",
            f"I like {word} very much.",
            f"She drinks {word} every morning.",
            f"He asked for {word}.",
            f"They don't have much {word} left.",
            f"The {word} tastes good.",
            f"We need more {word}."
        ]
        
        english_example = np.random.choice(templates)
        translated_example = self._translate(english_example, target_language)
        
        return {
            "english": english_example,
            "translated": translated_example,
            "source": "template_uncountable"
        }
    
    def _get_abstract_example(self, word, target_language):
        """Generate appropriate examples for abstract concepts."""
        templates = [
            f"{word.capitalize()} is important in our society.",
            f"We discussed {word} in our class.",
            f"Everyone needs {word} in their life.",
            f"She values {word} above all else.",
            f"The book is about {word} and its importance.",
            f"He spoke about the power of {word}.",
            f"Many people seek {word} in their lives.",
            f"Finding {word} can be difficult.",
            f"We should appreciate {word} more.",
            f"The importance of {word} cannot be overstated."
        ]
        
        english_example = np.random.choice(templates)
        translated_example = self._translate(english_example, target_language)
        
        return {
            "english": english_example,
            "translated": translated_example,
            "source": "template_abstract"
        }
    
    def _get_category_example(self, word, target_language, category, is_plural=False):
        """Generate category-specific examples."""
        # Use appropriate articles and verb forms
        if is_plural:
            articles = ["the", "these", "those", "some", ""]
            article = np.random.choice(articles)
            if article:
                article += " "
        else:
            if word[0].lower() in 'aeiou':
                articles = ["the", "an", "this", "that", "my", "your", "our"]
            else:
                articles = ["the", "a", "this", "that", "my", "your", "our"]
            article = np.random.choice(articles)
            article += " "
        
        # Get verbs appropriate for this category
        if category in self.category_verbs:
            verbs = self.category_verbs[category]
        else:
            verbs = self.category_verbs.get("default", ["using", "seeing", "looking at"])
        
        verb = np.random.choice(verbs)
        
        # Create varied templates using these components
        templates = [
            f"I like {article}{word}.",
            f"We are {verb} {article}{word}.",
            f"{article.capitalize()}{word} is in the room.",
            f"Do you see {article}{word}?",
            f"Where is {article}{word}?",
            f"I bought {article}{word} yesterday.",
            f"Can you show me {article}{word}?",
            f"Look at {article}{word}!",
            f"My friend has {article}{word}.",
            f"This is {article}{word} I was talking about."
        ]
        
        # Grammar adjustments for plural forms
        if is_plural:
            templates = [
                f"I like {article}{word}.",
                f"We are {verb} {article}{word}.",
                f"{article.capitalize()}{word} are in the room.",
                f"Do you see {article}{word}?",
                f"Where are {article}{word}?",
                f"I bought {article}{word} yesterday.",
                f"Can you show me {article}{word}?",
                f"Look at {article}{word}!",
                f"My friend has {article}{word}.",
                f"These are {article}{word} I was talking about."
            ]
        
        english_example = np.random.choice(templates)
        translated_example = self._translate(english_example, target_language)
        
        return {
            "english": english_example,
            "translated": translated_example,
            "source": f"template_{category}"
        }
    
    def _get_safe_fallback(self, word, target_language, category=None):
        """Provide a very simple, safe fallback example."""
        if word in self.person_words or category == "person":
            english_example = f"This is a {word}."
        elif word in self.uncountable_nouns:
            english_example = f"I like {word}."
        elif word in self.abstract_concepts:
            english_example = f"{word.capitalize()} is important."
        else:
            english_example = f"Here is a {word}."
            
        translated_example = self._translate(english_example, target_language)
        
        return {
            "english": english_example,
            "translated": translated_example,
            "source": "safe_fallback"
        }
    
    def _infer_category(self, word):
        """Try to infer word category for better templates."""
        # Check for people
        if word in self.person_words:
            return "person"
        
        # Check for uncountable nouns
        if word in self.uncountable_nouns:
            return "uncountable"
        
        # Check for abstract concepts
        if word in self.abstract_concepts:
            return "abstract"
        
        # Additional categories
        animals = ["dog", "cat", "bird", "fish", "horse", "cow", "elephant",
                  "lion", "tiger", "bear", "rabbit", "monkey", "mouse", "frog"]
        
        if word in animals:
            return "animal"
        
        vehicles = ["car", "bus", "train", "bike", "bicycle", "motorcycle",
                    "truck", "boat", "ship", "plane", "airplane"]
        
        if word in vehicles:
            return "vehicle"
        
        electronics = ["phone", "computer", "laptop", "tablet", "camera",
                       "television", "tv", "radio", "speaker", "headphones"]
        
        if word in electronics:
            return "electronics"
        
        clothing = ["shirt", "pants", "dress", "jacket", "coat", "hat",
                    "shoes", "boots", "socks", "gloves", "scarf"]
        
        if word in clothing:
            return "clothing"
        
        food = ["apple", "banana", "bread", "cake", "chicken", "coffee", 
                "pizza", "sandwich", "fruit", "vegetable", "meat", "fish"]
        
        if word in food:
            return "food"
        
        # Default category
        return "default"