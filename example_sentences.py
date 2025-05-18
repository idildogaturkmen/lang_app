"""
API-First Example Sentence Generator with Strong Context Filtering
-----------------------------------------------------------------
This implementation prioritizes API examples but ensures they're relevant
"""

import requests
import os
import json
import random
import re
import time
from functools import lru_cache

class ExampleSentenceGenerator:
    def __init__(self, translate_func=None, debug=False):
        """
        Initialize the example sentence generator.
        
        Args:
            translate_func: A function that takes (text, target_language) and returns translated text
            debug: Enable debug output
        """
        self.translate_func = translate_func
        self.debug = debug
        self.setup_cache_dir()
        self._initialize_word_data()
        
    def setup_cache_dir(self):
        """Create cache directory if it doesn't exist."""
        self.cache_dir = "sentence_cache"
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def _initialize_word_data(self):
        """Initialize word-specific data for better filtering."""
        # Words with special context requirements
        self.special_word_contexts = {
            # For each word, define required and forbidden context words
            "glasses": {
                "required_contexts": ["eye", "vision", "see", "wear", "read", "sight", "prescription", "lens", "optician"],
                "forbidden_contexts": ["glass", "fibre", "fiber", "window", "bottle", "drink", "mirror", "cup"]
            },
            "glass": {
                "required_contexts": ["drink", "window", "bottle", "cup", "mirror", "pour", "fill", "break"],
                "forbidden_contexts": ["eye", "vision", "wear", "read", "prescription", "lens"]
            },
            "top": {
                "required_contexts": ["wear", "shirt", "cloth", "outfit", "dress", "fashion", "bought", "blue", "red", "color", "colour"],
                "forbidden_contexts": ["hill", "mountain", "stop", "desktop", "laptop", "spin", "topped", "topping"]
            }
        }
        
        # Words that commonly appear in wrong contexts
        self.problematic_words = set(["glasses", "glass", "top", "fly", "bear", "chest", "box", "member"])
        
        # Base words that shouldn't be confused with their variants
        self.base_words = {
            "glasses": ["glass", "eyeglasses", "spectacles", "fiberglass", "fibreglass"],
            "glass": ["glasses", "eyeglasses", "fiberglass", "fibreglass"],
            "top": ["stop", "topped", "topping", "laptop", "desktop", "rooftop", "mountaintop"],
        }
        
        # Words that are typically plural
        self.plural_words = set([
            "glasses", "pants", "shorts", "scissors", "jeans", "trousers", "tights",
            "goggles", "eyeglasses", "spectacles", "sunglasses", "shades"
        ])
        
        # Word-to-category mapping for context hints and fallback templates
        self.word_to_category = {
            # Eyewear
            "glasses": "eyewear",
            "sunglasses": "eyewear",
            "spectacles": "eyewear",
            "eyeglasses": "eyewear",
            "goggles": "eyewear",
            
            # Clothing
            "top": "clothing",
            "shirt": "clothing",
            "pants": "clothing",
            "jeans": "clothing",
            "dress": "clothing",
            "shorts": "clothing",
            "sweater": "clothing",
            "jacket": "clothing",
            "coat": "clothing",
            "hat": "clothing",
            
            # Drink containers
            "glass": "drinkware",
            "cup": "drinkware",
            "mug": "drinkware",
            "bottle": "drinkware",
            
            # Additional mappings
            "dog": "animals",
            "cat": "animals",
            "chair": "furniture",
            "table": "furniture",
            "phone": "electronics",
            "computer": "electronics",
            "car": "vehicles",
            "bike": "vehicles"
        }
        
        # Template categories for fallback
        self.category_templates = {
            # Eyewear templates
            "eyewear": [
                "I can't find my {word} anywhere.",
                "These {word} help me see better.",
                "She wears {word} for reading.",
                "My {word} are scratched and need to be replaced.",
                "He forgot his {word} at home today."
            ],
            
            # Clothing templates
            "clothing": [
                "I bought a new {word} for the party.",
                "This {word} is very comfortable to wear.",
                "She likes wearing a blue {word} with jeans.",
                "The {word} is hanging in the closet.",
                "I need to wash my favorite {word}."
            ],
            
            # Animals
            "animals": [
                "The {word} is sleeping under the tree.",
                "I saw a {word} at the zoo yesterday.",
                "My friend has a {word} as a pet.",
                "The {word} was running in the park.",
                "That {word} looks very friendly."
            ],
            
            # Electronics
            "electronics": [
                "I need to charge my {word}.",
                "This {word} has many useful features.",
                "My {word} stopped working yesterday.",
                "She bought a new {word} online.",
                "The {word} comes with a one-year warranty."
            ],
            
            # Furniture
            "furniture": [
                "We placed the {word} near the window.",
                "This {word} is very comfortable.",
                "The {word} doesn't fit in the living room.",
                "We need to assemble the new {word}.",
                "I bought this {word} at a garage sale."
            ],
            
            # Vehicles
            "vehicles": [
                "I parked my {word} in the garage.",
                "The {word} needs to be serviced.",
                "She drives a blue {word} to work.",
                "We rented a {word} for our vacation.",
                "My brother's {word} is very fast."
            ],
            
            # Drinkware
            "drinkware": [
                "I filled the {word} with water.",
                "She dropped the {word} and it broke.",
                "This {word} keeps my coffee hot for hours.",
                "The {word} is made of ceramic.",
                "I need to wash this {word}."
            ],
            
            # General
            "general": [
                "I placed the {word} on the table.",
                "Can you pass me that {word}, please?",
                "The {word} is in the drawer.",
                "She's looking for her {word}.",
                "This {word} belongs to my brother."
            ]
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
            # Clean and normalize the word
            word = word.strip().lower()
            
            if self.debug:
                print(f"Getting example for: '{word}'")
            
            # Only use API for words that aren't known to cause problems
            if word not in self.problematic_words:
                # Get example from API (Free Dictionary or Wordnik)
                api_example = self._get_api_example(word, target_language)
                if api_example:
                    return api_example
            else:
                if self.debug:
                    print(f"'{word}' is a problematic word - looking for context-specific example")
                    
                # For problematic words, try to get context-specific examples
                context_example = self._get_context_specific_example(word, target_language)
                if context_example:
                    return context_example
            
            # If no API example or problematic word, use template
            template_example = self._get_template_example(word, target_language)
            return template_example
            
        except Exception as e:
            if self.debug:
                print(f"Error getting example sentence: {e}")
            
            # Fallback
            return {
                "english": f"This is a {word}.",
                "translated": self._translate(f"This is a {word}.", target_language),
                "source": "basic_fallback"
            }
    
    def _get_api_example(self, word, target_language):
        """Get example from APIs with basic filtering."""
        # Try Free Dictionary first
        example = self._get_free_dictionary_example(word, target_language)
        if example:
            return example
            
        # Then try Wordnik
        example = self._get_wordnik_example(word, target_language)
        if example:
            return example
            
        return None
    
    def _get_context_specific_example(self, word, target_language):
        """
        Get a context-specific example for problematic words.
        This method applies stricter filtering based on context words.
        """
        # Try Free Dictionary with context filtering
        example = self._get_free_dictionary_example_with_context(word, target_language)
        if example:
            return example
            
        # Try Wordnik with context filtering
        example = self._get_wordnik_example_with_context(word, target_language)
        if example:
            return example
            
        return None
    
    def _get_template_example(self, word, target_language):
        """Get a template example as fallback."""
        category = self.word_to_category.get(word, "general")
        templates = self.category_templates.get(category, self.category_templates["general"])
        
        # Select a random template
        template = random.choice(templates)
        
        # Handle plurals properly
        if word in self.plural_words:
            english_example = template.replace("a {word}", "{word}").replace("the {word}", "the {word}").format(word=word)
        else:
            english_example = template.format(word=word)
        
        # Translate
        translated_example = self._translate(english_example, target_language)
        
        return {
            "english": english_example,
            "translated": translated_example,
            "source": "template_" + category
        }
    
    def _contains_exact_word(self, text, word):
        """Check if text contains the exact word with proper word boundaries."""
        pattern = r'\b' + re.escape(word) + r'\b'
        return re.search(pattern, text.lower()) is not None
    
    def _contains_any_word(self, text, words):
        """Check if text contains any of the words in the list."""
        text_lower = text.lower()
        for word in words:
            if self._contains_exact_word(text_lower, word):
                return True
        return False
    
    def _is_valid_sentence(self, text):
        """Check if text is a valid, complete sentence."""
        # Must end with sentence-ending punctuation
        if not text.strip().endswith(('.', '!', '?')):
            return False
            
        # Must be a reasonable length
        words = text.split()
        if len(words) < 3 or len(words) > 20:
            return False
            
        # Shouldn't contain semicolons (often indicates a list, not a sentence)
        if ';' in text:
            return False
            
        # Shouldn't be metadata
        if "example of" in text.lower() or "examples of" in text.lower():
            return False
            
        return True
    
    def _check_context_requirements(self, text, word):
        """
        Check if a sentence meets the context requirements for a word.
        For problematic words, we have specific required and forbidden contexts.
        """
        if word not in self.special_word_contexts:
            return True
            
        context_rules = self.special_word_contexts[word]
        text_lower = text.lower()
        
        # Check for required context words
        required_contexts = context_rules.get("required_contexts", [])
        if required_contexts:
            if not any(context in text_lower for context in required_contexts):
                if self.debug:
                    print(f"Missing required context for '{word}' in: '{text}'")
                return False
                
        # Check for forbidden context words
        forbidden_contexts = context_rules.get("forbidden_contexts", [])
        if forbidden_contexts:
            if any(context in text_lower for context in forbidden_contexts):
                if self.debug:
                    print(f"Found forbidden context for '{word}' in: '{text}'")
                return False
                
        return True
    
    def _translate(self, text, target_language):
        """Translate text using the provided translation function."""
        if self.translate_func:
            try:
                return self.translate_func(text, target_language)
            except Exception as e:
                if self.debug:
                    print(f"Translation error: {e}")
                return ""
        else:
            return f"[Translation to {target_language}]"
    
    def _get_free_dictionary_example(self, word, target_language):
        """Get example from Free Dictionary API with basic filtering."""
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
                            example = definition['example']
                            if self._contains_exact_word(example, word) and self._is_valid_sentence(example):
                                examples.append(example)
            
            if examples:
                # Choose a random example
                english_example = random.choice(examples)
                
                # Clean up the example
                english_example = self._clean_example(english_example)
                
                # Translate
                translated_example = self._translate(english_example, target_language)
                
                return {
                    "english": english_example,
                    "translated": translated_example,
                    "source": "free_dictionary_api"
                }
            
            return None
        except Exception as e:
            if self.debug:
                print(f"Free Dictionary API error: {e}")
            return None
    
    def _get_free_dictionary_example_with_context(self, word, target_language):
        """Get example from Free Dictionary API with strict context filtering."""
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
                            example = definition['example']
                            # Apply all our filters
                            if (self._contains_exact_word(example, word) and 
                                self._is_valid_sentence(example) and
                                self._check_context_requirements(example, word)):
                                
                                # For base words, make sure their variants don't appear
                                if word in self.base_words:
                                    # Skip if the example contains any of the variants
                                    if not self._contains_any_word(example, self.base_words[word]):
                                        examples.append(example)
                                else:
                                    examples.append(example)
            
            if examples:
                # Choose a random example
                english_example = random.choice(examples)
                
                # Clean up the example
                english_example = self._clean_example(english_example)
                
                # Translate
                translated_example = self._translate(english_example, target_language)
                
                return {
                    "english": english_example,
                    "translated": translated_example,
                    "source": "free_dictionary_api_context_filtered"
                }
            
            return None
        except Exception as e:
            if self.debug:
                print(f"Free Dictionary API error: {e}")
            return None
    
    def _get_wordnik_example(self, word, target_language):
        """Get example from Wordnik API with basic filtering."""
        try:
            url = f"https://api.wordnik.com/v4/word.json/{word}/examples"
            response = requests.get(url, timeout=5)
            
            if response.status_code != 200:
                return None
                
            data = response.json()
            
            examples = []
            for example in data.get('examples', []):
                if 'text' in example and example['text']:
                    text = example['text']
                    if self._contains_exact_word(text, word) and self._is_valid_sentence(text):
                        examples.append(text)
            
            if examples:
                # Choose a random example
                english_example = random.choice(examples)
                
                # Clean up the example
                english_example = self._clean_example(english_example)
                
                # Translate
                translated_example = self._translate(english_example, target_language)
                
                return {
                    "english": english_example,
                    "translated": translated_example,
                    "source": "wordnik_api"
                }
            
            return None
        except Exception as e:
            if self.debug:
                print(f"Wordnik API error: {e}")
            return None
    
    def _get_wordnik_example_with_context(self, word, target_language):
        """Get example from Wordnik API with strict context filtering."""
        try:
            url = f"https://api.wordnik.com/v4/word.json/{word}/examples"
            response = requests.get(url, timeout=5)
            
            if response.status_code != 200:
                return None
                
            data = response.json()
            
            examples = []
            for example in data.get('examples', []):
                if 'text' in example and example['text']:
                    text = example['text']
                    # Apply all our filters
                    if (self._contains_exact_word(text, word) and 
                        self._is_valid_sentence(text) and
                        self._check_context_requirements(text, word)):
                        
                        # For base words, make sure their variants don't appear
                        if word in self.base_words:
                            # Skip if the example contains any of the variants
                            if not self._contains_any_word(text, self.base_words[word]):
                                examples.append(text)
                        else:
                            examples.append(text)
            
            if examples:
                # Choose a random example
                english_example = random.choice(examples)
                
                # Clean up the example
                english_example = self._clean_example(english_example)
                
                # Translate
                translated_example = self._translate(english_example, target_language)
                
                return {
                    "english": english_example,
                    "translated": translated_example,
                    "source": "wordnik_api_context_filtered"
                }
            
            return None
        except Exception as e:
            if self.debug:
                print(f"Wordnik API error: {e}")
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