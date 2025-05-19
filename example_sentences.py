"""
EMERGENCY FIX - May 18, 2025
----------------------------
Super simple implementation that ensures correct examples for problematic words
"""

import requests
import os
import random
import re
import time

# Global timestamp to confirm this is the latest version
IMPLEMENTATION_TIMESTAMP = "2025-05-18-08:30:25"

class ExampleSentenceGenerator:
    def __init__(self, translate_func=None, debug=False):
        """Initialize the generator."""
        print(f"\n\n!!!!!!! EMERGENCY FIX LOADED - {IMPLEMENTATION_TIMESTAMP} !!!!!!!\n\n")
        self.translate_func = translate_func
        self.debug = True  # Force debug mode
        
        # Mapping of problematic words to guaranteed good examples
        self.guaranteed_examples = {
            "glasses": [
                "I need my glasses to read this book.",
                "She wears glasses for driving.",
                "He forgot his glasses at home.",
                "These glasses help me see better.",
                "The glasses are on the table beside my bed."
            ],
            "glass": [
                "Please pour some water into this glass.",
                "The glass is half full.",
                "She dropped the glass and it broke.",
                "He's drinking from a glass of milk.",
                "The window is made of reinforced glass."
            ],
            "top": [
                "She was wearing a blue top with jeans.",
                "I bought this top at the mall yesterday.",
                "This is my favorite top for summer.",
                "The top doesn't match the skirt.",
                "He spilled coffee on his new top."
            ],
            "bear": [
                "The child hugged his teddy bear.",
                "We saw a bear at the zoo yesterday.",
                "The bear was eating berries in the forest.",
                "She has a collection of stuffed bears.",
                "The polar bear swam in the cold water."
            ],
            "person": [
                "I met an interesting person at the conference.",
                "She is a very kind person.",
                "The person at the reception desk helped me.",
                "He's the right person for this job.",
                "We need to find a qualified person for this position."
            ]
        }
        
        # Add all plurals of these words
        for word in list(self.guaranteed_examples.keys()):
            if word + "s" not in self.guaranteed_examples and word != "glasses":
                self.guaranteed_examples[word + "s"] = [
                    example.replace(word, word + "s") for example in self.guaranteed_examples[word]
                ]
        
        # Add related forms to check
        self.word_variants = {
            "glasses": ["glass", "fiberglass", "fibreglass", "eyeglass"],
            "glass": ["glasses", "fiberglass", "fibreglass"],
            "bear": ["bearing", "bearable", "unbearable", "bearer", "bore", "borne"],
            "top": ["topped", "topping", "laptop", "desktop", "stop"]
        }
    
    def get_example_sentence(self, word, target_language, category=None):
        """Get an example sentence for a word."""
        try:
            # Normalize word
            original_word = word
            word = word.strip().lower()
            
            print(f"\n\n>>> Getting example for: '{word}' <<<\n")
            
            # Check if this is a word with guaranteed examples
            if word in self.guaranteed_examples:
                print(f">>> Using guaranteed example for '{word}'")
                return self._use_guaranteed_example(word, target_language)
            
            # For all other words, try API with strict filtering
            print(f">>> Trying API for regular word '{word}'")
            example = self._get_safe_api_example(word, target_language)
            if example:
                return example
            
            # If everything fails, use a generic sentence
            print(f">>> Falling back to generic example for '{word}'")
            fallback = f"This is a {word}."
            return {
                "english": fallback,
                "translated": self._translate(fallback, target_language),
                "source": "fallback"
            }
            
        except Exception as e:
            print(f">>> ERROR: {e}")
            fallback = f"This is a {word}."
            return {
                "english": fallback,
                "translated": self._translate(fallback, target_language),
                "source": "error_fallback"
            }
    
    def _use_guaranteed_example(self, word, target_language):
        """Use a guaranteed example for a problematic word."""
        examples = self.guaranteed_examples[word]
        english_example = random.choice(examples)
        translated_example = self._translate(english_example, target_language)
        
        return {
            "english": english_example,
            "translated": translated_example,
            "source": "guaranteed_example"
        }
    
    def _get_safe_api_example(self, word, target_language):
        """Try to get a safe example from API with strict filtering."""
        # Try all possible API sources
        api_functions = [
            self._try_free_dictionary,
            self._try_wordnik
        ]
        
        for func in api_functions:
            example = func(word, target_language)
            if example:
                return example
        
        return None
    
    def _try_free_dictionary(self, word, target_language):
        """Try to get example from Free Dictionary API."""
        try:
            url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}"
            print(f">>> Requesting: {url}")
            
            response = requests.get(url, timeout=5)
            if response.status_code != 200:
                print(f">>> API returned non-200 status: {response.status_code}")
                return None
            
            data = response.json()
            if not isinstance(data, list) or len(data) == 0:
                print(">>> API returned empty or non-list data")
                return None
            
            # Extract examples
            examples = []
            for meaning in data[0].get('meanings', []):
                for definition in meaning.get('definitions', []):
                    if 'example' in definition and definition['example']:
                        example = definition['example']
                        print(f">>> Found example: '{example}'")
                        
                        # Check if it contains exact word
                        if not self._contains_exact_word(example, word):
                            print(f">>> Rejected: doesn't contain exact word '{word}'")
                            continue
                        
                        # Check if it contains any variants we want to avoid
                        if word in self.word_variants:
                            contains_variant = False
                            for variant in self.word_variants[word]:
                                if self._contains_exact_word(example, variant):
                                    print(f">>> Rejected: contains variant '{variant}'")
                                    contains_variant = True
                                    break
                            if contains_variant:
                                continue
                        
                        # Check general format
                        if not self._is_good_sentence(example):
                            print(f">>> Rejected: not a good sentence format")
                            continue
                        
                        # This example passed all checks
                        examples.append(example)
            
            if examples:
                # Select random example
                english = random.choice(examples)
                print(f">>> Selected example: '{english}'")
                
                # Clean up
                english = english.strip()
                if english and not english[0].isupper():
                    english = english[0].upper() + english[1:]
                if english and not english[-1] in '.!?':
                    english += '.'
                
                # Translate
                translated = self._translate(english, target_language)
                
                return {
                    "english": english,
                    "translated": translated,
                    "source": "free_dictionary_filtered"
                }
            
            print(">>> No valid examples found")
            return None
            
        except Exception as e:
            print(f">>> Error with Free Dictionary API: {e}")
            return None
    
    def _try_wordnik(self, word, target_language):
        """Try to get example from Wordnik API."""
        try:
            url = f"https://api.wordnik.com/v4/word.json/{word}/examples"
            print(f">>> Requesting: {url}")
            
            response = requests.get(url, timeout=5)
            if response.status_code != 200:
                print(f">>> API returned non-200 status: {response.status_code}")
                return None
            
            data = response.json()
            
            # Extract examples
            examples = []
            for example_obj in data.get('examples', []):
                if 'text' in example_obj and example_obj['text']:
                    example = example_obj['text']
                    print(f">>> Found example: '{example}'")
                    
                    # Check if it contains exact word
                    if not self._contains_exact_word(example, word):
                        print(f">>> Rejected: doesn't contain exact word '{word}'")
                        continue
                    
                    # Check if it contains any variants we want to avoid
                    if word in self.word_variants:
                        contains_variant = False
                        for variant in self.word_variants[word]:
                            if self._contains_exact_word(example, variant):
                                print(f">>> Rejected: contains variant '{variant}'")
                                contains_variant = True
                                break
                        if contains_variant:
                            continue
                    
                    # Check general format
                    if not self._is_good_sentence(example):
                        print(f">>> Rejected: not a good sentence format")
                        continue
                    
                    # This example passed all checks
                    examples.append(example)
            
            if examples:
                # Select random example
                english = random.choice(examples)
                print(f">>> Selected example: '{english}'")
                
                # Clean up
                english = english.strip()
                if english and not english[0].isupper():
                    english = english[0].upper() + english[1:]
                if english and not english[-1] in '.!?':
                    english += '.'
                
                # Translate
                translated = self._translate(english, target_language)
                
                return {
                    "english": english,
                    "translated": translated,
                    "source": "wordnik_filtered"
                }
            
            print(">>> No valid examples found")
            return None
            
        except Exception as e:
            print(f">>> Error with Wordnik API: {e}")
            return None
    
    def _contains_exact_word(self, text, word):
        """Check if text contains the exact word using word boundaries."""
        pattern = r'\b' + re.escape(word) + r'\b'
        result = re.search(pattern, text.lower())
        if result:
            return True
        return False
    
    def _is_good_sentence(self, text):
        """Basic quality checks for a sentence."""
        # Check for semicolons (often indicates a list)
        if ';' in text:
            return False
        
        # Check for reasonable length
        if len(text.split()) < 3 or len(text.split()) > 20:
            return False
        
        # Check for metadata-style text
        if "example:" in text.lower() or "examples:" in text.lower():
            return False
        
        return True
    
    def _translate(self, text, target_language):
        """Translate text using the provided translation function."""
        if self.translate_func:
            try:
                return self.translate_func(text, target_language)
            except Exception as e:
                print(f">>> Translation error: {e}")
                return f"[Translation error: {e}]"
        else:
            return f"[Translation to {target_language}]"