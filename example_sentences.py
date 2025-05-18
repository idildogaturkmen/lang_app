"""
Enhanced Example Sentence Generator
-----------------------------------
Fixed version with proper indentation for all methods
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
        self._initialize_category_templates()
        
    def setup_cache_dir(self):
        """Create cache directory if it doesn't exist."""
        self.cache_dir = "sentence_cache"
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def _initialize_category_templates(self):
        """Initialize templates organized by semantic category for better context matching."""
        self.category_templates = {
            # Clothing templates
            "clothing": [
                "I bought a new {word} for the party.",
                "This {word} is very comfortable to wear.",
                "She likes wearing a blue {word} with jeans.",
                "The {word} is hanging in the closet.",
                "I need to wash my favorite {word}.",
                "He gave me a {word} as a gift.",
                "We sell different styles of {word} in our store.",
                "My sister borrowed my {word} yesterday.",
                "This {word} doesn't fit me anymore.",
                "I spilled coffee on my {word} this morning."
            ],
            
            # Food templates
            "food": [
                "This {word} tastes delicious.",
                "I love eating {word} for breakfast.",
                "My mother makes the best {word} I've ever tasted.",
                "Would you like some {word} with your meal?",
                "The {word} is fresh from the market.",
                "She bought some {word} for tonight's dinner.",
                "We need more {word} for the recipe.",
                "The {word} smells wonderful.",
                "This restaurant serves excellent {word}.",
                "I learned how to cook {word} last year."
            ],
            
            # Animal templates
            "animals": [
                "The {word} is sleeping under the tree.",
                "I saw a {word} at the zoo yesterday.",
                "My friend has a {word} as a pet.",
                "The {word} was running in the park.",
                "That {word} looks very friendly.",
                "We watched the {word} playing in the garden.",
                "The {word} was eating some food.",
                "There's a {word} on the path ahead.",
                "The children love watching the {word}.",
                "The {word} is a fascinating animal."
            ],
            
            # Electronics templates
            "electronics": [
                "I need to charge my {word}.",
                "This {word} has many useful features.",
                "My {word} stopped working yesterday.",
                "She bought a new {word} online.",
                "The {word} comes with a one-year warranty.",
                "I use my {word} every day.",
                "Can you help me set up this {word}?",
                "The {word} is on sale this weekend.",
                "My brother got a new {word} for his birthday.",
                "This {word} is the latest model."
            ],
            
            # Furniture templates
            "furniture": [
                "We placed the {word} near the window.",
                "This {word} is very comfortable.",
                "The {word} doesn't fit in the living room.",
                "We need to assemble the new {word}.",
                "I bought this {word} at a garage sale.",
                "The {word} matches our other furniture.",
                "Can you help me move this {word}?",
                "This {word} has been in our family for generations.",
                "The {word} needs to be cleaned.",
                "We're looking for a new {word} for the bedroom."
            ],
            
            # Vehicle templates
            "vehicles": [
                "I parked my {word} in the garage.",
                "The {word} needs to be serviced.",
                "She drives a blue {word} to work.",
                "We rented a {word} for our vacation.",
                "My brother's {word} is very fast.",
                "The {word} ran out of fuel on the highway.",
                "This {word} can fit five passengers.",
                "I'm saving money to buy a new {word}.",
                "The {word} has a powerful engine.",
                "We took the {word} to the mountains last weekend."
            ],
            
            # Drinkware templates
            "drinkware": [
                "I filled the {word} with water.",
                "She dropped the {word} and it broke.",
                "This {word} keeps my coffee hot for hours.",
                "The {word} is made of glass.",
                "I need to wash this {word}.",
                "He drinks tea from his favorite {word}.",
                "We have a set of six {word}s for guests.",
                "The {word} is on the kitchen counter.",
                "My {word} has a small chip on the rim.",
                "This {word} can hold up to 12 ounces."
            ],
            
            # Eyewear templates
            "eyewear": [
                "I can't find my {word} anywhere.",
                "These {word} help me see clearly.",
                "She wears {word} for reading.",
                "My {word} are scratched and need to be replaced.",
                "He forgot his {word} at home.",
                "These {word} protect my eyes from the sun.",
                "I bought new {word} last month.",
                "The {word} are on the nightstand.",
                "She looks great in those {word}.",
                "I need to clean my {word}."
            ],
            
            # General object templates
            "object": [
                "I placed the {word} on the table.",
                "Can you pass me that {word}, please?",
                "The {word} is in the drawer.",
                "She's looking for her {word}.",
                "This {word} belongs to my brother.",
                "I found the {word} under the couch.",
                "We need to buy a new {word}.",
                "The {word} is very useful.",
                "He keeps the {word} in his office.",
                "I use this {word} almost every day."
            ]
        }
        
        # Map specific words to categories for better context matching
        self.word_to_category = {
            # Clothing
            "shirt": "clothing", "pants": "clothing", "dress": "clothing", "jacket": "clothing",
            "sweater": "clothing", "coat": "clothing", "hat": "clothing", "gloves": "clothing",
            "socks": "clothing", "shoes": "clothing", "boots": "clothing", "scarf": "clothing",
            "tie": "clothing", "belt": "clothing", "jeans": "clothing", "skirt": "clothing",
            "blouse": "clothing", "suit": "clothing", "top": "clothing", "shorts": "clothing",
            
            # Food
            "apple": "food", "banana": "food", "orange": "food", "bread": "food",
            "cheese": "food", "chicken": "food", "meat": "food", "fish": "food",
            "pasta": "food", "rice": "food", "vegetable": "food", "fruit": "food",
            "pizza": "food", "salad": "food", "sandwich": "food", "soup": "food",
            "cake": "food", "cookie": "food", "chocolate": "food", "candy": "food",
            
            # Animals
            "dog": "animals", "cat": "animals", "bird": "animals", "fish": "animals",
            "horse": "animals", "cow": "animals", "sheep": "animals", "pig": "animals",
            "lion": "animals", "tiger": "animals", "bear": "animals", "elephant": "animals",
            "monkey": "animals", "giraffe": "animals", "zebra": "animals", "rabbit": "animals",
            
            # Electronics
            "phone": "electronics", "computer": "electronics", "laptop": "electronics", "tablet": "electronics",
            "television": "electronics", "tv": "electronics", "camera": "electronics", "speaker": "electronics",
            "headphones": "electronics", "microphone": "electronics", "keyboard": "electronics", "mouse": "electronics",
            
            # Furniture
            "chair": "furniture", "table": "furniture", "desk": "furniture", "bed": "furniture",
            "sofa": "furniture", "couch": "furniture", "bookshelf": "furniture", "cabinet": "furniture",
            "drawer": "furniture", "wardrobe": "furniture", "dresser": "furniture", "lamp": "furniture",
            
            # Vehicles
            "car": "vehicles", "bike": "vehicles", "bicycle": "vehicles", "motorcycle": "vehicles",
            "bus": "vehicles", "train": "vehicles", "airplane": "vehicles", "boat": "vehicles",
            "ship": "vehicles", "truck": "vehicles", "van": "vehicles", "scooter": "vehicles",
            
            # Drinkware
            "cup": "drinkware", "glass": "drinkware", "mug": "drinkware", "bottle": "drinkware",
            "thermos": "drinkware", "flask": "drinkware", "teacup": "drinkware", "wineglass": "drinkware",
            
            # Eyewear
            "glasses": "eyewear", "sunglasses": "eyewear", "spectacles": "eyewear", "contacts": "eyewear",
            "goggles": "eyewear", "eyeglasses": "eyewear", "shades": "eyewear"
        }
        
        # List of inappropriate words to filter out
        self.inappropriate_words = [
            "sex", "sexy", "sexual", "naked", "nude", "porn", "adult", "xxx",
            "fuck", "shit", "damn", "ass", "bitch", "cunt", "dick", "cock", "pussy",
            "erotic", "intimate", "genital", "penis", "vagina", "breast", "intercourse",
            "oral", "anal", "bondage", "fetish", "masturbate", "masturbation", "orgasm",
            "gay", "lesbian", "transgender", "bisexual", "queer",  # These are not inappropriate on their own, but could lead to sensitive contexts
            "top", "bottom", "versatile", "dominant", "submissive"  # These have specific meanings in certain contexts
        ]
    
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
            
            # Special handling for known problematic words - ALWAYS use templates for these
            problematic_words = ["top", "glasses", "glass", "bottom", "fly", "chest", "box", "member"]
            if word in problematic_words:
                if self.debug:
                    print(f"Using template for known problematic word: {word}")
                return self._get_template_example(word, target_language, category)
            
            # Special handling for potentially inappropriate words
            if word in self.inappropriate_words:
                return self._get_safe_example(word, target_language, category)
            
            # Check cache first
            cached_example = self._get_cached_example(word, target_language)
            if cached_example:
                return cached_example
            
            # Try API methods first
            methods = [
                self._get_free_dictionary_example,
                self._get_wordnik_example
            ]
            
            for method in methods:
                example = method(word, target_language, category)
                if example and example["english"] and self._is_safe_content(example["english"]) and self._contains_exact_word(example["english"], word):
                    # Cache the example
                    self._cache_example(word, target_language, example)
                    return example
            
            # Fall back to template-based examples
            example = self._get_template_example(word, target_language, category)
            
            # Cache the example
            if example and example["english"]:
                self._cache_example(word, target_language, example)
                
            return example
            
        except Exception as e:
            if self.debug:
                print(f"Error getting example sentence: {e}")
                
            # Ultimate fallback
            return {
                "english": f"This is a {word}.",
                "translated": self._translate(f"This is a {word}.", target_language),
                "source": "basic_fallback"
            }
    
    def _contains_exact_word(self, text, word):
        """Check if text contains the exact word with proper word boundaries."""
        # Create regex pattern with word boundaries
        pattern = r'\b' + re.escape(word) + r'\b'
        
        # Check if the text contains the exact word (case insensitive)
        matches = re.search(pattern, text.lower())
        
        # For debugging
        if self.debug and not matches:
            print(f"Text doesn't contain exact word '{word}': '{text}'")
            
        return matches is not None
    
    def _get_template_example(self, word, target_language, hint_category=None):
        """Generate an example sentence using templates based on word category."""
        # Determine the appropriate category
        category = None
        
        # Use provided category hint if available
        if hint_category and hint_category in self.category_templates:
            category = hint_category
        # Otherwise look up in our mapping
        elif word in self.word_to_category:
            category = self.word_to_category[word]
        # Default to general object
        else:
            category = "object"
        
        # Get templates for this category
        templates = self.category_templates.get(category, self.category_templates["object"])
        
        # Select a random template
        template = random.choice(templates)
        
        # Handle plurals properly
        if word.endswith('s') and (word in ["glasses", "pants", "shorts", "scissors", "jeans"] or 
                                   category == "eyewear"):
            # These words are typically plural
            english_example = template.replace("a {word}", "{word}").replace("the {word}", "the {word}").format(word=word)
        else:
            english_example = template.format(word=word)
        
        # Translate to target language
        translated_example = self._translate(english_example, target_language)
        
        return {
            "english": english_example,
            "translated": translated_example,
            "source": f"{category}_template"
        }
    
    def _get_safe_example(self, word, target_language, category=None):
        """Generate a guaranteed safe example for potentially problematic words."""
        # For words we've identified as potentially problematic, use extremely simple templates
        safe_templates = [
            f"The {word} is on the table.",
            f"I need to buy a new {word}.",
            f"The {word} is blue.",
            f"Can you see the {word}?",
            f"I like this {word}.",
            f"The {word} is in the room.",
            f"She has a {word}.",
            f"Where is the {word}?",
            f"This is my {word}.",
            f"The {word} looks nice."
        ]
        
        english_example = random.choice(safe_templates)
        translated_example = self._translate(english_example, target_language)
        
        return {
            "english": english_example,
            "translated": translated_example,
            "source": "safe_template"
        }
    
    def _is_safe_content(self, text):
        """Check if content is appropriate and not offensive."""
        text_lower = text.lower()
        
        # General profanity check
        for word in self.inappropriate_words:
            if word in text_lower:
                if self.debug:
                    print(f"Rejected example due to inappropriate word: '{word}' in '{text}'")
                return False
        
        # Format check - avoid fragment examples like "Glass frog; glass shrimp"
        if ";" in text or not text.strip().endswith(('.', '!', '?')):
            if self.debug:
                print(f"Rejected example due to formatting: '{text}'")
            return False
        
        # Length check
        if len(text.split()) < 3 or len(text.split()) > 20:
            if self.debug:
                print(f"Rejected example due to length: '{text}'")
            return False
            
        # Basic relevance check
        if "example of" in text_lower or "examples of" in text_lower:
            if self.debug:
                print(f"Rejected example due to meta-reference: '{text}'")
            return False
            
        return True
    
    def _translate(self, text, target_language):
        """Translate text using the provided translation function."""
        if not text:
            return ""
            
        if self.translate_func:
            try:
                return self.translate_func(text, target_language)
            except Exception as e:
                if self.debug:
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
                    example = json.load(f)
                    
                    # Double-check safety even for cached examples
                    if not self._is_safe_content(example["english"]):
                        if self.debug:
                            print(f"Rejected cached example as unsafe: '{example['english']}'")
                        return None
                        
                    return example
            except Exception as e:
                if self.debug:
                    print(f"Cache read error: {e}")
        return None
    
    def _cache_example(self, word, target_language, example):
        """Cache an example for future use."""
        cache_file = f"{self.cache_dir}/{word}_{target_language}.json"
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(example, f, ensure_ascii=False)
        except Exception as e:
            if self.debug:
                print(f"Cache write error: {e}")
    
    def _get_free_dictionary_example(self, word, target_language, category=None):
        """Get example from Free Dictionary API with safety and relevance checks."""
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
                    # Check if the part of speech matches the expected category
                    part_of_speech = meaning.get('partOfSpeech', '')
                    
                    for definition in meaning.get('definitions', []):
                        if 'example' in definition and definition['example']:
                            example = definition['example']
                            # Check if it contains the exact word
                            if self._contains_exact_word(example, word):
                                # Add context about part of speech
                                examples.append((example, part_of_speech))
            
            if examples:
                # Filter for safe examples
                safe_examples = []
                for example, pos in examples:
                    if self._is_safe_content(example):
                        safe_examples.append((example, pos))
                
                if not safe_examples:
                    return None
                    
                # Prioritize examples with part of speech matching the category
                if category:
                    # Try to find examples that match the category
                    matched_examples = []
                    category_pos_map = {
                        "clothing": "noun", "food": "noun", "animals": "noun", 
                        "electronics": "noun", "furniture": "noun", "vehicles": "noun",
                        "drinkware": "noun", "eyewear": "noun", "object": "noun"
                    }
                    expected_pos = category_pos_map.get(category, None)
                    
                    if expected_pos:
                        for example, pos in safe_examples:
                            if pos == expected_pos:
                                matched_examples.append(example)
                        
                        if matched_examples:
                            english_example = random.choice(matched_examples)
                        else:
                            english_example = random.choice([ex for ex, _ in safe_examples])
                    else:
                        english_example = random.choice([ex for ex, _ in safe_examples])
                else:
                    english_example = random.choice([ex for ex, _ in safe_examples])
                
                # Double-check exact word matching
                if not self._contains_exact_word(english_example, word):
                    if self.debug:
                        print(f"Rejecting example that passed initial check but failed second check: '{english_example}'")
                    return None
                
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
    
    def _get_wordnik_example(self, word, target_language, category=None):
        """Get example from Wordnik API with safety and relevance checks."""
        try:
            # Public endpoint (no key needed)
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
                    # Check for exact word match using word boundaries
                    if self._contains_exact_word(text, word):
                        examples.append(text)
            
            if examples:
                # Filter for safe examples
                safe_examples = [ex for ex in examples if self._is_safe_content(ex)]
                
                if not safe_examples:
                    return None
                    
                # Choose an example with good length
                good_examples = [ex for ex in safe_examples if 5 <= len(ex.split()) <= 15]
                
                if good_examples:
                    english_example = random.choice(good_examples)
                else:
                    english_example = random.choice(safe_examples)
                
                # Double-check exact word matching
                if not self._contains_exact_word(english_example, word):
                    if self.debug:
                        print(f"Rejecting Wordnik example that passed initial check but failed second check: '{english_example}'")
                    return None
                
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