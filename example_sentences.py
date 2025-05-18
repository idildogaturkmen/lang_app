"""
Robust Example Sentence Generator
---------------------------------
Completely redesigned to guarantee proper word usage in appropriate contexts
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
        
        # Set debug to True temporarily to diagnose issues
        self.debug = True
        
        self.setup_cache_dir()
        self._initialize_category_data()
        
    def setup_cache_dir(self):
        """Create cache directory if it doesn't exist."""
        self.cache_dir = "sentence_cache"
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def _initialize_category_data(self):
        """Initialize comprehensive category data for context matching."""
        # Category templates - carefully crafted examples for each type of word
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
            
            # Eyewear templates - specially crafted for glasses and similar items
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
            
            # Drinkware templates
            "drinkware": [
                "I filled the {word} with water.",
                "She dropped the {word} and it broke.",
                "This {word} keeps my coffee hot for hours.",
                "The {word} is made of crystal.",
                "I need to wash this {word}.",
                "He drinks tea from his favorite {word}.",
                "We have a set of six {word}s for guests.",
                "The {word} is on the kitchen counter.",
                "My {word} has a small chip on the rim.",
                "This {word} can hold up to 12 ounces."
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
        
        # Category classification system - used for making decisions about which words skip the API
        # These categories will ALWAYS use templates, never API results
        self.TEMPLATE_ONLY_CATEGORIES = [
            "eyewear", "clothing", "position_words", "sexual_context_words"
        ]
        
        # Map words to their semantic categories
        self.word_to_category = {
            # Eyewear - ALWAYS use templates for these
            "glasses": "eyewear", 
            "sunglasses": "eyewear", 
            "spectacles": "eyewear", 
            "contacts": "eyewear",
            "goggles": "eyewear", 
            "eyeglasses": "eyewear", 
            "shades": "eyewear",
            
            # Clothing - ALWAYS use templates for these
            "top": "clothing", 
            "shirt": "clothing", 
            "pants": "clothing", 
            "dress": "clothing", 
            "jacket": "clothing",
            "sweater": "clothing", 
            "coat": "clothing", 
            "hat": "clothing", 
            "gloves": "clothing",
            "socks": "clothing", 
            "shoes": "clothing", 
            "boots": "clothing", 
            "scarf": "clothing",
            "tie": "clothing", 
            "belt": "clothing", 
            "jeans": "clothing", 
            "skirt": "clothing",
            "blouse": "clothing", 
            "suit": "clothing", 
            "shorts": "clothing",
            
            # Words with sexual meanings in certain contexts - ALWAYS use templates
            "bottom": "position_words",
            "positions": "position_words",
            "versatile": "position_words",
            "submissive": "position_words",
            "dominant": "position_words",
            
            # Food items
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
            
            # Drinkware - be careful about "glass" vs "glasses"
            "cup": "drinkware", "mug": "drinkware", "bottle": "drinkware",
            "thermos": "drinkware", "flask": "drinkware", "teacup": "drinkware", "wineglass": "drinkware"
        }
        
        # Add a separate explicit mapping for "glass" (singular)
        if "glass" not in self.word_to_category:
            self.word_to_category["glass"] = "drinkware"
            
        # List of words that are commonly confused (like glass/glasses)
        self.confusable_words = {
            "glasses": ["glass", "eyeglasses", "spectacles", "fiberglass", "fibreglass"],
            "glass": ["glasses", "eyeglasses", "fiberglass", "fibreglass"],
            "top": ["stop", "topped", "topping", "laptop", "desktop", "rooftop", "mountaintop"],
            "fly": ["flying", "butterfly", "dragonfly", "firefly", "flyer", "flier"],
            "bear": ["bearing", "bearable", "unbearable", "bearings"]
        }
        
        # Comprehensive list of inappropriate or ambiguous words that need careful handling
        self.problematic_words = set([
            # Sexual/inappropriate words
            "sex", "sexy", "sexual", "naked", "nude", "porn", "adult", "xxx",
            "fuck", "shit", "damn", "ass", "bitch", "cunt", "dick", "cock", "pussy",
            "erotic", "intimate", "genital", "penis", "vagina", "breast", "intercourse",
            "oral", "anal", "bondage", "fetish", "masturbate", "masturbation", "orgasm",
            # Words with dual meanings that could be inappropriate
            "top", "bottom", "versatile", "dominant", "submissive", "member", "head", 
            "package", "box", "cherry", "balls", "tool", "ride", "hole", "bang",
            # Words that need special handling for context
            "glasses", "glass", "fly", "chest", "pipe", "hot"
        ])
        
        # Words that we always want to use our templates for (never API)
        # This includes clothing, eyewear, and potential problem words
        self.template_only_words = set([
            # Eyewear
            "glasses", "sunglasses", "spectacles", "contacts", "goggles", "eyeglasses", "shades",
            # Clothing items (especially ones with other meanings)
            "top", "pants", "shorts", "jacket", "coat", "shirt", "dress", "skirt", "tie", "belt",
            # Words with dual meanings
            "bottom", "fly", "chest", "box", "member", "glasses", "glass"
        ])
        
        # Words that are normally plural (use special templates)
        self.plural_words = set([
            "glasses", "pants", "shorts", "scissors", "jeans", "trousers", "tights",
            "goggles", "eyeglasses", "spectacles", "sunglasses", "shades", "headphones"
        ])
    
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
            original_word = word
            word = word.strip().lower()
            
            if self.debug:
                print(f"\n>>> Getting example for: '{word}'")
            
            # Determine the appropriate category for this word
            determined_category = self._get_word_category(word)
            if determined_category and self.debug:
                print(f"Determined category: {determined_category}")
                
            # For words that should ALWAYS use our templates, skip the API
            if word in self.template_only_words or determined_category in self.TEMPLATE_ONLY_CATEGORIES:
                if self.debug:
                    print(f"'{word}' is a template-only word or in a template-only category")
                return self._get_template_example(word, target_language, determined_category)
            
            # Check cache first (but not for problematic words)
            if word not in self.problematic_words:
                cached_example = self._get_cached_example(word, target_language)
                if cached_example:
                    return cached_example
            
            # Try API methods first, but only if not a problematic word
            if word not in self.problematic_words:
                methods = [
                    self._get_free_dictionary_example,
                    self._get_wordnik_example
                ]
                
                for method in methods:
                    example = method(word, target_language, determined_category)
                    if example and example["english"]:
                        # Verify the example is appropriate and contains the exact word
                        if self._is_appropriate_example(example["english"], word):
                            # Cache the successful example
                            self._cache_example(word, target_language, example)
                            return example
            
            # Fall back to template-based examples
            return self._get_template_example(word, target_language, determined_category)
            
        except Exception as e:
            if self.debug:
                print(f"Error getting example sentence: {e}")
                
            # Ultimate fallback
            return {
                "english": f"This is a {word}.",
                "translated": self._translate(f"This is a {word}.", target_language),
                "source": "basic_fallback"
            }
    
    def _get_word_category(self, word):
        """Determine the semantic category of a word."""
        # First check our explicit mapping
        if word in self.word_to_category:
            return self.word_to_category[word]
        
        # No category found
        return None
    
    def _is_appropriate_example(self, text, word):
        """
        Comprehensive check if an example is appropriate for the word.
        Combines multiple checks into one decision.
        """
        # 1. Basic safety check
        if not self._is_safe_content(text):
            if self.debug:
                print(f"REJECTED - Unsafe content: '{text}'")
            return False
        
        # 2. Check for exact word match with boundaries
        if not self._contains_exact_word(text, word):
            if self.debug:
                print(f"REJECTED - Doesn't contain exact word: '{text}'")
            return False
        
        # 3. Check for confusable words
        if word in self.confusable_words:
            for confusable in self.confusable_words[word]:
                if self._contains_exact_word(text, confusable):
                    if self.debug:
                        print(f"REJECTED - Contains confusable word '{confusable}': '{text}'")
                    return False
        
        # 4. Length and quality checks
        word_count = len(text.split())
        if word_count < 4 or word_count > 18:
            if self.debug:
                print(f"REJECTED - Bad length ({word_count} words): '{text}'")
            return False
        
        # 5. Check for metadata patterns (not real examples)
        if "example of" in text.lower() or "examples of" in text.lower() or ":" in text:
            if self.debug:
                print(f"REJECTED - Contains metadata: '{text}'")
            return False
        
        # 6. Semantic context check for special words
        if word in self.word_to_category:
            category = self.word_to_category[word]
            if category == "eyewear" and any(w in text.lower() for w in ["window", "fiber", "fibre"]):
                if self.debug:
                    print(f"REJECTED - Wrong context for eyewear: '{text}'")
                return False
                
            if category == "clothing" and any(w in text.lower() for w in ["mountain", "hill", "stop"]):
                if self.debug:
                    print(f"REJECTED - Wrong context for clothing: '{text}'")
                return False
        
        # 7. Special case for "glasses" to avoid confusion with "glass"
        if word == "glasses":
            # Make sure it's actually about eyewear
            eyewear_indicators = ["see", "vision", "read", "eyes", "wear", "prescription", "lens"]
            if not any(indicator in text.lower() for indicator in eyewear_indicators):
                if self.debug:
                    print(f"REJECTED - 'glasses' without eyewear context: '{text}'")
                return False
        
        # Passed all checks
        if self.debug:
            print(f"ACCEPTED: '{text}'")
        return True
    
    def _contains_exact_word(self, text, word):
        """Check if text contains the exact word with proper word boundaries."""
        # Create regex pattern with word boundaries
        pattern = r'\b' + re.escape(word) + r'\b'
        
        # Check if the text contains the exact word (case insensitive)
        matches = re.search(pattern, text.lower())
        
        # For debugging
        if self.debug and not matches:
            print(f"No exact match for '{word}' in: '{text}'")
            
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
        if word in self.plural_words or (word.endswith('s') and word != "glass"):
            # These words are typically plural
            english_example = template.replace("a {word}", "{word}").replace("the {word}", "the {word}").format(word=word)
        else:
            english_example = template.format(word=word)
        
        # Translate to target language
        translated_example = self._translate(english_example, target_language)
        
        if self.debug:
            print(f"TEMPLATE EXAMPLE: '{english_example}'")
        
        return {
            "english": english_example,
            "translated": translated_example,
            "source": f"{category}_template"
        }
    
    def _is_safe_content(self, text):
        """Check if content is appropriate and not offensive."""
        text_lower = text.lower()
        
        # General profanity/inappropriate content check
        inappropriate_words = [
            "sex", "porn", "nude", "naked", "xxx", "adult", 
            "fuck", "shit", "damn", "ass", "bitch", "cunt", "dick", "cock", "pussy",
            "erotic", "masturbate", "orgasm"
        ]
        
        for word in inappropriate_words:
            if re.search(r'\b' + re.escape(word) + r'\b', text_lower):
                if self.debug:
                    print(f"Inappropriate word detected: '{word}' in '{text}'")
                return False
        
        # Format check - avoid fragment examples like "Glass frog; glass shrimp"
        if ";" in text or not text.strip().endswith(('.', '!', '?')):
            if self.debug:
                print(f"Bad format detected in: '{text}'")
            return False
        
        # Length check
        if len(text.split()) < 3:
            if self.debug:
                print(f"Too short: '{text}'")
            return False
            
        # Basic relevance check
        if "example:" in text_lower or "examples:" in text_lower:
            if self.debug:
                print(f"Contains metadata: '{text}'")
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
                    
                    # Double-check appropriateness even for cached examples
                    if not self._is_appropriate_example(example["english"], word):
                        if self.debug:
                            print(f"Rejected cached example: '{example['english']}'")
                        return None
                        
                    return example
            except Exception as e:
                if self.debug:
                    print(f"Cache read error: {e}")
        return None
    
    def _cache_example(self, word, target_language, example):
        """Cache an example for future use."""
        # Don't cache problematic words to ensure we always generate fresh examples for them
        if word in self.problematic_words:
            return
            
        cache_file = f"{self.cache_dir}/{word}_{target_language}.json"
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(example, f, ensure_ascii=False)
        except Exception as e:
            if self.debug:
                print(f"Cache write error: {e}")
    
    def _get_free_dictionary_example(self, word, target_language, category=None):
        """Get example from Free Dictionary API with strict appropriateness checks."""
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
                            # Only add if it contains the exact word
                            if self._contains_exact_word(example, word):
                                # Add context about part of speech
                                examples.append((example, part_of_speech))
            
            if examples:
                # Filter for appropriate examples
                good_examples = []
                for example, pos in examples:
                    if self._is_appropriate_example(example, word):
                        good_examples.append((example, pos))
                
                if not good_examples:
                    return None
                
                # Select a random example
                example_text, pos = random.choice(good_examples)
                
                # Clean up the example
                example_text = self._clean_example(example_text)
                
                # Translate
                translated_example = self._translate(example_text, target_language)
                
                return {
                    "english": example_text,
                    "translated": translated_example,
                    "source": "free_dictionary_api"
                }
            
            return None
        except Exception as e:
            if self.debug:
                print(f"Free Dictionary API error: {e}")
            return None
    
    def _get_wordnik_example(self, word, target_language, category=None):
        """Get example from Wordnik API with strict appropriateness checks."""
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
                    # Clean up the text
                    text = text.replace('"', '').replace('"', '')
                    # Only add if it contains the exact word
                    if self._contains_exact_word(text, word):
                        examples.append(text)
            
            if examples:
                # Filter for appropriate examples
                good_examples = [ex for ex in examples if self._is_appropriate_example(ex, word)]
                
                if not good_examples:
                    return None
                
                # Select a random example
                example_text = random.choice(good_examples)
                
                # Clean up the example
                example_text = self._clean_example(example_text)
                
                # Translate
                translated_example = self._translate(example_text, target_language)
                
                return {
                    "english": example_text,
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