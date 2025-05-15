"""
Extra-Strict Example Sentence Generator
--------------------------------------
Special hardcoded protection for the word "person" with explicit whitelist approach.
"""

import requests
import os
import json
import numpy as np
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
        self._initialize_semantic_lists()
        
    def setup_cache_dir(self):
        """Create cache directory if it doesn't exist."""
        self.cache_dir = "sentence_cache"
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def _initialize_semantic_lists(self):
        """Initialize comprehensive lists for semantic categorization."""
        # CRITICAL: List of person-related words that need special handling
        self.PERSON_WORDS = set([
            # General person terms
            "person", "man", "woman", "boy", "girl", "child", "adult", "teenager",
            "baby", "toddler", "senior", "individual", "citizen", "human", "people",
            
            # Family terms
            "mother", "father", "parent", "brother", "sister", "aunt", "uncle",
            "grandfather", "grandmother", "grandparent", "cousin", "relative", "family",
            
            # Professions/roles
            "doctor", "nurse", "teacher", "student", "professor", "lawyer", "chef",
            "driver", "worker", "engineer", "artist", "musician", "actor", "actress",
            "scientist", "researcher", "writer", "author", "athlete", "player", "coach",
            "banker", "clerk", "cashier", "manager", "employee", "boss", "secretary",
            "assistant", "guard", "officer", "police", "firefighter", "soldier", "pilot",
            "captain", "sailor", "farmer", "gardener", "architect", "designer", "dentist",
            "waiter", "waitress", "bartender", "cook", "baker", "butcher", "carpenter",
            "plumber", "electrician", "mechanic", "technician", "programmer", "developer",
            
            # Relationship terms
            "friend", "neighbor", "colleague", "partner", "spouse", "husband", "wife",
            "boyfriend", "girlfriend", "fiancé", "fiancée", "companion", "roommate",
            "guest", "host", "stranger", "acquaintance", "client", "customer", "patient",
            
            # General groups
            "group", "crowd", "team", "staff", "crew", "audience", "community", "society"
        ])
        
        # CRITICAL: List of forbidden patterns specifically for person words
        self.FORBIDDEN_PERSON_PATTERNS = [
            "need a new", "need new", "bought a new", "bought new", 
            "is very useful", "are very useful", "is useful", "are useful",
            "on the table", "in the kitchen", "in the house", "in the room",
            "I bought", "I sold", "I own", "I have a", "I have an",
            "I use", "I used", "I borrowed", "I returned", "I broke", "I fixed", 
            "I cleaned", "I washed", "I replaced", "I ordered", "I want",
            "I like this", "I like that", "I like the", "I need this", "I need that",
            "very useful", "quite useful", "so useful", "really useful",
            "this is a useful", "these are useful", "a useful",
            "used for", "can be used", "using the", "used the", 
            "belongs to", "new", "old", "fancy", "expensive", "cheap"
        ]

        # ABSOLUTELY GUARANTEED SAFE templates for person words
        # Using a whitelist approach rather than a blacklist
        self.PERSON_TEMPLATES = {
            # Default templates for most person words
            "default": [
                "The {word} is standing near the entrance.",
                "I saw a {word} at the park yesterday.",
                "A {word} asked me for directions.",
                "We met a friendly {word} on our trip.",
                "The {word} was waiting at the bus stop.",
                "My neighbor is a {word} who lives downtown.",
                "She has been a {word} for five years.",
                "The {word} waved hello to us.",
                "A {word} helped me find my way.",
                "The {word} was reading a book."
            ],
            
            # Special handling for "person" specifically to guarantee no issues
            "person": [
                "I met a person who speaks five languages.",
                "There was a person waiting at the reception desk.",
                "A person called to ask about the schedule.",
                "The person at the information desk was very helpful.",
                "I saw a person walking their dog in the park.",
                "There's a person here to see you.",
                "The first person to arrive gets the prize.",
                "A person from the company called earlier.",
                "Each person has their own unique talents.",
                "The person who found my wallet returned it to me."
            ]
        }
        
        # Animals list
        self.ANIMAL_WORDS = set([
            "dog", "cat", "bird", "fish", "horse", "cow", "elephant", "lion", "tiger",
            "bear", "rabbit", "monkey", "mouse", "rat", "frog", "turtle", "snake",
            "lizard", "alligator", "crocodile", "zebra", "giraffe", "deer", "fox",
            "wolf", "sheep", "goat", "pig", "chicken", "rooster", "duck", "goose",
            "penguin", "eagle", "hawk", "owl", "parrot", "canary", "hamster", "guinea pig",
            "squirrel", "butterfly", "bee", "ant", "spider", "scorpion", "crab", "lobster",
            "shrimp", "whale", "dolphin", "shark", "octopus", "squid", "seal", "walrus",
            "otter", "beaver", "hedgehog", "bat", "camel", "kangaroo", "koala", "panda"
        ])

        # Food list
        self.FOOD_WORDS = set([
            "apple", "banana", "orange", "grape", "strawberry", "blueberry", "raspberry",
            "watermelon", "melon", "pineapple", "peach", "pear", "cherry", "plum", "kiwi",
            "mango", "coconut", "avocado", "tomato", "potato", "carrot", "broccoli",
            "spinach", "lettuce", "cucumber", "onion", "garlic", "pepper", "corn",
            "rice", "wheat", "flour", "bread", "toast", "cake", "cookie", "pie",
            "chocolate", "candy", "sugar", "salt", "pepper", "cinnamon", "spice",
            "butter", "oil", "milk", "cheese", "yogurt", "cream", "egg", "meat",
            "beef", "pork", "chicken", "turkey", "fish", "salmon", "tuna", "shrimp",
            "soup", "stew", "salad", "sandwich", "hamburger", "pizza", "pasta", "noodle"
        ])

        # Uncountable nouns
        self.UNCOUNTABLE_NOUNS = set([
            "water", "air", "money", "food", "rice", "coffee", "tea", "milk", "bread", 
            "sugar", "salt", "oil", "vinegar", "soup", "advice", "information", "news", 
            "furniture", "luggage", "traffic", "weather", "homework", "work", "fun",
            "knowledge", "wisdom", "happiness", "sadness", "anger", "fear", "courage",
            "patience", "intelligence", "research", "education", "health", "safety", "time"
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
            word = word.strip().lower()
            
            # CRITICAL: Extra special handling for "person"
            if word == "person":
                if self.debug:
                    print("USING HARDCODED SAFE EXAMPLE FOR 'PERSON'")
                return self._get_person_specific_example(target_language)
            
            # Check if this is a person word that needs special handling
            is_person_word = word in self.PERSON_WORDS
            if self.debug and is_person_word:
                print(f"PERSON WORD DETECTED: '{word}'")
            
            # Check cache first (except for person words - always generate fresh)
            if not is_person_word:
                cached_example = self._get_cached_example(word, target_language)
                if cached_example:
                    return cached_example
            
            # For person words, always use safe templates
            if is_person_word:
                return self._get_safe_person_example(word, target_language)
            
            # For non-person words, try API methods first
            methods = [
                self._get_free_dictionary_example,
                self._get_wordnik_example
            ]
            
            for method in methods:
                example = method(word, target_language, category)
                if example and example["english"]:
                    # Verify safety even for API examples
                    if self._is_safe_example(example["english"], word):
                        self._cache_example(word, target_language, example)
                        return example
            
            # Fall back to template-based examples
            if word in self.ANIMAL_WORDS:
                example = self._get_animal_example(word, target_language)
            elif word in self.FOOD_WORDS:
                example = self._get_food_example(word, target_language)
            elif word in self.UNCOUNTABLE_NOUNS:
                example = self._get_uncountable_example(word, target_language)
            else:
                example = self._get_general_example(word, target_language, category)
                
            # Cache non-person examples
            if example and example["english"] and not is_person_word:
                self._cache_example(word, target_language, example)
                
            return example
            
        except Exception as e:
            print(f"Error getting example sentence: {e}")
            # Ultimate fallback
            if word == "person":
                return {
                    "english": "I met a person at the conference.",
                    "translated": self._translate("I met a person at the conference.", target_language),
                    "source": "error_fallback_person"
                }
            else:
                return {
                    "english": f"Here is the word '{word}'.",
                    "translated": self._translate(f"Here is the word '{word}'.", target_language),
                    "source": "error_fallback"
                }
    
    def _get_person_specific_example(self, target_language):
        """Get a guaranteed safe example specifically for the word 'person'."""
        # Use only from the dedicated "person" templates
        template = np.random.choice(self.PERSON_TEMPLATES["person"])
        english_example = template.format(word="person")
        translated_example = self._translate(english_example, target_language)
        
        if self.debug:
            print(f"PERSON-SPECIFIC EXAMPLE: '{english_example}'")
            
        return {
            "english": english_example,
            "translated": translated_example,
            "source": "safe_person_specific"
        }
    
    def _is_safe_example(self, example, word):
        """Check if an example is safe to use, especially for person words."""
        example_lower = example.lower()
        
        # Extra strict check for 'person'
        if word == "person":
            # For the word "person", ONLY allow examples from our explicit whitelist
            return False
        
        # Check if this is a person word
        if word in self.PERSON_WORDS:
            # For person words, check against forbidden patterns
            for pattern in self.FORBIDDEN_PERSON_PATTERNS:
                if pattern in example_lower:
                    if self.debug:
                        print(f"REJECTED PERSON EXAMPLE: '{example}' - contains '{pattern}'")
                    return False
        
        # General checks for all words
        if "example of" in example_lower or "examples of" in example_lower:
            return False
            
        if len(example.split()) < 3 or len(example.split()) > 20:
            return False
            
        return True
    
    def _get_safe_person_example(self, word, target_language):
        """Get a guaranteed safe example for a person word."""
        # Select a random safe template
        templates = self.PERSON_TEMPLATES["default"]  # Use default templates for most person words
        template = np.random.choice(templates)
        
        # Apply the template
        english_example = template.format(word=word)
        
        # Translate to target language
        translated_example = self._translate(english_example, target_language)
        
        if self.debug:
            print(f"SAFE PERSON EXAMPLE GENERATED: '{english_example}'")
            
        return {
            "english": english_example,
            "translated": translated_example,
            "source": "safe_person_template"
        }
    
    def _get_animal_example(self, word, target_language):
        """Safe examples for animals."""
        templates = [
            f"The {word} is running in the park.",
            f"I saw a {word} at the zoo yesterday.",
            f"The {word} is sleeping under the tree.",
            f"My friend has a {word} as a pet.",
            f"That {word} looks very friendly.",
            f"We watched the {word} playing in the garden.",
            f"The {word} was eating some food.",
            f"There's a {word} on the path ahead.",
            f"The children love watching the {word}.",
            f"Can you see the {word} over there?"
        ]
        
        english_example = np.random.choice(templates)
        translated_example = self._translate(english_example, target_language)
        
        return {
            "english": english_example,
            "translated": translated_example,
            "source": "animal_template"
        }
    
    def _get_food_example(self, word, target_language):
        """Safe examples for food."""
        templates = [
            f"This {word} tastes delicious.",
            f"I love eating {word} for breakfast.",
            f"My mom makes the best {word}.",
            f"Would you like some {word}?",
            f"The {word} is fresh from the market.",
            f"She bought some {word} yesterday.",
            f"We need more {word} for the recipe.",
            f"The {word} smells wonderful.",
            f"I prefer {word} over other foods.",
            f"They serve excellent {word} at that restaurant."
        ]
        
        english_example = np.random.choice(templates)
        translated_example = self._translate(english_example, target_language)
        
        return {
            "english": english_example,
            "translated": translated_example,
            "source": "food_template"
        }
    
    def _get_uncountable_example(self, word, target_language):
        """Safe examples for uncountable nouns."""
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
            "source": "uncountable_template"
        }
    
    def _get_general_example(self, word, target_language, category=None):
        """Safe general examples for other words."""
        # Choose appropriate article
        if word[0].lower() in 'aeiou':
            article = "an"
        else:
            article = "a"
            
        templates = [
            f"I see {article} {word} on the table.",
            f"She has {article} {word} in her bag.",
            f"The {word} is blue.",
            f"Can you give me the {word}?",
            f"Where is the {word}?",
            f"This {word} belongs to my friend.",
            f"I found {article} {word} in the drawer.",
            f"The {word} is very nice.",
            f"He likes that {word} very much.",
            f"Look at this {word}!"
        ]
        
        english_example = np.random.choice(templates)
        translated_example = self._translate(english_example, target_language)
        
        return {
            "english": english_example,
            "translated": translated_example,
            "source": "general_template"
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
        # Critical: Never use caching for "person" to be extra safe
        if word == "person":
            return None
            
        cache_file = f"{self.cache_dir}/{word}_{target_language}.json"
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    example = json.load(f)
                    
                    # Safety: Even for cached examples, verify they're safe
                    if not self._is_safe_example(example["english"], word):
                        if self.debug:
                            print(f"REJECTED CACHED EXAMPLE: '{example['english']}'")
                        return None
                        
                    return example
            except Exception as e:
                print(f"Cache read error: {e}")
        return None
    
    def _cache_example(self, word, target_language, example):
        """Cache an example for future use."""
        # Never cache person examples to ensure we always use fresh, safe examples
        if word in self.PERSON_WORDS or word == "person":
            return
            
        cache_file = f"{self.cache_dir}/{word}_{target_language}.json"
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(example, f, ensure_ascii=False)
        except Exception as e:
            print(f"Cache write error: {e}")
    
    def _get_free_dictionary_example(self, word, target_language, category=None):
        """Get example from Free Dictionary API with safety checks."""
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
                # Filter for safe examples
                safe_examples = []
                for example in examples:
                    if self._is_safe_example(example, word):
                        safe_examples.append(example)
                
                if not safe_examples:
                    return None
                    
                # Choose example with good length
                good_examples = [ex for ex in safe_examples if 4 <= len(ex.split()) <= 12]
                
                if good_examples:
                    english_example = np.random.choice(good_examples)
                else:
                    english_example = np.random.choice(safe_examples)
                
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
            print(f"Free Dictionary API error: {e}")
            return None
    
    def _get_wordnik_example(self, word, target_language, category=None):
        """Get example from Wordnik API with safety checks."""
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
                    examples.append(text)
            
            if examples:
                # Filter for safe examples that contain the word
                safe_examples = []
                for example in examples:
                    if word.lower() in example.lower() and self._is_safe_example(example, word):
                        safe_examples.append(example)
                
                if not safe_examples:
                    return None
                    
                # Choose an example with good length
                good_examples = [ex for ex in safe_examples if 5 <= len(ex.split()) <= 12]
                
                if good_examples:
                    english_example = np.random.choice(good_examples)
                else:
                    english_example = np.random.choice(safe_examples)
                
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