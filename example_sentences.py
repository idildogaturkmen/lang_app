"""
Context-Aware Example Sentence Generator
----------------------------------------
Prioritizes API examples with strong filtering, with specialized templates for different word types
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
        print("********** NEW IMPLEMENTATION LOADED! **********")
        self.translate_func = translate_func
        self.debug = debug
        self.setup_cache_dir()
        self._initialize_word_data()
        
    def setup_cache_dir(self):
        """Create cache directory if it doesn't exist."""
        self.cache_dir = "sentence_cache"
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def _initialize_word_data(self):
        """Initialize comprehensive word data for context-aware example generation."""
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
        
        # Person-related words that need special handling
        self.person_words = set([
            "person", "man", "woman", "boy", "girl", "child", "adult", "teenager",
            "baby", "toddler", "senior", "individual", "human", "people",
            "mother", "father", "parent", "brother", "sister", "aunt", "uncle",
            "grandfather", "grandmother", "cousin", "family", "friend", "neighbor",
            "student", "teacher", "doctor", "nurse", "patient", "client", "customer",
            "chef", "lawyer", "engineer", "scientist", "artist", "musician", "actor",
            "actress", "writer", "author", "athlete", "player", "coach", "police",
            "firefighter", "soldier", "pilot", "farmer", "gardener"
        ])
        
        # Word-to-category mapping
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
            
            # Additional mappings - we'll add person words dynamically
            "dog": "animals",
            "cat": "animals",
            "chair": "furniture",
            "table": "furniture",
            "phone": "electronics",
            "computer": "electronics",
            "car": "vehicles",
            "bike": "vehicles"
        }
        
        # Add all person words to the category mapping
        for word in self.person_words:
            self.word_to_category[word] = "person"
        
        # Template categories for fallback - with specialized templates for each category
        self.category_templates = {
            # Person templates - appropriate for humans
            "person": [
                "The {word} asked for directions to the station.",
                "I met a {word} who speaks five languages.",
                "A {word} was waiting at the bus stop.",
                "She is a friendly {word} who always smiles.",
                "The {word} helped me carry my groceries.",
                "We saw a {word} walking in the park yesterday.",
                "The {word} gave an interesting presentation.",
                "A {word} called to ask about the opening hours.",
                "The {word} at the reception desk was very helpful.",
                "My neighbor is a {word} who works at the hospital."
            ],
            
            # Eyewear templates
            "eyewear": [
                "I can't find my {word} anywhere.",
                "These {word} help me see better.",
                "She wears {word} for reading.",
                "My {word} are scratched and need to be replaced.",
                "He forgot his {word} at home today.",
                "The doctor prescribed new {word} for my vision.",
                "These {word} protect my eyes from the sun.",
                "I bought new {word} at the optician yesterday.",
                "The {word} are on the nightstand.",
                "I need to clean my {word}."
            ],
            
            # Clothing templates
            "clothing": [
                "I bought a new {word} for the party.",
                "This {word} is very comfortable to wear.",
                "She likes wearing a blue {word} with jeans.",
                "The {word} is hanging in the closet.",
                "I need to wash my favorite {word}.",
                "He gave me a {word} as a gift.",
                "The store sells different styles of {word}.",
                "My sister borrowed my {word} yesterday.",
                "This {word} doesn't fit me anymore.",
                "I spilled coffee on my {word} this morning."
            ],
            
            # Animals
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
            
            # Electronics
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
            
            # Furniture
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
            
            # Vehicles
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
            
            # Drinkware
            "drinkware": [
                "I filled the {word} with water.",
                "She dropped the {word} and it broke.",
                "This {word} keeps my coffee hot for hours.",
                "The {word} is made of ceramic.",
                "I need to wash this {word}.",
                "He drinks tea from his favorite {word}.",
                "We have a set of six {word}s for guests.",
                "The {word} is on the kitchen counter.",
                "My {word} has a small chip on the rim.",
                "This {word} can hold up to 12 ounces."
            ],
            
            # General
            "general": [
                "I placed the {word} on the table.",
                "Can you pass me that {word}, please?",
                "The {word} is in the drawer.",
                "She's looking for the {word}.",
                "This {word} belongs to my brother.",
                "I found the {word} under the couch.",
                "We need to buy a new {word}.",
                "The {word} is very useful.",
                "He keeps the {word} in his office.",
                "This {word} works very well."
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
            
            # Determine the appropriate category for this word
            determined_category = self._get_word_category(word)
            
            # For person words, prioritize templates or apply stricter filtering
            if word in self.person_words:
                # Try to get a quality API example first
                api_example = self._get_api_example_with_human_context(word, target_language)
                if api_example:
                    return api_example
                
                # Fall back to person-specific template
                return self._get_template_example(word, target_language, "person")
            
            # For problematic words like "glasses", "top", etc.
            elif word in self.problematic_words:
                # Try context-specific API example
                context_example = self._get_context_specific_example(word, target_language)
                if context_example:
                    return context_example
                
                # Fall back to category-specific template
                return self._get_template_example(word, target_language, determined_category)
            
            # For regular words, try API first
            else:
                # Try regular API example
                api_example = self._get_api_example(word, target_language)
                if api_example:
                    return api_example
                
                # Fall back to template
                return self._get_template_example(word, target_language, determined_category)
            
        except Exception as e:
            if self.debug:
                print(f"Error getting example sentence: {e}")
            
            # Fallback
            if word in self.person_words:
                fallback = f"The {word} is waiting outside."
            else:
                fallback = f"This is a {word}."
                
            return {
                "english": fallback,
                "translated": self._translate(fallback, target_language),
                "source": "error_fallback"
            }
    
    def _get_word_category(self, word):
        """Determine the semantic category of a word."""
        # Check explicit mapping first
        if word in self.word_to_category:
            return self.word_to_category[word]
        
        # Check if it's a person word
        if word in self.person_words:
            return "person"
        
        # Default category
        return "general"
    
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
    
    def _get_api_example_with_human_context(self, word, target_language):
        """Get example from APIs with human context filtering."""
        # Try to get examples from Free Dictionary
        examples = self._get_all_free_dictionary_examples(word)
        
        # Filter for examples that have appropriate human context
        good_examples = []
        for example in examples:
            # Check if the example uses the word in a human context
            if self._has_human_context(example, word):
                good_examples.append(example)
        
        if good_examples:
            # Choose a random example
            english_example = random.choice(good_examples)
            
            # Clean up the example
            english_example = self._clean_example(english_example)
            
            # Translate
            translated_example = self._translate(english_example, target_language)
            
            return {
                "english": english_example,
                "translated": translated_example,
                "source": "free_dictionary_api_human_context"
            }
        
        # If no good examples from Free Dictionary, try Wordnik
        examples = self._get_all_wordnik_examples(word)
        
        # Filter for examples that have appropriate human context
        good_examples = []
        for example in examples:
            # Check if the example uses the word in a human context
            if self._has_human_context(example, word):
                good_examples.append(example)
                
        if good_examples:
            # Choose a random example
            english_example = random.choice(good_examples)
            
            # Clean up the example
            english_example = self._clean_example(english_example)
            
            # Translate
            translated_example = self._translate(english_example, target_language)
            
            return {
                "english": english_example,
                "translated": translated_example,
                "source": "wordnik_api_human_context"
            }
        
        return None
    
    def _has_human_context(self, text, word):
        """Check if text has appropriate human context for person words."""
        # Skip examples that objectify people
        objectifying_patterns = [
            r"use.{0,10}" + re.escape(word),
            r"uses.{0,10}" + re.escape(word),
            r"using.{0,10}" + re.escape(word),
            r"need.{0,10}" + re.escape(word),
            r"needs.{0,10}" + re.escape(word),
            r"buy.{0,10}" + re.escape(word),
            r"buys.{0,10}" + re.escape(word),
            r"bought.{0,10}" + re.escape(word),
            r"sell.{0,10}" + re.escape(word),
            r"sells.{0,10}" + re.escape(word),
            r"sold.{0,10}" + re.escape(word),
        ]
        
        for pattern in objectifying_patterns:
            if re.search(pattern, text.lower()):
                if self.debug:
                    print(f"Rejected human example due to objectification: '{text}'")
                return False
        
        # Look for positive human context
        human_action_patterns = [
            r"" + re.escape(word) + r".{0,10}talk",
            r"" + re.escape(word) + r".{0,10}speak",
            r"" + re.escape(word) + r".{0,10}said",
            r"" + re.escape(word) + r".{0,10}asked",
            r"" + re.escape(word) + r".{0,10}help",
            r"" + re.escape(word) + r".{0,10}walk",
            r"" + re.escape(word) + r".{0,10}met",
            r"" + re.escape(word) + r".{0,10}saw",
            r"talk.{0,10}to.{0,10}" + re.escape(word),
            r"speak.{0,10}to.{0,10}" + re.escape(word),
            r"meet.{0,10}" + re.escape(word),
            r"saw.{0,10}" + re.escape(word),
            r"met.{0,10}" + re.escape(word),
            r"know.{0,10}" + re.escape(word),
            r"ask.{0,10}" + re.escape(word),
            r"help.{0,10}" + re.escape(word),
        ]
        
        # If we find any positive human context pattern, it's probably a good example
        for pattern in human_action_patterns:
            if re.search(pattern, text.lower()):
                return True
        
        # Check if the sentence is generally appropriate
        # (doesn't contain obviously objectifying language)
        if self._is_valid_sentence(text) and self._contains_exact_word(text, word):
            return True
            
        return False
    
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
    
    def _get_template_example(self, word, target_language, category=None):
        """Get a template example based on word category."""
        # Determine category if not provided
        if not category:
            category = self._get_word_category(word)
        
        # Get templates for this category
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
            "source": f"template_{category}"
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
    
    def _get_all_free_dictionary_examples(self, word):
        """Get all examples from Free Dictionary API."""
        try:
            url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}"
            response = requests.get(url, timeout=5)
            
            if response.status_code != 200:
                return []
                
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
            
            return examples
        except Exception as e:
            if self.debug:
                print(f"Free Dictionary API error: {e}")
            return []
    
    def _get_all_wordnik_examples(self, word):
        """Get all examples from Wordnik API."""
        try:
            url = f"https://api.wordnik.com/v4/word.json/{word}/examples"
            response = requests.get(url, timeout=5)
            
            if response.status_code != 200:
                return []
                
            data = response.json()
            
            examples = []
            for example in data.get('examples', []):
                if 'text' in example and example['text']:
                    text = example['text']
                    if self._contains_exact_word(text, word) and self._is_valid_sentence(text):
                        examples.append(text)
            
            return examples
        except Exception as e:
            if self.debug:
                print(f"Wordnik API error: {e}")
            return []
    
    def _get_free_dictionary_example(self, word, target_language):
        """Get example from Free Dictionary API with basic filtering."""
        examples = self._get_all_free_dictionary_examples(word)
        
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
    
    def _get_free_dictionary_example_with_context(self, word, target_language):
        """Get example from Free Dictionary API with strict context filtering."""
        examples = self._get_all_free_dictionary_examples(word)
        
        # Filter for examples that meet context requirements
        good_examples = []
        for example in examples:
            if self._check_context_requirements(example, word):
                # For base words, make sure their variants don't appear
                if word in self.base_words:
                    # Skip if the example contains any of the variants
                    if not self._contains_any_word(example, self.base_words[word]):
                        good_examples.append(example)
                else:
                    good_examples.append(example)
        
        if good_examples:
            # Choose a random example
            english_example = random.choice(good_examples)
            
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
    
    def _get_wordnik_example(self, word, target_language):
        """Get example from Wordnik API with basic filtering."""
        examples = self._get_all_wordnik_examples(word)
        
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
    
    def _get_wordnik_example_with_context(self, word, target_language):
        """Get example from Wordnik API with strict context filtering."""
        examples = self._get_all_wordnik_examples(word)
        
        # Filter for examples that meet context requirements
        good_examples = []
        for example in examples:
            if self._check_context_requirements(example, word):
                # For base words, make sure their variants don't appear
                if word in self.base_words:
                    # Skip if the example contains any of the variants
                    if not self._contains_any_word(example, self.base_words[word]):
                        good_examples.append(example)
                else:
                    good_examples.append(example)
        
        if good_examples:
            # Choose a random example
            english_example = random.choice(good_examples)
            
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