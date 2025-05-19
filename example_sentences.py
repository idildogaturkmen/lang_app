"""
Filter Pipeline Example Sentence Generator
------------------------------------------
A systematic approach to ensuring relevant examples for any word
"""

import requests
import os
import random
import re
import time
from functools import lru_cache

class ExampleSentenceGenerator:
    def __init__(self, translate_func=None, debug=False):
        """Initialize the generator with a multi-stage filter pipeline."""
        print("\n>>> FILTER PIPELINE GENERATOR LOADED <<<\n")
        self.translate_func = translate_func
        self.debug = debug  # Set to True to see detailed rejection reasons
        self.setup_cache_dir()
        self._initialize_templates()
    
    def setup_cache_dir(self):
        """Create cache directory if it doesn't exist."""
        self.cache_dir = "sentence_cache"
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def _initialize_templates(self):
        """Initialize templates for fallback cases."""
        # Templates for different parts of speech
        self.templates = {
            "noun": [
                "The {word} is on the table.",
                "I can see a {word} in the room.",
                "She bought a new {word} yesterday.",
                "This {word} is very useful.",
                "We need another {word} for this project.",
                "My friend has a {word} at home.",
                "The {word} costs twenty dollars.",
                "I like this {word} very much.",
                "Can you pass me the {word}, please?",
                "He gave me a {word} as a present."
            ],
            "person": [
                "The {word} asked for directions.",
                "I met a {word} at the conference.",
                "The {word} was very helpful.",
                "She is a kind {word} who helps others.",
                "The {word} speaks three languages.",
                "We need to hire a new {word} for this position.",
                "The {word} left a few minutes ago.",
                "I saw the {word} in the park yesterday.",
                "The {word} smiled and waved hello.",
                "A {word} called to ask about our services."
            ],
            "animal": [
                "The {word} is sleeping under the tree.",
                "I saw a {word} at the zoo yesterday.",
                "My friend has a {word} as a pet.",
                "The {word} was eating some food.",
                "That {word} looks very friendly.",
                "The {word} ran quickly across the field.",
                "We watched the {word} playing in the garden.",
                "The {word} has beautiful fur.",
                "The children love watching the {word}.",
                "The {word} made a loud noise."
            ],
            "clothing": [
                "I bought a new {word} for the party.",
                "This {word} is very comfortable to wear.",
                "She likes wearing a blue {word}.",
                "The {word} is hanging in the closet.",
                "I need to wash my favorite {word}.",
                "He gave me a {word} as a gift.",
                "The store sells different styles of {word}.",
                "My sister borrowed my {word} yesterday.",
                "This {word} doesn't fit me anymore.",
                "I spilled coffee on my {word} this morning."
            ],
            "eyewear": [
                "I can't find my {word} anywhere.",
                "These {word} help me see better.",
                "She wears {word} for reading.",
                "My {word} are scratched and need to be replaced.",
                "He forgot his {word} at home today.",
                "The doctor prescribed new {word} for me.",
                "These {word} protect my eyes from the sun.",
                "I bought new {word} at the optician yesterday.",
                "The {word} are on the nightstand.",
                "I need to clean my {word}."
            ],
            "verb": [
                "I {word} every morning before breakfast.",
                "She {word} whenever she has free time.",
                "They {word} together every weekend.",
                "He will {word} later this afternoon.",
                "We {word} yesterday for hours.",
                "Do you {word} often?",
                "I don't usually {word} on Sundays.",
                "They are {word}ing right now.",
                "She has {word}ed three times today.",
                "We should {word} tomorrow if the weather is good."
            ],
            "adjective": [
                "That is a very {word} book.",
                "She wore a {word} dress to the party.",
                "The weather is {word} today.",
                "I bought a {word} car last week.",
                "This soup tastes {word}.",
                "He lives in a {word} neighborhood.",
                "The children were very {word} today.",
                "We stayed at a {word} hotel during our vacation.",
                "She has a {word} voice when she sings.",
                "The movie was {word} and entertaining."
            ],
            "general": [
                "The {word} is important in this context.",
                "I learned about {word} in school.",
                "She mentioned {word} in her presentation.",
                "We should consider {word} for this project.",
                "The book explains {word} in detail.",
                "He talked about {word} during the meeting.",
                "I'm interested in learning more about {word}.",
                "The article discusses {word} and its implications.",
                "She has experience with {word}.",
                "This relates to {word} in many ways."
            ],

            "tools": [
                "I use these {word} to cut paper.",
                "These {word} are very sharp.",
                "Can you hand me the {word}, please?",
                "The {word} are in the drawer.",
                "She's looking for her {word}.",
                "I need to buy new {word}.",
                "The {word} are perfect for this craft project.",
                "Be careful with those {word}.",
                "Do you have a pair of {word} I can borrow?",
                "These {word} are getting dull and need sharpening."
            ],
            "jewelry": [
                "She wore a beautiful {word} to the party.",
                "This {word} was a gift from my grandmother.",
                "The {word} matches her dress perfectly.",
                "I love your new {word}.",
                "That {word} looks expensive.",
                "She keeps her favorite {word} in a special box.",
                "He bought her a {word} for their anniversary.",
                "This {word} is made of gold.",
                "The {word} has a sparkling gemstone in the center.",
                "She never takes off her lucky {word}."
            ],
            "toys": [
                "The child loves playing with the {word}.",
                "This {word} is my favorite toy.",
                "She hugs her {word} when she sleeps.",
                "The {word} sits on a shelf in the child's room.",
                "He gave his little brother a {word} for his birthday.",
                "This {word} is soft and cuddly.",
                "My daughter won't go anywhere without her {word}.",
                "The {word} has been in our family for generations.",
                "We bought a new {word} at the toy store.",
                "The children share their {word} with each other."
            ],

            
        }
        
        # Common words in each category
        self.category_words = {
            "person": [
                "person", "man", "woman", "boy", "girl", "child", "teacher", "doctor", 
                "student", "friend", "neighbor", "parent", "employee", "worker"
            ],
            "animal": [
                "dog", "cat", "bird", "fish", "horse", "cow", "elephant", "lion", 
                "tiger", "bear", "rabbit", "monkey", "mouse", "frog", "snake"
            ],
            "clothing": [
                "shirt", "pants", "dress", "jacket", "coat", "hat", "gloves", "socks",
                "shoes", "boots", "sweater", "skirt", "jeans", "top", "scarf", "tie"
            ],
            "eyewear": [
                "glasses", "sunglasses", "contacts", "goggles", "spectacles", "eyeglasses"
            ],
            # Add these after your existing category_words definitions
            "jewelry": [
                "necklace", "ring", "bracelet", "earrings", "watch", "pendant", 
                "brooch", "pin", "chain", "locket", "anklet", "cufflinks"
            ],
            "tools": [
                "scissors", "knife", "hammer", "screwdriver", "wrench", "pliers",
                "saw", "drill", "tape measure", "level", "chisel", "clamp"
            ],
            "toys": [
                "teddy bear", "doll", "ball", "blocks", "action figure", "puzzle",
                "toy car", "stuffed animal", "plush toy", "game", "toy"
            ],
        }
        
        # Words that are typically used in plural form
        self.plural_words = [
            "glasses", "pants", "shorts", "jeans", "scissors", "trousers", "sunglasses",
            "goggles", "spectacles", "eyeglasses", "headphones", "tights", "leggings"
        ]
    
    def get_example_sentence(self, word, target_language, category=None):
        """Get an example sentence for a word using the filter pipeline."""
        try:
            # Normalize word
            original_word = word
            word = word.strip().lower()
            
            if self.debug:
                print(f"\n>>> Getting example for: '{word}' <<<\n")
            
            # Try to get example from API with filter pipeline
            examples = self._get_api_examples_with_pipeline(word)
            
            # If we have valid examples, choose one randomly and return
            if examples:
                example = random.choice(examples)
                if self.debug:
                    print(f">>> Selected API example: '{example}'")
                    
                # Clean up the example
                example = self._clean_sentence(example)
                
                # Translate
                translated = self._translate(example, target_language)
                
                return {
                    "english": example,
                    "translated": translated,
                    "source": "filtered_api"
                }
            
            # If no valid API examples, use template
            if self.debug:
                print(f">>> No valid API examples found, using template")
                
            # Determine appropriate template category
            template_category = self._get_template_category(word)
            
            # Get template and generate example
            templates = self.templates.get(template_category, self.templates["general"])
            template = random.choice(templates)
            
            # Handle plurals
            if word in self.plural_words:
                example = template.replace("a {word}", "{word}").replace("the {word}", "the {word}").format(word=word)
            else:
                example = template.format(word=word)
            
            # Translate
            translated = self._translate(example, target_language)
            
            return {
                "english": example,
                "translated": translated,
                "source": "template_" + template_category
            }
            
        except Exception as e:
            if self.debug:
                print(f">>> Error: {e}")
                
            # Ultimate fallback
            fallback = f"This is a {word}."
            return {
                "english": fallback,
                "translated": self._translate(fallback, target_language),
                "source": "error_fallback"
            }
    
    def _get_api_examples_with_pipeline(self, word):
        """Get examples from APIs and run them through the filter pipeline."""
        # Get raw examples from all API sources
        raw_examples = []
        
        # Try Free Dictionary API
        dict_examples = self._get_free_dictionary_examples(word)
        if dict_examples:
            raw_examples.extend(dict_examples)
            
        # Try Wordnik API
        wordnik_examples = self._get_wordnik_examples(word)
        if wordnik_examples:
            raw_examples.extend(wordnik_examples)
            
        if not raw_examples:
            if self.debug:
                print(">>> No examples found from APIs")
            return []
            
        # Apply filter pipeline to raw examples
        filtered_examples = []
        for example in raw_examples:
            # Run through all filters
            if self._filter_pipeline(example, word):
                filtered_examples.append(example)
                
        return filtered_examples
    
    def _filter_pipeline(self, example, word):
        """
        Run an example through the multi-stage filter pipeline.
        Returns True if the example passes all filters, False otherwise.
        """
        # Filter 1: Basic quality check
        if not self._basic_quality_check(example):
            if self.debug:
                print(f">>> Rejected (Basic quality): '{example}'")
            return False
            
        # Filter 2: Exact word match
        if not self._contains_exact_word(example, word):
            if self.debug:
                print(f">>> Rejected (No exact word match): '{example}'")
            return False
            
        # Filter 3: Compound word check
        if self._contains_compound_words(example, word):
            if self.debug:
                print(f">>> Rejected (Contains compound word): '{example}'")
            return False
            
        # Filter 4: Variant form check
        if self._contains_variant_forms(example, word):
            if self.debug:
                print(f">>> Rejected (Contains variant form): '{example}'")
            return False
            
        # Filter 5: Context appropriateness check
        if not self._check_context_appropriate(example, word):
            if self.debug:
                print(f">>> Rejected (Inappropriate context): '{example}'")
            return False
            
        # All filters passed
        if self.debug:
            print(f">>> Accepted: '{example}'")
        return True
    
    def _basic_quality_check(self, text):
        """Check if a sentence meets basic quality standards."""
        # Must be a reasonable length
        words = text.split()
        if len(words) < 3 or len(words) > 20:
            return False
            
        # Shouldn't contain semicolons (often indicates a list, not a sentence)
        if ';' in text:
            return False
            
        # Shouldn't be metadata
        if "example of" in text.lower() or "examples of" in text.lower() or "example:" in text.lower():
            return False
            
        # Check if it has a proper end punctuation
        if not text.strip().endswith(('.', '!', '?')):
            return False
            
        return True
    
    def _contains_exact_word(self, text, word):
        """Check if text contains the exact word with proper word boundaries."""
        pattern = r'\b' + re.escape(word) + r'\b'
        return re.search(pattern, text.lower()) is not None
    
    def _contains_compound_words(self, text, word):
        """Check if text contains compound words that include the target word."""
        # Skip for very short words (2 chars or less) to avoid too many false positives
        if len(word) <= 2:
            return False
            
        # Common prefixes and suffixes for compound words
        prefixes = ["re", "un", "in", "im", "dis", "non", "over", "under", "super", "sub"]
        suffixes = ["ing", "ed", "er", "ize", "ise", "able", "ible", "ful", "less", "ness", "ly", "ment", "tion", "sion", "ship"]
        
        # Check for exact word first (we want to exclude exact matches from this check)
        exact_pattern = r'\b' + re.escape(word) + r'\b'
        
        # Find all words in the text
        all_words = re.findall(r'\b\w+\b', text.lower())
        
        for text_word in all_words:
            # Skip exact match
            if text_word == word:
                continue
                
            # Check if the word is part of a longer word
            if word in text_word:
                # It's a compound word
                return True
        
        return False
    
    def _contains_variant_forms(self, text, word):
        """Check if text contains variant forms of the word."""
        # Common variant forms for different word types
        variants = set()
        
        # Plural forms (for nouns)
        if word.endswith('s'):
            # If word ends with 's', consider singular form as variant
            singular = word[:-1]
            if len(singular) > 2:  # Only add if meaningful
                variants.add(singular)
        else:
            # Add potential plural forms
            variants.add(word + "s")
            if word.endswith('y'):
                variants.add(word[:-1] + "ies")
            elif word.endswith(('ch', 'sh', 'x', 'z', 's')):
                variants.add(word + "es")
                
        # Verb forms
        variants.add(word + "ing")
        variants.add(word + "ed")
        if word.endswith('e'):
            variants.add(word[:-1] + "ing")
            variants.add(word + "d")
        elif word.endswith(('y')):
            variants.add(word[:-1] + "ied")
        elif len(word) > 3 and word[-1] not in 'aeiou' and word[-2] in 'aeiou' and word[-3] not in 'aeiou':
            variants.add(word + word[-1] + "ing")
            variants.add(word + word[-1] + "ed")
            
        # Adjective forms
        variants.add(word + "er")
        variants.add(word + "est")
        if word.endswith('e'):
            variants.add(word + "r")
            variants.add(word + "st")
        elif word.endswith('y'):
            variants.add(word[:-1] + "ier")
            variants.add(word[:-1] + "iest")
        elif len(word) > 3 and word[-1] not in 'aeiou' and word[-2] in 'aeiou' and word[-3] not in 'aeiou':
            variants.add(word + word[-1] + "er")
            variants.add(word + word[-1] + "est")
            
        # Special case handling for certain words
        if word == "bear":
            variants.update(["bore", "bearing", "bearable", "unbearable"])
        elif word == "glasses":
            variants.update(["glass", "eyeglass", "fiberglass", "fibreglass"])
        elif word == "glass":
            variants.update(["glasses", "fiberglass", "fibreglass"])
        elif word == "top":
            variants.update(["topping", "topped", "topmost"])
            
        # Check if text contains any variants
        for variant in variants:
            if len(variant) <= 2:
                continue  # Skip very short variants
                
            if self._contains_exact_word(text, variant):
                if self.debug:
                    print(f">>> Found variant form: '{variant}'")
                return True
                
        return False
    
    def _check_context_appropriate(self, text, word):
        """Check if the context is appropriate for this word."""
        text_lower = text.lower()
        
        # Special case for eyewear "glasses"
        if word == "glasses":
            # Check if it has eyewear context
            eyewear_contexts = ["see", "vision", "read", "eye", "wear", "sight", "prescription", "lens", "optician"]
            wrong_contexts = ["fill", "empty", "drink", "beverage", "water", "wine", "window", "fiber", "fibre"]
            
            # Must have at least one eyewear context word
            has_eyewear_context = any(context in text_lower for context in eyewear_contexts)
            
            # Must not have conflicting contexts
            has_wrong_context = any(context in text_lower for context in wrong_contexts)
            
            return has_eyewear_context and not has_wrong_context
            
        # Special case for "bear" (animal vs. verb)
        elif word == "bear":
            # Check for animal context
            animal_contexts = ["zoo", "animal", "wild", "fur", "cub", "paw", "den", "forest", "grizzly", "polar", "teddy"]
            verb_contexts = ["burden", "weight", "load", "responsibility", "stand", "support", "carry", "bore", "market", "stock"]
            
            has_animal_context = any(context in text_lower for context in animal_contexts)
            has_verb_context = any(context in text_lower for context in verb_contexts)
            
            # Prefer animal context, reject verb context
            return has_animal_context or not has_verb_context
            
        # Special case for "top" (clothing vs. position)
        elif word == "top":
            # Check for clothing context
            clothing_contexts = ["wear", "shirt", "outfit", "fashion", "dress", "color", "colour", "style", "clothes", "wardrobe", "buy", "bought", "new"]
            position_contexts = ["mountain", "hill", "climb", "reached", "leadership", "ranked", "ceiling", "position", "over", "above", "surface"]
            
            has_clothing_context = any(context in text_lower for context in clothing_contexts)
            has_position_context = any(context in text_lower for context in position_contexts)
            
            return has_clothing_context or not has_position_context
            
        # For person words, check that context is not objectifying
        elif word in self.category_words["person"]:
            objectifying_contexts = ["use", "using", "used", "utilize", "buy", "bought", "sell", "sold", "cost", "price", "cheap", "expensive"]
            
            for context in objectifying_contexts:
                if context in text_lower and abs(text_lower.find(context) - text_lower.find(word)) < 10:
                    return False
        
        # Special case for "scissors"
        elif word == "scissors":
            # Check for appropriate contexts
            tool_contexts = ["cut", "cutting", "paper", "fabric", "hair", "sharp", "blade", "trim"]
            inappropriate_contexts = ["executed", "jump", "kick", "position", "technique"]
            
            has_tool_context = any(context in text_lower for context in tool_contexts)
            has_inappropriate_context = any(context in text_lower for context in inappropriate_contexts)
            
            return has_tool_context and not has_inappropriate_context

        # Special case for jewelry items
        elif word in self.category_words.get("jewelry", []):
            # Check for appropriate contexts
            jewelry_contexts = ["wear", "wore", "beautiful", "gold", "silver", "gift", "precious", "expensive", "jewelry", "accessory"]
            inappropriate_contexts = ["need another", "project", "useful", "tool"]
            
            has_jewelry_context = any(context in text_lower for context in jewelry_contexts)
            has_inappropriate_context = any(context in text_lower for context in inappropriate_contexts)
            
            return has_jewelry_context and not has_inappropriate_context

        # Special case for toys like teddy bear
        elif word in self.category_words.get("toys", []) or "teddy" in word or "toy" in word:
            # Check for appropriate contexts
            toy_contexts = ["play", "child", "soft", "cuddly", "stuffed", "toy", "kids", "gift", "hug"]
            inappropriate_contexts = ["hunt", "wild", "attack", "killed", "animal", "zoo"]
            
            has_toy_context = any(context in text_lower for context in toy_contexts)
            has_inappropriate_context = any(context in text_lower for context in inappropriate_contexts)
            
            return has_toy_context and not has_inappropriate_context
            
        # For most words, no special context check needed
        return True
    
    def _get_template_category(self, word):
        """Determine the most appropriate template category for a word."""
        # Check predefined categories
        for category, words in self.category_words.items():
            if word in words:
                return category
                
        # Heuristic guesses based on common patterns
        if word in self.plural_words:
            return "noun"
        elif word.endswith('ing') and len(word) > 5:
            return "verb"
        elif word.endswith(('er', 'est')) and len(word) > 4:
            return "adjective"
        else:
            # Default to noun for most words
            return "noun"
    
    def _get_free_dictionary_examples(self, word):
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
                            examples.append(definition['example'])
            
            return examples
        except Exception as e:
            if self.debug:
                print(f">>> Free Dictionary API error: {e}")
            return []
    
    def _get_wordnik_examples(self, word):
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
                    examples.append(example['text'])
            
            return examples
        except Exception as e:
            if self.debug:
                print(f">>> Wordnik API error: {e}")
            return []
    
    def _clean_sentence(self, sentence):
        """Clean and format a sentence."""
        # Remove quotes, extra spaces, etc.
        sentence = sentence.strip()
        sentence = re.sub(r'\s+', ' ', sentence)
        
        # Capitalize first letter
        if sentence and not sentence[0].isupper():
            sentence = sentence[0].upper() + sentence[1:]
        
        # Ensure ending punctuation
        if sentence and not sentence[-1] in '.!?':
            sentence += '.'
        
        return sentence
    
    def _translate(self, text, target_language):
        """Translate text using the provided translation function."""
        if self.translate_func:
            try:
                return self.translate_func(text, target_language)
            except Exception as e:
                if self.debug:
                    print(f">>> Translation error: {e}")
                return ""
        else:
            return f"[Translation to {target_language}]"