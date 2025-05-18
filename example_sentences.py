"""
Simplified but Guaranteed Example Sentence Generator
----------------------------------------------------
This implementation takes a different approach: using high-quality templates
for all words to guarantee appropriate examples.
"""

import os
import json
import random
from functools import lru_cache

class ExampleSentenceGenerator:
    def __init__(self, translate_func=None, debug=False):
        """Initialize the example sentence generator."""
        self.translate_func = translate_func
        self.debug = debug
        self.setup_cache_dir()
        
    def setup_cache_dir(self):
        """Create cache directory if it doesn't exist."""
        self.cache_dir = "sentence_cache"
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def get_example_sentence(self, word, target_language, category=None):
        """Get an example sentence for a word with translation."""
        try:
            # Normalize word
            original_word = word
            word = word.strip().lower()
            
            # Get appropriate template category
            category = self._get_category(word, category)
            
            # Generate example
            english_example = self._get_template_example(word, category)
            
            # Translate
            translated_example = self._translate(english_example, target_language)
            
            return {
                "english": english_example,
                "translated": translated_example,
                "source": f"{category}_template"
            }
            
        except Exception as e:
            if self.debug:
                print(f"Error getting example sentence: {e}")
            
            # Simple fallback
            return {
                "english": f"This is a {word}.",
                "translated": self._translate(f"This is a {word}.", target_language),
                "source": "basic_fallback"
            }
    
    def _get_category(self, word, hint_category=None):
        """Determine the appropriate category for the word."""
        # Check explicit mappings first
        if word in self.word_category_map:
            return self.word_category_map[word]
            
        # Use hint if provided and valid
        if hint_category and hint_category in self.template_map:
            return hint_category
            
        # Default to general
        return "general"
    
    def _get_template_example(self, word, category):
        """Get a template example for the word."""
        # Get templates for this category
        templates = self.template_map.get(category, self.template_map["general"])
        
        # Select a random template
        template = random.choice(templates)
        
        # Handle plurals properly
        if word in self.plural_words:
            # These words are typically plural
            english_example = template.replace("a {word}", "{word}").replace("the {word}", "the {word}").format(word=word)
        else:
            english_example = template.format(word=word)
        
        return english_example
    
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
    
    # Words that are typically plural
    plural_words = {
        "glasses", "pants", "shorts", "scissors", "jeans", "trousers",
        "goggles", "sunglasses", "spectacles", "eyeglasses", "headphones"
    }
    
    # Comprehensive mapping of words to categories
    word_category_map = {
        # Eyewear
        "glasses": "eyewear",
        "sunglasses": "eyewear",
        "spectacles": "eyewear",
        "eyeglasses": "eyewear",
        "goggles": "eyewear",
        "contacts": "eyewear",
        "shades": "eyewear",
        
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
        "gloves": "clothing",
        "socks": "clothing",
        "shoes": "clothing",
        "boots": "clothing",
        "scarf": "clothing",
        "tie": "clothing",
        "belt": "clothing",
        "skirt": "clothing",
        "blouse": "clothing",
        "suit": "clothing",
        
        # Food
        "apple": "food",
        "banana": "food",
        "orange": "food",
        "bread": "food",
        "cheese": "food",
        "chicken": "food",
        "meat": "food",
        "fish": "food",
        "pasta": "food",
        "rice": "food",
        "vegetable": "food",
        "fruit": "food",
        "pizza": "food",
        "salad": "food",
        "sandwich": "food",
        "soup": "food",
        "cake": "food",
        "cookie": "food",
        "chocolate": "food",
        "candy": "food",
        
        # Animals
        "dog": "animals",
        "cat": "animals",
        "bird": "animals",
        "fish": "animals",
        "horse": "animals",
        "cow": "animals",
        "sheep": "animals",
        "pig": "animals",
        "lion": "animals",
        "tiger": "animals",
        "bear": "animals",
        "elephant": "animals",
        "monkey": "animals",
        "giraffe": "animals",
        "zebra": "animals",
        "rabbit": "animals",
        
        # Electronics
        "phone": "electronics",
        "computer": "electronics",
        "laptop": "electronics",
        "tablet": "electronics",
        "television": "electronics",
        "tv": "electronics",
        "camera": "electronics",
        "speaker": "electronics",
        "headphones": "electronics",
        "microphone": "electronics",
        "keyboard": "electronics",
        "mouse": "electronics",
        
        # Furniture
        "chair": "furniture",
        "table": "furniture",
        "desk": "furniture",
        "bed": "furniture",
        "sofa": "furniture",
        "couch": "furniture",
        "bookshelf": "furniture",
        "cabinet": "furniture",
        "drawer": "furniture",
        "wardrobe": "furniture",
        "dresser": "furniture",
        "lamp": "furniture",
        
        # Vehicles
        "car": "vehicles",
        "bike": "vehicles",
        "bicycle": "vehicles",
        "motorcycle": "vehicles",
        "bus": "vehicles",
        "train": "vehicles",
        "airplane": "vehicles",
        "boat": "vehicles",
        "ship": "vehicles",
        "truck": "vehicles",
        "van": "vehicles",
        "scooter": "vehicles",
        
        # Drinks/Containers
        "cup": "drinkware",
        "mug": "drinkware",
        "glass": "drinkware",  # Singular glass, not eyewear
        "bottle": "drinkware",
        "thermos": "drinkware",
        "flask": "drinkware",
        "teacup": "drinkware",
        "wineglass": "drinkware",
        "water": "drinks",
        "coffee": "drinks",
        "tea": "drinks",
        "juice": "drinks",
        "milk": "drinks",
        "soda": "drinks",
        "wine": "drinks",
        "beer": "drinks"
    }
    
    # Comprehensive template map organized by category
    template_map = {
        # Eyewear templates - specifically crafted for glasses
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
        
        # Clothing templates - specifically for apparel
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
        
        # Food templates
        "food": [
            "This {word} tastes delicious.",
            "I love eating {word} for breakfast.",
            "My mother makes the best {word} I've ever tasted.",
            "Would you like some {word} with your meal?",
            "The {word} is fresh from the market.",
            "She bought some {word} for dinner tonight.",
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
            "The {word} is made of ceramic.",
            "I need to wash this {word}.",
            "He drinks tea from his favorite {word}.",
            "We have a set of six {word}s for guests.",
            "The {word} is on the kitchen counter.",
            "My {word} has a small chip on the rim.",
            "This {word} can hold up to 12 ounces."
        ],
        
        # Drinks templates
        "drinks": [
            "I like to drink {word} in the morning.",
            "Would you like some {word}?",
            "She prefers {word} without sugar.",
            "This {word} tastes really good.",
            "I spilled some {word} on my shirt.",
            "We ran out of {word} yesterday.",
            "The {word} is too hot to drink right now.",
            "Can you pour me a glass of {word}, please?",
            "The {word} in this restaurant is excellent.",
            "My father drinks {word} every evening."
        ],
        
        # General templates for any object
        "general": [
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