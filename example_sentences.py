"""
Enhanced Example Sentence Generator
------------------------------------------
A sophisticated approach to generating diverse, natural example sentences
"""

import requests
import os
import random
import re
import time
import json
from functools import lru_cache

class EnhancedExampleGenerator:
    def __init__(self, translate_func=None, debug=False):
        """Initialize the generator with a multi-stage filter pipeline and diverse templates."""
        print("\n>>> ENHANCED SENTENCE GENERATOR LOADED <<<\n")
        self.translate_func = translate_func
        self.debug = debug  # Set to True to see detailed rejection reasons
        self.setup_cache_dir()
        self._initialize_templates()
        self.recent_templates = {}  # Store recently used templates to avoid repetition
        self.max_recent_templates = 10  # How many recent templates to remember per category
    
    def setup_cache_dir(self):
        """Create cache directory if it doesn't exist."""
        self.cache_dir = "sentence_cache"
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def _initialize_templates(self):
        """Initialize extensive template library with varied sentence structures."""
        # CORE TEMPLATES
        
        # NOUN TEMPLATES - Basic, Intermediate, Advanced
        self.templates = {
            # NOUN TEMPLATES - Various complexity levels and contexts
            "noun": {
                "basic": [
                    "The {word} is on the table.",
                    "I can see a {word} in the room.",
                    "She bought a new {word} yesterday.",
                    "This {word} is very useful.",
                    "Do you have a {word}?",
                    "Where is my {word}?",
                    "I need to buy a {word}.",
                    "That {word} belongs to me.",
                    "His {word} is very old.",
                    "Can I borrow your {word}?"
                ],
                "intermediate": [
                    "I've been looking for a {word} like this for months.",
                    "She inherited an antique {word} from her grandmother.",
                    "The {word} was displayed prominently in the center of the room.",
                    "My brother accidentally broke my favorite {word} last week.",
                    "We should consider getting a different {word} for this purpose.",
                    "They sell handmade {word}s at that store downtown.",
                    "The museum has a rare {word} from the 16th century.",
                    "I'm thinking about replacing my {word} soon.",
                    "That {word} reminds me of my childhood home.",
                    "Would you mind passing me that {word} on the shelf?"
                ],
                "advanced": [
                    "The antique {word} that I found at the flea market turned out to be quite valuable.",
                    "She explained that the unusual {word} had been in her family for generations.",
                    "Despite its age, the {word} functions perfectly after the restoration.",
                    "The artist incorporated a {word} into his latest sculpture, giving it an unexpected twist.",
                    "After searching through numerous stores, I finally found a {word} that matched my requirements.",
                    "The {word} caught my attention because of its unique design and craftsmanship.",
                    "They discussed replacing the old {word} with something more modern and practical.",
                    "I was surprised to discover that the {word} was actually made in my hometown.",
                    "The {word} would be perfect for our new apartment, if only it weren't so expensive.",
                    "When traveling abroad, he always brings back a traditional {word} as a souvenir."
                ]
            },
            
            # CLOTHING TEMPLATES - With proper context for clothing items
            "clothing": {
                "basic": [
                    "I like your {word}.",
                    "This {word} is too small.",
                    "She bought a new {word}.",
                    "The {word} is blue.",
                    "Can I borrow your {word}?",
                    "I need a warm {word}.",
                    "That {word} looks great on you.",
                    "My {word} is too big.",
                    "Where did you get that {word}?",
                    "He lost his favorite {word}."
                ],
                "intermediate": [
                    "She wore her favorite {word} to the dinner party last night.",
                    "This {word} would look perfect with your new jeans.",
                    "I bought this {word} on sale last weekend.",
                    "The designer creates unique {word}s for celebrities.",
                    "I need to iron my {word} before the interview tomorrow.",
                    "This {word} is made of high-quality cotton.",
                    "That {word} really brings out the color of your eyes.",
                    "He spilled coffee on his new {word} this morning.",
                    "My mother gave me this {word} for my birthday.",
                    "Do you think this {word} is appropriate for a formal event?"
                ],
                "advanced": [
                    "The intricate embroidery on her {word} caught everyone's attention at the gallery opening.",
                    "I've been searching for a {word} in this particular shade of blue for months.",
                    "The vintage {word} I found at the thrift store needed just minor alterations to fit perfectly.",
                    "She designs and sews her own {word}s using sustainable fabrics and traditional techniques.",
                    "After the washing machine incident, my favorite {word} was never quite the same.",
                    "When traveling to colder climates, I always pack my warmest {word} just in case.",
                    "The designer's latest collection features {word}s with bold geometric patterns and vibrant colors.",
                    "The celebrity was photographed wearing a custom-made {word} at the awards ceremony.",
                    "I've had this {word} for years, and it still remains one of my favorite pieces in my wardrobe.",
                    "The tailor suggested a different style of {word} that would better complement my body type."
                ]
            },
            
            # UNCOUNTABLE CLOTHING TEMPLATES - No articles
            "uncountable_clothing": {
                "basic": [
                    "The {word} feels very soft.",
                    "I need new {word} for winter.",
                    "This {word} is expensive.",
                    "She likes wearing {word} from that brand.",
                    "Where did you buy this {word}?",
                    "The {word} is made in Italy.",
                    "I prefer cotton {word}.",
                    "This {word} is too warm for summer.",
                    "The store sells {word} for all seasons.",
                    "That {word} looks comfortable."
                ],
                "intermediate": [
                    "She designs {word} for a high-end fashion label in Milan.",
                    "I'm looking for {word} that's both stylish and comfortable.",
                    "The {word} she wore to the event received many compliments.",
                    "This brand specializes in {word} for outdoor activities.",
                    "The {word} in the display window caught my attention immediately.",
                    "Quality {word} is worth the investment in the long run.",
                    "They import {word} from several European countries.",
                    "The {word} feels incredibly soft and lightweight.",
                    "We need to buy warmer {word} before our trip to Norway.",
                    "The latest {word} collection features bold patterns and colors."
                ],
                "advanced": [
                    "The designer's new collection features {word} made entirely from sustainable and recycled materials.",
                    "She has a keen eye for quality {word} that will last for many seasons.",
                    "The boutique specializes in handcrafted {word} using traditional techniques passed down for generations.",
                    "When traveling to varied climates, I always pack versatile {word} that can be layered as needed.",
                    "The exhibition showcased how {word} has evolved through different cultural contexts over the centuries.",
                    "The company has revolutionized athletic {word} with their innovative moisture-wicking fabrics.",
                    "After studying fashion design, she launched her own line of {word} inspired by architectural elements.",
                    "The {word} displayed in the museum demonstrates the craftsmanship of that historical period.",
                    "I prefer investing in timeless {word} rather than following fast-fashion trends.",
                    "The documentary explores how modern {word} production impacts environmental sustainability."
                ]
            },
            
            # PERSON TEMPLATES - For words describing people
            "person": {
                "basic": [
                    "The {word} is waiting outside.",
                    "I met a friendly {word} today.",
                    "She is a {word} at the university.",
                    "My neighbor is a {word}.",
                    "A {word} helped me find my way.",
                    "The {word} waved hello.",
                    "I saw a {word} at the store.",
                    "The {word} was very kind.",
                    "Do you know that {word}?",
                    "The {word} is reading a book."
                ],
                "intermediate": [
                    "The {word} I met at the conference gave an interesting presentation about climate change.",
                    "My cousin recently became a {word} at the new hospital downtown.",
                    "The {word} who lives next door often helps me with my gardening questions.",
                    "We interviewed several {word}s for the position, but none had the right qualifications.",
                    "The {word} who wrote this article has a unique perspective on the issue.",
                    "A {word} approached me in the park asking for directions to the museum.",
                    "The community center needs more volunteer {word}s to help with the after-school program.",
                    "The {word} who served us at the restaurant was very attentive and friendly.",
                    "My daughter wants to become a {word} when she grows up.",
                    "The {word} standing by the entrance seemed to be waiting for someone."
                ],
                "advanced": [
                    "The {word} who mentored me during my early career had a profound influence on my professional development.",
                    "After twenty years as a dedicated {word}, she decided to pursue a completely different career path.",
                    "The documentary features interviews with {word}s from diverse backgrounds sharing their unique experiences.",
                    "The renowned {word} delivered a compelling speech about the importance of community engagement.",
                    "The organization provides resources and support to {word}s who are navigating challenging circumstances.",
                    "My grandparent was a {word} during a pivotal time in our country's history.",
                    "Research suggests that {word}s who maintain strong social connections tend to live longer, healthier lives.",
                    "The book chronicles the journey of a {word} who overcame significant obstacles to achieve their dreams.",
                    "The panel discussion included perspectives from {word}s representing various fields and specialties.",
                    "As an experienced {word}, she offers valuable insights that only come with years of practice."
                ]
            },
            
            # ANIMAL TEMPLATES - For animal words
            "animal": {
                "basic": [
                    "The {word} is sleeping.",
                    "I saw a {word} at the zoo.",
                    "My friend has a pet {word}.",
                    "The {word} runs very fast.",
                    "The {word} has beautiful fur.",
                    "I like watching {word}s.",
                    "The {word} is eating.",
                    "That {word} is very large.",
                    "The {word} lives in the forest.",
                    "Can you see the {word}?"
                ],
                "intermediate": [
                    "We observed a {word} in its natural habitat during our safari trip.",
                    "My daughter is learning interesting facts about the {word} for her science project.",
                    "The documentary showed how {word}s adapt to changing environmental conditions.",
                    "The zoo's new exhibit features {word}s from the rainforest region.",
                    "Scientists are working to protect the endangered {word} from extinction.",
                    "The {word} we spotted on our hike yesterday was with its young.",
                    "The photographer spent months tracking the elusive {word} in the mountains.",
                    "The behavior of the {word} in captivity differs greatly from those in the wild.",
                    "My neighbor's {word} escaped from their yard yesterday afternoon.",
                    "The children were fascinated by the {word} at the wildlife sanctuary."
                ],
                "advanced": [
                    "Researchers have documented unique communication patterns among {word}s that suggest a complex social structure.",
                    "The conservation program has successfully increased the population of the endangered {word} in its native habitat.",
                    "The migration patterns of the {word} have been significantly affected by climate change in recent decades.",
                    "The documentary explores the fascinating symbiotic relationship between the {word} and its ecosystem.",
                    "After decades of studying {word} behavior, the biologist published a comprehensive field guide.",
                    "The ancient mythology of this region often features the {word} as a symbol of wisdom and strength.",
                    "The unusual appearance of this rare {word} has evolved specifically to help it survive in harsh conditions.",
                    "The wildlife photographer spent three years capturing the elusive {word} in its remote mountain habitat.",
                    "Recent genetic studies have revealed that the {word} shares a common ancestor with several other species.",
                    "The rehabilitation center works specifically with injured {word}s before releasing them back into the wild."
                ]
            },
            
            # EYEWEAR TEMPLATES - For glasses, sunglasses, etc.
            "eyewear": {
                "basic": [
                    "I need my {word} to read.",
                    "She wears {word} all the time.",
                    "These {word} are new.",
                    "My {word} are broken.",
                    "Where are my {word}?",
                    "I lost my {word} yesterday.",
                    "These {word} help me see better.",
                    "The {word} are on the table.",
                    "His {word} have blue frames.",
                    "I need to clean my {word}."
                ],
                "intermediate": [
                    "I finally found a pair of {word} that fit comfortably on my face.",
                    "She needs to wear prescription {word} while driving.",
                    "The optometrist recommended these {word} for reading and computer work.",
                    "My {word} have transition lenses that darken in the sunlight.",
                    "He's been wearing the same style of {word} for over twenty years.",
                    "I keep forgetting where I put my {word} down around the house.",
                    "These designer {word} were quite expensive but should last for years.",
                    "My new {word} have anti-glare coating which helps reduce eye strain.",
                    "She has several pairs of {word} that she coordinates with different outfits.",
                    "The {word} case protects them when I'm not wearing them."
                ],
                "advanced": [
                    "After trying countless styles, I finally found {word} that complement my face shape and personal style.",
                    "The optometrist explained that my progressive {word} would take some time to adjust to, but would ultimately be more convenient.",
                    "She collects vintage {word} from the 1960s and has an impressive display of unique frames.",
                    "The technology behind these specialized {word} allows people with color blindness to perceive a wider range of colors.",
                    "My grandfather has worn the same tortoiseshell {word} for decades, refusing all suggestions to update his style.",
                    "The museum exhibition featured {word} worn by famous historical figures, showing how eyewear fashion has evolved.",
                    "Modern {word} manufacturing combines traditional craftsmanship with innovative materials for both durability and comfort.",
                    "The ophthalmologist suggested I try computer {word} specifically designed to filter blue light and reduce digital eye strain.",
                    "After laser surgery, she only needs to wear {word} occasionally for night driving or extended reading sessions.",
                    "The artisan creates custom {word} using sustainable materials and traditional techniques passed down for generations."
                ]
            },
            
            # JEWELRY TEMPLATES - For necklaces, rings, bracelets, etc.
            "jewelry": {
                "basic": [
                    "She wears a beautiful {word}.",
                    "The {word} is made of gold.",
                    "I lost my {word} yesterday.",
                    "This {word} was a gift.",
                    "The {word} has a diamond.",
                    "I like your {word}.",
                    "That {word} looks expensive.",
                    "My grandmother gave me this {word}.",
                    "She never takes off her {word}.",
                    "Where did you buy that {word}?"
                ],
                "intermediate": [
                    "She wore her grandmother's antique {word} to the wedding ceremony.",
                    "The handcrafted {word} features stones collected from the local riverbed.",
                    "I had my {word} repaired at a specialty shop downtown.",
                    "He surprised her with a {word} for their anniversary celebration.",
                    "The museum displays {word}s from various historical periods and cultures.",
                    "She designs custom {word}s using traditional techniques she learned as an apprentice.",
                    "The {word} catches the light beautifully when she moves.",
                    "This {word} has been passed down through four generations of my family.",
                    "I'm looking for a {word} that I can wear with both casual and formal outfits.",
                    "The artisan market had a vendor selling handmade {word}s from recycled materials."
                ],
                "advanced": [
                    "The intricate {word} she wore to the gala featured rare gemstones arranged in a pattern inspired by celestial constellations.",
                    "Archaeological discoveries revealed that ancient civilizations crafted {word}s using techniques that still challenge modern artisans.",
                    "The exhibition showcased how {word} designs have evolved across cultures while maintaining symbolic significance.",
                    "After studying under master craftspeople, she now creates custom {word}s that blend traditional methods with contemporary aesthetics.",
                    "The family heirloom {word} not only holds sentimental value but has been appraised as a rare historical piece.",
                    "The artisan's {word} collection features materials sourced ethically from sustainable mines and workshops.",
                    "Cultural traditions surrounding the gifting of {word}s reveal fascinating insights about historical social structures.",
                    "The delicate balance between craftsmanship and wearability is evident in every {word} the designer creates.",
                    "Her travels around the world have inspired a unique collection of {word}s that incorporate diverse cultural elements.",
                    "The restoration of the antique {word} required specialized techniques to preserve its original character and value."
                ]
            },
            
            # TOOLS TEMPLATES - For scissors, hammers, etc.
            "tools": {
                "basic": [
                    "I need the {word} to finish this.",
                    "The {word} is in the toolbox.",
                    "Can you pass me the {word}?",
                    "This {word} is very sharp.",
                    "Where are the {word}?",
                    "I bought a new {word} yesterday.",
                    "These {word} are old but work well.",
                    "Be careful with the {word}.",
                    "The {word} is broken.",
                    "I can't find my {word}."
                ],
                "intermediate": [
                    "I need a specific type of {word} for this delicate repair work.",
                    "My grandfather's {word} has lasted for decades because of its quality craftsmanship.",
                    "The workshop has a specialized {word} that makes this process much easier.",
                    "She keeps her {word} organized in a custom-built storage system.",
                    "The artisan demonstrated how to properly use the {word} for best results.",
                    "I borrowed my neighbor's {word} to finish the project yesterday.",
                    "These professional-grade {word}s cost more but will last much longer.",
                    "The {word} needs to be sharpened before we continue with the project.",
                    "You'll need a different {word} for working with that material.",
                    "The antique {word} is still perfectly functional despite its age."
                ],
                "advanced": [
                    "The craftsman hand-forges each {word} using techniques that have been passed down for generations.",
                    "Modern manufacturing has changed the design of the {word}, but professional artisans often prefer traditional models.",
                    "The museum's collection includes {word}s dating back to the industrial revolution, showing the evolution of tool design.",
                    "After apprenticing with a master, she developed her own modified {word} that addresses common issues with the standard design.",
                    "The specialized {word} was developed specifically for this type of precision work and requires significant training to use properly.",
                    "Restoring antique furniture requires period-appropriate {word}s to maintain historical accuracy and preservation standards.",
                    "Different regions developed unique variations of the {word} based on local materials and specific cultural needs.",
                    "The ergonomic design of this modern {word} reduces strain during extended periods of use.",
                    "The documentary explores how the invention of the {word} revolutionized this particular craft and industry.",
                    "Professional artisans often modify their {word}s to suit their specific working style and the unique demands of their specialty."
                ]
            },
            
            # TOYS TEMPLATES - For toys, games, etc.
            "toys": {
                "basic": [
                    "The child loves this {word}.",
                    "I had a similar {word} when I was young.",
                    "This {word} is very popular.",
                    "She plays with her {word} every day.",
                    "The {word} is broken.",
                    "Where is your {word}?",
                    "He got a new {word} for his birthday.",
                    "The {word} is on the shelf.",
                    "This {word} is for children over three.",
                    "Can I play with your {word}?"
                ],
                "intermediate": [
                    "My daughter refuses to go to sleep without her favorite {word} beside her.",
                    "The vintage {word} I found at the flea market is similar to one I had as a child.",
                    "Educational {word}s like this one help develop important cognitive skills.",
                    "The handmade {word} was passed down through several generations of our family.",
                    "This interactive {word} responds to voice commands and movement.",
                    "The children take turns playing with the {word} during recess.",
                    "We donated gently used {word}s to the children's hospital last month.",
                    "The popular {word} was sold out in stores for months after its release.",
                    "The museum has a collection of {word}s from different historical periods.",
                    "My son spends hours creating elaborate stories with his {word}."
                ],
                "advanced": [
                    "The exhibition showcases how {word}s have evolved over the centuries, reflecting changing attitudes toward childhood and education.",
                    "Researchers study how children interact with {word}s like this one to better understand cognitive development and creativity.",
                    "The handcrafted {word} combines traditional craftsmanship with modern safety standards and sustainable materials.",
                    "The documentary explores how this iconic {word} has influenced generations of children across diverse cultures.",
                    "Collectors value vintage {word}s from this era not only for nostalgia but as artifacts of cultural history.",
                    "The therapeutic benefits of specialized {word}s have been documented in numerous studies with children on the autism spectrum.",
                    "The company redesigned their classic {word} to incorporate feedback from child development specialists and parents.",
                    "The family tradition involves giving each child a personalized {word} that reflects their unique interests and personality.",
                    "The restoration of antique {word}s requires specialized knowledge to preserve their historical integrity while making them safe for display.",
                    "The innovative {word} was designed by educators to support multiple learning styles and developmental stages."
                ]
            },
            
            # VERB TEMPLATES - For action words
            "verb": {
                "basic": [
                    "I {word} every morning.",
                    "She {word}s regularly.",
                    "They {word} on weekends.",
                    "Do you {word} often?",
                    "He doesn't {word}.",
                    "We {word}ed yesterday.",
                    "I will {word} tomorrow.",
                    "She is {word}ing now.",
                    "Have you ever {word}ed?",
                    "They are {word}ing."
                ],
                "intermediate": [
                    "I try to {word} at least three times a week for my health.",
                    "She {word}s whenever she has a free moment in her busy schedule.",
                    "The instructor taught us the proper technique to {word} effectively.",
                    "After the accident, he couldn't {word} for several months during recovery.",
                    "We {word} together every Saturday as part of our weekend routine.",
                    "The guidelines recommend that beginners {word} no more than twice weekly.",
                    "I've been {word}ing regularly since January and have noticed significant improvements.",
                    "Children naturally learn to {word} through observation and practice.",
                    "The app helps you track how often you {word} and measures your progress.",
                    "They {word} competitively at the national level."
                ],
                "advanced": [
                    "Research indicates that those who {word} consistently throughout their lives maintain better cognitive function as they age.",
                    "After decades of {word}ing professionally, she now teaches advanced techniques to dedicated students.",
                    "The documentary explores how cultural differences influence the way people {word} across various societies.",
                    "The ancient practice of {word}ing has evolved considerably but maintains its core principles and benefits.",
                    "His innovative method of {word}ing challenged conventional wisdom and eventually transformed the entire field.",
                    "The rehabilitation program incorporates modified ways to {word} that accommodate various physical limitations.",
                    "Neuroscience research reveals that {word}ing activates multiple brain regions simultaneously, creating new neural pathways.",
                    "The philosophy behind this approach suggests that how you {word} reflects fundamental aspects of your worldview.",
                    "Traditional communities have {word}ed as part of seasonal rituals for countless generations.",
                    "The longitudinal study tracked participants who {word} regularly, finding significant long-term health benefits."
                ]
            },
            
            # ADJECTIVE TEMPLATES - For descriptive words
            "adjective": {
                "basic": [
                    "The sky is {word} today.",
                    "She looks very {word}.",
                    "That is a {word} book.",
                    "The food tastes {word}.",
                    "I feel {word} this morning.",
                    "The music sounds {word}.",
                    "What a {word} day!",
                    "The water is {word}.",
                    "His house is very {word}.",
                    "The flowers smell {word}."
                ],
                "intermediate": [
                    "The critics described the film as surprisingly {word} despite its low budget.",
                    "After the renovation, the room feels much more {word} and welcoming.",
                    "The chef is known for creating {word} dishes that combine unexpected flavors.",
                    "The {word} atmosphere of the café makes it perfect for quiet conversation.",
                    "Her writing style is refreshingly {word} compared to other authors in the genre.",
                    "The landscape becomes increasingly {word} as you travel further north.",
                    "We were impressed by the {word} performance of the young musician.",
                    "The weather turned unexpectedly {word} during our vacation.",
                    "She chose a more {word} approach to solving the complex problem.",
                    "The {word} quality of the light at sunset inspired many of his paintings."
                ],
                "advanced": [
                    "The architectural design achieves a harmonious balance between {word} elements and practical functionality.",
                    "The region's cuisine is characterized by {word} flavors that reflect its unique cultural heritage and local ingredients.",
                    "Critics have noted the increasingly {word} tone of the author's work as it evolved over decades.",
                    "The exhibition juxtaposes {word} contemporary pieces with classical works to create thought-provoking contrasts.",
                    "Researchers documented the {word} conditions that contribute to this rare ecological phenomenon.",
                    "The performance was notable for its {word} interpretation of the traditional composition.",
                    "The memoir reveals the {word} complexity of relationships formed during periods of historical upheaval.",
                    "Urban planners aim to create more {word} community spaces that encourage diverse interactions.",
                    "The documentary captures the {word} beauty of remote landscapes rarely witnessed by human eyes.",
                    "Medical researchers have identified several {word} biomarkers associated with early stages of the condition."
                ]
            },
            
            # GENERAL TEMPLATES - Fallback for everything else
            "general": {
                "basic": [
                    "Let's talk about {word}.",
                    "I'm interested in {word}.",
                    "Do you know about {word}?",
                    "{word} is important.",
                    "I learned about {word} yesterday.",
                    "She mentioned {word}.",
                    "They're studying {word}.",
                    "What do you think about {word}?",
                    "{word} has changed over time.",
                    "The book explains {word}."
                ],
                "intermediate": [
                    "We discussed the implications of {word} during yesterday's meeting.",
                    "The article explores different perspectives on {word} in modern society.",
                    "She's been researching {word} for her upcoming paper.",
                    "Recent developments in {word} have changed our understanding of the field.",
                    "The podcast featured experts discussing the future of {word}.",
                    "Understanding {word} requires knowledge of several related concepts.",
                    "The course offers an introduction to the basics of {word}.",
                    "There are various approaches to addressing issues related to {word}.",
                    "The conference will focus on recent advances in {word}.",
                    "They published a comprehensive study about {word} last year."
                ],
                "advanced": [
                    "The interdisciplinary approach to {word} reveals connections that weren't apparent within traditional academic frameworks.",
                    "Historical perspectives on {word} provide valuable context for understanding contemporary challenges in the field.",
                    "The symposium brought together experts from diverse backgrounds to address emerging questions about {word}.",
                    "Technological innovations have fundamentally transformed how researchers approach the study of {word}.",
                    "Cultural attitudes toward {word} vary significantly across different societies and historical periods.",
                    "The foundation funds projects that explore ethical implications of {word} in various contexts.",
                    "Policymakers must consider multiple stakeholder perspectives when developing regulations related to {word}.",
                    "The textbook presents a comprehensive framework for analyzing {word} across different scenarios.",
                    "Ongoing debates about {word} reflect deeper philosophical questions about our societal values.",
                    "The documentary examines how public understanding of {word} has evolved over the past century."
                ]
            }
        }
        
        # Common words in each category
        self.category_words = {
            "person": [
                "person", "man", "woman", "boy", "girl", "child", "teacher", "doctor", 
                "student", "friend", "neighbor", "parent", "employee", "worker", "artist",
                "musician", "writer", "chef", "athlete", "scientist", "engineer", "nurse"
            ],
            "animal": [
                "dog", "cat", "bird", "fish", "horse", "cow", "elephant", "lion", 
                "tiger", "bear", "rabbit", "monkey", "mouse", "frog", "snake", "wolf",
                "fox", "deer", "giraffe", "zebra", "penguin", "eagle", "owl", "turtle"
            ],
            "clothing": [
                "shirt", "pants", "dress", "jacket", "coat", "hat", "gloves", "socks",
                "shoes", "boots", "sweater", "skirt", "jeans", "top", "scarf", "tie",
                "blouse", "suit", "belt", "vest", "hoodie", "shorts", "pajamas", "uniform"
            ],
            "uncountable_clothing": [
                "clothing", "outerwear", "underwear", "sportswear", "footwear", "swimwear", 
                "knitwear", "loungewear", "sleepwear", "activewear", "winterwear", "beachwear"
            ],
            "eyewear": [
                "glasses", "sunglasses", "contacts", "goggles", "spectacles", "eyeglasses", 
                "shades", "reading glasses", "bifocals", "prescription glasses"
            ],
            "jewelry": [
                "necklace", "ring", "bracelet", "earrings", "watch", "pendant", 
                "brooch", "pin", "chain", "locket", "anklet", "cufflinks", "tiara",
                "crown", "medallion", "choker", "bangle", "charm bracelet"
            ],
            "tools": [
                "scissors", "knife", "hammer", "screwdriver", "wrench", "pliers",
                "saw", "drill", "tape measure", "level", "chisel", "clamp", "ruler",
                "axe", "shovel", "rake", "trowel", "sander", "nail gun", "paintbrush"
            ],
            "toys": [
                "teddy bear", "doll", "ball", "blocks", "action figure", "puzzle",
                "toy car", "stuffed animal", "plush toy", "game", "toy", "robot",
                "kite", "yo-yo", "train set", "board game", "video game", "rattle",
                "building set", "puppet", "play set", "model kit"
            ]
        }
        
        # Words that are typically used in plural form
        self.plural_words = [
            "glasses", "pants", "shorts", "jeans", "scissors", "trousers", "sunglasses",
            "goggles", "spectacles", "eyeglasses", "headphones", "tights", "leggings",
            "pliers", "binoculars", "tweezers", "pajamas", "shorts", "overalls", "trunks",
            "boxers", "briefs", "clippers", "shears", "earrings", "earbuds", "headphones"
        ]
        
        # Special cases that need specific handling
        self.special_cases = {
            "top": "clothing",
            "glasses": "eyewear",
            "teddy bear": "toys",
            "teddy": "toys",
            "clothing": "uncountable_clothing",
            "bear": "animal",
            "watch": "jewelry"
        }
        
        # Initialize the history of used templates
        self.template_history = {}
    
    def get_example_sentence(self, word, target_language, category=None):
        """Get an example sentence for a word using the enhanced algorithm."""
        try:
            # Normalize word
            original_word = word
            word = word.strip().lower()
            
            if self.debug:
                print(f"\n>>> Getting example for: '{word}' <<<\n")
            
            # Try to get example from API with filter pipeline
            examples = self._get_api_examples_with_pipeline(word, category)
            
            # If we have valid examples, choose one randomly and return
            if examples:
                example = self._select_diverse_example(examples, word)
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
            
            # If no valid API examples, use templates with complex selection
            if self.debug:
                print(f">>> No valid API examples found, using enhanced templates")
                
            # Use category if provided, otherwise determine category
            word_category = category if category else self._get_word_category(word)
            if self.debug:
                print(f">>> Word category: {word_category}")
                
            # Get appropriate templates and select a diverse one
            template, complexity = self._select_diverse_template(word, word_category)
            
            # Create example from template with appropriate handling
            example = self._create_example_from_template(template, word, word_category)
            
            # Translate the example
            translated = self._translate(example, target_language)
            
            return {
                "english": example,
                "translated": translated,
                "source": f"template_{word_category}_{complexity}"
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
    
    def _get_api_examples_with_pipeline(self, word, category=None):
        """Get examples from APIs and run them through the enhanced filter pipeline."""
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
            
        # Apply enhanced filter pipeline to raw examples
        filtered_examples = []
        for example in raw_examples:
            # Run through all filters
            if self._enhanced_filter_pipeline(example, word, category):
                filtered_examples.append(example)
                
        return filtered_examples
    
    def _enhanced_filter_pipeline(self, example, word, category=None):
        """
        Enhanced filter pipeline with improved context awareness.
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
            
        # Filter 5: Context appropriateness check based on category
        identified_category = category if category else self._get_word_category(word)
        if not self._check_advanced_context(example, word, identified_category):
            if self.debug:
                print(f">>> Rejected (Inappropriate context for {identified_category}): '{example}'")
            return False
            
        # Filter 6: Complexity and educational value check
        if not self._check_complexity_value(example, word):
            if self.debug:
                print(f">>> Rejected (Not suitable complexity/value): '{example}'")
            return False
            
        # All filters passed
        if self.debug:
            print(f">>> Accepted: '{example}'")
        return True
    
    def _basic_quality_check(self, text):
        """Check if a sentence meets basic quality standards."""
        # Must be a reasonable length
        words = text.split()
        if len(words) < 3 or len(words) > 25:
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
            
        # Shouldn't contain inappropriate language
        inappropriate_terms = ["kill", "die", "death", "murder", "suicide", "sex", "porn", "explicit", "violent"]
        if any(term in text.lower() for term in inappropriate_terms):
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
            
        # Find all words in the text
        all_words = re.findall(r'\b\w+\b', text.lower())
        
        for text_word in all_words:
            # Skip exact match
            if text_word == word:
                continue
                
            # Check if the word is part of a longer word
            if word in text_word:
                # Check for special cases where it's ok
                if word == "glass" and text_word == "glasses" and re.search(r'\b(eye|read|see|vision)', text.lower()):
                    continue
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
            
        # Check for special case exceptions
        if word == "glasses" and "eyeglasses" in text.lower():
            return False
            
        # Check if text contains any variants
        for variant in variants:
            if len(variant) <= 2:
                continue  # Skip very short variants
                
            if self._contains_exact_word(text, variant):
                if self.debug:
                    print(f">>> Found variant form: '{variant}'")
                return True
                
        return False
    
    def _check_advanced_context(self, text, word, category):
        """Enhanced context check that considers word category and typical usage patterns."""
        text_lower = text.lower()
        
        # Special case for eyewear "glasses"
        if category == "eyewear":
            # Check if it has eyewear context
            eyewear_contexts = ["see", "vision", "read", "eye", "wear", "sight", "prescription", 
                              "lens", "optician", "frame", "vision", "optometrist", "sunglasses"]
            wrong_contexts = ["fill", "empty", "drink", "beverage", "water", "wine", "window", 
                            "fiber", "fibre", "cup", "mug", "liquid", "pour"]
            
            # Must have at least one eyewear context word or phrase
            has_eyewear_context = any(context in text_lower for context in eyewear_contexts)
            
            # Must not have conflicting contexts
            has_wrong_context = any(context in text_lower for context in wrong_contexts)
            
            return has_eyewear_context or not has_wrong_context
            
        # Special case for "bear" (animal vs. verb)
        elif word == "bear" or category == "animal" and word == "bear":
            # Check for animal context
            animal_contexts = ["zoo", "animal", "wild", "fur", "cub", "paw", "den", "forest", 
                             "grizzly", "polar", "teddy", "pet", "wildlife", "nature"]
            verb_contexts = ["burden", "weight", "load", "responsibility", "stand", "support", 
                           "carry", "bore", "market", "stock", "bear with", "bear in mind"]
            
            has_animal_context = any(context in text_lower for context in animal_contexts)
            has_verb_context = any(context in text_lower for context in verb_contexts)
            
            # Prefer animal context, reject verb context
            return has_animal_context and not has_verb_context
            
        # Special case for "top" (clothing vs. position)
        elif word == "top" or category == "clothing" and word == "top":
            # Check for clothing context
            clothing_contexts = ["wear", "shirt", "outfit", "fashion", "dress", "color", "colour", 
                               "style", "clothes", "wardrobe", "buy", "bought", "new", "fabric", 
                               "cotton", "silk", "button", "sleeve", "collar", "blouse"]
            position_contexts = ["mountain", "hill", "climb", "reached", "leadership", "ranked", 
                               "ceiling", "position", "over", "above", "surface", "highest", 
                               "best", "leading", "foremost", "premier", "superior", "chief"]
            
            has_clothing_context = any(context in text_lower for context in clothing_contexts)
            has_position_context = any(context in text_lower for context in position_contexts)
            
            # For "top", strongly prefer clothing context for a language learning app
            return has_clothing_context and not has_position_context
            
        # For person words, check that context is not objectifying or inappropriate
        elif category == "person":
            objectifying_contexts = ["use", "using", "used", "utilize", "buy", "bought", "sell", 
                                   "sold", "cost", "price", "cheap", "expensive", "owned"]
            
            # Check proximity of objectifying words to the person word
            for context in objectifying_contexts:
                if context in text_lower:
                    # Check if the objectifying word is close to the person word
                    context_index = text_lower.find(context)
                    word_index = text_lower.find(word)
                    if abs(context_index - word_index) < 15:  # Within ~3-4 words
                        return False
        
        # Special case for tools
        elif category == "tools" and word == "scissors":
            wrong_contexts = ["executed", "perfect", "jump", "kick", "position", "technique", "sports"]
            tool_contexts = ["cut", "cutting", "paper", "fabric", "hair", "sharp", "blade", "trim"]
            
            has_wrong_context = any(context in text_lower for context in wrong_contexts)
            has_tool_context = any(context in text_lower for context in tool_contexts)
            
            return has_tool_context or not has_wrong_context
            
        # Special case for jewelry
        elif category == "jewelry":
            jewelry_contexts = ["wear", "gold", "silver", "diamond", "gem", "stone", "gift", 
                              "beautiful", "elegant", "accessory", "decorated", "adorned"]
            
            has_jewelry_context = any(context in text_lower for context in jewelry_contexts)
            return has_jewelry_context
            
        # Clothing checks
        elif category in ["clothing", "uncountable_clothing"]:
            clothing_contexts = ["wear", "fashion", "style", "outfit", "dressed", "clothes", 
                               "wardrobe", "fabric", "color", "comfortable", "fit", "size"]
            
            has_clothing_context = any(context in text_lower for context in clothing_contexts)
            return has_clothing_context
            
        # For most words, more relaxed context check
        return True
    
    def _check_complexity_value(self, text, word):
        """Check if the example has appropriate complexity and educational value."""
        # Count words for basic complexity check
        word_count = len(text.split())
        
        # Very short examples may not provide enough context
        if word_count < 5:
            return False
            
        # Very long examples might be too complex for beginners
        if word_count > 20:
            return False
            
        # Check for overly complex words that might be difficult for language learners
        complex_words = 0
        for w in text.split():
            w = w.lower().strip('.,!?":;()')
            if len(w) > 8 and w != word.lower():  # Don't count the target word as complex
                complex_words += 1
                
        # If more than 20% of words are complex, maybe too difficult
        if complex_words / word_count > 0.2:
            return False
            
        # Prefer examples where the target word appears near the beginning or middle
        # for better context understanding
        word_position = text.lower().find(word.lower())
        if word_position > len(text) * 0.7:  # If word appears only near the end
            return False
            
        return True
    
    def _get_word_category(self, word):
        """Determine the most appropriate category for a word with enhanced detection."""
        # Check special cases first
        if word in self.special_cases:
            return self.special_cases[word]
            
        # Check predefined categories
        for category, words in self.category_words.items():
            if word in words:
                return category
                
        # Handle compound words
        if "teddy bear" in word or "teddy" in word:
            return "toys"
        
        # Check word endings and patterns
        if word in self.plural_words:
            # Check specific types of plurals
            if word in ["scissors", "pliers", "clippers", "shears"]:
                return "tools"
            elif word in ["glasses", "sunglasses", "spectacles", "eyeglasses"]:
                return "eyewear"
            elif word in ["pants", "shorts", "jeans", "trousers", "tights", "leggings"]:
                return "clothing"
            else:
                return "noun"
        elif word.endswith('ing') and len(word) > 5:
            if word in ["clothing"]:
                return "uncountable_clothing"
            return "verb"
        elif word.endswith(('er', 'est')) and len(word) > 4:
            # Check if it's actually a comparative/superlative adjective
            # or a noun ending in 'er' (like "computer")
            if word.endswith('er') and word[:-2] + 'e' in self.category_words.get("verb", []):
                return "person"  # Like "teacher" from "teach"
            return "adjective"
        
        # Default to noun for most words
        return "noun"
    
    def _select_diverse_example(self, examples, word):
        """Select a diverse example, avoiding recent patterns if possible."""
        if not examples:
            return None
            
        # If only one example, return it
        if len(examples) == 1:
            return examples[0]
            
        # Analyze examples for diversity factors
        analyzed_examples = []
        for example in examples:
            # Calculate distance from word to beginning
            word_pos = example.lower().find(word.lower())
            total_length = len(example)
            position_ratio = word_pos / total_length if total_length > 0 else 0
            
            # Count sentence complexity (length, word variety)
            words = example.split()
            sentence_length = len(words)
            unique_words = len(set([w.lower() for w in words]))
            word_variety = unique_words / sentence_length if sentence_length > 0 else 0
            
            # Detect sentence structure
            has_question = example.endswith('?')
            has_dialogue = '"' in example or "'" in example and "said" in example.lower()
            has_conjunction = any(conj in example.lower() for conj in [" and ", " but ", " or ", " because ", " however "])
            
            # Create a feature vector for this example
            features = {
                "position_ratio": position_ratio,
                "sentence_length": sentence_length,
                "word_variety": word_variety,
                "has_question": has_question,
                "has_dialogue": has_dialogue,
                "has_conjunction": has_conjunction
            }
            
            analyzed_examples.append((example, features))
        
        # Get previously used patterns if available
        prev_patterns = self.recent_templates.get(word, [])
        
        # Score examples based on diversity from previous patterns
        scored_examples = []
        for example, features in analyzed_examples:
            # Start with base score
            score = 10
            
            # Prefer moderate length sentences (not too short, not too long)
            length_score = 0
            if 7 <= features["sentence_length"] <= 15:
                length_score = 5
            elif 5 <= features["sentence_length"] < 7 or 15 < features["sentence_length"] <= 20:
                length_score = 3
            score += length_score
            
            # Prefer good word variety
            if features["word_variety"] > 0.8:
                score += 3
            elif features["word_variety"] > 0.6:
                score += 2
            
            # Slight bonus for different sentence structures
            if features["has_question"]:
                score += 1
            if features["has_dialogue"]:
                score += 1
            if features["has_conjunction"]:
                score += 2
            
            # Major bonus for differing from recent patterns
            for prev_pattern in prev_patterns:
                # Compare this example to previous patterns
                pattern_similarity = 0
                
                # Position similarity
                if abs(features["position_ratio"] - prev_pattern.get("position_ratio", 0)) < 0.2:
                    pattern_similarity += 1
                
                # Length similarity
                if abs(features["sentence_length"] - prev_pattern.get("sentence_length", 0)) < 3:
                    pattern_similarity += 1
                
                # Structure similarity
                if features["has_question"] == prev_pattern.get("has_question", False):
                    pattern_similarity += 1
                if features["has_dialogue"] == prev_pattern.get("has_dialogue", False):
                    pattern_similarity += 1
                if features["has_conjunction"] == prev_pattern.get("has_conjunction", False):
                    pattern_similarity += 1
                
                # Penalize similarity to recent patterns
                score -= pattern_similarity
            
            scored_examples.append((example, features, score))
        
        # Sort by score (highest first)
        scored_examples.sort(key=lambda x: x[2], reverse=True)
        
        # Get the highest scored example
        best_example = scored_examples[0][0]
        best_features = scored_examples[0][1]
        
        # Update recent patterns
        if word not in self.recent_templates:
            self.recent_templates[word] = []
        
        self.recent_templates[word].append(best_features)
        
        # Keep only the most recent patterns
        if len(self.recent_templates[word]) > self.max_recent_templates:
            self.recent_templates[word] = self.recent_templates[word][-self.max_recent_templates:]
        
        return best_example
    
    def _select_diverse_template(self, word, category):
        """Select a diverse template based on complexity and avoiding recent patterns."""
        # Determine appropriate template category
        template_category = category
        
        # If no templates for this category, use general
        if template_category not in self.templates:
            template_category = "noun" if category in ["clothing", "tools", "jewelry", "eyewear"] else "general"
        
        # Get template subcategories for this type (basic, intermediate, advanced)
        if template_category in self.templates:
            subcats = list(self.templates[template_category].keys())
        else:
            subcats = ["basic", "intermediate", "advanced"]
        
        # Weighted selection of complexity level (more beginner-focused)
        # 50% basic, 30% intermediate, 20% advanced
        weights = [0.5, 0.3, 0.2]
        complexity = random.choices(subcats, weights=weights[:len(subcats)])[0]
        
        # Get templates for this category and complexity
        templates = self.templates.get(template_category, self.templates["general"]).get(complexity, self.templates["general"]["basic"])
        
        # Initialize template history for this word if not exists
        word_key = f"{word}_{template_category}"
        if word_key not in self.template_history:
            self.template_history[word_key] = []
        
        # Try to find a template that hasn't been used recently
        available_templates = [t for t in templates if t not in self.template_history[word_key]]
        
        # If all templates have been used, reset history
        if not available_templates:
            available_templates = templates
            self.template_history[word_key] = []
        
        # Select a template
        template = random.choice(available_templates)
        
        # Update history
        self.template_history[word_key].append(template)
        if len(self.template_history[word_key]) > min(5, len(templates)):
            self.template_history[word_key] = self.template_history[word_key][-5:]
        
        return template, complexity
    
    def _create_example_from_template(self, template, word, category):
        """Create an example from a template with proper handling of word forms."""
        # Special handling for plural words
        if word in self.plural_words:
            # Replace articles appropriately for plurals
            example = template.replace("a {word}", "{word}").replace("an {word}", "{word}")
            example = example.replace("the {word} is", "the {word} are")
            example = example.replace("This {word} is", "These {word} are")
            example = example.replace("this {word} is", "these {word} are")
        # Special handling for uncountable nouns
        elif category == "uncountable_clothing":
            # Remove articles for uncountable nouns
            example = template.replace("a {word}", "{word}").replace("an {word}", "{word}")
        # Regular handling for other words
        else:
            example = template
        
        # Insert word into template
        example = example.format(word=word)
        
        # Ensure first letter is capitalized
        if example and not example[0].isupper():
            example = example[0].upper() + example[1:]
        
        # Ensure ending punctuation
        if example and not example[-1] in '.!?':
            example += '.'
        
        return example
    
    def _get_free_dictionary_examples(self, word):
        """Get all examples from Free Dictionary API."""
        try:
            cache_path = os.path.join(self.cache_dir, f"freedict_{word}.json")
            
            # Try to load from cache first
            if os.path.exists(cache_path):
                with open(cache_path, 'r') as f:
                    cached_data = json.load(f)
                    if cached_data.get('examples'):
                        return cached_data['examples']
            
            # If not in cache, fetch from API
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
            
            # Cache results
            if examples:
                with open(cache_path, 'w') as f:
                    json.dump({'examples': examples}, f)
            
            return examples
        except Exception as e:
            if self.debug:
                print(f">>> Free Dictionary API error: {e}")
            return []
    
    def _get_wordnik_examples(self, word):
        """Get all examples from Wordnik API."""
        try:
            cache_path = os.path.join(self.cache_dir, f"wordnik_{word}.json")
            
            # Try to load from cache first
            if os.path.exists(cache_path):
                with open(cache_path, 'r') as f:
                    cached_data = json.load(f)
                    if cached_data.get('examples'):
                        return cached_data['examples']
            
            # If not in cache, fetch from API
            # Note: This API might need an API key in practice
            url = f"https://api.wordnik.com/v4/word.json/{word}/examples"
            response = requests.get(url, timeout=5)
            
            if response.status_code != 200:
                return []
                
            data = response.json()
            
            examples = []
            for example in data.get('examples', []):
                if 'text' in example and example['text']:
                    examples.append(example['text'])
            
            # Cache results
            if examples:
                with open(cache_path, 'w') as f:
                    json.dump({'examples': examples}, f)
            
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
        
        # Remove metadata markers
        sentence = re.sub(r'\[.*?\]', '', sentence)
        
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