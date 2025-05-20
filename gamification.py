"""
Gamification Module for Vocam Language Learning App

This module implements gamification features:
1. Achievement Badges & Milestones
2. Daily Challenges
3. Streaks & Consistency Rewards
4. Progress Bars & Visual Feedback
"""

import streamlit as st
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import base64
import json
import random
import tempfile
import time
from datetime import date, datetime, timedelta
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont
import io
import sqlite3

class GamificationSystem:
    def __init__(self, db_path="language_learning.db", translate_func=None):
        """Initialize the gamification system."""
        self.db_path = db_path
        self.translate_func = translate_func  # Can be None initially
        self.initialize_state()
        self.load_game_state()
        
        # Create asset directories if they don't exist
        self.ensure_asset_directories()
        
        # Check if we need to update streak and challenges for today
        self.check_daily_updates()
    
    def ensure_asset_directories(self):
        """Create necessary directories for gamification assets."""
        os.makedirs("assets/badges", exist_ok=True)
        os.makedirs("assets/tree", exist_ok=True)
    
    def initialize_state(self):
        """Initialize session state variables for gamification."""
        # Achievement and badge system
        if 'achievements' not in st.session_state:
            st.session_state['achievements'] = {}
        if 'badges' not in st.session_state:
            st.session_state['badges'] = {}
        if 'displayed_achievements' not in st.session_state:
            st.session_state['displayed_achievements'] = set()
        
        # Daily challenges
        if 'daily_challenges' not in st.session_state:
            st.session_state['daily_challenges'] = []
        if 'daily_challenges_completed' not in st.session_state:
            st.session_state['daily_challenges_completed'] = set()
        if 'word_of_the_day' not in st.session_state:
            st.session_state['word_of_the_day'] = None
        if 'word_of_the_day_date' not in st.session_state:
            st.session_state['word_of_the_day_date'] = None
            
        # Streak system
        if 'streak_days' not in st.session_state:
            st.session_state['streak_days'] = 0
        if 'last_active_date' not in st.session_state:
            st.session_state['last_active_date'] = None
        if 'streak_savers' not in st.session_state:
            st.session_state['streak_savers'] = 0
            
        # Progress tracking
        if 'points' not in st.session_state:
            st.session_state['points'] = 0
        if 'level' not in st.session_state:
            st.session_state['level'] = 1
        if 'category_progress' not in st.session_state:
            st.session_state['category_progress'] = {}
        if 'vocabulary_tree' not in st.session_state:
            st.session_state['vocabulary_tree'] = {
                'size': 1,
                'leaves': 0,
                'fruit': 0,
                'level': 1
            }
    
    def check_daily_updates(self):
        """Check and update streak and challenges for the current day."""
        self.check_streak()
        self.generate_daily_challenges()
        self.generate_word_of_the_day()
        self.update_category_progress()
    
    #=========================================================================
    # 1. Achievement Badges & Milestones
    #=========================================================================
    
    def award_achievement(self, achievement_id, title, description, points=25):
        """Award an achievement to the user if they don't already have it."""
        if achievement_id not in st.session_state.achievements:
            st.session_state.achievements[achievement_id] = {
                "id": achievement_id,
                "title": title,
                "description": description,
                "date_earned": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            st.session_state.displayed_achievements.add(achievement_id)
            
            # Award points
            self.add_points(points, f"Achievement: {title}")
            
            # Show notification
            st.toast(f"🏆 Achievement Unlocked: {title}")
            
            # Save game state
            self.save_game_state()
            return True
        return False
    
    def award_badge(self, badge_id, title, category="general"):
        """Award a badge to the user if they don't already have it."""
        if badge_id not in st.session_state.badges:
            # Create badge image path
            badge_image = self.create_badge_image(badge_id, title, category)
            
            # Store badge info
            st.session_state.badges[badge_id] = {
                "id": badge_id,
                "title": title,
                "category": category,
                "image_path": badge_image,
                "date_earned": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Show notification
            st.toast(f"🏅 Badge Earned: {title}")
            
            # Save game state
            self.save_game_state()
            return True
        return False
    
    def create_badge_image(self, badge_id, title, category):
        """Create a badge image and return the path."""
        try:
            badge_path = f"assets/badges/{badge_id}.png"
            
            # If badge already exists, return path
            if os.path.exists(badge_path):
                return badge_path
                
            # Create a new badge image
            size = 200
            img = Image.new('RGBA', (size, size), (255, 255, 255, 0))
            draw = ImageDraw.Draw(img)
            
            # Select color based on category
            colors = {
                "general": (52, 152, 219),     # Blue
                "streak": (231, 76, 60),       # Red
                "vocabulary": (46, 204, 113),  # Green
                "quiz": (155, 89, 182),        # Purple
                "challenge": (241, 196, 15)    # Yellow
            }
            
            color = colors.get(category, colors["general"])
            
            # Draw badge circle
            draw.ellipse([(10, 10), (size-10, size-10)], fill=color, outline=(255, 255, 255), width=3)
            
            # Add badge initial (first letter of title)
            try:
                # Try to load font, fallback to default if not available
                try:
                    font = ImageFont.truetype("Arial.ttf", 80)
                except:
                    font = ImageFont.load_default()
                
                # Get initial from title
                initial = title[0].upper()
                
                # Calculate text position (center)
                text_width, text_height = font.getsize(initial) if hasattr(font, 'getsize') else (50, 50)
                text_position = ((size - text_width) // 2, (size - text_height) // 2)
                
                # Draw text
                draw.text(text_position, initial, fill=(255, 255, 255), font=font)
            except Exception as e:
                print(f"Error adding text to badge: {e}")
                # Add a simple mark if text fails
                draw.rectangle([(size//2-20, size//2-20), (size//2+20, size//2+20)], fill=(255, 255, 255))
            
            # Save image
            img.save(badge_path)
            return badge_path
            
        except Exception as e:
            print(f"Error creating badge image: {e}")
            return None
    
    def check_achievements(self, action_type, **kwargs):
        """Check for achievements based on user actions."""
        if action_type == "word_learned":
            word = kwargs.get('word', '')
            category = kwargs.get('category', '')
            language = kwargs.get('language', '')
            
            # First word achievement
            if len(st.session_state.achievements) == 0:
                self.award_achievement("first_word", "First Steps", "Learn your first word")
                self.award_badge("first_word", "First Word", "vocabulary")
            
            # Category-specific achievements
            self.check_category_achievements(category)
            
            # Language-specific achievements
            self.check_language_achievements(language)
            
        elif action_type == "quiz_completed":
            score = kwargs.get('score', 0)
            total = kwargs.get('total', 0)
            
            # First quiz achievement
            if "first_quiz" not in st.session_state.achievements:
                self.award_achievement("first_quiz", "Quiz Taker", "Complete your first quiz")
                self.award_badge("quiz_taker", "Quiz Taker", "quiz")
            
            # Perfect quiz achievement
            if score == total and total >= 5:
                self.award_achievement("perfect_quiz", "Perfect Score", "Get a perfect score on a quiz")
                self.award_badge("perfect_score", "Perfect Score", "quiz")
    
    def check_category_achievements(self, category):
        """Check for category-specific achievements."""
        if not category:
            return
            
        # Get all vocabulary
        vocab_data = self.get_all_vocabulary()
        
        # Count words in this category
        category_count = sum(1 for word in vocab_data if word and 'category' in word and word['category'] == category)
        
        # Award achievements based on count
        if category_count >= 5:
            self.award_achievement(
                f"{category}_5", 
                f"{category.title()} Collector", 
                f"Learn 5 {category} words"
            )
            
        if category_count >= 10:
            self.award_achievement(
                f"{category}_10", 
                f"{category.title()} Master", 
                f"Learn 10 {category} words"
            )
            self.award_badge(f"{category}_master", f"{category.title()} Master", "vocabulary")
    
    def check_language_achievements(self, language):
        """Check for language-specific achievements."""
        if not language:
            return
            
        # Get all vocabulary
        vocab_data = self.get_all_vocabulary()
        
        # Count words in this language
        language_count = sum(1 for word in vocab_data 
                           if word and 'language_translated' in word and word['language_translated'] == language)
        
        # Get language name
        language_name = self.get_language_name(language)
        
        # Award achievements based on count
        if language_count >= 10:
            self.award_achievement(
                f"{language}_10", 
                f"{language_name} Beginner", 
                f"Learn 10 words in {language_name}"
            )
            
        if language_count >= 25:
            self.award_achievement(
                f"{language}_25", 
                f"{language_name} Enthusiast", 
                f"Learn 25 words in {language_name}"
            )
            self.award_badge(f"{language}_enthusiast", f"{language_name} Enthusiast", "vocabulary")
            
        if language_count >= 50:
            self.award_achievement(
                f"{language}_50", 
                f"{language_name} Speaker", 
                f"Learn 50 words in {language_name}"
            )
            self.award_badge(f"{language}_speaker", f"{language_name} Speaker", "vocabulary")
    
    def display_achievements(self):
        """Display earned achievements in the UI."""
        st.subheader("🏆 Achievements")
        
        if not st.session_state.achievements:
            st.info("You haven't earned any achievements yet. Keep learning to unlock them!")
            return
        
        # Group achievements by type
        achievement_types = {
            "Streaks": [],
            "Vocabulary": [],
            "Quizzes": [],
            "Challenges": [],
            "Other": []
        }
        
        for achievement_id, achievement in st.session_state.achievements.items():
            if "streak" in achievement_id:
                achievement_types["Streaks"].append(achievement)
            elif "word" in achievement_id or "_5" in achievement_id or "_10" in achievement_id or "_25" in achievement_id or "_50" in achievement_id:
                achievement_types["Vocabulary"].append(achievement)
            elif "quiz" in achievement_id:
                achievement_types["Quizzes"].append(achievement)
            elif "challenge" in achievement_id:
                achievement_types["Challenges"].append(achievement)
            else:
                achievement_types["Other"].append(achievement)
        
        # Display achievements by type
        for category, achievements in achievement_types.items():
            if achievements:
                with st.expander(f"{category} ({len(achievements)})", expanded=(category == "Recent")):
                    for achievement in sorted(achievements, key=lambda x: x.get("date_earned", ""), reverse=True):
                        st.markdown(f"**{achievement['title']}**: {achievement['description']}")
                        st.markdown(f"*Earned on {achievement.get('date_earned', 'Unknown').split()[0]}*")
                        st.markdown("---")
    
    def display_badges(self):
        """Display earned badges in the UI."""
        st.subheader("🏅 Badges")
        
        if not st.session_state.badges:
            st.info("You haven't earned any badges yet. Complete achievements to earn badges!")
            return
        
        # Display badges in a grid
        badges = list(st.session_state.badges.values())
        
        # Split badges into rows of 3
        badge_rows = [badges[i:i+3] for i in range(0, len(badges), 3)]
        
        for row in badge_rows:
            cols = st.columns(3)
            
            for i, badge in enumerate(row):
                with cols[i]:
                    try:
                        if os.path.exists(badge["image_path"]):
                            st.image(badge["image_path"], width=100)
                    except:
                        st.markdown("🏅")
                    
                    st.markdown(f"**{badge['title']}**")
                    earned_date = badge.get('date_earned', 'Unknown').split()[0]
                    st.markdown(f"*Earned: {earned_date}*")
    
    #=========================================================================
    # 2. Daily Challenges
    #=========================================================================
    
    def generate_daily_challenges(self):
        """Generate daily challenges based on user's vocabulary and progress."""
        today = date.today().strftime("%Y-%m-%d")
        
        # If we already generated challenges today, return those
        if st.session_state.daily_challenges and len(st.session_state.daily_challenges) > 0:
            if st.session_state.daily_challenges[0].get('date') == today:
                return st.session_state.daily_challenges
        
        # Get vocabulary data
        vocab_data = self.get_all_vocabulary()
        
        # Count words by category and language
        categories = defaultdict(int)
        languages = defaultdict(int)
        
        for word in vocab_data:
            if word and 'category' in word and word['category']:
                categories[word['category']] += 1
            
            if word and 'language_translated' in word:
                languages[word['language_translated']] += 1
        
        # Generate 3 daily challenges
        challenges = []
        
        # Challenge 1: Category-based challenge
        if categories:
            # Find categories with fewer items for targeted learning
            sorted_categories = sorted(categories.items(), key=lambda x: x[1])
            target_category = sorted_categories[0][0] if sorted_categories else "food"
            target_count = random.randint(2, 4)
            
            challenges.append({
                "id": f"category_{today}_{target_category}",
                "date": today,
                "title": f"Learn {target_count} {target_category} words",
                "description": f"Add {target_count} new {target_category} words to your vocabulary",
                "type": "category",
                "target": target_category,
                "target_count": target_count,
                "current_count": 0,
                "completed": False,
                "points": 20
            })
        
        # Challenge 2: Quiz challenge
        challenges.append({
            "id": f"quiz_{today}",
            "date": today,
            "title": "Ace a Quiz",
            "description": "Complete a quiz with at least 80% accuracy",
            "type": "quiz",
            "target_accuracy": 80,
            "completed": False,
            "points": 25
        })
        
        # Challenge 3: Language-specific challenge
        if languages:
            user_language = st.session_state.get('target_language', 'es')  # Default to Spanish
            target_count = random.randint(3, 5)
            
            challenges.append({
                "id": f"language_{today}_{user_language}",
                "date": today,
                "title": f"Learn {target_count} words in {self.get_language_name(user_language)}",
                "description": f"Add {target_count} new words in {self.get_language_name(user_language)}",
                "type": "language",
                "target": user_language,
                "target_count": target_count,
                "current_count": 0,
                "completed": False,
                "points": 15
            })
        
        # Save the challenges
        st.session_state.daily_challenges = challenges
        st.session_state.daily_challenges_completed = set()
        
        # Save state
        self.save_game_state()
        
        return challenges
    
    def complete_challenge(self, challenge_id):
        """Mark a challenge as completed and award points."""
        for challenge in st.session_state.daily_challenges:
            if challenge["id"] == challenge_id and not challenge["completed"]:
                challenge["completed"] = True
                st.session_state.daily_challenges_completed.add(challenge_id)
                
                # Award points
                self.add_points(challenge["points"], f"Challenge: {challenge['title']}")
                
                # Show notification
                st.toast(f"🎯 Challenge completed: {challenge['title']}")
                
                # Check for achievement
                if len(st.session_state.daily_challenges_completed) >= 10:
                    self.award_achievement(
                        "challenge_master", 
                        "Challenge Master", 
                        "Complete 10 daily challenges",
                        50
                    )
                    self.award_badge("challenge_master", "Challenge Master", "challenge")
                
                # Add to vocabulary tree
                self.update_vocabulary_tree(0.05)
                
                # Save state
                self.save_game_state()
                return True
        return False
    
    def check_challenge_progress(self, word_original=None, word_translated=None, language=None, quiz_score=None, quiz_total=None):
        """Check if any challenges are completed by user actions."""
        # Word learning challenges
        if word_original and language:
            category = self.get_object_category(word_original)
            
            for challenge in st.session_state.daily_challenges:
                if challenge["completed"]:
                    continue
                    
                if challenge["type"] == "category" and challenge["target"] == category:
                    challenge["current_count"] += 1
                    if challenge["current_count"] >= challenge["target_count"]:
                        self.complete_challenge(challenge["id"])
                        
                elif challenge["type"] == "language" and challenge["target"] == language:
                    challenge["current_count"] += 1
                    if challenge["current_count"] >= challenge["target_count"]:
                        self.complete_challenge(challenge["id"])
            
            # Check word of the day
            wotd = st.session_state.word_of_the_day
            if wotd and not wotd["saved"] and word_original.lower() == wotd["original"].lower():
                wotd["saved"] = True
                self.add_points(wotd["points"], "Word of the Day bonus")
                st.toast(f"🌟 Word of the Day bonus: +{wotd['points']} points!")
                
                # Award achievement if this is first time
                if "word_of_the_day" not in st.session_state.achievements:
                    self.award_achievement(
                        "word_of_the_day",
                        "Word Collector",
                        "Learn the Word of the Day"
                    )
        
        # Quiz challenges
        if quiz_score is not None and quiz_total is not None and quiz_total > 0:
            accuracy = (quiz_score / quiz_total) * 100
            
            for challenge in st.session_state.daily_challenges:
                if challenge["type"] == "quiz" and not challenge["completed"]:
                    if accuracy >= challenge["target_accuracy"]:
                        self.complete_challenge(challenge["id"])
    
    def generate_word_of_the_day(self):
        """Generate a word of the day."""
        today = date.today().strftime("%Y-%m-%d")
        
        # Return existing word if already generated today
        if st.session_state.word_of_the_day and st.session_state.word_of_the_day_date == today:
            return st.session_state.word_of_the_day
        
        # Words to potentially use as word of the day (common useful words)
        potential_words = [
            {"word": "hello", "category": "greeting"},
            {"word": "thank you", "category": "courtesy"},
            {"word": "please", "category": "courtesy"},
            {"word": "friend", "category": "people"},
            {"word": "food", "category": "essential"},
            {"word": "water", "category": "essential"},
            {"word": "help", "category": "emergency"},
            {"word": "good", "category": "description"},
            {"word": "bad", "category": "description"},
            {"word": "yes", "category": "response"},
            {"word": "no", "category": "response"},
            {"word": "goodbye", "category": "greeting"},
            {"word": "sorry", "category": "courtesy"},
            {"word": "excuse me", "category": "courtesy"},
            {"word": "love", "category": "emotion"},
            {"word": "book", "category": "object"},
            {"word": "car", "category": "vehicle"},
            {"word": "house", "category": "place"},
            {"word": "family", "category": "people"},
            {"word": "work", "category": "activity"}
        ]
        
        # Find words the user doesn't already have
        vocab_data = self.get_all_vocabulary()
        existing_words = set(word['word_original'].lower() for word in vocab_data if 'word_original' in word)
        
        # Filter to words not already in vocabulary 
        available_words = [word for word in potential_words if word["word"].lower() not in existing_words]
        
        # If all words are known, just pick any
        if not available_words:
            available_words = potential_words
        
        # Pick a random word
        word_of_the_day = random.choice(available_words)
        
        # Try to translate it
        translated_word = self.translate_placeholder(word_of_the_day["word"], st.session_state.get('target_language', 'es'))
        
        # Set word of the day
        st.session_state.word_of_the_day = {
            "original": word_of_the_day["word"],
            "translated": translated_word,
            "category": word_of_the_day["category"],
            "points": 10,
            "saved": False
        }
        
        st.session_state.word_of_the_day_date = today
        
        # Save state
        self.save_game_state()
        
        return st.session_state.word_of_the_day
    
    def display_daily_challenges(self):
        """Display daily challenges in the UI."""
        challenges = self.generate_daily_challenges()
        
        st.subheader("📅 Today's Challenges")
        
        # Display word of the day first
        if st.session_state.word_of_the_day:
            st.markdown("### ✨ Word of the Day")
            wotd = st.session_state.word_of_the_day
            
            st.markdown(f"**{wotd['original']}** → {wotd['translated']} ({wotd['category']})")
            
            if wotd["saved"]:
                st.success(f"✅ Added to vocabulary (+{wotd['points']} points)")
            else:
                st.info(f"Add this word to your vocabulary for {wotd['points']} bonus points!")
        
        # Display challenges
        if challenges:
            for challenge in challenges:
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    if challenge["completed"] or challenge["id"] in st.session_state.daily_challenges_completed:
                        st.markdown(f"✅ **{challenge['title']}**")
                        st.markdown(f"*{challenge['description']}* (Completed)")
                    else:
                        st.markdown(f"⭐ **{challenge['title']}**")
                        st.markdown(f"*{challenge['description']}*")
                        
                        # Show progress for challenges with a count
                        if challenge["type"] in ["category", "language"]:
                            current = challenge.get("current_count", 0)
                            target = challenge.get("target_count", 1)
                            progress = min(1.0, current / target)
                            st.progress(progress)
                            st.markdown(f"Progress: {current}/{target}")
                
                with col2:
                    st.markdown(f"+{challenge['points']} pts")
                
                st.markdown("---")
        else:
            st.info("No challenges available for today.")
    
    #=========================================================================
    # 3. Streaks & Consistency Rewards
    #=========================================================================
    
    def check_streak(self):
        """Update user streak based on last active date."""
        today = date.today()
        
        # Initialize streak if first time
        if st.session_state.last_active_date is None:
            st.session_state.streak_days = 1
            st.session_state.last_active_date = today
            self.save_game_state()
            return True
        
        # Convert string date to date object if needed
        last_date = st.session_state.last_active_date
        if isinstance(last_date, str):
            try:
                last_date = datetime.strptime(last_date, "%Y-%m-%d").date()
            except:
                last_date = today
        
        # Calculate days since last activity
        days_passed = (today - last_date).days
        
        # If already visited today, no streak change
        if days_passed == 0:
            return False
        
        # Streak continues if visited yesterday
        if days_passed == 1:
            st.session_state.streak_days += 1
            st.session_state.last_active_date = today
            
            # Award streak achievements
            if st.session_state.streak_days == 3:
                self.award_achievement("3_day_streak", "On Fire!", "Maintained a 3-day learning streak")
                self.award_badge("fire_starter", "Fire Starter", "streak")
            elif st.session_state.streak_days == 7:
                self.award_achievement("7_day_streak", "Week Warrior", "Maintained a 7-day learning streak")
                self.award_badge("week_warrior", "Week Warrior", "streak")
                # Award a streak saver at 7 days
                st.session_state.streak_savers += 1
                st.toast("🛟 You earned a Streak Saver! Use it to maintain your streak if you miss a day.")
            elif st.session_state.streak_days == 30:
                self.award_achievement("30_day_streak", "Monthly Master", "Maintained a 30-day learning streak")
                self.award_badge("monthly", "Monthly Master", "streak")
                # Award more streak savers at 30 days
                st.session_state.streak_savers += 2
                st.toast("🛟 You earned 2 Streak Savers for your amazing dedication!")
            
            # Add bonus points for consecutive days
            bonus_points = min(st.session_state.streak_days, 10)  # Cap at 10 points
            self.add_points(bonus_points, f"Streak bonus (Day {st.session_state.streak_days})")
            
            # Update vocabulary tree for streak maintenance
            self.update_vocabulary_tree(0.02)
            
            self.save_game_state()
            return True
            
        # Missed a day but have a streak saver
        elif days_passed == 2 and st.session_state.streak_savers > 0:
            st.session_state.streak_savers -= 1
            st.session_state.last_active_date = today
            st.toast("🛟 Used a Streak Saver to maintain your streak!")
            self.save_game_state()
            return True
            
        # Streak broken
        else:
            st.session_state.streak_days = 1
            st.session_state.last_active_date = today
            self.save_game_state()
            return False
    
    def add_points(self, amount, reason=""):
        """Add points to the user's account and check for level ups."""
        st.session_state.points += amount
        
        # Check for level up
        old_level = st.session_state.level
        new_level = 1 + int(st.session_state.points / 100)  # Level up every 100 points
        
        if new_level > old_level:
            st.session_state.level = new_level
            st.toast(f"🎉 Level Up! You're now level {new_level}")
            
            # Award level-up achievement
            if new_level == 5:
                self.award_achievement("level_5", "Level 5", "Reach level 5 in your learning journey", 50)
                self.award_badge("level_5", "Level 5", "general")
            elif new_level == 10:
                self.award_achievement("level_10", "Level 10", "Reach level 10 in your learning journey", 100)
                self.award_badge("level_10", "Level 10", "general")
            
            # Award streak saver on level up
            if new_level > old_level:
                st.session_state.streak_savers += 1
                st.toast("🛟 You earned a Streak Saver for leveling up!")
            
            # Update vocabulary tree
            self.update_vocabulary_tree(0.2)  # Big growth on level up
        
        # Save state
        self.save_game_state()
    
    def display_streak_info(self):
        """Display streak information in the UI."""
        st.subheader("🔥 Learning Streak")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**Current Streak**: {st.session_state.streak_days} days")
            if st.session_state.streak_days >= 7:
                st.success(f"Amazing! Keep up the great work! 🎉")
            elif st.session_state.streak_days >= 3:
                st.success(f"You're on fire! 🔥")
        
        with col2:
            if st.session_state.streak_savers > 0:
                st.markdown(f"**Streak Savers**: {st.session_state.streak_savers}")
                st.info("Streak savers help maintain your streak if you miss a day.")
        
        # Next streak milestone
        if st.session_state.streak_days < 3:
            target = 3
        elif st.session_state.streak_days < 7:
            target = 7
        elif st.session_state.streak_days < 30:
            target = 30
        else:
            target = 100
        
        # Progress to next milestone
        progress = st.session_state.streak_days / target
        st.markdown(f"**Next milestone**: {target} days")
        st.progress(min(1.0, progress))
        
        # Days to go
        days_to_go = target - st.session_state.streak_days
        if days_to_go > 0:
            st.markdown(f"*{days_to_go} days to go!*")
    
    #=========================================================================
    # 4. Progress Bars & Visual Feedback
    #=========================================================================
    
    def update_category_progress(self):
        """Update progress for different categories based on vocabulary."""
        vocab_data = self.get_all_vocabulary()
        
        # Count totals by category and language
        category_counts = defaultdict(int)
        language_counts = defaultdict(lambda: defaultdict(int))
        
        for word in vocab_data:
            if word and 'category' in word and word['category']:
                category_counts[word['category']] += 1
            
            if word and 'language_translated' in word and 'category' in word and word['category']:
                language_counts[word['language_translated']][word['category']] += 1
        
        # Calculate progress for each category (assume targets)
        category_targets = {
            "food": 15,
            "animals": 10,
            "vehicles": 8,
            "electronics": 10,
            "furniture": 8,
            "personal": 6,
            "sports": 8,
            "household": 10,
            "other": 10,
            "manual": 5,
            "text": 10
        }
        
        # Calculate overall progress
        total_words = sum(category_counts.values())
        overall_target = sum(category_targets.values())
        overall_progress = min(1.0, total_words / max(1, overall_target))
        
        # Update progress state
        st.session_state.category_progress = {
            "categories": {cat: min(1.0, category_counts[cat] / max(1, category_targets.get(cat, 10))) 
                          for cat in set(list(category_counts.keys()) + list(category_targets.keys()))},
            "languages": {lang: {cat: count for cat, count in cats.items()} 
                         for lang, cats in language_counts.items()},
            "overall": overall_progress,
            "total_words": total_words
        }
        
        # Save state
        self.save_game_state()
    
    def update_vocabulary_tree(self, growth_amount):
        """Update the vocabulary tree based on learning progress."""
        tree = st.session_state.vocabulary_tree
        
        # Add growth to the tree
        tree['size'] += growth_amount
        
        # Check for size thresholds to add leaves and fruit
        if tree['size'] >= 1.5 and tree['leaves'] < 5:
            # Add leaves as tree grows
            leaves_to_add = min(5 - tree['leaves'], int(tree['size'] - 1))
            if leaves_to_add > 0:
                tree['leaves'] += leaves_to_add
                st.toast(f"🌿 Your vocabulary tree is growing new leaves!")
        
        if tree['size'] >= 3 and tree['fruit'] < 10:
            # Add fruit as tree grows even larger
            fruit_to_add = min(10 - tree['fruit'], int((tree['size'] - 2) / 0.5))
            if fruit_to_add > 0:
                tree['fruit'] += fruit_to_add
                st.toast(f"🍎 Your vocabulary tree is growing fruit!")
        
        # Level up tree at certain sizes
        if tree['size'] >= 5 and tree['level'] == 1:
            tree['level'] = 2
            st.toast(f"🌳 Your vocabulary tree has grown to level 2!")
            self.award_achievement("tree_level_2", "Growing Tree", "Grow your vocabulary tree to level 2", 30)
        elif tree['size'] >= 10 and tree['level'] == 2:
            tree['level'] = 3
            st.toast(f"🌳 Your vocabulary tree has grown to level 3!")
            self.award_achievement("tree_level_3", "Thriving Tree", "Grow your vocabulary tree to level 3", 50)
            self.award_badge("tree_master", "Tree Master", "vocabulary")
        
        # Save state
        self.save_game_state()
    
    def generate_tree_image(self):
        """Generate an image of the vocabulary tree."""
        try:
            tree = st.session_state.vocabulary_tree
            
            # Create a base image
            width, height = 400, 400
            img = Image.new('RGBA', (width, height), (255, 255, 255, 0))
            draw = ImageDraw.Draw(img)
            
            # Tree properties based on level and size
            trunk_width = 20 + (tree['level'] * 10)
            trunk_height = 100 + (tree['size'] * 20)
            canopy_size = 50 + (tree['size'] * 30)
            
            # Draw trunk
            trunk_color = (139, 69, 19)  # Brown
            trunk_x = width // 2
            trunk_bottom = height - 50
            trunk_top = trunk_bottom - trunk_height
            
            draw.rectangle(
                [(trunk_x - trunk_width//2, trunk_top), 
                 (trunk_x + trunk_width//2, trunk_bottom)],
                fill=trunk_color
            )
            
            # Draw canopy (tree top)
            canopy_color = (34, 139, 34)  # Forest Green
            draw.ellipse(
                [(trunk_x - canopy_size, trunk_top - canopy_size),
                 (trunk_x + canopy_size, trunk_top + canopy_size//2)],
                fill=canopy_color
            )
            
            # Add leaves
            if tree['leaves'] > 0:
                leaf_color = (50, 205, 50)  # Lime Green
                for i in range(tree['leaves']):
                    angle = (i / tree['leaves']) * 360
                    radius = canopy_size * 0.8
                    x = trunk_x + int(radius * np.cos(np.radians(angle)))
                    y = trunk_top - int(radius * np.sin(np.radians(angle)) * 0.5)
                    
                    draw.ellipse(
                        [(x - 15, y - 10), (x + 15, y + 10)],
                        fill=leaf_color
                    )
            
            # Add fruit
            if tree['fruit'] > 0:
                fruit_color = (255, 0, 0)  # Red
                for i in range(tree['fruit']):
                    angle = (i / tree['fruit']) * 360
                    radius = canopy_size * 0.6
                    x = trunk_x + int(radius * np.cos(np.radians(angle)))
                    y = trunk_top - int(radius * np.sin(np.radians(angle)) * 0.7)
                    
                    draw.ellipse(
                        [(x - 8, y - 8), (x + 8, y + 8)],
                        fill=fruit_color
                    )
            
            # Convert to bytes for Streamlit
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            byte_im = buf.getvalue()
            
            return byte_im
        except Exception as e:
            print(f"Error generating tree image: {e}")
            return None
    
    def display_progress_visuals(self):
        """Display progress visualizations in the UI."""
        st.subheader("📊 Learning Progress")
        
        # Display overall progress
        total_words = st.session_state.category_progress.get("total_words", 0)
        st.markdown(f"### Total Words Learned: {total_words}")
        
        # Target goals by level
        target_words = 10 * st.session_state.level  # Increases with level
        progress = min(1.0, total_words / target_words)
        
        st.markdown(f"**Level Goal**: {total_words}/{target_words} words")
        st.progress(progress)
        
        # Display category progress
        st.markdown("### Category Progress")
        
        categories = st.session_state.category_progress.get("categories", {})
        if categories:
            sorted_categories = sorted(
                [(cat, prog) for cat, prog in categories.items()],
                key=lambda x: x[1],
                reverse=True
            )
            
            for category, progress in sorted_categories:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**{category.title()}**")
                    st.progress(progress)
                with col2:
                    percentage = int(progress * 100)
                    st.markdown(f"{percentage}%")
        else:
            st.info("Start learning words to see category progress!")
        
        # Display vocabulary tree
        st.markdown("### 🌳 Vocabulary Tree")
        
        tree_img = self.generate_tree_image()
        if tree_img:
            st.image(tree_img, width=300)
            
            # Tree stats
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Tree Level**: {st.session_state.vocabulary_tree['level']}")
                st.markdown(f"**Size**: {st.session_state.vocabulary_tree['size']:.1f}")
            with col2:
                st.markdown(f"**Leaves**: {st.session_state.vocabulary_tree['leaves']}")
                st.markdown(f"**Fruit**: {st.session_state.vocabulary_tree['fruit']}")
            
            st.info("Your tree grows as you learn more words, complete challenges, and maintain your streak!")
        else:
            st.info("Keep learning to grow your vocabulary tree!")
    
    #=========================================================================
    # Gamification Dashboard & Integration
    #=========================================================================
    
    def render_dashboard(self):
        """Render the complete gamification dashboard."""
        st.title("🎮 My Progress")
        
        # User level and points at the top
        col1, col2 = st.columns([1, 3])
        
        with col1:
            # Show level badge
            if st.session_state.level >= 5 and "level_5" in st.session_state.badges:
                try:
                    badge_path = st.session_state.badges["level_5"]["image_path"]
                    if os.path.exists(badge_path):
                        st.image(badge_path, width=100)
                except:
                    st.markdown(f"## Level {st.session_state.level}")
            else:
                st.markdown(f"## Level {st.session_state.level}")
        
        with col2:
            # Show progress to next level
            next_level = st.session_state.level + 1
            points_needed = (next_level - 1) * 100
            current_level_points = st.session_state.points - ((st.session_state.level - 1) * 100)
            progress_percentage = current_level_points / 100
            
            st.markdown(f"### {st.session_state.points} total points")
            st.markdown(f"{current_level_points}/100 to level {next_level}")
            st.progress(progress_percentage)
        
        # Create tabs for different sections
        tab1, tab2, tab3, tab4 = st.tabs(["Challenges", "Streaks", "Progress", "Achievements"])
        
        with tab1:
            self.display_daily_challenges()
        
        with tab2:
            self.display_streak_info()
        
        with tab3:
            self.display_progress_visuals()
        
        with tab4:
            col1, col2 = st.columns(2)
            with col1:
                self.display_achievements()
            with col2:
                self.display_badges()
    
    def update_sidebar(self):
        """Update the sidebar with gamification information."""
        try:
            with st.sidebar.expander("🏆 My Progress", expanded=True):
                # Use get() with default values to safely access session state
                level = st.session_state.get("level", 1)  # Default to 1 if not exists
                points = st.session_state.get("points", 0)  # Default to 0 if not exists
                
                # Display level and points
                st.markdown(f"**Level {level}** | {points} points")
                
                # Streak - check if it exists first
                streak_days = st.session_state.get("streak_days", 0)
                if streak_days > 0:
                    st.markdown(f"🔥 **{streak_days}** day streak")
                
                # Word of the day preview - check if it exists first
                wotd = st.session_state.get("word_of_the_day")
                if wotd and isinstance(wotd, dict):  # Ensure it's a valid dictionary
                    st.markdown("**✨ Word of the Day**")
                    original = wotd.get('original', '?')
                    translated = wotd.get('translated', '?')
                    st.markdown(f"{original} → {translated}")
                
                # Challenge preview - check if they exist first
                daily_challenges = st.session_state.get("daily_challenges", [])
                if daily_challenges and isinstance(daily_challenges, list) and len(daily_challenges) > 0:
                    completed = sum(1 for c in daily_challenges if c.get("completed", False))
                    total = len(daily_challenges)
                    st.markdown(f"**📅 Challenges**: {completed}/{total} completed")
        except Exception as e:
            # Provide a fallback if anything goes wrong
            st.sidebar.markdown("🏆 **Progress system is initializing...**")
            print(f"Error in update_sidebar: {e}")  # Log the error
    
    #=========================================================================
    # Helper Functions
    #=========================================================================
    
    def get_all_vocabulary_direct(self):
        """Get all vocabulary items directly from SQLite."""
        try:
            # Connect to the database
            conn = sqlite3.connect(self.db_path)
            
            # Use dictionary cursor for easier access
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Get all vocabulary with user progress info
            cursor.execute('''
            SELECT v.id, v.word_original, v.word_translated, v.language_translated,
                v.category, v.image_path, v.date_added,
                up.proficiency_level, up.review_count, up.correct_count, up.last_reviewed
            FROM vocabulary v
            LEFT JOIN user_progress up ON v.id = up.vocabulary_id
            ORDER BY v.date_added DESC
            ''')
            
            # Fetch all results
            results = cursor.fetchall()
            
            # Convert to list of dictionaries
            vocabulary = []
            for row in results:
                # Convert row to dictionary
                word = dict(row)
                vocabulary.append(word)
            
            conn.close()
            return vocabulary
        except Exception as e:
            print(f"Error retrieving vocabulary: {str(e)}")
            return []
    
    def get_all_vocabulary(self):
        """Get all vocabulary, handling different potential sources."""
        # Try the direct method
        vocabulary = self.get_all_vocabulary_direct()
        
        # If that fails, try the session state
        if not vocabulary and 'vocabulary' in st.session_state:
            vocabulary = st.session_state.vocabulary
        
        return vocabulary
    
    def get_object_category(self, label):
        """Get the category for a detected object label."""
        label = label.lower()
        for category, items in OBJECT_CATEGORIES.items():
            if label in items:
                return category
        return "other"
    
    def get_language_name(self, language_code):
        """Get language name from code."""
        language_map = {
            "es": "Spanish",
            "fr": "French",
            "de": "German",
            "it": "Italian",
            "pt": "Portuguese",
            "ru": "Russian",
            "ja": "Japanese",
            "zh-CN": "Chinese"
        }
        return language_map.get(language_code, language_code)
    
    def set_translate_func(self, translate_func):
        """Set or update the translation function after initialization."""
        self.translate_func = translate_func
    
    def translate_placeholder(self, text, target_language):
        """Translate text using the provided translation function or fallback to placeholders."""
        # If we have a translation function, use it
        if self.translate_func:
            try:
                return self.translate_func(text, target_language)
            except Exception as e:
                print(f"Translation error: {e}")
                # Fall through to placeholder if translation fails
        
        # Fallback to simple dictionary for common words
        language_examples = {
            "es": {"hello": "hola", "thank you": "gracias", "please": "por favor", "book": "libro"},
            "fr": {"hello": "bonjour", "thank you": "merci", "please": "s'il vous plaît", "book": "livre"},
            "de": {"hello": "hallo", "thank you": "danke", "please": "bitte", "book": "Buch"},
            "it": {"hello": "ciao", "thank you": "grazie", "please": "per favore", "book": "libro"}
        }
        
        # Try to get a canned translation
        if target_language in language_examples and text.lower() in language_examples[target_language]:
            return language_examples[target_language][text.lower()]
            
        # If no translation available, return placeholder
        return f"[{text} in {self.get_language_name(target_language)}]"
    
    #=========================================================================
    # State Management
    #=========================================================================
    
    def save_game_state(self):
        """Save game state to the database."""
        try:
            # Create a JSON representation of the gamification state
            game_state = {
                "achievements": st.session_state.achievements,
                "badges": st.session_state.badges,
                "streak_days": st.session_state.streak_days,
                "last_active_date": str(st.session_state.last_active_date) if st.session_state.last_active_date else None,
                "streak_savers": st.session_state.streak_savers,
                "points": st.session_state.points,
                "level": st.session_state.level,
                "daily_challenges": st.session_state.daily_challenges,
                "daily_challenges_completed": list(st.session_state.daily_challenges_completed),
                "word_of_the_day": st.session_state.word_of_the_day,
                "word_of_the_day_date": st.session_state.word_of_the_day_date,
                "category_progress": st.session_state.category_progress,
                "vocabulary_tree": st.session_state.vocabulary_tree
            }
            
            # Convert to JSON string
            game_state_json = json.dumps(game_state)
            
            # Save to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if game_state table exists, create if not
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS game_state (
                id INTEGER PRIMARY KEY,
                state_json TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            ''')
            
            # Update or insert game state
            cursor.execute("SELECT id FROM game_state WHERE id = 1")
            if cursor.fetchone():
                cursor.execute("UPDATE game_state SET state_json = ?, updated_at = ? WHERE id = 1", 
                            (game_state_json, datetime.now()))
            else:
                cursor.execute("INSERT INTO game_state (id, state_json, updated_at) VALUES (1, ?, ?)", 
                            (game_state_json, datetime.now()))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            # If database operations fail, just log and continue
            print(f"Error saving game state: {e}")
    
    def load_game_state(self):
        """Load game state from the database."""
        try:
            # Connect to the database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='game_state'")
            if not cursor.fetchone():
                conn.close()
                return False
            
            # Get the state
            cursor.execute("SELECT state_json FROM game_state WHERE id = 1")
            result = cursor.fetchone()
            conn.close()
            
            if not result:
                return False
            
            # Parse the state
            game_state = json.loads(result[0])
            
            # Update session state
            st.session_state.achievements = game_state.get("achievements", {})
            st.session_state.badges = game_state.get("badges", {})
            st.session_state.streak_days = game_state.get("streak_days", 0)
            st.session_state.last_active_date = game_state.get("last_active_date")
            st.session_state.streak_savers = game_state.get("streak_savers", 0)
            st.session_state.points = game_state.get("points", 0)
            st.session_state.level = game_state.get("level", 1)
            st.session_state.daily_challenges = game_state.get("daily_challenges", [])
            st.session_state.daily_challenges_completed = set(game_state.get("daily_challenges_completed", []))
            st.session_state.word_of_the_day = game_state.get("word_of_the_day")
            st.session_state.word_of_the_day_date = game_state.get("word_of_the_day_date")
            st.session_state.category_progress = game_state.get("category_progress", {})
            st.session_state.vocabulary_tree = game_state.get("vocabulary_tree", {
                'size': 1, 'leaves': 0, 'fruit': 0, 'level': 1
            })
            
            return True
        except Exception as e:
            print(f"Error loading game state: {e}")
            return False

# Define Object Categories (copied from main.py)
OBJECT_CATEGORIES = {
    "food": ["apple", "banana", "orange", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", 
             "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake"],
    
    "animals": ["bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"],
    
    "vehicles": ["bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"],
    
    "electronics": ["tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", 
                   "toaster", "sink", "refrigerator"],
    
    "furniture": ["chair", "couch", "potted plant", "bed", "dining table", "toilet"],
    
    "personal": ["backpack", "umbrella", "handbag", "tie", "suitcase"],
    
    "sports": ["frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", 
              "baseball glove", "skateboard", "surfboard", "tennis racket"],
    
    "household": ["bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "book", "clock", 
                 "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
}