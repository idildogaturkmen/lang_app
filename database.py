import sqlite3
import os
import datetime
import hashlib

class LanguageLearningDB:
    def __init__(self, db_path):
        """Initialize the database connection."""
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()
        
        # Create tables if they don't exist
        self._create_tables()
        self._create_indexes()
    
    def _create_tables(self):
        """Create necessary database tables if they don't exist."""
        self.cursor.executescript('''
        -- Users table for Supabase user data
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            email TEXT UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP,
            display_name TEXT,
            preferences TEXT  -- JSON string for user preferences
        );
        
        -- Vocabulary table with user_id foreign key
        CREATE TABLE IF NOT EXISTS vocabulary (
            id INTEGER PRIMARY KEY,
            user_id TEXT NOT NULL,
            word_original TEXT NOT NULL,
            word_translated TEXT NOT NULL,
            language_translated TEXT NOT NULL,
            category TEXT,
            image_path TEXT,
            date_added TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            source TEXT DEFAULT 'manual',
            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
        );
        
        -- User progress table with user_id foreign key
        CREATE TABLE IF NOT EXISTS user_progress (
            id INTEGER PRIMARY KEY,
            user_id TEXT NOT NULL,
            vocabulary_id INTEGER,
            review_count INTEGER DEFAULT 0,
            correct_count INTEGER DEFAULT 0,
            last_reviewed TIMESTAMP,
            proficiency_level INTEGER DEFAULT 0,
            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE,
            FOREIGN KEY (vocabulary_id) REFERENCES vocabulary (id) ON DELETE CASCADE
        );
        
        -- Sessions table with user_id foreign key
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY,
            user_id TEXT NOT NULL,
            start_time TIMESTAMP,
            end_time TIMESTAMP,
            words_studied INTEGER DEFAULT 0,
            words_learned INTEGER DEFAULT 0,
            session_type TEXT DEFAULT 'general',  -- 'general', 'quiz', 'pronunciation', etc.
            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
        );
        
        -- Camera translations table with user_id foreign key
        CREATE TABLE IF NOT EXISTS camera_translations (
            id INTEGER PRIMARY KEY,
            user_id TEXT NOT NULL,
            image_path TEXT,
            detected_text TEXT,
            translated_text TEXT,
            source_language TEXT,
            target_language TEXT,
            date_captured TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_saved_to_vocabulary BOOLEAN DEFAULT 0,
            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
        );
        
        -- User achievements table for gamification
        CREATE TABLE IF NOT EXISTS user_achievements (
            id INTEGER PRIMARY KEY,
            user_id TEXT NOT NULL,
            achievement_id TEXT NOT NULL,
            achievement_name TEXT NOT NULL,
            achieved_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            achievement_data TEXT,  -- JSON string for additional data
            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE,
            UNIQUE(user_id, achievement_id)
        );
        
        -- User statistics table for tracking progress
        CREATE TABLE IF NOT EXISTS user_statistics (
            id INTEGER PRIMARY KEY,
            user_id TEXT NOT NULL,
            stat_date DATE DEFAULT CURRENT_DATE,
            words_learned_today INTEGER DEFAULT 0,
            time_spent_minutes INTEGER DEFAULT 0,
            quiz_accuracy REAL DEFAULT 0.0,
            streak_days INTEGER DEFAULT 0,
            total_points INTEGER DEFAULT 0,
            level INTEGER DEFAULT 1,
            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE,
            UNIQUE(user_id, stat_date)
        );
        
        -- Quiz results table
        CREATE TABLE IF NOT EXISTS quiz_results (
            id INTEGER PRIMARY KEY,
            user_id TEXT NOT NULL,
            session_id INTEGER,
            quiz_type TEXT NOT NULL,
            total_questions INTEGER NOT NULL,
            correct_answers INTEGER NOT NULL,
            completion_time INTEGER,  -- in seconds
            date_completed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE,
            FOREIGN KEY (session_id) REFERENCES sessions (id) ON DELETE SET NULL
        );
        ''')
        self.conn.commit()
    
    def _create_indexes(self):
        """Create database indexes for better performance."""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_vocabulary_user_id ON vocabulary (user_id);",
            "CREATE INDEX IF NOT EXISTS idx_vocabulary_language ON vocabulary (language_translated);",
            "CREATE INDEX IF NOT EXISTS idx_vocabulary_category ON vocabulary (category);",
            "CREATE INDEX IF NOT EXISTS idx_user_progress_user_id ON user_progress (user_id);",
            "CREATE INDEX IF NOT EXISTS idx_user_progress_vocab_id ON user_progress (vocabulary_id);",
            "CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions (user_id);",
            "CREATE INDEX IF NOT EXISTS idx_sessions_date ON sessions (start_time);",
            "CREATE INDEX IF NOT EXISTS idx_camera_translations_user_id ON camera_translations (user_id);",
            "CREATE INDEX IF NOT EXISTS idx_user_achievements_user_id ON user_achievements (user_id);",
            "CREATE INDEX IF NOT EXISTS idx_user_statistics_user_id ON user_statistics (user_id);",
            "CREATE INDEX IF NOT EXISTS idx_quiz_results_user_id ON quiz_results (user_id);"
        ]
        
        for index_sql in indexes:
            try:
                self.cursor.execute(index_sql)
            except sqlite3.Error as e:
                print(f"Warning: Could not create index: {e}")
        
        self.conn.commit()
    
    def create_or_update_user(self, user_id, email, display_name=None):
        """Create or update a user record from Supabase authentication."""
        try:
            # Check if user exists
            self.cursor.execute("SELECT id FROM users WHERE id = ?", (user_id,))
            existing_user = self.cursor.fetchone()
            
            if existing_user:
                # Update existing user
                self.cursor.execute('''
                UPDATE users 
                SET email = ?, display_name = ?, last_login = CURRENT_TIMESTAMP
                WHERE id = ?
                ''', (email, display_name, user_id))
            else:
                # Create new user
                self.cursor.execute('''
                INSERT INTO users (id, email, display_name, last_login)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                ''', (user_id, email, display_name))
            
            self.conn.commit()
            return True
        except sqlite3.Error as e:
            print(f"Error creating/updating user: {e}")
            return False
    
    def get_user(self, user_id):
        """Get user information by ID."""
        try:
            self.cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
            return self.cursor.fetchone()
        except sqlite3.Error as e:
            print(f"Error getting user: {e}")
            return None
    
    def add_vocabulary(self, user_id, word_original, word_translated, language_translated, category=None, image_path=None, source='manual'):
        """Add a new vocabulary entry to the database for a specific user."""
        try:
            vocab_id = self.cursor.execute('''
            INSERT INTO vocabulary 
            (user_id, word_original, word_translated, language_translated, category, image_path, source)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (user_id, word_original, word_translated, language_translated, category, image_path, source)).lastrowid
            
            # Initialize user progress for this vocabulary
            self.cursor.execute('''
            INSERT INTO user_progress (user_id, vocabulary_id, last_reviewed)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            ''', (user_id, vocab_id))
            
            self.conn.commit()
            return vocab_id
        except sqlite3.Error as e:
            print(f"Error adding vocabulary: {e}")
            return None
    
    def get_vocabulary(self, vocabulary_id, user_id):
        """Get a specific vocabulary entry by ID for a specific user."""
        try:
            self.cursor.execute('''
            SELECT v.*, p.proficiency_level, p.review_count, p.correct_count, p.last_reviewed
            FROM vocabulary v
            LEFT JOIN user_progress p ON v.id = p.vocabulary_id AND v.user_id = p.user_id
            WHERE v.id = ? AND v.user_id = ?
            ''', (vocabulary_id, user_id))
            
            return self.cursor.fetchone()
        except sqlite3.Error as e:
            print(f"Error getting vocabulary: {e}")
            return None
    
    def get_all_vocabulary(self, user_id, category=None, language=None):
        """Get all vocabulary entries for a specific user, optionally filtered by category and/or language."""
        try:
            query = '''
            SELECT v.*, p.proficiency_level, p.review_count, p.correct_count, p.last_reviewed
            FROM vocabulary v
            LEFT JOIN user_progress p ON v.id = p.vocabulary_id AND v.user_id = p.user_id
            WHERE v.user_id = ?
            '''
            params = [user_id]
            
            if category and language:
                query += " AND v.category = ? AND v.language_translated = ?"
                params.extend([category, language])
            elif category:
                query += " AND v.category = ?"
                params.append(category)
            elif language:
                query += " AND v.language_translated = ?"
                params.append(language)
                
            query += " ORDER BY v.date_added DESC"
            
            self.cursor.execute(query, params)
            return self.cursor.fetchall()
        except sqlite3.Error as e:
            print(f"Error getting vocabulary: {e}")
            return []
    
    def update_vocabulary(self, vocabulary_id, user_id, word_original=None, word_translated=None, 
                          language_translated=None, category=None, image_path=None):
        """Update an existing vocabulary entry for a specific user."""
        try:
            # Get current values first
            current = self.get_vocabulary(vocabulary_id, user_id)
            if not current:
                return False
                
            # Use current values for any parameter not provided
            word_original = word_original if word_original is not None else current['word_original']
            word_translated = word_translated if word_translated is not None else current['word_translated']
            language_translated = language_translated if language_translated is not None else current['language_translated']
            category = category if category is not None else current['category']
            image_path = image_path if image_path is not None else current['image_path']
            
            self.cursor.execute('''
            UPDATE vocabulary 
            SET word_original = ?, word_translated = ?, 
                language_translated = ?, category = ?, image_path = ?
            WHERE id = ? AND user_id = ?
            ''', (word_original, word_translated, language_translated, 
                  category, image_path, vocabulary_id, user_id))
            
            self.conn.commit()
            return True
        except sqlite3.Error as e:
            print(f"Error updating vocabulary: {e}")
            return False
    
    def delete_vocabulary(self, vocabulary_id, user_id):
        """Delete a vocabulary entry by ID for a specific user and associated progress."""
        try:
            # Delete associated progress first (due to foreign key constraint)
            self.cursor.execute('''
            DELETE FROM user_progress WHERE vocabulary_id = ? AND user_id = ?
            ''', (vocabulary_id, user_id))
            
            # Delete vocabulary entry
            self.cursor.execute('''
            DELETE FROM vocabulary WHERE id = ? AND user_id = ?
            ''', (vocabulary_id, user_id))
            
            self.conn.commit()
            return True
        except sqlite3.Error as e:
            print(f"Error deleting vocabulary: {e}")
            return False
    
    def search_vocabulary(self, user_id, search_term, language=None):
        """Search vocabulary entries for a term in original or translated word for a specific user."""
        try:
            query = '''
            SELECT v.*, p.proficiency_level, p.review_count, p.correct_count, p.last_reviewed
            FROM vocabulary v
            LEFT JOIN user_progress p ON v.id = p.vocabulary_id AND v.user_id = p.user_id
            WHERE v.user_id = ? AND (v.word_original LIKE ? OR v.word_translated LIKE ?)
            '''
            params = [user_id, f'%{search_term}%', f'%{search_term}%']
            
            if language:
                query += " AND v.language_translated = ?"
                params.append(language)
                
            self.cursor.execute(query, params)
            return self.cursor.fetchall()
        except sqlite3.Error as e:
            print(f"Error searching vocabulary: {e}")
            return []
    
    def update_word_progress(self, vocabulary_id, is_correct):
        """Update the progress for a vocabulary word after review."""
        try:
            # Get the user_id for this vocabulary item first
            self.cursor.execute("SELECT user_id FROM vocabulary WHERE id = ?", (vocabulary_id,))
            vocab_result = self.cursor.fetchone()
            if not vocab_result:
                return False
            
            user_id = vocab_result['user_id']
            
            # Get current progress
            self.cursor.execute('''
            SELECT * FROM user_progress WHERE vocabulary_id = ? AND user_id = ?
            ''', (vocabulary_id, user_id))
            
            progress = self.cursor.fetchone()
            if not progress:
                # Create new progress entry if it doesn't exist
                self.cursor.execute('''
                INSERT INTO user_progress 
                (user_id, vocabulary_id, review_count, correct_count, last_reviewed, proficiency_level)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, ?)
                ''', (user_id, vocabulary_id, 1, 1 if is_correct else 0, 0))
            else:
                # Update existing progress
                review_count = progress['review_count'] + 1
                correct_count = progress['correct_count'] + (1 if is_correct else 0)
                
                # Calculate new proficiency level (0-5)
                proficiency = min(5, int((correct_count / review_count) * 5))
                
                self.cursor.execute('''
                UPDATE user_progress 
                SET review_count = ?, correct_count = ?, 
                    last_reviewed = CURRENT_TIMESTAMP, proficiency_level = ?
                WHERE vocabulary_id = ? AND user_id = ?
                ''', (review_count, correct_count, proficiency, vocabulary_id, user_id))
            
            self.conn.commit()
            return True
        except sqlite3.Error as e:
            print(f"Error updating word progress: {e}")
            return False
    
    def get_word_progress(self, vocabulary_id, user_id):
        """Get the progress for a specific vocabulary word for a specific user."""
        try:
            self.cursor.execute('''
            SELECT * FROM user_progress WHERE vocabulary_id = ? AND user_id = ?
            ''', (vocabulary_id, user_id))
            
            return self.cursor.fetchone()
        except sqlite3.Error as e:
            print(f"Error getting word progress: {e}")
            return None
    
    def get_words_for_review(self, user_id, limit=10, min_proficiency=None, max_proficiency=None):
        """Get words for review based on proficiency level for a specific user."""
        try:
            query = '''
            SELECT v.*, p.proficiency_level, p.last_reviewed, p.review_count, p.correct_count
            FROM vocabulary v
            JOIN user_progress p ON v.id = p.vocabulary_id AND v.user_id = p.user_id
            WHERE v.user_id = ?
            '''
            params = [user_id]
            
            if min_proficiency is not None:
                query += " AND p.proficiency_level >= ?"
                params.append(min_proficiency)
                
            if max_proficiency is not None:
                query += " AND p.proficiency_level <= ?"
                params.append(max_proficiency)
                
            # Order by last reviewed (oldest first) and proficiency level (lowest first)
            query += " ORDER BY p.last_reviewed ASC, p.proficiency_level ASC LIMIT ?"
            params.append(limit)
            
            self.cursor.execute(query, params)
            return self.cursor.fetchall()
        except sqlite3.Error as e:
            print(f"Error getting words for review: {e}")
            return []
    
    def start_session(self, user_id, session_type='general'):
        """Start a new learning session for a specific user."""
        try:
            session_id = self.cursor.execute('''
            INSERT INTO sessions (user_id, start_time, words_studied, words_learned, session_type)
            VALUES (?, CURRENT_TIMESTAMP, 0, 0, ?)
            ''', (user_id, session_type)).lastrowid
            
            self.conn.commit()
            return session_id
        except sqlite3.Error as e:
            print(f"Error starting session: {e}")
            return None
    
    def end_session(self, session_id, words_studied, words_learned):
        """End a learning session with statistics."""
        try:
            self.cursor.execute('''
            UPDATE sessions 
            SET end_time = CURRENT_TIMESTAMP, words_studied = ?, words_learned = ?
            WHERE id = ?
            ''', (words_studied, words_learned, session_id))
            
            self.conn.commit()
            return True
        except sqlite3.Error as e:
            print(f"Error ending session: {e}")
            return False
    
    def get_session_stats(self, user_id, days=30):
        """Get statistics from sessions in the last N days for a specific user."""
        try:
            cutoff_date = datetime.datetime.now() - datetime.timedelta(days=days)
            
            self.cursor.execute('''
            SELECT 
                COUNT(*) as total_sessions,
                SUM(words_studied) as total_words_studied,
                SUM(words_learned) as total_words_learned,
                AVG(words_studied) as avg_words_per_session,
                AVG(words_learned) as avg_learned_per_session,
                AVG(CASE 
                    WHEN end_time IS NOT NULL AND start_time IS NOT NULL 
                    THEN (julianday(end_time) - julianday(start_time)) * 24 * 60 
                    ELSE NULL 
                END) as avg_session_minutes
            FROM sessions
            WHERE user_id = ? AND start_time >= ?
            ''', (user_id, cutoff_date))
            
            return self.cursor.fetchone()
        except sqlite3.Error as e:
            print(f"Error getting session stats: {e}")
            return None
    
    def save_camera_translation(self, user_id, image_path, detected_text, translated_text, 
                               source_language, target_language):
        """Save a translation from camera capture for a specific user."""
        try:
            translation_id = self.cursor.execute('''
            INSERT INTO camera_translations
            (user_id, image_path, detected_text, translated_text, source_language, target_language)
            VALUES (?, ?, ?, ?, ?, ?)
            ''', (user_id, image_path, detected_text, translated_text, source_language, target_language)).lastrowid
            
            self.conn.commit()
            return translation_id
        except sqlite3.Error as e:
            print(f"Error saving camera translation: {e}")
            return None
    
    def get_camera_translations(self, user_id, limit=50):
        """Get recent camera translations for a specific user."""
        try:
            self.cursor.execute('''
            SELECT * FROM camera_translations
            WHERE user_id = ?
            ORDER BY date_captured DESC
            LIMIT ?
            ''', (user_id, limit))
            
            return self.cursor.fetchall()
        except sqlite3.Error as e:
            print(f"Error getting camera translations: {e}")
            return []
    
    def add_camera_translation_to_vocabulary(self, translation_id, user_id, category=None):
        """Save a camera translation to vocabulary for a specific user."""
        try:
            # Get the translation
            self.cursor.execute('''
            SELECT * FROM camera_translations WHERE id = ? AND user_id = ?
            ''', (translation_id, user_id))
            
            translation = self.cursor.fetchone()
            if not translation:
                return False
            
            # Add to vocabulary
            vocab_id = self.add_vocabulary(
                user_id,
                translation['detected_text'],
                translation['translated_text'],
                translation['target_language'],
                category,
                translation['image_path'],
                'camera'
            )
            
            # Mark as saved to vocabulary
            if vocab_id:
                self.cursor.execute('''
                UPDATE camera_translations
                SET is_saved_to_vocabulary = 1
                WHERE id = ? AND user_id = ?
                ''', (translation_id, user_id))
                
                self.conn.commit()
                return vocab_id
            
            return None
        except sqlite3.Error as e:
            print(f"Error adding camera translation to vocabulary: {e}")
            return None
    
    def save_achievement(self, user_id, achievement_id, achievement_name, achievement_data=None):
        """Save a user achievement."""
        try:
            self.cursor.execute('''
            INSERT OR IGNORE INTO user_achievements 
            (user_id, achievement_id, achievement_name, achievement_data)
            VALUES (?, ?, ?, ?)
            ''', (user_id, achievement_id, achievement_name, achievement_data))
            
            self.conn.commit()
            return True
        except sqlite3.Error as e:
            print(f"Error saving achievement: {e}")
            return False
    
    def get_user_achievements(self, user_id):
        """Get all achievements for a specific user."""
        try:
            self.cursor.execute('''
            SELECT * FROM user_achievements 
            WHERE user_id = ? 
            ORDER BY achieved_at DESC
            ''', (user_id,))
            
            return self.cursor.fetchall()
        except sqlite3.Error as e:
            print(f"Error getting user achievements: {e}")
            return []
    
    def update_user_statistics(self, user_id, words_learned_today=0, time_spent_minutes=0, 
                              quiz_accuracy=None, streak_days=None, total_points=None, level=None):
        """Update daily statistics for a user."""
        try:
            # Get current stats for today
            today = datetime.date.today()
            self.cursor.execute('''
            SELECT * FROM user_statistics WHERE user_id = ? AND stat_date = ?
            ''', (user_id, today))
            
            existing_stats = self.cursor.fetchone()
            
            if existing_stats:
                # Update existing record
                update_fields = []
                params = []
                
                if words_learned_today > 0:
                    update_fields.append("words_learned_today = words_learned_today + ?")
                    params.append(words_learned_today)
                
                if time_spent_minutes > 0:
                    update_fields.append("time_spent_minutes = time_spent_minutes + ?")
                    params.append(time_spent_minutes)
                
                if quiz_accuracy is not None:
                    update_fields.append("quiz_accuracy = ?")
                    params.append(quiz_accuracy)
                
                if streak_days is not None:
                    update_fields.append("streak_days = ?")
                    params.append(streak_days)
                
                if total_points is not None:
                    update_fields.append("total_points = ?")
                    params.append(total_points)
                
                if level is not None:
                    update_fields.append("level = ?")
                    params.append(level)
                
                if update_fields:
                    query = f"UPDATE user_statistics SET {', '.join(update_fields)} WHERE user_id = ? AND stat_date = ?"
                    params.extend([user_id, today])
                    self.cursor.execute(query, params)
            else:
                # Create new record
                self.cursor.execute('''
                INSERT INTO user_statistics 
                (user_id, stat_date, words_learned_today, time_spent_minutes, 
                 quiz_accuracy, streak_days, total_points, level)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (user_id, today, words_learned_today or 0, time_spent_minutes or 0,
                      quiz_accuracy or 0.0, streak_days or 0, total_points or 0, level or 1))
            
            self.conn.commit()
            return True
        except sqlite3.Error as e:
            print(f"Error updating user statistics: {e}")
            return False
    
    def get_user_statistics(self, user_id, days=30):
        """Get user statistics for the last N days."""
        try:
            cutoff_date = datetime.date.today() - datetime.timedelta(days=days)
            
            self.cursor.execute('''
            SELECT * FROM user_statistics 
            WHERE user_id = ? AND stat_date >= ?
            ORDER BY stat_date DESC
            ''', (user_id, cutoff_date))
            
            return self.cursor.fetchall()
        except sqlite3.Error as e:
            print(f"Error getting user statistics: {e}")
            return []
    
    def save_quiz_result(self, user_id, session_id, quiz_type, total_questions, 
                        correct_answers, completion_time=None):
        """Save quiz results for a user."""
        try:
            result_id = self.cursor.execute('''
            INSERT INTO quiz_results 
            (user_id, session_id, quiz_type, total_questions, correct_answers, completion_time)
            VALUES (?, ?, ?, ?, ?, ?)
            ''', (user_id, session_id, quiz_type, total_questions, correct_answers, completion_time)).lastrowid
            
            self.conn.commit()
            return result_id
        except sqlite3.Error as e:
            print(f"Error saving quiz result: {e}")
            return None
    
    def get_quiz_results(self, user_id, limit=50):
        """Get recent quiz results for a user."""
        try:
            self.cursor.execute('''
            SELECT * FROM quiz_results 
            WHERE user_id = ?
            ORDER BY date_completed DESC
            LIMIT ?
            ''', (user_id, limit))
            
            return self.cursor.fetchall()
        except sqlite3.Error as e:
            print(f"Error getting quiz results: {e}")
            return []
    
    def cleanup_expired_sessions(self):
        """Clean up old session data if needed."""
        try:
            # Delete sessions older than 90 days
            cutoff_date = datetime.datetime.now() - datetime.timedelta(days=90)
            self.cursor.execute('''
            DELETE FROM sessions 
            WHERE start_time < ?
            ''', (cutoff_date,))
            
            deleted = self.cursor.rowcount
            self.conn.commit()
            
            if deleted > 0:
                print(f"Cleaned up {deleted} old sessions")
        except Exception as e:
            print(f"Error cleaning up sessions: {e}")
    
    def get_database_stats(self):
        """Get overall database statistics."""
        try:
            stats = {}
            
            # Count users
            self.cursor.execute("SELECT COUNT(*) as count FROM users")
            stats['total_users'] = self.cursor.fetchone()['count']
            
            # Count vocabulary words
            self.cursor.execute("SELECT COUNT(*) as count FROM vocabulary")
            stats['total_vocabulary'] = self.cursor.fetchone()['count']
            
            # Count sessions
            self.cursor.execute("SELECT COUNT(*) as count FROM sessions")
            stats['total_sessions'] = self.cursor.fetchone()['count']
            
            # Count achievements
            self.cursor.execute("SELECT COUNT(*) as count FROM user_achievements")
            stats['total_achievements'] = self.cursor.fetchone()['count']
            
            return stats
        except sqlite3.Error as e:
            print(f"Error getting database stats: {e}")
            return {}
    
    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()

# Example usage and testing
if __name__ == "__main__":
    # Create a test database
    db = LanguageLearningDB("test_language_learning.db")
    
    # Test user creation
    test_user_id = "test_user_123"
    test_email = "test@example.com"
    
    print("Testing user creation...")
    db.create_or_update_user(test_user_id, test_email, "Test User")
    
    # Test vocabulary addition
    print("Testing vocabulary addition...")
    vocab_id = db.add_vocabulary(test_user_id, "apple", "manzana", "es", "food")
    print(f"Added vocabulary with ID: {vocab_id}")
    
    # Test progress update
    print("Testing progress update...")
    db.update_word_progress(vocab_id, True)
    
    # Test getting vocabulary
    print("Testing vocabulary retrieval...")
    vocab_list = db.get_all_vocabulary(test_user_id)
    for word in vocab_list:
        print(f"Word: {word['word_original']} -> {word['word_translated']} (Proficiency: {word['proficiency_level']})")
    
    # Test session management
    print("Testing session management...")
    session_id = db.start_session(test_user_id)
    print(f"Started session with ID: {session_id}")
    
    db.end_session(session_id, 1, 1)
    print("Ended session")
    
    # Get session stats
    print("Testing session statistics...")
    stats = db.get_session_stats(test_user_id)
    print(f"Session stats: {dict(stats) if stats else 'No stats'}")
    
    # Get database stats
    print("Testing database statistics...")
    db_stats = db.get_database_stats()
    print(f"Database stats: {db_stats}")
    
    # Close the connection
    db.close()
    print("Database connection closed")