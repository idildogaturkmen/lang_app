import sqlite3
import os
import datetime

class LanguageLearningDB:
    def __init__(self, db_path):
        """Initialize the database connection."""
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()
        
        # Create tables if they don't exist
        self._create_tables()
    
    def _create_tables(self):
        """Create necessary database tables if they don't exist."""
        self.cursor.executescript('''
        CREATE TABLE IF NOT EXISTS vocabulary (
            id INTEGER PRIMARY KEY,
            word_original TEXT NOT NULL,
            word_translated TEXT NOT NULL,
            language_translated TEXT NOT NULL,
            category TEXT,
            image_path TEXT,
            date_added TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            source TEXT DEFAULT 'manual'
        );
        
        CREATE TABLE IF NOT EXISTS user_progress (
            id INTEGER PRIMARY KEY,
            vocabulary_id INTEGER,
            review_count INTEGER DEFAULT 0,
            correct_count INTEGER DEFAULT 0,
            last_reviewed TIMESTAMP,
            proficiency_level INTEGER DEFAULT 0,
            FOREIGN KEY (vocabulary_id) REFERENCES vocabulary (id)
        );
        
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY,
            start_time TIMESTAMP,
            end_time TIMESTAMP,
            words_studied INTEGER DEFAULT 0,
            words_learned INTEGER DEFAULT 0
        );
        
        CREATE TABLE IF NOT EXISTS camera_translations (
            id INTEGER PRIMARY KEY,
            image_path TEXT,
            detected_text TEXT,
            translated_text TEXT,
            source_language TEXT,
            target_language TEXT,
            date_captured TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_saved_to_vocabulary BOOLEAN DEFAULT 0
        );
        ''')
        self.conn.commit()
    
    def add_vocabulary(self, word_original, word_translated, language_translated, category=None, image_path=None, source='manual'):
        """Add a new vocabulary entry to the database."""
        try:
            vocab_id = self.cursor.execute('''
            INSERT INTO vocabulary 
            (word_original, word_translated, language_translated, category, image_path, source)
            VALUES (?, ?, ?, ?, ?, ?)
            ''', (word_original, word_translated, language_translated, category, image_path, source)).lastrowid
            
            # Initialize user progress for this vocabulary
            self.cursor.execute('''
            INSERT INTO user_progress (vocabulary_id, last_reviewed)
            VALUES (?, ?)
            ''', (vocab_id, datetime.datetime.now()))
            
            self.conn.commit()
            return vocab_id
        except sqlite3.Error as e:
            print(f"Error adding vocabulary: {e}")
            return None
    
    def get_vocabulary(self, vocabulary_id):
        """Get a specific vocabulary entry by ID."""
        try:
            self.cursor.execute('''
            SELECT * FROM vocabulary WHERE id = ?
            ''', (vocabulary_id,))
            
            return self.cursor.fetchone()
        except sqlite3.Error as e:
            print(f"Error getting vocabulary: {e}")
            return None
    
    def get_all_vocabulary(self, category=None, language=None):
        """Get all vocabulary entries, optionally filtered by category and/or language."""
        try:
            query = "SELECT * FROM vocabulary"
            params = []
            
            if category and language:
                query += " WHERE category = ? AND language_translated = ?"
                params = [category, language]
            elif category:
                query += " WHERE category = ?"
                params = [category]
            elif language:
                query += " WHERE language_translated = ?"
                params = [language]
                
            query += " ORDER BY date_added DESC"
            
            self.cursor.execute(query, params)
            return self.cursor.fetchall()
        except sqlite3.Error as e:
            print(f"Error getting vocabulary: {e}")
            return []
    
    def update_vocabulary(self, vocabulary_id, word_original=None, word_translated=None, 
                          language_translated=None, category=None, image_path=None):
        """Update an existing vocabulary entry."""
        try:
            # Get current values first
            current = self.get_vocabulary(vocabulary_id)
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
            WHERE id = ?
            ''', (word_original, word_translated, language_translated, 
                  category, image_path, vocabulary_id))
            
            self.conn.commit()
            return True
        except sqlite3.Error as e:
            print(f"Error updating vocabulary: {e}")
            return False
    
    def delete_vocabulary(self, vocabulary_id):
        """Delete a vocabulary entry by ID and associated progress."""
        try:
            # Delete associated progress first (due to foreign key constraint)
            self.cursor.execute('''
            DELETE FROM user_progress WHERE vocabulary_id = ?
            ''', (vocabulary_id,))
            
            # Delete vocabulary entry
            self.cursor.execute('''
            DELETE FROM vocabulary WHERE id = ?
            ''', (vocabulary_id,))
            
            self.conn.commit()
            return True
        except sqlite3.Error as e:
            print(f"Error deleting vocabulary: {e}")
            return False
    
    def search_vocabulary(self, search_term, language=None):
        """Search vocabulary entries for a term in original or translated word."""
        try:
            query = '''
            SELECT * FROM vocabulary 
            WHERE (word_original LIKE ? OR word_translated LIKE ?)
            '''
            params = [f'%{search_term}%', f'%{search_term}%']
            
            if language:
                query += " AND language_translated = ?"
                params.append(language)
                
            self.cursor.execute(query, params)
            return self.cursor.fetchall()
        except sqlite3.Error as e:
            print(f"Error searching vocabulary: {e}")
            return []
    
    def update_word_progress(self, vocabulary_id, is_correct):
        """Update the progress for a vocabulary word after review."""
        try:
            # Get current progress
            self.cursor.execute('''
            SELECT * FROM user_progress WHERE vocabulary_id = ?
            ''', (vocabulary_id,))
            
            progress = self.cursor.fetchone()
            if not progress:
                # Create new progress entry if it doesn't exist
                self.cursor.execute('''
                INSERT INTO user_progress 
                (vocabulary_id, review_count, correct_count, last_reviewed, proficiency_level)
                VALUES (?, ?, ?, ?, ?)
                ''', (vocabulary_id, 1, 1 if is_correct else 0, datetime.datetime.now(), 0))
            else:
                # Update existing progress
                review_count = progress['review_count'] + 1
                correct_count = progress['correct_count'] + (1 if is_correct else 0)
                
                # Calculate new proficiency level (0-5)
                # Simple algorithm: proficiency is percentage of correct answers, mapped to 0-5 scale
                proficiency = min(5, int((correct_count / review_count) * 5))
                
                self.cursor.execute('''
                UPDATE user_progress 
                SET review_count = ?, correct_count = ?, 
                    last_reviewed = ?, proficiency_level = ?
                WHERE vocabulary_id = ?
                ''', (review_count, correct_count, datetime.datetime.now(), 
                      proficiency, vocabulary_id))
            
            self.conn.commit()
            return True
        except sqlite3.Error as e:
            print(f"Error updating word progress: {e}")
            return False
    
    def get_word_progress(self, vocabulary_id):
        """Get the progress for a specific vocabulary word."""
        try:
            self.cursor.execute('''
            SELECT * FROM user_progress WHERE vocabulary_id = ?
            ''', (vocabulary_id,))
            
            return self.cursor.fetchone()
        except sqlite3.Error as e:
            print(f"Error getting word progress: {e}")
            return None
    
    def get_words_for_review(self, limit=10, min_proficiency=None, max_proficiency=None):
        """Get words for review based on proficiency level."""
        try:
            query = '''
            SELECT v.*, p.proficiency_level, p.last_reviewed, p.review_count, p.correct_count
            FROM vocabulary v
            JOIN user_progress p ON v.id = p.vocabulary_id
            WHERE 1=1
            '''
            params = []
            
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
    
    def start_session(self):
        """Start a new learning session."""
        try:
            session_id = self.cursor.execute('''
            INSERT INTO sessions (start_time, words_studied, words_learned)
            VALUES (?, 0, 0)
            ''', (datetime.datetime.now(),)).lastrowid
            
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
            SET end_time = ?, words_studied = ?, words_learned = ?
            WHERE id = ?
            ''', (datetime.datetime.now(), words_studied, words_learned, session_id))
            
            self.conn.commit()
            return True
        except sqlite3.Error as e:
            print(f"Error ending session: {e}")
            return False
    
    def get_session_stats(self, days=30):
        """Get statistics from sessions in the last N days."""
        try:
            cutoff_date = datetime.datetime.now() - datetime.timedelta(days=days)
            
            self.cursor.execute('''
            SELECT 
                COUNT(*) as total_sessions,
                SUM(words_studied) as total_words_studied,
                SUM(words_learned) as total_words_learned,
                AVG(words_studied) as avg_words_per_session,
                AVG(words_learned) as avg_learned_per_session,
                SUM(CAST(strftime('%s', end_time) - strftime('%s', start_time) AS REAL)) / 60 as total_minutes
            FROM sessions
            WHERE start_time >= ?
            ''', (cutoff_date,))
            
            return self.cursor.fetchone()
        except sqlite3.Error as e:
            print(f"Error getting session stats: {e}")
            return None
    
    def save_camera_translation(self, image_path, detected_text, translated_text, 
                               source_language, target_language):
        """Save a translation from camera capture."""
        try:
            translation_id = self.cursor.execute('''
            INSERT INTO camera_translations
            (image_path, detected_text, translated_text, source_language, target_language)
            VALUES (?, ?, ?, ?, ?)
            ''', (image_path, detected_text, translated_text, source_language, target_language)).lastrowid
            
            self.conn.commit()
            return translation_id
        except sqlite3.Error as e:
            print(f"Error saving camera translation: {e}")
            return None
    
    def get_camera_translations(self, limit=50):
        """Get recent camera translations."""
        try:
            self.cursor.execute('''
            SELECT * FROM camera_translations
            ORDER BY date_captured DESC
            LIMIT ?
            ''', (limit,))
            
            return self.cursor.fetchall()
        except sqlite3.Error as e:
            print(f"Error getting camera translations: {e}")
            return []
    
    def add_camera_translation_to_vocabulary(self, translation_id, category=None):
        """Save a camera translation to vocabulary."""
        try:
            # Get the translation
            self.cursor.execute('''
            SELECT * FROM camera_translations WHERE id = ?
            ''', (translation_id,))
            
            translation = self.cursor.fetchone()
            if not translation:
                return False
            
            # Add to vocabulary
            vocab_id = self.add_vocabulary(
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
                WHERE id = ?
                ''', (translation_id,))
                
                self.conn.commit()
                return vocab_id
            
            return None
        except sqlite3.Error as e:
            print(f"Error adding camera translation to vocabulary: {e}")
            return None
    
    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()

# Example usage
if __name__ == "__main__":
    # Create a test database in memory
    db = LanguageLearningDB(":memory:")
    
    # Add some test vocabulary
    db.add_vocabulary("apple", "manzana", "es", "food")
    db.add_vocabulary("book", "libro", "es", "objects")
    
    # Update progress
    db.update_word_progress(1, True)  # Correct answer for word ID 1
    
    # Save a camera translation
    translation_id = db.save_camera_translation(
        "/path/to/image.jpg",
        "hello world",
        "hola mundo",
        "en",
        "es"
    )
    
    # Add translation to vocabulary
    db.add_camera_translation_to_vocabulary(translation_id, "phrases")
    
    # Get all vocabulary
    vocab = db.get_all_vocabulary()
    for word in vocab:
        print(f"{word['word_original']} -> {word['word_translated']} ({word['language_translated']})")
    
    # Close the connection
    db.close()