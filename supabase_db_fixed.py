import streamlit as st
import requests
import json
from datetime import datetime, date
import uuid

class SupabaseDB:
    def __init__(self):
        self.supabase_url = "https://csszlzpsfwmsezursivk.supabase.co"
        self.supabase_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImNzc3psenBzZndtc2V6dXJzaXZrIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTA1Mjg1MjEsImV4cCI6MjA2NjEwNDUyMX0.gIi0Q_pifYpXeM1r8kWlgTO1LD8bc91lQ3suH8OWDKI"

        try:
            from supabase import create_client
            self.supabase = create_client(self.supabase_url, self.supabase_key)
            print("✅ Supabase client initialized successfully")
        except ImportError:
            print("⚠️ Supabase client not available, using requests only")
            self.supabase = None

    def get_user_id(self):
        """Get current user ID from authentication."""
        try:
            user = self.get_authenticated_user()
            if user:
                return str(user.get('id'))
            return None
        except Exception as e:
            print(f"❌ Error getting user ID: {e}")
            return None
    
    def get_authenticated_user(self):
        """Get authenticated user from session state."""
        try:
            # Check session state for user
            if hasattr(st.session_state, 'user') and st.session_state.user:
                return st.session_state.user
            
            # Check for authentication parameters
            if hasattr(st.session_state, 'auth_token') and st.session_state.auth_token:
                return {
                    'id': st.session_state.get('user_id'),
                    'email': st.session_state.get('user_email'),
                    'auth_token': st.session_state.auth_token
                }
            
            return None
        except Exception as e:
            print(f"❌ Error getting authenticated user: {e}")
            return None
    
    def get_headers(self):
        """Get headers with proper authentication."""
        user = self.get_authenticated_user()
        auth_token = user.get('auth_token') if user else None
        
        headers = {
            'apikey': self.supabase_key,
            'Content-Type': 'application/json',
            'Prefer': 'return=representation'
        }
        
        if auth_token:
            headers['Authorization'] = f'Bearer {auth_token}'
        else:
            headers['Authorization'] = f'Bearer {self.supabase_key}'
        
        return headers

    def check_word_exists(self, word_original, word_translated, language_translated):
        """Check if a word already exists in the user's vocabulary."""
        user_id = self.get_user_id()
        if not user_id:
            return False
            
        try:
            if self.supabase:
                # Use Supabase client
                response = self.supabase.table('vocabulary').select('id').eq('user_id', user_id).eq('word_original', word_original).eq('word_translated', word_translated).eq('language_translated', language_translated).execute()
                return len(response.data) > 0
            else:
                # Use requests
                headers = self.get_headers()
                url = f'{self.supabase_url}/rest/v1/vocabulary?user_id=eq.{user_id}&word_original=eq.{word_original}&word_translated=eq.{word_translated}&language_translated=eq.{language_translated}&select=id'
                
                response = requests.get(url, headers=headers, timeout=30)
                
                if response.status_code == 200:
                    result = response.json()
                    return len(result) > 0
                return False
                
        except Exception as e:
            print(f"❌ Error checking for existing word: {e}")
            return False

    def add_vocabulary(self, word_original, word_translated, language_translated, category=None, image_path=None, source='manual'):
        """Add vocabulary to Supabase with duplicate prevention and user progress tracking."""
        user_id = self.get_user_id()
        if not user_id:
            print("❌ No user ID available for vocabulary save")
            return None
            
        try:
            # Check for duplicates first
            if self.check_word_exists(word_original, word_translated, language_translated):
                print(f"⚠️ Duplicate detected: {word_original} → {word_translated}")
                return 'duplicate'
            
            # Prepare vocabulary data
            vocab_data = {
                'user_id': user_id,
                'word_original': word_original,
                'word_translated': word_translated,
                'language_translated': language_translated,
                'category': category,
                'image_path': image_path,
                'source': source,
                'date_added': datetime.now().isoformat(),
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat()
            }
            
            if self.supabase:
                # Use Supabase client
                print(f"🔄 Saving vocabulary using Supabase client...")
                response = self.supabase.table('vocabulary').insert(vocab_data).execute()
                
                if response.data and len(response.data) > 0:
                    vocab_id = response.data[0]['id']
                    print(f"✅ Vocabulary saved with ID: {vocab_id}")
                    
                    # Create user progress entry
                    self.create_user_progress(vocab_id)
                    
                    return vocab_id
                else:
                    print(f"❌ Failed to save vocabulary: {response}")
                    return None
            else:
                # Use requests
                print(f"🔄 Saving vocabulary using requests...")
                headers = self.get_headers()
                
                response = requests.post(
                    f'{self.supabase_url}/rest/v1/vocabulary',
                    headers=headers,
                    json=vocab_data,
                    timeout=30
                )
                
                if response.status_code in [200, 201]:
                    result = response.json()
                    if result and len(result) > 0:
                        vocab_id = result[0]['id']
                        print(f"✅ Vocabulary saved with ID: {vocab_id}")
                        
                        # Create user progress entry
                        self.create_user_progress(vocab_id)
                        
                        return vocab_id
                else:
                    print(f"❌ Failed to save vocabulary: {response.status_code} - {response.text}")
                    return None
                
        except Exception as e:
            print(f"❌ Error adding vocabulary: {e}")
            import traceback
            print(f"📋 Full traceback: {traceback.format_exc()}")
            return None

    def create_user_progress(self, vocabulary_id):
        """Create a user progress entry for new vocabulary."""
        user_id = self.get_user_id()
        if not user_id or not vocabulary_id:
            return False
            
        try:
            progress_data = {
                'user_id': user_id,
                'vocabulary_id': vocabulary_id,
                'review_count': 0,
                'correct_count': 0,
                'proficiency_level': 0,
                'last_reviewed': None,
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat()
            }
            
            if self.supabase:
                response = self.supabase.table('user_progress').insert(progress_data).execute()
                if response.data:
                    print(f"✅ User progress created for vocabulary {vocabulary_id}")
                    return True
            else:
                headers = self.get_headers()
                response = requests.post(
                    f'{self.supabase_url}/rest/v1/user_progress',
                    headers=headers,
                    json=progress_data,
                    timeout=30
                )
                
                if response.status_code in [200, 201]:
                    print(f"✅ User progress created for vocabulary {vocabulary_id}")
                    return True
            
            return False
                
        except Exception as e:
            print(f"❌ Error creating user progress: {e}")
            return False

    def get_all_vocabulary(self):
        """Get all vocabulary for the authenticated user with progress data."""
        user_id = self.get_user_id()
        if not user_id:
            return []
            
        try:
            if self.supabase:
                # Use Supabase client with join
                response = self.supabase.table('vocabulary').select(
                    '*, user_progress(*)'
                ).eq('user_id', user_id).order('date_added', desc=True).execute()
                
                if response.data:
                    print(f"✅ Retrieved {len(response.data)} vocabulary items using Supabase client")
                    return response.data
            else:
                # Use requests
                headers = self.get_headers()
                url = f'{self.supabase_url}/rest/v1/vocabulary?user_id=eq.{user_id}&select=*,user_progress(*)&order=date_added.desc'
                
                response = requests.get(url, headers=headers, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    print(f"✅ Retrieved {len(data)} vocabulary items using requests")
                    return data
            
            return []
                
        except Exception as e:
            print(f"❌ Error getting vocabulary: {e}")
            return []

    def start_session(self):
        """Start a new learning session."""
        user_id = self.get_user_id()
        if not user_id:
            return None
            
        try:
            session_data = {
                'user_id': user_id,
                'start_time': datetime.now().isoformat(),
                'words_studied': 0,
                'words_learned': 0
            }
            
            if self.supabase:
                response = self.supabase.table('sessions').insert(session_data).execute()
                if response.data and len(response.data) > 0:
                    session_id = response.data[0]['id']
                    print(f"✅ Session started with ID: {session_id}")
                    return session_id
            else:
                headers = self.get_headers()
                
                response = requests.post(
                    f'{self.supabase_url}/rest/v1/sessions',
                    headers=headers,
                    json=session_data,
                    timeout=30
                )
                
                if response.status_code in [200, 201]:
                    result = response.json()
                    if result and len(result) > 0:
                        session_id = result[0]['id']
                        print(f"✅ Session started with ID: {session_id}")
                        return session_id
            
            return None
                
        except Exception as e:
            print(f"❌ Error starting session: {e}")
            return None

    def end_session(self, session_id, words_studied, words_learned):
        """End a learning session."""
        if not session_id:
            return False
            
        try:
            update_data = {
                'end_time': datetime.now().isoformat(),
                'words_studied': words_studied,
                'words_learned': words_learned
            }
            
            if self.supabase:
                response = self.supabase.table('sessions').update(update_data).eq('id', session_id).execute()
                if response.data:
                    print(f"✅ Session {session_id} ended successfully")
                    return True
            else:
                headers = self.get_headers()
                
                response = requests.patch(
                    f'{self.supabase_url}/rest/v1/sessions?id=eq.{session_id}',
                    headers=headers,
                    json=update_data,
                    timeout=30
                )
                
                if response.status_code in [200, 204]:
                    print(f"✅ Session {session_id} ended successfully")
                    return True
            
            return False
                
        except Exception as e:
            print(f"❌ Error ending session: {e}")
            return False

    def get_user_streak_data(self):
        """Get user's streak data from Supabase."""
        user_id = self.get_user_id()
        if not user_id:
            return None
            
        try:
            if self.supabase:
                response = self.supabase.table('user_streaks').select('*').eq('user_id', user_id).execute()
                if response.data and len(response.data) > 0:
                    return response.data[0]
            else:
                headers = self.get_headers()
                url = f'{self.supabase_url}/rest/v1/user_streaks?user_id=eq.{user_id}'
                
                response = requests.get(url, headers=headers, timeout=30)
                
                if response.status_code == 200:
                    result = response.json()
                    if result and len(result) > 0:
                        return result[0]
            
            return None
                
        except Exception as e:
            print(f"❌ Error getting streak data: {e}")
            return None

    def update_user_streak(self, streak_days, last_active_date, streak_savers=0):
        """Update user's streak data."""
        user_id = self.get_user_id()
        if not user_id:
            return False
            
        try:
            streak_data = {
                'user_id': user_id,
                'streak_days': streak_days,
                'last_active_date': str(last_active_date),
                'streak_savers': streak_savers,
                'updated_at': datetime.now().isoformat()
            }
            
            if self.supabase:
                response = self.supabase.table('user_streaks').upsert(
                    streak_data,
                    on_conflict='user_id'
                ).execute()
                
                if response.data:
                    print(f"✅ Streak data updated: {streak_days} days")
                    return True
            else:
                headers = self.get_headers()
                
                response = requests.post(
                    f'{self.supabase_url}/rest/v1/user_streaks',
                    headers=headers,
                    json=streak_data,
                    timeout=30
                )
                
                if response.status_code in [200, 201]:
                    print(f"✅ Streak data updated: {streak_days} days")
                    return True
            
            return False
                
        except Exception as e:
            print(f"❌ Error updating streak: {e}")
            return False

    def update_user_progress(self, vocabulary_id, correct_answer):
        """Update user progress for a specific vocabulary word."""
        user_id = self.get_user_id()
        if not user_id:
            return False
            
        try:
            # First, get current progress
            if self.supabase:
                current_response = self.supabase.table('user_progress').select('*').eq('user_id', user_id).eq('vocabulary_id', vocabulary_id).execute()
                
                if current_response.data and len(current_response.data) > 0:
                    current = current_response.data[0]
                    new_review_count = current['review_count'] + 1
                    new_correct_count = current['correct_count'] + (1 if correct_answer else 0)
                    new_proficiency = min(5, current['proficiency_level'] + (1 if correct_answer else -1))
                    new_proficiency = max(0, new_proficiency)
                    
                    update_data = {
                        'review_count': new_review_count,
                        'correct_count': new_correct_count,
                        'proficiency_level': new_proficiency,
                        'last_reviewed': datetime.now().isoformat(),
                        'updated_at': datetime.now().isoformat()
                    }
                    
                    response = self.supabase.table('user_progress').update(update_data).eq('user_id', user_id).eq('vocabulary_id', vocabulary_id).execute()
                    
                    if response.data:
                        print(f"✅ Progress updated for vocabulary {vocabulary_id}")
                        return True
            
            return False
                
        except Exception as e:
            print(f"❌ Error updating progress: {e}")
            return False