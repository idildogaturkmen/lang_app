import streamlit as st
import json
import uuid
from datetime import datetime, date
from main import SupabaseDB

class SupabaseGamificationManager:
    def __init__(self):
        self.db = SupabaseDB()
    
    def get_user_id(self):
        """Get properly formatted user ID."""
        user = st.session_state.get('user')
        if not user:
            return None
        
        user_id = user.get('id')
        if not user_id:
            return None
        
        # Ensure it's a proper UUID string
        if isinstance(user_id, str):
            try:
                # Validate it's a proper UUID
                uuid.UUID(user_id)
                return user_id
            except ValueError:
                print(f"❌ Invalid UUID format: {user_id}")
                return None
        
        return str(user_id)
    
    def save_streak_data(self):
        """Save streak data to Supabase user_streaks table."""
        try:
            user_id = self.get_user_id()
            if not user_id:
                print("❌ No valid user ID found")
                return False
            
            streak_data = {
                'user_id': user_id,
                'streak_days': st.session_state.get('streak_days', 0),
                'last_active_date': str(st.session_state.get('last_active_date', date.today())),
                'streak_savers': st.session_state.get('streak_savers', 0),
                'updated_at': datetime.now().isoformat()
            }
            
            # Use upsert with proper conflict resolution
            response = self.db.supabase.table('user_streaks').upsert(
                streak_data,
                on_conflict='user_id'
            ).execute()
            
            if response.data:
                print(f"✅ Streak data saved to Supabase")
                return True
            else:
                print(f"❌ Failed to save streak data: {response}")
                return False
                
        except Exception as e:
            print(f"❌ Error saving streak data: {e}")
            return False
    
    def load_streak_data(self):
        """Load streak data from Supabase."""
        try:
            user_id = self.get_user_id()
            if not user_id:
                print("❌ No valid user ID found")
                return False
            
            # Use proper UUID comparison
            response = self.db.supabase.table('user_streaks').select('*').eq('user_id', user_id).execute()
            
            if response.data and len(response.data) > 0:
                streak_data = response.data[0]
                st.session_state.streak_days = streak_data.get('streak_days', 0)
                st.session_state.last_active_date = streak_data.get('last_active_date')
                st.session_state.streak_savers = streak_data.get('streak_savers', 0)
                print(f"✅ Streak data loaded from Supabase: {streak_data}")
                return True
            else:
                # Initialize new streak data
                st.session_state.streak_days = 0
                st.session_state.last_active_date = None
                st.session_state.streak_savers = 0
                print(f"🆕 No existing streak data, initialized defaults")
                return False
                
        except Exception as e:
            print(f"❌ Error loading streak data: {e}")
            return False