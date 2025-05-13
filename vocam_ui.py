import streamlit as st
import base64

# Define color palette
COLORS = {
    "primary_dark": "#074173",
    "primary_medium": "#1679AB",
    "accent_light": "#5DEBD7",
    "accent_lighter": "#C5FF95"
}

def apply_custom_css():
    """Apply custom CSS for Vocam UI enhancements."""
    st.markdown("""
    <style>
        /* Main color palette */
        :root {
            --primary-dark: #074173;
            --primary-medium: #1679AB;
            --accent-light: #5DEBD7;
            --accent-lighter: #C5FF95;
            --text-light: #FFFFFF;
            --text-dark: #333333;
            --bg-light: #F8F9FA;
            --bg-medium: #EFF3F6;
            --card-bg: #FFFFFF;
            --shadow: rgba(0, 0, 0, 0.1);
        }
        
        /* Base app styling */
        .main .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
        }
        
        /* Streamlit elements styling */
        div.stButton > button {
            background-color: var(--primary-medium);
            color: white;
            border-radius: 8px;
            border: none;
            padding: 0.5rem 1rem;
            font-weight: 500;
            box-shadow: 0 2px 4px var(--shadow);
            transition: all 0.3s;
        }
        
        div.stButton > button:hover {
            background-color: var(--primary-dark);
            transform: translateY(-2px);
            box-shadow: 0 4px 8px var(--shadow);
        }
        
        div.stButton > button[data-baseweb="button"][kind="primary"] {
            background-color: var(--accent-light);
            color: var(--primary-dark);
            font-weight: bold;
        }
        
        div.stButton > button[data-baseweb="button"][kind="primary"]:hover {
            background-color: var(--accent-lighter);
        }
        
        /* Sidebar styling */
        section[data-testid="stSidebar"] {
            background-color: var(--primary-dark);
        }
        section[data-testid="stSidebar"] > div {
            background-color: var(--primary-dark);
        }
        .sidebar-content {
            color: var(--text-light);
        }
        
        /* Expander styling */
        .streamlit-expanderHeader {
            background-color: var(--bg-medium);
            border-radius: 5px;
        }
        .streamlit-expanderHeader:hover {
            background-color: var(--bg-light);
        }
        
        /* Card containers */
        .vocam-card {
            background-color: var(--card-bg);
            border-radius: 10px;
            padding: 1.5rem;
            box-shadow: 0 4px 6px var(--shadow);
            margin-bottom: 1rem;
            border-left: 4px solid var(--primary-medium);
        }
        
        .vocam-card-accent {
            border-left: 4px solid var(--accent-light);
        }
        
        /* Progress indicators */
        .stProgress > div > div {
            background-color: var(--accent-light);
        }
        
        /* Status messages */
        .success-box {
            background-color: #DFF2BF;
            color: #4F8A10;
            padding: 0.75rem;
            border-radius: 8px;
            margin: 1rem 0;
            border-left: 4px solid #4F8A10;
        }
        
        .info-box {
            background-color: #BDE5F8;
            color: #00529B;
            padding: 0.75rem;
            border-radius: 8px;
            margin: 1rem 0;
            border-left: 4px solid #00529B;
        }
        
        .warning-box {
            background-color: #FEEFB3;
            color: #9F6000;
            padding: 0.75rem;
            border-radius: 8px;
            margin: 1rem 0;
            border-left: 4px solid #9F6000;
        }
        
        .error-box {
            background-color: #FFBABA;
            color: #D8000C;
            padding: 0.75rem;
            border-radius: 8px;
            margin: 1rem 0;
            border-left: 4px solid #D8000C;
        }
        
        /* Loading indicator */
        .loading-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 2rem;
            background-color: rgba(255, 255, 255, 0.9);
            border-radius: 10px;
            box-shadow: 0 4px 6px var(--shadow);
            margin: 1rem 0;
            text-align: center;
        }
        
        .loading-spinner {
            border: 6px solid #f3f3f3;
            border-top: 6px solid var(--primary-medium);
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 2s linear infinite;
            margin-bottom: 1rem;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        /* Mobile optimizations */
        @media (max-width: 768px) {
            .main .block-container {
                padding: 0.5rem;
            }
            
            .vocam-card {
                padding: 1rem;
                margin-bottom: 0.75rem;
            }
            
            div.stButton > button {
                width: 100%;
                padding: 0.75rem;
                margin-bottom: 0.5rem;
            }
            
            .result-container {
                margin-top: 2rem;
                padding-top: 1rem;
                border-top: 2px dashed var(--primary-medium);
            }
            
            /* Ensure scroll indicators are visible */
            .scroll-indicator {
                display: flex;
                justify-content: center;
                margin: 0.5rem 0;
                color: var(--primary-medium);
                font-size: 1.5rem;
            }
        }
        
        /* Custom components */
        .word-card {
            background-color: white;
            border-radius: 8px;
            padding: 1rem;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            margin-bottom: 1rem;
            border-left: 4px solid var(--primary-medium);
            transition: transform 0.2s;
        }
        
        .word-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        }
        
        /* Navigation tabs */
        .nav-tab {
            background-color: var(--bg-medium);
            padding: 0.5rem 1rem;
            border-radius: 5px 5px 0 0;
            font-weight: 500;
            cursor: pointer;
        }
        
        .nav-tab-active {
            background-color: var(--primary-medium);
            color: white;
        }
        
        /* Help tooltips */
        .help-tooltip {
            display: inline-block;
            background-color: var(--primary-dark);
            color: white;
            width: 18px;
            height: 18px;
            text-align: center;
            border-radius: 50%;
            font-size: 12px;
            margin-left: 5px;
            cursor: help;
        }
        
        /* Quiz styling */
        .quiz-question {
            background-color: var(--bg-light);
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            border-left: 4px solid var(--primary-medium);
        }
        
        .quiz-option {
            background-color: white;
            border: 2px solid var(--primary-medium);
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 10px;
            cursor: pointer;
            transition: all 0.2s;
        }
        
        .quiz-option:hover {
            background-color: var(--bg-light);
            transform: translateY(-2px);
        }
        
        .quiz-option-correct {
            border-color: #4CAF50;
            background-color: rgba(76, 175, 80, 0.1);
        }
        
        .quiz-option-incorrect {
            border-color: #F44336;
            background-color: rgba(244, 67, 54, 0.1);
        }
        
        /* Statistics styling */
        .stat-card {
            background-color: white;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            text-align: center;
            border-top: 4px solid var(--primary-medium);
        }
        
        .stat-value {
            font-size: 2rem;
            font-weight: bold;
            color: var(--primary-dark);
        }
        
        .stat-label {
            color: var(--primary-medium);
            font-size: 0.9rem;
        }
        
        /* Profile section styling */
        .profile-header {
            background-color: var(--primary-dark);
            color: white;
            border-radius: 10px 10px 0 0;
            padding: 20px;
            text-align: center;
        }
        
        .profile-body {
            background-color: white;
            border-radius: 0 0 10px 10px;
            padding: 20px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        
        .achievement-badge {
            background-color: var(--accent-light);
            color: var(--primary-dark);
            border-radius: 30px;
            padding: 5px 15px;
            display: inline-block;
            margin: 5px;
            font-weight: bold;
        }
    </style>
    """, unsafe_allow_html=True)

# UI Helper Functions
def show_loading_spinner(text="Processing your image..."):
    """Display a custom loading spinner with scroll indication on mobile"""
    spinner_html = f"""
    <div class="loading-container">
        <div class="loading-spinner"></div>
        <p>{text}</p>
        <div class="scroll-indicator">
            <span>⬇️ Scroll down to view results ⬇️</span>
        </div>
    </div>
    """
    return st.markdown(spinner_html, unsafe_allow_html=True)

def success_message(text):
    """Display a stylized success message"""
    return st.markdown(f'<div class="success-box">✅ {text}</div>', unsafe_allow_html=True)

def info_message(text):
    """Display a stylized info message"""
    return st.markdown(f'<div class="info-box">ℹ️ {text}</div>', unsafe_allow_html=True)

def warning_message(text):
    """Display a stylized warning message"""
    return st.markdown(f'<div class="warning-box">⚠️ {text}</div>', unsafe_allow_html=True)

def error_message(text):
    """Display a stylized error message"""
    return st.markdown(f'<div class="error-box">❌ {text}</div>', unsafe_allow_html=True)

def vocam_card(content_function, accent=False):
    """Create a card container with consistent styling"""
    card_class = "vocam-card-accent" if accent else "vocam-card"
    st.markdown(f'<div class="{card_class}">', unsafe_allow_html=True)
    content_function()
    st.markdown('</div>', unsafe_allow_html=True)

def word_card(word_original, word_translated, audio_html=None, category=None):
    """Display a stylized word card"""
    html = f"""
    <div class="word-card">
        <h3>{word_original} → {word_translated}</h3>
        {f'<p>Category: {category}</p>' if category else ''}
        {audio_html if audio_html else ''}
    </div>
    """
    return st.markdown(html, unsafe_allow_html=True)

def add_result_separator():
    """Add a visual separator between input and results for mobile"""
    return st.markdown('<div class="result-container"></div>', unsafe_allow_html=True)

def add_scroll_indicator():
    """Add a mobile-friendly scroll indicator"""
    return st.markdown('<div class="scroll-indicator">⬇️ Results Below ⬇️</div>', unsafe_allow_html=True)

def style_title(title_text):
    """Create a styled page title"""
    return st.markdown(f'<h1 style="color: #074173;">{title_text}</h1>', unsafe_allow_html=True)

def style_section_title(title_text):
    """Create a styled section title"""
    return st.markdown(f'<h3 style="color: #1679AB; margin-top: 20px;">{title_text}</h3>', unsafe_allow_html=True)

def style_audio_player(audio_bytes):
    """Get HTML for a styled audio player"""
    if audio_bytes:
        audio_base64 = base64.b64encode(audio_bytes).decode()
        audio_tag = f'<audio src="data:audio/mp3;base64,{audio_base64}" controls style="width: 100%; max-width: 300px;"></audio>'
        return audio_tag
    return ""

def style_sidebar_header():
    """Style the sidebar header"""
    st.sidebar.markdown('<h1 style="color: #5DEBD7;">🌍 Vocam</h1>', unsafe_allow_html=True)

def add_current_mode_indicator(app_mode):
    """Add a visual indicator for the current mode in the sidebar"""
    st.sidebar.markdown(f"""
    <div style="background-color: #5DEBD7; padding: 8px; border-radius: 5px; margin-top: 10px; margin-bottom: 20px;">
        <p style="margin: 0; color: #074173; font-weight: bold;">Currently in: {app_mode}</p>
    </div>
    """, unsafe_allow_html=True)

def setup_sidebar_language_section(languages, st_session_state):
    """Set up the language selection section in the sidebar"""
    st.sidebar.markdown('<h3 style="color: #5DEBD7; margin-top: 20px;">Language Settings</h3>', unsafe_allow_html=True)
    selected_language = st.sidebar.selectbox(
        "Select target language",
        list(languages.keys()),
        index=list(languages.values()).index(st_session_state.target_language) if st_session_state.target_language in languages.values() else 0
    )
    return selected_language

def display_session_info(st_session_state):
    """Display session information in the sidebar"""
    st.sidebar.markdown('<h3 style="color: #5DEBD7; margin-top: 20px;">Session Info</h3>', unsafe_allow_html=True)
    if st_session_state.session_id:
        st.sidebar.markdown(f"""
        <div style="background-color: #1679AB; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
            <p style="color: white; margin: 0;"><strong>✅ Session active</strong></p>
        </div>
        <div style="background-color: #074173; padding: 10px; border-radius: 5px;">
            <p style="color: white; margin: 0;">📚 Words studied: {st_session_state.words_studied}</p>
            <p style="color: white; margin: 0;">🧠 Words learned: {st_session_state.words_learned}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.sidebar.markdown(f"""
        <div style="background-color: #1679AB; padding: 10px; border-radius: 5px;">
            <p style="color: white; margin: 0;">⚠️ No active session</p>
            <p style="color: #C5FF95; margin-top: 5px; font-size: 0.9em;">Start a session in Camera Mode to track progress</p>
        </div>
        """, unsafe_allow_html=True)

def add_help_section():
    """Add a help section to the sidebar"""
    with st.sidebar.expander("ℹ️ Need Help?"):
        st.markdown("""
        **Quick Tips:**
        - 📸 Use **Camera Mode** to capture objects and learn new words
        - 📚 Review your words in **My Vocabulary**
        - 🎮 Test yourself in **Quiz Mode**
        - 📊 Track your progress in **Statistics**
        
        **On Mobile:**
        - After taking a picture, scroll down to see results
        - Tap buttons to navigate between sections
        """)

def add_footer():
    """Add a footer to the page"""
    st.markdown("""
    <div style="text-align: center; margin-top: 40px; padding: 20px; color: #777;">
        <p>Vocam - Language Learning App | Created with ❤️</p>
    </div>
    """, unsafe_allow_html=True)

def empty_state(title, message, action_text=None, action_link=None):
    """Create a styled empty state container"""
    html = f"""
    <div style="text-align: center; padding: 50px 20px; background-color: #f8f9fa; border-radius: 10px; margin: 30px 0;">
        <h3 style="color: #1679AB;">{title}</h3>
        <p style="margin-bottom: 30px;">{message}</p>
    """
    
    if action_text and action_link:
        html += f"""
        <button style="background-color: #5DEBD7; color: #074173; border: none; padding: 10px 20px; border-radius: 8px; font-weight: bold; cursor: pointer;" 
                onclick="document.querySelector('span:contains(\'{action_link}\')').click();">
            {action_text}
        </button>
        """
    
    html += "</div>"
    return st.markdown(html, unsafe_allow_html=True)

def style_plot(fig, title):
    """Apply consistent styling to matplotlib plots"""
    ax = fig.gca()
    
    # Style the chart
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#DDDDDD')
    ax.spines['bottom'].set_color('#DDDDDD')
    ax.set_xlabel('Category', fontsize=12, color='#333333')
    ax.set_ylabel('Value', fontsize=12, color='#333333')
    ax.set_title(title, fontsize=14, color='#074173')
    return fig