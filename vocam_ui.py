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
            
            /* New highlight colors */
            --highlight-orange: #FF9800;
            --highlight-gold: #FFC107;
            --highlight-aqua: #00FFFF;
            --highlight-turquoise: #40E0D0;
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
            border-radius: 12px;
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
        
        /* Sidebar styling - FIXED with ROUNDED CORNERS */
        section[data-testid="stSidebar"] {
            background-color: var(--primary-dark);
        }
        section[data-testid="stSidebar"] > div {
            background-color: var(--primary-dark);
        }
        
        /* Add rounded corners to all sidebar elements */
        section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] > div {
            border-radius: 12px !important;
            margin-bottom: 8px;
        }
        
        /* Special styling for My Progress section - BRIGHTER ORANGE */
        section[data-testid="stSidebar"] div:has(h3:contains("My Progress")),
        section[data-testid="stSidebar"] div:has(div:contains("My Progress")) {
            background-color: var(--highlight-orange) !important;
            border-radius: 12px !important;
            padding: 10px !important;
            margin-top: 12px !important;
            margin-bottom: 12px !important;
            box-shadow: 0 0 10px rgba(255, 152, 0, 0.5);
        }
        
        /* Make My Progress text pop */
        section[data-testid="stSidebar"] div:has(h3:contains("My Progress")) h3,
        section[data-testid="stSidebar"] div:has(div:contains("My Progress")) div {
            color: var(--primary-dark) !important;
            font-weight: bold !important;
            text-shadow: 0 0 3px rgba(255, 255, 255, 0.5);
        }
        
        /* Special styling for Need Help section - BRIGHT AQUA */
        section[data-testid="stSidebar"] div.stExpander:has(div:contains("Need Help")),
        section[data-testid="stSidebar"] details:has(summary:contains("Need Help")) {
            background-color: var(--highlight-aqua) !important;
            border-radius: 12px !important;
            margin-top: 12px !important;
            box-shadow: 0 0 10px rgba(0, 255, 255, 0.5); 
        }
        
        /* Make Need Help title more visible */
        section[data-testid="stSidebar"] .streamlit-expanderHeader:contains("Need Help"),
        section[data-testid="stSidebar"] summary:contains("Need Help") {
            background-color: var(--highlight-turquoise) !important;
            color: var(--primary-dark) !important;
            font-weight: bold !important;
            border-radius: 10px !important;
            text-shadow: 0 0 3px rgba(255, 255, 255, 0.5);
        }
        
        /* White text for standard elements */
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] h4,
        section[data-testid="stSidebar"] h5,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] span:not(.st-emotion-cache-10trblm),
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] .stMarkdown {
            color: var(--text-light) !important;
        }
        
        /* Fix dropdown styling */
        section[data-testid="stSidebar"] .stSelectbox label {
            color: var(--text-light) !important;
        }
        
        /* Style the selectbox itself with rounded corners */
        section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] {
            background-color: #ffffff !important;
            border-radius: 10px !important;
            overflow: hidden;
        }
        
        /* Style the selectbox text */
        section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] span {
            color: var(--text-dark) !important;
        }
        
        /* Ensure dropdown options are visible */
        section[data-testid="stSidebar"] .stSelectbox option,
        section[data-testid="stSidebar"] select option,
        section[data-testid="stSidebar"] div[role="listbox"] *,
        section[data-testid="stSidebar"] ul[role="listbox"] * {
            color: var(--text-dark) !important;
            background-color: #ffffff !important;
        }
        
        /* Rest of the CSS remains unchanged... */
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
        <p>Vocam | Created by İdil Doğa Türkmen</p>
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