import streamlit as st
import requests
import json
import time
from io import BytesIO
import base64

# Audio recording is available through st.audio_input (built-in)
AUDIO_RECORDER_AVAILABLE = True

# Page configuration
st.set_page_config(
    page_title="Cyber Security Assistant",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load custom CSS
def load_css():
    with open('streamlit_style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_mode" not in st.session_state:
    st.session_state.chat_mode = "Text Chat"

if "api_url" not in st.session_state:
    st.session_state.api_url = "http://localhost:8000"

# Header with floating sparkles
st.markdown("""
<!-- Floating sparkles -->
<div class="sparkle s1"></div>
<div class="sparkle s2"></div>
<div class="sparkle s3"></div>
<div class="sparkle s4"></div>
<div class="sparkle s5"></div>

<div class="header-container">
    <div class="badge">🛡️</div>
    <div>
        <h1 class="title">Cyber Security Assistant</h1>
        <p class="subtitle">Ask me anything Cyber Security ✨</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Mode selection
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    chat_mode = st.selectbox(
        "Chat Mode",
        ["Text Chat", "Voice Chat"],
        index=0 if st.session_state.chat_mode == "Text Chat" else 1,
        key="mode_selector"
    )

    if chat_mode != st.session_state.chat_mode:
        st.session_state.chat_mode = chat_mode
        st.rerun()

# Status indicator
mode_text = "Text Chat Mode" if st.session_state.chat_mode == "Text Chat" else "Voice Chat Mode"
st.markdown(f"""
<div style="text-align: center;">
    <div class="status-indicator">
        <div class="status-dot"></div>
        {mode_text}
    </div>
</div>
""", unsafe_allow_html=True)

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Function to format messages (similar to original HTML version)
def format_message(text):
    """Format message text for better display"""
    if not text:
        return text

    # Handle code blocks - Streamlit will handle this automatically with markdown
    # Handle numbered sections - make them bold
    import re
    text = re.sub(r'^(\d+\)\s+[^\n]+)', r'**\1**', text, flags=re.MULTILINE)

    return text

# Function to send text chat
def send_text_chat(user_input):
    """Send text chat to the FastAPI backend"""
    try:
        response = requests.post(
            f"{st.session_state.api_url}/chat/",
            data={"user_text": user_input},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=60  # 60 second timeout
        )

        if response.status_code == 200:
            data = response.json()
            bot_text = data.get("bot_text", "No response received")
            return format_message(bot_text)
        else:
            return f"**Error:** {response.status_code} - {response.text}"

    except requests.exceptions.ConnectionError:
        return "**Error:** Unable to connect to the backend server. Please make sure the server is running on port 8000."
    except requests.exceptions.Timeout:
        return "**Error:** Request timed out. The server might be processing a complex query."
    except Exception as e:
        return f"**Error:** {str(e)}"

# Function to send voice chat
def send_voice_chat(audio_data):
    """Send voice chat to the FastAPI backend"""
    try:
        # Create BytesIO object from audio data
        if isinstance(audio_data, bytes):
            audio_file = BytesIO(audio_data)
        else:
            audio_file = audio_data

        files = {"file": ("recording.wav", audio_file, "audio/wav")}
        response = requests.post(
            f"{st.session_state.api_url}/voice-chat/",
            files=files,
            timeout=120  # 2 minute timeout for voice processing
        )

        if response.status_code == 200:
            data = response.json()
            user_text = data.get("user_text", "")
            bot_text = data.get("bot_text", "No response received")
            audio_url = data.get("audio_url", None)

            return user_text, format_message(bot_text), audio_url
        else:
            return "", f"**Error:** {response.status_code} - {response.text}", None

    except requests.exceptions.ConnectionError:
        return "", "**Error:** Unable to connect to the backend server. Please make sure the server is running on port 8000.", None
    except requests.exceptions.Timeout:
        return "", "**Error:** Voice processing timed out. Please try again.", None
    except Exception as e:
        return "", f"**Error:** {str(e)}", None

# Chat input handling
if st.session_state.chat_mode == "Text Chat":
    # Text chat input
    if prompt := st.chat_input("Type your cyber security question here..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get assistant response
        with st.chat_message("assistant"):
            with st.spinner("Processing..."):
                response = send_text_chat(prompt)

            st.markdown(response)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

else:
    # Voice chat interface
    st.markdown("""
    <div style="text-align: center; padding: 10px;">
        <p style="color: var(--muted); font-size: 0.95rem; margin-bottom: 16px;">
            Click the record button below to record your voice message.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Audio input component (built-in Streamlit)
    audio_data = st.audio_input("Record your voice message")

    # Process audio when recorded
    if audio_data is not None:
        # Show processing message
        with st.spinner("Processing your voice message..."):
            user_text, bot_response, audio_url = send_voice_chat(audio_data)

        if user_text:  # Only add messages if we got valid transcription
            # Add user message (transcription)
            st.session_state.messages.append({"role": "user", "content": f"🎤 **Voice:** {user_text}"})

            # Display user message
            with st.chat_message("user"):
                st.markdown(f"🎤 **Voice:** {user_text}")

            # Display assistant response
            with st.chat_message("assistant"):
                st.markdown(bot_response)

            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": bot_response})

            # Play audio response if available
            if audio_url:
                try:
                    # Create full URL
                    if not audio_url.startswith('http'):
                        full_audio_url = f"{st.session_state.api_url}{audio_url}"
                    else:
                        full_audio_url = audio_url

                    # Display audio player
                    st.audio(full_audio_url)

                    st.markdown("""
                    <p style="text-align: center; color: var(--muted); font-size: 0.85rem; margin-top: 8px;">
                        🔊 Audio response generated
                    </p>
                    """, unsafe_allow_html=True)

                except Exception as e:
                    st.warning(f"Audio playback not available: {str(e)}")

        elif bot_response and "Error:" in bot_response:
            # Display error message
            with st.chat_message("assistant"):
                st.markdown(bot_response)

        # Clear the audio input after processing
        st.rerun()

    # Instructions
    st.markdown("""
    <div style="text-align: center; margin-top: 20px;">
        <p style="color: var(--muted); font-size: 0.85rem;">
            💡 Click the record button above to capture your voice message.<br>
            The assistant will transcribe your speech and provide both text and voice responses.
        </p>
    </div>
    """, unsafe_allow_html=True)

# Footer - using a simpler approach that works better with Streamlit
st.markdown("""
<div style="height: 60px;"></div>
""", unsafe_allow_html=True)