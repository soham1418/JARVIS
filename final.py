import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import streamlit as st
import numpy as np
from scipy.io.wavfile import read as wav_read
import soundfile as sf
from audiorecorder import audiorecorder
import whisper
import google.generativeai as genai
import sqlite3 
import httpcore
import requests
from googletrans import Translator
from gtts import gTTS
from langdetect import detect
translator = Translator()

setattr(httpcore, 'SyncHTTPTransport', 'AsyncHTTPProxy')

GOOGLE_API_KEY = '<your_api_key>'
genai.configure(api_key=GOOGLE_API_KEY)
model= genai.GenerativeModel("gemini-1.5-flash")
INITIAL_PROMPT = '' # give extra prior information like 'Act like a news reporter'
WHISPER_MODEL = 'small' # change this to medium or large for better results
history=[]

# Language options
language_options = {
    "Auto-detect": "auto",
    "English": "en",
    "Hindi": "hi",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Japanese": "ja",
    "Korean": "ko",
    "Chinese": "zh",
    "Arabic": "ar"
}

conn=sqlite3.connect("chatbot.db")
cursor=conn.cursor()
#cursor.execute('DROP table users')

cursor.execute("""create table if not exists Chat_History(ID integer Primary Key, User_Input text, Bot_Response text, Date datetime default current_timestamp)""")


def elevenlabs_tts(speech):
        CHUNK_SIZE = 1024
        url = "https://api.elevenlabs.io/v1/text-to-speech/rW2lcIFbB5AVdzWcOG9n"

        headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": "87b68d0f8fb0ea99cd14b79d84dd58f5"
        }

        data = {
        "text": speech,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": 0.3,
            "similarity_boost": 0.5
        }
        }

        response = requests.post(url, json=data, headers=headers)
        with open('output.mp3', 'wb') as f:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    f.write(chunk)
            # audio_file = open("output.mp3", "rb")
            # audio_bytes = audio_file.read()

            # st.audio(audio_bytes, format="audio/mp3")

            st.audio("output.mp3",format="audio/mpeg")



# Function to save recorded audio to a file
def save_audio_to_file(filepath, data, samplerate):
    sf.write(filepath, data, samplerate)


def google_tts(text):
     tts=gTTS(text)
     tts.save ("output.mp3")
     st.audio("output.mp3",format="audio/mpeg")



# Main function for the Streamlit app

def main():
    st.set_page_config(layout="wide", initial_sidebar_state="collapsed")

    st.title("Multi-Lingual Voicebot")

    # Custom CSS for the mic button and wave animation
    st.markdown("""
    <style>
    .stApp {
        display: flex;
        flex-direction: column;
        height: 100vh;
    }
    .main {
        flex-grow: 1;
        overflow: auto;
    }
    .input-container {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        padding: 1rem;
        background-color: white;
        z-index: 1000;
    }
    .btn-outline-secondary {
        width: 60px;
        height: 60px;
        border-radius: 50%;
        background-color: #f0f0f0;
        display: flex;
        justify-content: center;
        align-items: center;
        cursor: pointer;
        transition: background-color 0.3s;
    }
    .btn-outline-secondary:hover {
        background-color: #e0e0e0;
    }
    .mic-icon {
        width: 30px;
        height: 30px;
        fill: #333;
    }
    .wave {
        width: 60px;
        height: 60px;
        border-radius: 50%;
        position: absolute;
        background-color: rgba(255, 0, 0, 0.3);
        opacity: 0;
        animation: wave 1.5s infinite;
    }
    @keyframes wave {
        0% {
            transform: scale(1);
            opacity: 0.8;
        }
        100% {
            transform: scale(1.5);
            opacity: 0;
        }
    }
    </style>
    """, unsafe_allow_html=True)

    
    

    # Initialize session state for chat history if it doesn't exist
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Chat history container
    chat_container = st.container()

    # Input container
    input_container = st.container()

    # Display chat history
    with chat_container:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])

    # Input area at the bottom
    with input_container:
        st.markdown('<div class="input-container">', unsafe_allow_html=True)
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            user_input = st.text_input("Type your message here:", key="user_input")
        
        with col2:
            st.markdown("""
            <div class="mic-button">
                <br>
            </div>
            """, unsafe_allow_html=True)
            audio = audiorecorder("🎙️", "🛑")  # Hide the default audio recorder buttons
        st.markdown('</div>', unsafe_allow_html=True)

        with col3:
            selected_language = st.selectbox(
                "Select Language",
                options=list(language_options.keys()),
                index=0,
                key="language_selector"
            )

    try:
        if user_input or len(audio) > 0:
            user_language = language_options[selected_language]
            if len(audio) > 0:
                # Process audio input
                audio.export("audio.mp3", format="mp3")
                transcription_model = whisper.load_model(WHISPER_MODEL)
                audio = whisper.load_audio("audio.mp3")
                audio = whisper.pad_or_trim(audio)
                if user_language == "auto":
                    mel = whisper.log_mel_spectrogram(audio).to(transcription_model.device)
                    _, probs = transcription_model.detect_language(mel)
                    user_language = str(max(probs, key=probs.get))
                
                result = transcription_model.transcribe("audio.mp3", language=user_language, fp16=False)
                user_input = result['text']
            else:
                if user_language == "auto":
                    user_language = str(detect(user_input))


            # Add user message to chat history
            st.session_state.chat_history.append({"role": f"User (lang:{user_language})", "content": user_input})

            # Generate bot response
            user_input=INITIAL_PROMPT + ' '+  user_input
            response = model.generate_content(user_input)
            ollama_speech = response.text
            speech = translator.translate(ollama_speech, dest=user_language).text

            # Add bot response to chat history
            st.session_state.chat_history.append({"role": "Jarvis", "content": speech})

            # Display the latest messages
            with chat_container:
                with st.chat_message(f"User (lang:{user_language})"):
                    st.write(user_input)
                    if len(audio) > 0:
                        st.audio("audio.mp3",format="audio/mpeg")
                with st.chat_message("Jarvis"):
                    st.write(speech)
                    google_tts(speech)

            # Save to database
            cursor.execute('insert into Chat_History(User_Input, Bot_Response) values (?,?)', (user_input, speech))
            conn.commit()
        
    except KeyError:
        st.write("Oops! There was some problem. Please trying once again;)")

    # Clean up
    conn.close()

    footer_html = """
    <div style='position: fixed; bottom: 5px; left: 40%; text-align: center; padding: 10px;'>
        <p>Developed with ❤️ by Soham Sabharwal.</p>
    </div>
    """

    # Render the footer 
    st.markdown(footer_html, unsafe_allow_html=True)



# Run the app
if __name__ == "__main__":
    main()
