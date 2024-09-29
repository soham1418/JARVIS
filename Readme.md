# Voice Assistant Application

This project implements a *multilingual voice assistant* using various Python libraries to handle audio input, translation, speech-to-text, and text-to-speech features. It leverages APIs like Whisper and Google Translate to facilitate real-time voice interactions in multiple languages.

## Features

- *Audio Input Recording*: The application captures audio input via an integrated microphone and processes the audio for further analysis.
- *Speech-to-Text*: Utilizing Whisper AI, it converts the recorded voice into text for processing.
- *Text Translation*: The text can be translated into different languages using Google Translate.
- *Text-to-Speech*: The translated or recognized text is converted back into speech using Google Text-to-Speech (gTTS).
- *Multilingual Detection*: Automatic language detection using langdetect.

## Prerequisites

- *Python 3.11*
- *Streamlit*: For building the web interface.
- *Whisper*: For speech recognition.
- *Google Translate*: For text translation.
- *gTTS*: For converting text to speech.
- *SQLite*: For database operations, if required.
- *Soundfile, **audiorecorder*: For handling audio input and output.

## Setup

1. Clone the repository:
   bash
   git clone JARVIS
   cd JARVIS
   

2. Install the required dependencies:
   bash
   pip install -r requirements.txt
   

3. Run the application using Streamlit:
   bash
   streamlit run final.py
   

## Usage

- Open the web interface generated by Streamlit.
- Record your voice by pressing the *Record* button.
- The application will process the audio, detect the spoken language, translate it if required, and output the translation in both text and speech formats.

## Key Libraries

- streamlit: For web-based user interface.
- whisper: AI-powered speech-to-text recognition.
- googletrans: For translating recognized text into different languages.
- gTTS: Text-to-speech conversion.
- langdetect: For language detection.
- sqlite3: If any database operations are needed for storing user data.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
