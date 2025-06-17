import nltk
from nltk.corpus import wordnet
import pyttsx3
from deep_translator import GoogleTranslator
from gtts import gTTS
import tempfile
import os
import pygame

# Download WordNet corpus if not installed
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    print("Downloading NLTK WordNet corpus...")
    nltk.download('wordnet')
    print("WordNet corpus downloaded successfully.")

# Initialize text-to-speech engine for English
engine = pyttsx3.init()
engine.setProperty('rate', 140)
engine.setProperty('volume', 0.9)
voices = engine.getProperty('voices')
for voice in voices:
    if "english" in voice.name.lower():
        engine.setProperty('voice', voice.id)
        break

# Initialize Google Translator
translator = GoogleTranslator(source='en', target='bn')

# Function to check if a word is valid using WordNet
def is_valid_english_word(word):
    return bool(wordnet.synsets(word.lower()))

# Function to speak Bengali text using gTTS and pygame
def speak_bengali(text):
    try:
        pygame.mixer.init()
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
            temp_file_path = temp_file.name
            tts = gTTS(text=text, lang='bn')
            tts.save(temp_file_path)
        
        pygame.mixer.music.load(temp_file_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        
        pygame.mixer.quit()
        try:
            os.unlink(temp_file_path)
        except Exception as e:
            print(f"Error deleting temporary file: {e}")
    except Exception as e:
        print(f"Error in Bengali TTS: {e}")

# Test function to process a sample word
def test_word(sample_word):
    print(f"Testing word: {sample_word}")
    if is_valid_english_word(sample_word):
        print(f"Valid English word: {sample_word.lower()}")
        try:
            bengali_word = translator.translate(sample_word)
            print(f"Bengali translation: {bengali_word}")
            # Speak English word
            engine.say(sample_word.lower())
            engine.runAndWait()
            # Speak Bengali word
            speak_bengali(bengali_word)
        except Exception as e:
            print(f"Translation error for '{sample_word}': {e}")
    else:
        print(f"'{sample_word}' is not a valid English word.")
        engine.say(f"{sample_word} is not a valid word.")
        engine.runAndWait()
        speak_bengali(f"{sample_word} একটি বৈধ শব্দ নয়।")

# Sample word to test
sample_word = "RABBIT"  # Change to any word, e.g., "SORRY" or "XYZ"
test_word(sample_word)