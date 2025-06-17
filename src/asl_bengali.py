import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import os
import pyttsx3
import nltk
from nltk.corpus import wordnet
from deep_translator import GoogleTranslator
from gtts import gTTS
import tempfile
import pygame

# Download WordNet corpus if not already installed
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

# Define custom functions for model loading
def mish(x):
    return x * tf.math.tanh(tf.math.softplus(x))

def mixed_pooling(x, pool_size=(2, 2), alpha=0.5):
    max_pool = tf.keras.layers.MaxPooling2D(pool_size=pool_size)(x)
    avg_pool = tf.keras.layers.AveragePooling2D(pool_size=pool_size)(x)
    return alpha * max_pool + (1 - alpha) * avg_pool

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Initialize OpenCV video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Get video frame dimensions
ret, frame = cap.read()
if not ret:
    print("Error: Could not read video frame.")
    cap.release()
    exit()
frame_height, frame_width = frame.shape[:2]

# Create canvas
canvas_height = frame_height + 200
canvas = np.zeros((canvas_height, frame_width, 3), dtype=np.uint8)

# Ensure models directory exists
model_path = 'models/asl_model.keras'
if not os.path.exists(os.path.dirname(model_path)):
    print(f"Error: Models directory {os.path.dirname(model_path)} does not exist.")
    cap.release()
    exit()

# Load trained model
if not os.path.exists(model_path):
    print(f"Error: Model file {model_path} not found.")
    cap.release()
    exit()
custom_objects = {
    'CustomLayers>mish': mish,
    'CustomLayers>mixed_pooling': mixed_pooling
}
model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
labels = [chr(i) for i in range(65, 91)]  # A-Z

# Function to process hand landmarks
def process_landmarks(hand_landmarks):
    # Extract x, y, z for each landmark
    landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
    landmarks = np.array(landmarks).flatten()
    wrist = landmarks[0:3]
    landmarks = landmarks.reshape(-1, 3) - wrist
    landmarks = landmarks.flatten()
    return landmarks

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

# Function to process accumulated characters into words
def process_word_buffer(buffer):
    detected_words = []
    word_to_check = buffer.lower()
    max_length = len(word_to_check)
    index = 0

    while index < max_length:
        valid_word = None
        split_index = max_length

        # Check for the longest valid word starting from index
        for i in range(max_length, index, -1):
            prefix = word_to_check[index:i]
            if len(prefix) >= 2 and is_valid_english_word(prefix):
                valid_word = prefix
                split_index = i
                break

        if valid_word:
            detected_words.append(valid_word)
            index = split_index
        else:
            index += 1

    return detected_words

# Variables for stable prediction, history, and word formation
current_char = "None"
char_buffer = []
buffer_size = 3
hold_frames = 30
frame_counter = 0
char_history = []
word_buffer = ""
detected_words = []  # For display purposes, empty during session

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read video frame.")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    canvas = np.zeros((canvas_height, frame_width, 3), dtype=np.uint8)
    canvas[0:frame_height, 0:frame_width] = frame

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(canvas[0:frame_height, 0:frame_width], hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if frame_counter <= 0:
                landmarks = process_landmarks(hand_landmarks)
                landmarks_input = landmarks.reshape(1, 9, 7, 1)
                pred = model.predict(landmarks_input, verbose=0)
                predicted_char = labels[np.argmax(pred)]
                char_buffer.append(predicted_char)

                if len(char_buffer) >= buffer_size:
                    if all(c == char_buffer[-1] for c in char_buffer[-buffer_size:]):
                        current_char = char_buffer[-1]
                        char_history.append(current_char)
                        if len(char_history) > 10:
                            char_history.pop(0)

                        word_buffer += current_char
                        frame_counter = hold_frames
                    char_buffer = char_buffer[-buffer_size:]
            else:
                frame_counter -= 1

    # Display current character
    cv2.putText(canvas, f"ASL: {current_char}", (10, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display character history
    history_start_y = frame_height + 30
    cv2.putText(canvas, "Detected Characters:", (10, history_start_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    for i, char in enumerate(char_history):
        cv2.putText(canvas, char, (10 + i * 30, history_start_y + 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Display detected words (empty during session)
    words_start_y = history_start_y + 80
    cv2.putText(canvas, "Detected Words:", (10, words_start_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    for i, word in enumerate(detected_words[-5:]):
        cv2.putText(canvas, word.upper(), (10 + i * 60, words_start_y + 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

    cv2.imshow('ASL Character Detection', canvas)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
hands.close()

# Process accumulated characters into words after window closes
if word_buffer:
    detected_words = process_word_buffer(word_buffer)

# Print and speak detected words
print("Detected Words in Session (up to last 10):")
if detected_words:
    detected_words_str = " ".join(detected_words[-10:])
    print(detected_words_str)
    try:
        bengali_words_str = translator.translate(detected_words_str)
        print(f"Bengali translation: {bengali_words_str}")
        for word in detected_words[-10:]:
            # Speak each word exactly once in English and Bengali
            print(f"Speaking word: {word}")
            engine.say(word)
            engine.runAndWait()
            try:
                bengali_word = translator.translate(word)
                print(f"Bengali translation for {word}: {bengali_word}")
                speak_bengali(bengali_word)
            except Exception as e:
                print(f"Translation error for '{word}': {e}")
    except Exception as e:
        print(f"Translation error for session summary: {e}")
else:
    print("No words were detected during the session.")
    engine.say("No words were detected during the session.")
    engine.runAndWait()
    print("কোনো শব্দ সনাক্ত করা হয়নি।")
    speak_bengali("কোনো শব্দ সনাক্ত করা হয়নি।")

print("Real-time detection completed successfully!")