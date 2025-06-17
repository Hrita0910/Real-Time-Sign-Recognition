import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import os
import pyttsx3
from deep_translator import GoogleTranslator
from gtts import gTTS
import tempfile
import pygame

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

# Function to compute hand orientation angles
def compute_hand_angles(hand_landmarks):
    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
    wrist = landmarks[0]
    index_mcp = landmarks[5]  # Index finger MCP
    thumb_mcp = landmarks[2]  # Thumb MCP
    # Compute vectors
    wrist_to_index = index_mcp - wrist
    wrist_to_thumb = thumb_mcp - wrist
    # Normalize vectors
    wrist_to_index = wrist_to_index / (np.linalg.norm(wrist_to_index) + 1e-6)
    wrist_to_thumb = wrist_to_thumb / (np.linalg.norm(wrist_to_thumb) + 1e-6)
    # Compute angle between vectors
    dot_product = np.dot(wrist_to_index, wrist_to_thumb)
    angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
    return np.array([angle])

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
model_path = 'models/asl_model_with_space_backspace.keras'
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
labels = [chr(i) for i in range(65, 91)] + ['space', 'backspace']  # A-Z + space + backspace

# Function to process hand landmarks with angle feature
def process_landmarks(hand_landmarks):
    landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
    landmarks = np.array(landmarks)
    wrist = landmarks[0]
    landmarks = landmarks - wrist  # Normalize by wrist
    landmarks = landmarks.flatten()
    angles = compute_hand_angles(hand_landmarks)
    return np.concatenate([landmarks, angles])

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

# Variables for stable prediction, history, and phrase formation
current_char = "None"
char_buffer = []
buffer_size = 3
hold_frames = 30
frame_counter = 0
char_history = []
phrase_buffer = ""
detected_phrase = ""  # For final output

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
                landmarks_input = landmarks.reshape(1, 8, 8, 1)  # Match training input shape
                pred = model.predict(landmarks_input, verbose=0)
                predicted_char = labels[np.argmax(pred)]
                char_buffer.append(predicted_char)

                if len(char_buffer) >= buffer_size:
                    if all(c == char_buffer[-1] for c in char_buffer[-buffer_size:]):
                        current_char = char_buffer[-1]
                        char_history.append(current_char if current_char not in ['space', 'backspace'] else '_' if current_char == 'space' else '<-')
                        if len(char_history) > 10:
                            char_history.pop(0)

                        if current_char == 'space':
                            phrase_buffer += '_'
                        elif current_char == 'backspace':
                            if phrase_buffer:
                                phrase_buffer = phrase_buffer[:-1]
                        else:
                            phrase_buffer += current_char
                        frame_counter = hold_frames
                    char_buffer = char_buffer[-buffer_size:]
            else:
                frame_counter -= 1

    # Display current character
    display_char = current_char if current_char not in ['space', 'backspace'] else '_' if current_char == 'space' else '<-'
    cv2.putText(canvas, f"ASL: {display_char}", (10, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display character history
    history_start_y = frame_height + 30
    cv2.putText(canvas, "Detected Characters:", (10, history_start_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    for i, char in enumerate(char_history):
        cv2.putText(canvas, char, (10 + i * 30, history_start_y + 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Display phrase buffer
    words_start_y = history_start_y + 80
    cv2.putText(canvas, "Current Buffer:", (10, words_start_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    display_buffer = phrase_buffer.replace('_', ' ')
    cv2.putText(canvas, display_buffer.upper(), (10, words_start_y + 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

    cv2.imshow('ASL Character Detection', canvas)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
hands.close()

# Use the phrase buffer directly as the final detected phrase
if phrase_buffer:
    detected_phrase = phrase_buffer.replace('_', ' ')
else:
    detected_phrase = ""

# Print and speak the detected phrase
print("Detected Phrase in Session:")
if detected_phrase:
    print(detected_phrase)
    try:
        # Speak the entire phrase in English first
        print(f"Speaking phrase in English: {detected_phrase}")
        engine.say(detected_phrase)
        engine.runAndWait()
        
        # Translate and speak the entire phrase in Bengali
        bengali_phrase = translator.translate(detected_phrase)
        print(f"Bengali translation: {bengali_phrase}")
        speak_bengali(bengali_phrase)
    except Exception as e:
        print(f"Translation error for session summary: {e}")
else:
    print("No phrase was detected during the session.")
    engine.say("No phrase was detected during the session.")
    engine.runAndWait()
    print("কোনো বাক্য সনাক্ত করা হয়নি।")
    speak_bengali("কোনো বাক্য সনাক্ত করা হয় নি।")

print("Real-time detection completed successfully!")