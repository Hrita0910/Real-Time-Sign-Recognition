import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import os
import pyttsx3
# Import enchant for English word validation (optional, comment out if not using)
import enchant

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 140)  # Speed of speech (words per minute)
engine.setProperty('volume', 0.9)  # Volume (0.0 to 1.0)
voices = engine.getProperty('voices')
for voice in voices:
    if "english" in voice.name.lower():
        engine.setProperty('voice', voice.id)  # Set to an English voice
        break

# Initialize English dictionary for word validation (optional, comment out if not using)
try:
    dictionary = enchant.Dict("en_US")
except enchant.errors.DictNotFoundError:
    print("Warning: English dictionary not found for pyenchant. Falling back to manual word list.")
    dictionary = None

# Fallback list of valid words if pyenchant is not available
VALID_WORDS = {"say", "hi", "yes", "no", "bye", "hello", "ok", "good", "bad", "stop","boy"}  # Add more as needed

# Define custom functions (same as in train_asl_model.py) for loading
def mish(x):
    return x * tf.math.tanh(tf.math.softplus(x))

def mixed_pooling(x, pool_size=(2, 2), alpha=0.5):
    max_pool = tf.keras.layers.MaxPooling2D(pool_size=pool_size)(x)  # Fixed typo
    avg_pool = tf.keras.layers.AveragePooling2D(pool_size=pool_size)(x)
    return alpha * max_pool + (1 - alpha) * avg_pool

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Initialize OpenCV video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam. Please check if the webcam is connected and accessible.")
    exit()

# Get video frame dimensions
ret, frame = cap.read()
if not ret:
    print("Error: Could not read video frame. Exiting...")
    cap.release()
    exit()
frame_height, frame_width = frame.shape[:2]

# Create a larger canvas: video feed on top, history below
canvas_height = frame_height + 200
canvas = np.zeros((canvas_height, frame_width, 3), dtype=np.uint8)

# Ensure models directory exists
model_path = 'models/asl_model.keras'
if not os.path.exists(os.path.dirname(model_path)):
    print(f"Error: Models directory {os.path.dirname(model_path)} does not exist. Please train the model first.")
    cap.release()
    exit()

# Load trained model with custom objects
if not os.path.exists(model_path):
    print(f"Error: Model file {model_path} not found. Please train the model first.")
    cap.release()
    exit()
custom_objects = {
    'CustomLayers>mish': mish,
    'CustomLayers>mixed_pooling': mixed_pooling
}
model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
labels = [chr(i) for i in range(65, 91)]  # A-Z, including J and Z

# Function to process hand landmarks
def process_landmarks(hand_landmarks):
    landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
    landmarks = np.array(landmarks).flatten()
    wrist = landmarks[0:3]
    landmarks = landmarks.reshape(-1, 3) - wrist
    landmarks = landmarks.flatten()
    return landmarks

# Variables for stable prediction, history, and word formation
current_char = "None"
char_buffer = []
buffer_size = 3
hold_frames = 30
frame_counter = 0
char_history = []
word_buffer = ""  # To accumulate characters into a potential word
detected_words = []  # To store recognized words

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read video frame. Breaking loop...")
        break

    # Convert frame to RGB for MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # Reset canvas: video feed on top, history below
    canvas = np.zeros((canvas_height, frame_width, 3), dtype=np.uint8)
    canvas[0:frame_height, 0:frame_width] = frame

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on frame
            mp_drawing.draw_landmarks(canvas[0:frame_height, 0:frame_width], hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Process landmarks for classification
            if frame_counter <= 0:
                # Call process_landmarks to create the landmarks variable
                landmarks = process_landmarks(hand_landmarks)
                landmarks_input = landmarks.reshape(1, 9, 7, 1)  # Reshape for CNN
                pred = model.predict(landmarks_input, verbose=0)
                predicted_char = labels[np.argmax(pred)]
                char_buffer.append(predicted_char)

                # Check if the last buffer_size predictions are the same
                if len(char_buffer) >= buffer_size:
                    if all(c == char_buffer[-1] for c in char_buffer[-buffer_size:]):
                        current_char = char_buffer[-1]
                        char_history.append(current_char)
                        if len(char_history) > 10:
                            char_history.pop(0)

                        # Add the character to the word buffer
                        word_buffer += current_char

                        # Check if the word buffer forms a valid English word
                        if len(word_buffer) >= 2:  # Minimum length to check for a word
                            word_to_check = word_buffer.lower()
                            is_valid_word = (dictionary.check(word_to_check) if dictionary else word_to_check in VALID_WORDS)
                            if is_valid_word:
                                detected_words.append(word_to_check)
                                print(f"Detected word: {word_to_check}")
                                word_buffer = ""  # Reset the word buffer after detecting a word
                            elif len(word_buffer) > 5:  # Reset if too long and not a word
                                word_buffer = current_char  # Keep the last character to start a new word

                        frame_counter = hold_frames
                    char_buffer = char_buffer[-buffer_size:]
            else:
                frame_counter -= 1

    # Display current character on video feed
    cv2.putText(canvas, f"ASL: {current_char}", (10, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display character history below video feed
    history_start_y = frame_height + 30
    cv2.putText(canvas, "Detected Characters:", (10, history_start_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    for i, char in enumerate(char_history):
        cv2.putText(canvas, char, (10 + i * 30, history_start_y + 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Display detected words below character history
    words_start_y = history_start_y + 80
    cv2.putText(canvas, "Detected Words:", (10, words_start_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    for i, word in enumerate(detected_words[-5:]):  # Show last 5 words
        cv2.putText(canvas, word.upper(), (10 + i * 60, words_start_y + 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

    # Show canvas
    cv2.imshow('ASL Character Detection', canvas)

    # Exit on 'Esc' key
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
hands.close()

# Print and speak detected words
print("Detected Words in Session (up to last 10):")
if detected_words:
    detected_words_str = " ".join(detected_words[-10:])
    print(detected_words_str)
    # Speak the detected words
    engine.say("The detected words are: " + detected_words_str)
    engine.runAndWait()
else:
    print("No words were detected during the session.")
    engine.say("No words were detected during the session.")
    engine.runAndWait()

# If there are remaining characters in the word buffer, treat them as a final word
if word_buffer:
    word_to_check = word_buffer.lower()
    is_valid_word = (dictionary.check(word_to_check) if dictionary else word_to_check in VALID_WORDS)
    if is_valid_word:
        detected_words.append(word_to_check)
        print(f"Final detected word: {word_to_check}")
        engine.say(word_to_check)
        engine.runAndWait()

print("Real-time detection completed successfully!")