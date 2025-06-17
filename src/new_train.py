import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt
import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import uuid

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.3)

# Mish activation function
@keras.saving.register_keras_serializable(package="CustomLayers")
def mish(x):
    return x * tf.math.tanh(tf.math.softplus(x))

# Mixed pooling layer
@keras.saving.register_keras_serializable(package="CustomLayers")
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

# Function to process landmarks with additional features
def process_landmarks(hand_landmarks):
    landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
    landmarks = np.array(landmarks)
    wrist = landmarks[0]
    landmarks = landmarks - wrist  # Normalize by wrist
    landmarks = landmarks.flatten()
    angles = compute_hand_angles(hand_landmarks)
    return np.concatenate([landmarks, angles])

# Custom data augmentation for landmarks
def augment_landmarks(landmarks, translation_range=0.05, scale_range=0.1, rotation_range=15):
    # Reshape to (21, 3) for 21 landmarks with x, y, z
    landmarks_3d = landmarks[:-1].reshape(-1, 3)  # Exclude angle feature
    # Translation
    translation = np.random.uniform(-translation_range, translation_range, size=(1, 3))
    landmarks_3d += translation
    # Scaling
    scale = np.random.uniform(1 - scale_range, 1 + scale_range)
    landmarks_3d *= scale
    # Rotation (around z-axis for simplicity)
    angle = np.radians(np.random.uniform(-rotation_range, rotation_range))
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])
    landmarks_3d = np.dot(landmarks_3d, rotation_matrix)
    # Flatten and append angle feature
    return np.concatenate([landmarks_3d.flatten(), landmarks[-1:]])

# Generator for data augmentation
def data_generator(X, y, batch_size, augment=True):
    while True:
        indices = np.random.randint(0, len(X), batch_size)
        batch_X = X[indices]
        batch_y = y[indices]
        if augment:
            batch_X = np.array([augment_landmarks(x) for x in batch_X])
        # Reshape for CNN input
        batch_X = batch_X.reshape(-1, 8, 8, 1)  # Adjusted for new feature size
        yield batch_X, batch_y

# Load dataset (ASL Alphabet Dataset)
data_dir = "C:/Users/HRITAJIT/OneDrive/Desktop/asl_grok/data/asl_alphabet_train/asl_alphabet_train"
labels = [chr(i) for i in range(65, 91)]  # A-Z
X, y = [], []
max_images_per_letter = 1000
image_counts = {label: {'valid': 0, 'total': 0, 'skipped': 0} for label in labels}

for label in labels:
    label_dir = os.path.join(data_dir, label)
    if not os.path.exists(label_dir):
        print(f"Directory {label_dir} not found, skipping...")
        image_counts[label]['skipped'] = -1
        continue
    
    img_count = 0
    total_images = 0
    skipped_images = 0
    print(f"Processing images for letter {label}...")
    for img_name in os.listdir(label_dir):
        if img_count >= max_images_per_letter:
            break
        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            print(f"Skipping non-image file: {os.path.join(label_dir, img_name)}")
            skipped_images += 1
            total_images += 1
            continue
        img_path = os.path.join(label_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load image: {img_path}")
            skipped_images += 1
            total_images += 1
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = process_landmarks(hand_landmarks)
                X.append(landmarks)
                y.append(labels.index(label))
                img_count += 1
                if img_count % 10 == 0:
                    print(f"Processed {img_count} images for letter {label}")
        else:
            print(f"No hand landmarks detected in: {img_path}")
            skipped_images += 1
        total_images += 1
    image_counts[label]['valid'] = img_count
    image_counts[label]['total'] = total_images
    image_counts[label]['skipped'] = skipped_images
    print(f"Finished processing {img_count} images for letter {label} (skipped {skipped_images} images)")

# Save image counts to file
os.makedirs('../outputs', exist_ok=True)
with open('../outputs/image_counts.txt', 'w') as f:
    f.write("Summary of Processed Images:\n")
    f.write("Letter | Valid Images | Total Images | Skipped Images\n")
    f.write("-" * 45 + "\n")
    for label in labels:
        if image_counts[label]['skipped'] == -1:
            f.write(f"{label:6} | {'N/A':12} | {'N/A':12} | {'Directory missing':15}\n")
        else:
            f.write(f"{label:6} | {image_counts[label]['valid']:12} | {image_counts[label]['total']:12} | {image_counts[label]['skipped']:15}\n")

# Print image counts summary
print("\nSummary of Processed Images:")
print("Letter | Valid Images | Total Images | Skipped Images")
print("-" * 45)
for label in labels:
    if image_counts[label]['skipped'] == -1:
        print(f"{label:6} | {'N/A':12} | {'N/A':12} | {'Directory missing':15}")
    else:
        print(f"{label:6} | {image_counts[label]['valid']:12} | {image_counts[label]['total']:12} | {image_counts[label]['skipped']:15}")

# Check if data was loaded
if len(X) == 0 or len(y) == 0:
    print("Error: No data loaded. Please check the dataset path and ensure images are present.")
    exit()

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# Build CNN model with adjusted input shape (64 features = 63 landmarks + 1 angle)
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation=mish, input_shape=(8, 8, 1), padding='same'),
    tf.keras.layers.Lambda(mixed_pooling),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Conv2D(64, (3, 3), activation=mish, padding='same'),
    tf.keras.layers.Lambda(mixed_pooling),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Conv2D(128, (3, 3), activation=mish, padding='same'),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation=mish),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(labels), activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Training parameters
batch_size = 32
steps_per_epoch = len(X_train) // batch_size
validation_steps = len(X_val) // batch_size

# Create data generators
train_generator = data_generator(X_train, y_train, batch_size, augment=True)
val_generator = data_generator(X_val, y_val, batch_size, augment=False)

# Train model
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=20,
    validation_data=val_generator,
    validation_steps=validation_steps
)

# Evaluate model
X_test_reshaped = X_test.reshape(-1, 8, 8, 1)
test_loss, test_acc = model.evaluate(X_test_reshaped, y_test)
print(f"Test accuracy: {test_acc}")

# Create models directory if it doesn't exist
os.makedirs('../models', exist_ok=True)

# Save model in native Keras format
model.save('models/asl_model_augmented.keras')

# Create outputs directory if it doesn't exist
os.makedirs('../outputs', exist_ok=True)

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')

plt.savefig('outputs/training_history_augmented.png')

hands.close()
print("Training completed successfully!")
print("Image count summary saved to ../outputs/image_counts.txt")
print("Model saved to ../models/asl_model_augmented.keras")
print("Training history plot saved to ../outputs/training_history_augmented.png")