# Real-Time Sign Character Recognition

A real-time Sign Language character recognition system using computer vision and deep learning to detect hand gestures and convert them to text with English and Bengali text-to-speech output.

## Features

- Real-time sign languaage recognition (A-Z + space + backspace)
- Custom CNN with MediaPipe hand landmark detection
- Phrase formation with visual feedback
- Multilingual TTS (English and Bengali)
- Custom dataset trained model

## Sample Video

https://drive.google.com/file/d/17KE8h6T_9mtK2m3Ekbdw5Hex8W4NUhXS/view?usp=sharing

## Installation

```bash
pip install opencv-python mediapipe numpy tensorflow scikit-learn matplotlib pyttsx3 deep-translator gtts pygame
```

## Dataset Structure

```
data/asl_alphabet_train/asl_alphabet_train/
├── A/ (your custom gesture images)
├── B/
├── ...
├── Z/
├── space/
└── backspace/
```

**Note**: This project uses a custom dataset created with personal gesture images, not open-source data.

## Usage

### 1. Train Model
```bash
python train_model.py
```
- Processes your custom gesture images
- Creates CNN model with hand landmark features
- Saves model to `models/asl_model_with_space_backspace.keras`

### 2. Run Real-time Detection
```bash
python real_time_detection.py
```
- Press **ESC** to exit
- Automatic phrase formation and TTS output

## Model Architecture

- **Input**: 8x8x1 (processed hand landmarks from MediaPipe)
- **Features**: 21 hand landmarks + orientation angle
- **Architecture**: Custom CNN with Mish activation and mixed pooling
- **Output**: 28 classes (A-Z + space + backspace)

## Key Components

- **Hand Processing**: MediaPipe landmark detection with custom angle computation
- **Stability**: Frame buffering for consistent predictions
- **TTS**: pyttsx3 (English) + gTTS (Bengali)
- **Augmentation**: Translation, scaling, rotation during training

## Configuration

```python
# Training
max_images_per_letter = 1000
batch_size = 32
epochs = 20

# Detection
buffer_size = 3        # Frames for stable prediction
hold_frames = 30       # Hold duration
```

## File Structure

```
├── train_model.py              # Training script
├── real_time_detection.py      # Detection app
├── models/                     # Trained model
├── outputs/                    # Training logs/plots
├── data/                       # Your custom dataset
└── README.md
```

## Troubleshooting

- **Camera issues**: Check permissions and camera index
- **Model errors**: Verify model file exists and dependencies installed
- **Poor accuracy**: Ensure good lighting and clear hand positioning
- **TTS issues**: Check internet connection for Bengali TTS

## Technical Details

- **Custom Mish activation** for better gradient flow
- **Mixed pooling** combining max and average pooling
- **Hand angle computation** for gesture orientation
- **Real-time stability** through prediction buffering

---

**Custom Dataset**: This project uses personally created gesture images for training, providing tailored recognition for the specific gesture style used.
