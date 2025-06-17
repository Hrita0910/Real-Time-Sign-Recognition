import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Define lip landmark indices (Mediapipe provides 468 landmarks, lips are specific indices)
LIP_OUTER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
LIP_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Error: Could not read frame.")
        break

    # Convert BGR to RGB (Mediapipe requires RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_rgb.flags.writeable = False  # Prevent unintended modifications
    results = face_mesh.process(frame_rgb)
    frame_rgb.flags.writeable = True
    frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape

            # Extract lip landmarks
            lip_outer_points = []
            lip_inner_points = []
            for idx in LIP_OUTER:
                x = int(face_landmarks.landmark[idx].x * w)
                y = int(face_landmarks.landmark[idx].y * h)
                lip_outer_points.append((x, y))
            for idx in LIP_INNER:
                x = int(face_landmarks.landmark[idx].x * w)
                y = int(face_landmarks.landmark[idx].y * h)
                lip_inner_points.append((x, y))

            # Convert points to numpy arrays for OpenCV
            lip_outer_points = np.array(lip_outer_points, np.int32)
            lip_inner_points = np.array(lip_inner_points, np.int32)

            # Draw lip contours
            cv2.polylines(frame, [lip_outer_points], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.polylines(frame, [lip_inner_points], isClosed=True, color=(0, 0, 255), thickness=2)

            # Optional: Fill the lip region (between outer and inner lips)
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [lip_outer_points], 255)
            cv2.fillPoly(mask, [lip_inner_points], 0)
            frame[mask == 255] = frame[mask == 255] * 0.5 + np.array([255, 0, 0]) * 0.5

    # Display the frame
    cv2.imshow('Real-Time Lip Detection', frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
face_mesh.close()