import cv2
import mediapipe as mp
import time
import streamlit as st

# Helper functions
def is_thumb_index_finger_touching(landmarks):
    """Check if the thumb and index finger are touching."""
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    distance = ((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2) ** 0.5
    return distance < 0.04

def is_thumb_pinky_finger_touching(landmarks):
    """Check if the thumb and pinky finger are touching."""
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP]
    distance = ((thumb_tip.x - pinky_tip.x) ** 2 + (thumb_tip.y - pinky_tip.y) ** 2) ** 0.5
    return distance < 0.03

# Streamlit app title
st.title("Hand Gesture Control App")

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Set desired FPS
desired_fps = 30
frame_duration = 1.0 / desired_fps

# State to track if thumb-index detection is active
thumb_index_active = True
toggle_delay = 0.5  # Delay to avoid repeated toggling
last_toggle_time = time.time()

# Note to cloud users
st.info("This app is optimized for local environments. In a cloud environment, live video feed and certain actions may be restricted.")

# Webcam functionality replacement
uploaded_file = st.file_uploader("Upload an image of your hand", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Load and process the uploaded image
    file_bytes = uploaded_file.read()
    frame = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
    
    # Flip and process the frame
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw the hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Check thumb-pinky touch for toggling
            if time.time() - last_toggle_time > toggle_delay:
                if is_thumb_pinky_finger_touching(hand_landmarks.landmark):
                    thumb_index_active = not thumb_index_active
                    st.write(f"Thumb-index detection {'activated' if thumb_index_active else 'deactivated'}")
                    last_toggle_time = time.time()
            
            # Simulate spacebar action as a log message
            if thumb_index_active and is_thumb_index_finger_touching(hand_landmarks.landmark):
                st.write("Simulated spacebar press!")  # Logs instead of using `pyautogui`

    # Display the processed image with landmarks
    st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Processed Image with Hand Landmarks")
