import cv2
import mediapipe as mp
import numpy as np
import time
import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

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

# State to track if thumb-index detection is active
thumb_index_active = True
toggle_delay = 0.5  # Delay to avoid repeated toggling
last_toggle_time = time.time()

# Define a video transformer using streamlit-webrtc
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.last_toggle_time = time.time()
        self.thumb_index_active = True

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Flip and process the frame
        img = cv2.flip(img, 1)
        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Draw the hand landmarks
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Check thumb-pinky touch for toggling
                if time.time() - self.last_toggle_time > toggle_delay:
                    if is_thumb_pinky_finger_touching(hand_landmarks.landmark):
                        self.thumb_index_active = not self.thumb_index_active
                        st.write(f"Thumb-index detection {'activated' if self.thumb_index_active else 'deactivated'}")
                        self.last_toggle_time = time.time()

                # Simulate spacebar action as a log message
                if self.thumb_index_active and is_thumb_index_finger_touching(hand_landmarks.landmark):
                    st.write("Simulated spacebar press!")  # Logs instead of using `pyautogui`

        return img

# Start the webcam stream using webrtc_streamer
webrtc_streamer(key="gesture-control", video_transformer_factory=VideoTransformer)
