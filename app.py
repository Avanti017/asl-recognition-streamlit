import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import deque 
from PIL import Image

st.title("ASL Recognition Demo")

model = joblib.load("asl_model.pkl")
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode = True, 
    max_num_hands = 1,
    min_detection_confidence = 0.7
)

mp_draw = mp.solutions.drawing_utils

prediction_buffer = deque(maxlen=7)

img_file_buffer = st.camera_input("Show your hand")

output_text = st.session_state.get("output_text", "")

if img_file_buffer:
    image = Image.open(img_file_buffer)
    frame = np.array(image)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        handLms = results.multi_hand_landmarks[0]

        landmarks = []

    for lm in handLms.landmark:
        landmarks.extend([lm.x, lm.y, lm.z])
    X = np.array(landmarks).reshape(1,-1)
    prediction = model.predict(X)[0]
    prediction_buffer.append[prediction]
    
    final_prediction = max(set(prediction_buffer),
                            key=prediction_buffer.count)
    output_text += final_prediction
    output_text = output_text[-19:]
    st.session_state["output_text"] = output_text

    st.subheader(f"Text: {output_text}")


