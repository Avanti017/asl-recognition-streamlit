import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import deque
from PIL import Image

# ---------------- CONSTANTS ----------------
STABLE_FRAMES_REQUIRED = 10
NO_HAND_FRAMES_REQUIRED = 10
BACKSPACE_FRAMES_REQUIRED = 10
MAX_CHARS = 19

# ---------------- SESSION STATE INIT ----------------
defaults = {
    "output_text": "",
    "last_prediction": None,
    "stable_frames": 0,
    "letter_locked": False,
    "no_hand_frames": 0,
    "space_locked": False,
    "backspace_frames": 0,
    "backspace_locked": False,
    "prediction_buffer": deque(maxlen=7),
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---------------- SETUP ----------------
st.title("ASL Recognition Demo")

model = joblib.load("asl_model.pkl")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.7
)

# ---------------- HELPERS ----------------
def is_open_hand(handLms):
    tips = [4, 8, 12, 16, 20]
    pips = [3, 6, 10, 14, 18]

    for tip, pip in zip(tips, pips):
        if handLms.landmark[tip].y > handLms.landmark[pip].y:
            return False

    thumb_tip = handLms.landmark[4]
    index_mcp = handLms.landmark[5]

    if abs(thumb_tip.x - index_mcp.x) < 0.04:
        return False

    spread = abs(handLms.landmark[8].x - handLms.landmark[20].x)
    return spread >= 0.12

# ---------------- CAMERA INPUT ----------------
img_file_buffer = st.camera_input("Show your hand")

if img_file_buffer:
    image = Image.open(img_file_buffer)
    frame = np.array(image)
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_RGB2RGB)

    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        st.session_state.no_hand_frames = 0
        st.session_state.space_locked = False

        handLms = results.multi_hand_landmarks[0]

        # extract landmarks
        landmarks = []
        for lm in handLms.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
        X = np.array(landmarks).reshape(1, -1)

        # ---------- BACKSPACE ----------
        if is_open_hand(handLms):
            st.session_state.backspace_frames += 1

            if (
                st.session_state.backspace_frames >= BACKSPACE_FRAMES_REQUIRED
                and not st.session_state.backspace_locked
            ):
                st.session_state.output_text = st.session_state.output_text[:-1]
                st.session_state.backspace_locked = True
                st.session_state.backspace_frames = 0
                st.session_state.prediction_buffer.clear()

        else:
            st.session_state.backspace_frames = 0
            st.session_state.backspace_locked = False

            # ---------- LETTER PREDICTION ----------
            prediction = model.predict(X)[0]
            st.session_state.prediction_buffer.append(prediction)

            final_prediction = max(
                set(st.session_state.prediction_buffer),
                key=st.session_state.prediction_buffer.count
            )

            if final_prediction == st.session_state.last_prediction:
                st.session_state.stable_frames += 1
            else:
                st.session_state.stable_frames = 0
                st.session_state.letter_locked = False

            st.session_state.last_prediction = final_prediction

            if (
                st.session_state.stable_frames >= STABLE_FRAMES_REQUIRED
                and not st.session_state.letter_locked
            ):
                st.session_state.output_text += final_prediction
                st.session_state.output_text = st.session_state.output_text[-MAX_CHARS:]
                st.session_state.letter_locked = True

    else:
        # ---------- NO HAND (SPACE) ----------
        st.session_state.no_hand_frames += 1
        st.session_state.stable_frames = 0
        st.session_state.letter_locked = False
        st.session_state.last_prediction = None
        st.session_state.prediction_buffer.clear()
        st.session_state.backspace_frames = 0
        st.session_state.backspace_locked = False

        if (
            st.session_state.no_hand_frames >= NO_HAND_FRAMES_REQUIRED
            and not st.session_state.space_locked
        ):
            if (
                len(st.session_state.output_text) > 0
                and st.session_state.output_text[-1] != " "
            ):
                st.session_state.output_text += " "
                st.session_state.output_text = st.session_state.output_text[-MAX_CHARS:]

            st.session_state.space_locked = True

# ---------------- OUTPUT ----------------
st.subheader(f"Text: {st.session_state.output_text}")
