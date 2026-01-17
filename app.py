import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import deque

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

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

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="ASL Live Recognition", layout="centered")
st.title("✋ ASL Live Recognition (Webcam)")

# ---------------- LOAD MODEL ----------------
model = joblib.load("asl_model.pkl")

# ---------------- MEDIAPIPE SETUP ----------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
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


# ---------------- VIDEO PROCESSOR ----------------
class ASLProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(imgRGB)

        if results.multi_hand_landmarks:
            st.session_state.no_hand_frames = 0
            st.session_state.space_locked = False

            handLms = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

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

                prediction = model.predict(X)[0]
                st.session_state.prediction_buffer.append(prediction)

                final_prediction = max(
                    set(st.session_state.prediction_buffer),
                    key=st.session_state.prediction_buffer.count,
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
            # ---------- NO HAND → SPACE ----------
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

        # ---------- DRAW TEXT ----------
        cv2.putText(
            img,
            f"Text: {st.session_state.output_text}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (255, 0, 0),
            2,
        )

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ---------------- START WEBRTC ----------------
webrtc_streamer(
    key="asl-live",
    video_processor_factory=ASLProcessor,
    media_stream_constraints={"video": True, "audio": False},
)

# ---------------- OUTPUT TEXT ----------------
st.subheader("Recognized Text")
st.write(st.session_state.output_text)

if st.button("Clear text"):
    st.session_state.output_text = ""
