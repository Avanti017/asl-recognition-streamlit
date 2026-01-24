import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import deque
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

# ================== CONFIG ==================
STABLE_FRAMES_REQUIRED = 10
NO_HAND_FRAMES_REQUIRED = 10
BACKSPACE_FRAMES_REQUIRED = 10
MAX_CHARS = 19

model = joblib.load("asl_model.pkl")

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# ================== HELPERS ==================
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
    if spread < 0.12:
        return False

    return True


# ================== VIDEO PROCESSOR ==================
class ASLProcessor(VideoProcessorBase):
    def __init__(self):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        )

        self.prediction_buffer = deque(maxlen=7)
        self.last_prediction = None
        self.stable_frames = 0
        self.letter_locked = False

        self.no_hand_frames = 0
        self.space_locked = False

        self.backspace_frames = 0
        self.backspace_locked = False

        self.output_text = ""

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = self.hands.process(imgRGB)

        if results.multi_hand_landmarks:
            self.no_hand_frames = 0
            self.space_locked = False

            handLms = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(
                img, handLms, mp_hands.HAND_CONNECTIONS
            )

            landmarks = []
            for lm in handLms.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            X = np.array(landmarks).reshape(1, -1)

            if is_open_hand(handLms):
                self.backspace_frames += 1
                if (
                    self.backspace_frames >= BACKSPACE_FRAMES_REQUIRED
                    and not self.backspace_locked
                ):
                    self.output_text = self.output_text[:-1]
                    self.backspace_locked = True
                    self.backspace_frames = 0
                    self.prediction_buffer.clear()
            else:
                self.backspace_frames = 0
                self.backspace_locked = False

                prediction = model.predict(X)[0]
                self.prediction_buffer.append(prediction)

                final_prediction = max(
                    set(self.prediction_buffer),
                    key=self.prediction_buffer.count,
                )

                if final_prediction == self.last_prediction:
                    self.stable_frames += 1
                else:
                    self.stable_frames = 0
                    self.letter_locked = False

                self.last_prediction = final_prediction

                if (
                    self.stable_frames >= STABLE_FRAMES_REQUIRED
                    and not self.letter_locked
                ):
                    self.output_text += final_prediction
                    self.output_text = self.output_text[-MAX_CHARS:]
                    self.letter_locked = True

        else:
            self.no_hand_frames += 1
            self.stable_frames = 0
            self.letter_locked = False
            self.last_prediction = None
            self.prediction_buffer.clear()
            self.backspace_frames = 0
            self.backspace_locked = False

            if (
                self.no_hand_frames >= NO_HAND_FRAMES_REQUIRED
                and not self.space_locked
            ):
                if len(self.output_text) > 0 and self.output_text[-1] != " ":
                    self.output_text += " "
                    self.output_text = self.output_text[-MAX_CHARS:]
                self.space_locked = True

        cv2.putText(
            img,
            f"Text: {self.output_text}",
            (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255, 0, 0),
            2,
        )

        st.session_state["output_text"] = self.output_text
        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ================== STREAMLIT UI ==================
st.set_page_config(page_title="ASL Alphabet Recognition", layout="centered")
st.title("🤟 ASL Alphabet Recognition")

if "output_text" not in st.session_state:
    st.session_state["output_text"] = ""

webrtc_streamer(
    key="asl",
    video_processor_factory=ASLProcessor,
    media_stream_constraints={"video": True, "audio": False},
)

st.markdown("### 📝 Recognised Text")
st.text_area(
    "Output",
    st.session_state["output_text"],
    height=100,
)
