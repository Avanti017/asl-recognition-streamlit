import os
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2

class TestProcessor(VideoProcessorBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

st.title("Camera Test")
webrtc_streamer(
    key="test",
    video_processor_factory=TestProcessor,
    media_stream_constraints={"video": True, "audio": False},
)
