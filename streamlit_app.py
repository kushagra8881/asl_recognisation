import cv2
import streamlit as st
import mediapipe as mp
import time
import numpy as np
from keras.models import load_model
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av

# RTC Configuration for WebRTC
RTC_CONFIGURATION = {
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}],
    "media_stream_constraints": {"audio": False, "video": True},
}

# Load the ASL model
model_1 = load_model("asl_detection_model.h5")

# ASL dictionary
asl_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
    19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'del', 27: 'nothing', 28: 'space'
}

# Initialize Mediapipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Initialize timing variables
prediction_timer = time.time()
text_output = []

class ASLVideoProcessor(VideoProcessorBase):
    def __init__(self):
        super().__init__()

    def recv(self, frame):
        global prediction_timer, text_output

        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                h, w, c = img.shape
                bbox = [w, h, 0, 0]

                for lm in handLms.landmark:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    bbox[0] = min(bbox[0], cx)
                    bbox[1] = min(bbox[1], cy)
                    bbox[2] = max(bbox[2], cx)
                    bbox[3] = max(bbox[3], cy)

                cv2.rectangle(img, (bbox[0] - 30, bbox[1] - 30), (bbox[2] + 30, bbox[3] + 30), (255, 0, 255), 2)

                if time.time() - prediction_timer >= 2:
                    hand_img = img[bbox[1] - 30:bbox[3] + 30, bbox[0] - 30:bbox[2] + 30]
                    hand_img = cv2.resize(hand_img, (64, 64))
                    hand_img = np.expand_dims(hand_img, axis=0)
                    hand_img = hand_img.astype('float32') / 255.0

                    # Predict the sign
                    result = model_1.predict(hand_img)
                    result = np.argmax(result)
                    print(result)
                    if result in asl_dict:
                        result_text = asl_dict[result]
                        if result_text == 'nothing':
                            text_output = []
                        elif result_text == 'space':
                            text_output.append(' ')
                        elif result_text == 'del':
                            if text_output:
                                text_output = text_output[:-1]
                        else:
                            text_output.append(result_text)

                        prediction_timer = time.time()

        # Draw text on the main image
        cv2.putText(img, ' '.join(text_output), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 3)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

def main():
    st.title("ASL Recognition with Streamlit and WebRTC")

    webrtc_streamer(
        key="example",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=ASLVideoProcessor,
        async_processing=True,
    )
    st.image("model.png", use_column_width=True)
    st.image("image_1.png", use_column_width=True)
if __name__ == "__main__":
    main()
