import streamlit as st
import cv2
from mediapipe.solutions import hands, drawing_utils
import numpy as np
from keras.models import load_model
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, VideoTransformerContext
import av
import time

# Load the model
model_1 = load_model("asl_detection_model.h5")

# ASL dictionary
asl_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
    19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'del', 27: 'nothing', 28: 'space'
}

# Initialize mediapipe hands
mpHands = hands
hands = mpHands.Hands()
mpDraw = drawing_utils

# Timing variables
prediction_timer = time.time()
text_output = []

# White screen for displaying recognized text
white_screen = np.ones((300, 800, 3), np.uint8) * 255

def draw_text_on_white_screen(white_screen, text_output):
    max_words_per_line = 30
    lines = [' '.join(text_output[i:i + max_words_per_line]) for i in range(0, len(text_output), max_words_per_line)]
    for idx, line in enumerate(lines):
        y_position = 70 + idx * 30  # Adjust the y position for each line
        cv2.putText(white_screen, line, (10, y_position), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 3)

class ASLVideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.prediction_timer = time.time()
        self.text_output = []
        self.white_screen = np.ones((300, 800, 3), np.uint8) * 255
        self.ptime = 0

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                h, w, c = img.shape
                bbox = [w, h, 0, 0]  # Initialize bounding box coordinates
                for lm in handLms.landmark:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    bbox[0] = min(bbox[0], cx)
                    bbox[1] = min(bbox[1], cy)
                    bbox[2] = max(bbox[2], cx)
                    bbox[3] = max(bbox[3], cy)

                cv2.rectangle(img, (bbox[0] - 30, bbox[1] - 30), (bbox[2] + 30, bbox[3] + 30), (255, 0, 255), 2)

                if time.time() - self.prediction_timer >= 2:
                    hand_img = img[bbox[1] - 30:bbox[3] + 30, bbox[0] - 30:bbox[2] + 30]
                    hand_img = cv2.resize(hand_img, (64, 64))
                    hand_img = np.expand_dims(hand_img, axis=0)
                    hand_img = hand_img.astype('float32') / 255.0

                    result = model_1.predict(hand_img)
                    result = np.argmax(result)

                    if result in asl_dict:
                        result_text = asl_dict[result]
                        if result_text == 'nothing':
                            continue
                        elif result_text == 'space':
                            self.text_output.append(' ')
                        elif result_text == 'del':
                            self.text_output = self.text_output[:-1]
                        else:
                            self.text_output.append(result_text)

                        self.white_screen = np.ones((300, 800, 3), np.uint8) * 255
                        draw_text_on_white_screen(self.white_screen, self.text_output)
                    self.prediction_timer = time.time()

        ctime = time.time()
        fps = 1 / (ctime - self.ptime)
        self.ptime = ctime
        cv2.putText(img, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

def main():
    st.title("ASL Recognition with Streamlit")

    webrtc_ctx = webrtc_streamer(key="example", video_transformer_factory=ASLVideoTransformer)

    if webrtc_ctx.video_transformer:
        stframe2 = st.empty()
        while webrtc_ctx.state.playing:
            white_screen_rgb = cv2.cvtColor(webrtc_ctx.video_transformer.white_screen, cv2.COLOR_BGR2RGB)
            stframe2.image(white_screen_rgb, channels='RGB')

if __name__ == "__main__":
    main()
