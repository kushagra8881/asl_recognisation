import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from keras.models import load_model
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import time
from PIL import Image, ImageDraw, ImageFont

# Load the model
model_1 = load_model("asl_detection_model.h5")

# ASL dictionary
asl_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
    19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'del', 27: 'nothing', 28: 'space'
}

# Initialize mediapipe hands
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

# Timing variables
prediction_timer = time.time()
text_output = []

# White screen for displaying recognized text
white_screen = Image.new("RGB", (800, 300), "white")
font = ImageFont.load_default()

def draw_text_on_white_screen(white_screen, text_output):
    max_words_per_line = 30
    lines = [' '.join(text_output[i:i + max_words_per_line]) for i in range(0, len(text_output), max_words_per_line)]
    draw = ImageDraw.Draw(white_screen)
    for idx, line in enumerate(lines):
        y_position = 70 + idx * 30  # Adjust the y position for each line
        draw.text((10, y_position), line, fill="black", font=font)

class ASLVideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.prediction_timer = time.time()
        self.text_output = []
        self.white_screen = Image.new("RGB", (800, 300), "white")
        self.ptime = 0

    def transform(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        imgRGB = np.flip(img, axis=2)  # Convert BGR to RGB
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

                if time.time() - self.prediction_timer >= 2:
                    hand_img = img[bbox[1] - 30:bbox[3] + 30, bbox[0] - 30:bbox[2] + 30]
                    hand_img = Image.fromarray(hand_img)
                    hand_img = hand_img.resize((64, 64))
                    hand_img = np.expand_dims(hand_img, axis=0)
                    hand_img = np.array(hand_img).astype('float32') / 255.0

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

                        self.white_screen = Image.new("RGB", (800, 300), "white")
                        draw_text_on_white_screen(self.white_screen, self.text_output)
                    self.prediction_timer = time.time()

        ctime = time.time()
        fps = 1 / (ctime - self.ptime)
        self.ptime = ctime
        fps_text = f'FPS: {int(fps)}'
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)
        draw.text((10, 70), fps_text, fill="magenta", font=font)
        img = np.array(img_pil)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

def main():
    st.title("ASL Recognition with Streamlit")

    webrtc_ctx = webrtc_streamer(key="example", video_transformer_factory=ASLVideoTransformer)

    if webrtc_ctx.video_transformer:
        stframe2 = st.empty()
        while webrtc_ctx.state.playing:
            white_screen_rgb = np.array(webrtc_ctx.video_transformer.white_screen)
            stframe2.image(white_screen_rgb)

if __name__ == "__main__":
    main()
