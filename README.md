ASL Hand Sign to Text Translator

This project uses a Convolutional Neural Network (CNN) integrated with Long Short-Term Memory (LSTM) to translate American Sign Language (ASL) hand signs into text. The application utilizes OpenCV and MediaPipe to capture and process video input, detecting hand landmarks to predict ASL signs in real-time.
Table of Contents

    Introduction
    Features
    Setup
    Usage
    Model Training
    Model Architecture
    Demo
    Contributing
    License

Introduction

The goal of this project is to provide a tool for translating ASL hand signs into text. It uses a pre-trained model to recognize various hand gestures and converts them into corresponding text in real-time.
Features

    Real-time ASL hand sign recognition using OpenCV and MediaPipe.
    Pre-trained CNN-LSTM model for accurate hand sign prediction.
    Displays translated text on the screen.
    Supports 26 ASL signs corresponding to the English alphabet, as well as 'space', 'delete', and 'nothing'.

Setup
Requirements

    Python 3.7+
    OpenCV
    TensorFlow
    Keras
    MediaPipe
    NumPy

Installation

    Clone the repository:

    sh

git clone https://github.com/yourusername/asl-hand-sign-to-text.git
cd asl-hand-sign-to-text

Install the required packages:

sh

pip install -r requirements.txt

Download the pre-trained model and place it in the appropriate directory:

sh

    # Ensure you have the model at the specified path
    /home/kushagra/Documents/code/AI/project/asl_recognisation/asl_detection_model.h5

Usage

    Run the script to start the application:

    sh

    python asl_to_text.py

    Use your webcam to show ASL signs. The application will display the recognized sign and translate it into text on the screen.

    Press 'q' to quit the application.

Code Overview

python

import cv2
import mediapipe as mp
import numpy as np
from keras.models import load_model
import time

# Initialize camera
cap = cv2.VideoCapture(2)

# Initialize MediaPipe Hands
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

# Load pre-trained model
model_1 = load_model("/home/kushagra/Documents/code/AI/project/asl_recognisation/asl_detection_model.h5")

# ASL dictionary mapping
asl_dict = {0: 'A', 1: 'B', 2: 'C', ...}

# Timing and text variables
ptime = 0
prediction_timer = time.time()
a = []
white_screen = np.ones((300, 800, 3), np.uint8) * 255

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

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

                result = model_1.predict(hand_img)
                result = np.argmax(result)

                if result in asl_dict:
                    result_text = asl_dict[result]
                    if result_text == 'nothing':
                        break
                    elif result_text == 'space':
                        a.append(' ')
                    elif result_text == 'del':
                        white_screen = np.ones((300, 800, 3), np.uint8) * 255
                        a = a[:-1]
                    else:
                        a.append(result_text)

                    cv2.putText(white_screen, ''.join(a), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 3)

                prediction_timer = time.time()

    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = ctime
    cv2.putText(img, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("Image", img)
    cv2.imshow("Predictions", white_screen)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

Model Training

If you wish to train the model yourself, refer to the following steps:

    Data Collection: Gather a dataset of ASL hand signs. Ensure the images are labeled correctly.

    Data Preprocessing: Resize images, normalize pixel values, and split the dataset into training and testing sets.

    Model Architecture: Define a CNN-LSTM model using TensorFlow/Keras. Example:

    python

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Reshape, LSTM, Dense

model = Sequential()

model.add(Conv2D(filters=64, kernel_size=5, padding='same', activation='relu', input_shape=(64, 64, 3)))
model.add(Conv2D(filters=64, kernel_size=5, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(4, 4)))
model.add(Dropout(0.5))
model.add(Conv2D(filters=128, kernel_size=5, padding='same', activation='relu'))
model.add(Conv2D(filters=128, kernel_size=5, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(4, 4)))
model.add(Dropout(0.5))
model.add(Conv2D(filters=256, kernel_size=5, padding='same', activation='relu'))
model.add(Dropout(0.5))
model.add(Flatten())

model.add(Reshape((1, 256 * 4 * 4)))  

model.add(LSTM(128, dropout=0.5, return_sequences=False))
model.add(Dropout(0.5))

model.add(Dense(29, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

Training: Train the model with your dataset.

python

model.fit(train_images, train_labels, epochs=10, validation_split=0.2)

Saving the Model: Save the trained model:

python

    model.save('asl_detection_model.h5')

Model Architecture

The model architecture is designed to capture spatial features with CNN layers and temporal dependencies with an LSTM layer. Here is a summary of the architecture:

    Convolutional Layers: Extract spatial features from the input images.
    MaxPooling Layers: Reduce spatial dimensions and retain important features.
    Dropout Layers: Prevent overfitting.
    Flatten Layer: Convert 3D feature maps to 1D feature vectors.
    Reshape Layer: Prepare data for LSTM layer.
    LSTM Layer: Capture temporal dependencies in the sequence of feature vectors.
    Dense Layer: Output the probability distribution over the 29 classes (26 letters, 'space', 'delete', 'nothing').

Demo

A demo video showcasing the application's functionality can be found here.
[![ASL Hand Sign to Text Translator]](https://github.com/kushagra8881/asl_recognisation/blob/main/asl.mp4)
Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any improvements or bug fixes.
