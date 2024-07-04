import cv2
import mediapipe as mp
import time
import numpy as np
from keras.models import load_model

# Load the model
model_1 = load_model("/home/kushagra/Documents/code/AI/project/asl_recognisation/asl_detection_model.h5")

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

# Open the camera
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

# Initialize timing variables
ptime = 0
prediction_timer = time.time()
text_output = []

# White screen for displaying recognized text
white_screen = np.ones((300, 800, 3), np.uint8) * 255

while True:
    success, img = cap.read()
    if not success:
        print("Error: Failed to capture image.")
        break

    # Flip and rotate the image for correct orientation
    img = cv2.flip(img, 1)
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process the image and find hand landmarks
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

            # Draw bounding box around the hand
            cv2.rectangle(img, (bbox[0] - 30, bbox[1] - 30), (bbox[2] + 30, bbox[3] + 30), (255, 0, 255), 2)

            # Perform prediction every 2 seconds
            if time.time() - prediction_timer >= 2:
                hand_img = img[bbox[1] - 30:bbox[3] + 30, bbox[0] - 30:bbox[2] + 30]
                hand_img = cv2.resize(hand_img, (64, 64))
                hand_img = np.expand_dims(hand_img, axis=0)
                hand_img = hand_img.astype('float32') / 255.0

                # Predict the sign
                result = model_1.predict(hand_img)
                result = np.argmax(result)

                if result in asl_dict:
                    result_text = asl_dict[result]
                    print(result_text)
                    if result_text == 'nothing':
                        continue
                    elif result_text == 'space':
                        text_output.append(' ')
                    elif result_text == 'del':
                        text_output = text_output[:-1]
                    else:
                        text_output.append(result_text)

                    # Update the white screen with recognized text
                    white_screen = np.ones((300, 800, 3), np.uint8) * 255
                    cv2.putText(white_screen, ''.join(text_output), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 3)
                    print(text_output)
                prediction_timer = time.time()

    # Calculate and display FPS
    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = ctime
    cv2.putText(img, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    # Display the image and predictions
    cv2.imshow("Image", img)
    cv2.imshow("Predictions", white_screen)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
