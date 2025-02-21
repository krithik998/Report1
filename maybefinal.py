import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf  # Import TensorFlow

# Load the trained model
model = tf.keras.models.load_model("sign_language_model.h5")  # Load your saved model

# MediaPipe Hands setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Sign labels (must match the order used during training)
signs = ["hello", "home", "i love you"]  # Updated vocabulary

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error reading frame.")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Preprocess the hand landmarks for prediction (if you trained with landmarks)
            # hand_landmarks_array = []
            # for landmark in hand_landmarks.landmark:
            #     hand_landmarks_array.append([landmark.x, landmark.y, landmark.z])  # Use normalized coordinates
            # hand_landmarks_array = np.array(hand_landmarks_array).flatten()  # Flatten the array

            # Image Preprocessing (if you trained with image data, which is more common)
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            img_resized = cv2.resize(img, (64, 64))  # Same size as during training
            img_normalized = img_resized / 255.0  # Normalize pixel values
            img_reshaped = img_normalized.reshape(1, 64, 64, 1)  # Reshape for CNN

            # Make prediction
            prediction = model.predict(img_reshaped)  # Predict using the model

            predicted_class = np.argmax(prediction)  # Get the predicted class index
            predicted_sign = signs[predicted_class]  # Get the sign name

            # Display the prediction on the frame
            cv2.putText(frame, predicted_sign, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Sign Language Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

hands.close()
cap.release()
cv2.destroyAllWindows()
print("Sign language recognition stopped.")