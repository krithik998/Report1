import cv2
import mediapipe as mp
import os

# ... (Your data collection code from before goes here) ...

# After data collection is complete:
print("Data collection complete. Starting hand detection.")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils  # Import drawing_utils correctly
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam for hand detection.")
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
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)  # Use mp_drawing here

            # Accessing landmark coordinates (unchanged):
            for landmark in hand_landmarks.landmark:
                x_px = int(landmark.x * frame.shape[1])
                y_px = int(landmark.y * frame.shape[0])
                #print(f"Landmark: x={x_px}, y={y_px}")

    cv2.imshow('Hand Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

hands.close()
cap.release()
cv2.destroyAllWindows()
print("Hand detection stopped.")