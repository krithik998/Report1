import cv2
import os

signs = ["hello", "home", "i love you"]  # Updated vocabulary
data_dir = "sign_language_data"

if not os.path.exists(data_dir):
    os.makedirs(data_dir)
    for sign in signs:
        os.makedirs(os.path.join(data_dir, sign))

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

for sign in signs:
    count = 0
    folder_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error reading frame. Check your webcam.")
            break

        cv2.putText(frame, f"Collecting data for: {sign} (Folder {folder_count})", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Collecting Data", frame)

        key = cv2.waitKey(1)

        if key == ord('s'):
            image_path = os.path.join(data_dir, sign, f"folder_{folder_count}", f"{sign}_{count}.jpg")
            os.makedirs(os.path.dirname(image_path), exist_ok=True)
            cv2.imwrite(image_path, frame)
            count += 1
            print(f"Captured image {count} for {sign} in folder {folder_count}")
        elif key == ord('q'):
            folder_count += 1
            count = 0
            print(f"Moving to next folder: {folder_count}")
        elif key == ord('x'):
            break

    cv2.destroyAllWindows()
    if key == ord('x'):
        continue
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Data collection complete.")