import cv2
import mediapipe as mp
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# 1. Data Loading and Preprocessing
data_dir = "sign_language_data"  # Or your absolute path
signs = ["hello", "home", "i love you"]  # Updated vocabulary
print(f"Signs list: {signs}")

data = []
labels = []
img_size = 64  # Adjust as needed

try:
    for sign in signs:
        print(f"Processing sign: {sign}")
        for folder_count_str in os.listdir(os.path.join(data_dir, sign)):  # Iterate through folders
            if not folder_count_str.startswith("folder_"):
                continue
            try:
                folder_count = int(folder_count_str[len("folder_"):])  # Extract folder number
            except ValueError:
                print(f"Skipping invalid folder name: {folder_count_str}")
                continue
            folder_path = os.path.join(data_dir, sign, folder_count_str)
            print(f"  Processing folder: {folder_path}")
            for filename in os.listdir(folder_path):
                if not filename.endswith(".jpg"):
                    continue
                image_path = os.path.join(folder_path, filename)
                print(f"    Found image: {image_path}")
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"      ERROR: Could not load image: {image_path}")
                    continue
                img_resized = cv2.resize(img, (img_size, img_size))
                data.append(img_resized)
                labels.append(signs.index(sign))

except Exception as e:
    print(f"An error occurred during data loading: {e}")
    exit()  # Exit if data loading fails

data = np.array(data)
labels = np.array(labels)

if len(data) == 0:  # Check if any data was loaded
    print("Error: No data loaded. Check your data directory and file names.")
    exit()

# Normalize pixel values
data = data / 255.0

# Reshape for CNN (add channel dimension)
data = data.reshape(-1, img_size, img_size, 1)

# 2. Data Splitting
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# 3. Model Building (CNN)
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(signs), activation='softmax')  # Output layer size adjusted
])

# 4. Model Compilation
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 5. Model Training
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# 6. Model Evaluation
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

# 7. Model Saving
model.save("sign_language_model.h5")
print("Model saved as sign_language_model.h5")