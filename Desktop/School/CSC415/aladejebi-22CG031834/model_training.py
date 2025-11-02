import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

# Path to your dataset folders
data_dir = "dataset/train"
categories = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# Prepare data and labels
data = []
labels = []

for category in categories:
    folder_path = os.path.join(data_dir, category)
    label = categories.index(category)

    for img_file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_file)
        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # convert to grayscale
            img = cv2.resize(img, (48, 48))  # resize for uniformity
            data.append(img.flatten())
            labels.append(label)
        except Exception as e:
            print("Error loading image:", img_path, e)

# Convert to NumPy arrays
X = np.array(data)
y = np.array(labels)

# Split into training/testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple Support Vector Machine model
print("Training the model, please wait...")
model = SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save the model
joblib.dump(model, "face_emotionModel.pkl")
print("Model saved as face_emotionModel.pkl")
