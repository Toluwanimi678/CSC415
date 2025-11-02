from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
model = load_model('face_emotionModel.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return render_template('index.html', emotion="No file uploaded")

    file = request.files['image']
    if file.filename == '':
        return render_template('index.html', emotion="No file selected")

    # Save file temporarily
    filepath = os.path.join('static', file.filename)
    file.save(filepath)

    # Read and preprocess image
    img = cv2.imread(filepath)
    img = cv2.resize(img, (48, 48))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0

    # Predict
    prediction = model.predict(img)
    emotion = np.argmax(prediction)
    emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

    return render_template('index.html', emotion=emotions[emotion])

if __name__ == '__main__':
    app.run(debug=True)
