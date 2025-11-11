import os
import sqlite3
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
from datetime import datetime

# ------------------ Configuration ------------------
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load the trained model
model = load_model("face_emotionModel.h5")

# Emotion labels (must match the folder names in your training data)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# ------------------ Database Setup ------------------
db_name = "database.db"

def init_db():
    conn = sqlite3.connect(db_name)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            email TEXT,
            image_path TEXT,
            emotion TEXT,
            timestamp TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()  # create table if not exists

# ------------------ Routes ------------------
@app.route("/", methods=["GET", "POST"])
def index():
    message = ""
    if request.method == "POST":
        name = request.form.get("name")
        email = request.form.get("email")
        image_file = request.files.get("image")

        if not name or not email or not image_file:
            message = "Please fill in all fields and upload an image."
            return render_template("index.html", message=message)

        # Save uploaded image
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"{timestamp}_{image_file.filename}"
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        image_file.save(filepath)

        # Load image for prediction
        img = load_img(filepath, color_mode="grayscale", target_size=(48, 48))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # batch dimension

        # Predict emotion
        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions)
        emotion = emotion_labels[predicted_index]

        # Create friendly message
        emotion_messages = {
            "Angry": "You look angry. Take a deep breath!",
            "Disgust": "Hmm, something is bothering you.",
            "Fear": "You seem scared. Stay calm!",
            "Happy": "You are smiling! Keep shining :)",
            "Neutral": "You look calm and neutral.",
            "Sad": "You are frowning. Why are you sad?",
            "Surprise": "Wow, something surprised you!"
        }
        friendly_message = emotion_messages.get(emotion, "Emotion detected!")

        # Save info to database
        conn = sqlite3.connect(db_name)
        c = conn.cursor()
        c.execute('''
            INSERT INTO users (name, email, image_path, emotion, timestamp)
            VALUES (?, ?, ?, ?, ?)
        ''', (name, email, filepath, emotion, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        conn.commit()
        conn.close()

        message = friendly_message

    return render_template("index.html", message=message)

# ------------------ Run App ------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Use the PORT from Render
    app.run(host="0.0.0.0", port=port)