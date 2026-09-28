from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import os
import cv2

app = Flask(__name__)

# ===========================
# Load CNN Model
# ===========================
model = tf.keras.models.load_model("emotion_model.keras")

# Emotion Labels
class_names = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]

# Upload Folder
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Face Detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)


@app.route("/", methods=["GET", "POST"])
def home():

    emotion = None
    confidence = None
    message = None
    image_path = None

    if request.method == "POST":

        if "image" not in request.files:
            message = "Please select an image."
            return render_template(
                "index.html",
                message=message
            )

        file = request.files["image"]

        if file.filename == "":
            message = "Please select an image."
            return render_template(
                "index.html",
                message=message
            )

        # Save Uploaded Image
        filepath = os.path.join(
            UPLOAD_FOLDER,
            file.filename
        )

        file.save(filepath)

        # Used to display image in HTML
        image_path = filepath.replace("\\", "/")

        # Read Image
        image = cv2.imread(filepath)

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        # Detect Face
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        if len(faces) == 0:

            message = "No face detected."

            return render_template(
                "index.html",
                message=message,
                image_path=image_path
            )

        # Largest Face
        x, y, w, h = max(
            faces,
            key=lambda r: r[2] * r[3]
        )

        face = gray[
            y:y+h,
            x:x+w
        ]

        # Resize
        face = cv2.resize(
            face,
            (48, 48)
        )

        # Normalize
        face = face.astype("float32") / 255.0

        face = np.expand_dims(face, axis=0)
        face = np.expand_dims(face, axis=-1)

        # Prediction
        prediction = model.predict(
            face,
            verbose=0
        )

        predicted_index = np.argmax(prediction)

        emotion = class_names[predicted_index]

        confidence = round(
            prediction[0][predicted_index] * 100,
            2
        )

    return render_template(
        "index.html",
        emotion=emotion,
        confidence=confidence,
        image_path=image_path,
        message=message
    )


if __name__ == "__main__":
    app.run(debug=True)