# 😊 FeelVision - Deep Learning-Based Facial Emotion Recognition

![Python](https://img.shields.io/badge/Python-3.13-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Flask](https://img.shields.io/badge/Flask-Web%20Application-black)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 📌 Project Overview

**FeelVision** is an AI-powered Facial Emotion Recognition System that detects human emotions from facial images using **Deep Learning**. The application uses a **Convolutional Neural Network (CNN)** trained on the **FER2013 dataset** to classify facial expressions into seven emotions.

Users can upload a facial image through a Flask-based web application. The system detects the face using **OpenCV Haar Cascade**, preprocesses the image, predicts the emotion, and displays the predicted emotion along with its confidence score.

---

## 🎯 Objectives

- Detect human emotions from facial images.
- Build a CNN model for emotion classification.
- Develop an interactive web application using Flask.
- Display emotion prediction with confidence.
- Demonstrate the application of Deep Learning and Computer Vision.

---

## 😊 Supported Emotions

The model predicts the following seven emotions:

- 😠 Angry
- 🤢 Disgust
- 😨 Fear
- 😊 Happy
- 😐 Neutral
- 😢 Sad
- 😲 Surprise

---

## 🛠 Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- Flask
- NumPy
- HTML5
- CSS3
- FER2013 Dataset

---

## 📂 Project Structure

```
FeelVision/
│
├── dataset/
│   ├── train/
│   └── test/
│
├── static/
│   ├── style.css
│   └── uploads/
│
├── templates/
│   └── index.html
│
├── app.py
├── train_model.py
├── emotion_model.keras
├── check_dataset.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/FeelVision.git

cd FeelVision
```

---

### 2️⃣ Create Virtual Environment

Windows

```bash
python -m venv .venv
```

Activate

```bash
.venv\Scripts\activate
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Dataset

Download the **FER2013 Dataset** and place it inside:

```
dataset/

    train/

    test/
```

Folder structure:

```
dataset/

    train/

        angry/

        disgust/

        fear/

        happy/

        neutral/

        sad/

        surprise/

    test/

        angry/

        disgust/

        fear/

        happy/

        neutral/

        sad/

        surprise/
```

---

### 5️⃣ Train the Model

```bash
python train_model.py
```

After training, the model will be saved as

```
emotion_model.keras
```

---

### 6️⃣ Run the Web Application

```bash
python app.py
```

Open your browser and visit:

```
http://127.0.0.1:5000
```

---

## 🔄 Workflow

```
Facial Image
      │
      ▼
Upload through Flask
      │
      ▼
OpenCV Face Detection
      │
      ▼
Image Preprocessing
      │
      ▼
CNN Emotion Prediction
      │
      ▼
Display Emotion
      │
      ▼
Confidence Score
```

---

## ✨ Features

- Face Detection using OpenCV
- CNN-based Emotion Classification
- Upload Facial Images
- Displays Predicted Emotion
- Confidence Percentage
- Simple and User-Friendly Interface
- Deep Learning Model using TensorFlow

---

## 📊 Model Information

**Model:** Convolutional Neural Network (CNN)

**Input Size:** 48 × 48 Grayscale Image

**Dataset:** FER2013

**Output Classes:** 7

Activation Function:

- ReLU
- Softmax

Optimizer:

- Adam

Loss Function:

- Categorical Crossentropy

---

## 📸 Sample Output

```
Uploaded Image

Detected Emotion

😊 Happy

Confidence

92.15%
```

---

## 🚀 Future Enhancements

- Real-time Emotion Detection using Webcam
- Live Video Stream Prediction
- Improved CNN Architecture
- Mobile Application
- Emotion Analytics Dashboard
- Face Recognition Integration
- Higher Prediction Accuracy

---

## 📚 Applications

- Human-Computer Interaction
- Healthcare
- Online Learning
- Smart Surveillance
- Customer Experience Analysis
- Driver Monitoring Systems

