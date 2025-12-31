# 🧠 Object Detection

A simple Python-based object detection project using OpenCV’s Haarcascade classifiers to detect faces and eyes in images or video streams.

## 📌 Overview

This project demonstrates basic object detection using Haar Cascade models from OpenCV.  
It can detect:

- 👤 Faces  
- 👁️ Eyes

The detection runs in real-time using your webcam or from static images.

## 🛠️ Technologies Used

- 🐍 Python  
- 📸 OpenCV  
- 📁 Haarcascade XML classifiers

## 📁 Project Structure

object-detection/
├── face_detector.py
├── haarcascade_eye.xml
├── haarcascade_frontalface_default.xml (if included)
└── README.md



## 🚀 Installation

1. **Clone the repository**
```bash
git clone https://github.com/Chidambaram2701/object-detection.git
Navigate to the folder

cd object-detection
Install required dependencies

pip install opencv-python

📦 Usage
Run the detection script:
python face_detector.py
