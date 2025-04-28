# YVES – Playfully Teaching AI Principles through Interactive Machine Teaching

## 📖 Overview

This project started in June 2022 and was completed in February 2023

YVES was developed as part of a Bachelor's thesis at the Ludwig-Maximilians-Universität (LMU) Munich, under the Human-Centered Ubiquitous Media group.
It was evaluated at a public Open Lab Day event, demonstrating that playful interaction successfully improves non-experts’ understanding of AI. It was later on tested and placed in the Copernicus Science Center in Warsaw, Poland. Finally, it was further developed by the Ubiquitous Media group, without me, and placed in the Deutsches Museum, Munich, to further teach young people about AI.

## About

YVES is a tangible, playful prototype designed to teach non-experts — especially young users — key principles of Artificial Intelligence (AI) and Machine Learning (ML) through hands-on interaction.
It focuses on the concepts of object detection and dataset quality, using a simulated training and testing pipeline in an engaging, museum-style setting. The system includes three pre-trained object detection models which are used at each interaction step to demonstrate the development of an object detection model to the user.
Built with **Flask**, **TensorFlow**, and **YOLO**, YVES allows users to create an object detection model in an interactive and fun way, thereby teaching themselves basic principles of how AI systems learn from data and what makes data "good" training data.

## 🧪 Features

- Hands-on half simulated object detection model training.
Two interaction types:
- Guided Interaction (automated dataset augmentation).
- Unguided Interaction (user-driven dataset diversification).
- Real-time feedback through bounding boxes and detection accuracy scores.
- Designed for museum-style exhibits and public short term engagement with young viewers.

## 🛠️ Technologies Used

- Flask – backend web framework.
- TensorFlow – object detection model training.
- OpenCV – camera capture and processing.
- HTML/CSS/JavaScript – frontend interface.
- YOLO (You Only Look Once) – real-time object detection.


## 📦 Project Structure

```
├── app.py            # Main Flask application
├── models/           # Pre-trained object detection models
├── templates/        # HTML templates
├── static/           # CSS, JavaScript, and static files
├── uploads/          # Temporary storage for user-uploaded images
├── utils/            # Helper functions (if applicable)
└── requirements.txt  # Python dependencies
```

## Future Work

- Real-time training (if computational constraints are overcome).
- More AI principles (e.g., bounding box accuracy, adversarial examples).

## 🎯 Motivation

Despite AI systems influencing daily life (social media feeds, face recognition, spam filters), most people lack an understanding of how they work.
YVES addresses this gap by providing an approachable and playful experience where users:
Create their own datasets.
Simulate training an AI model.
Test and experience model performance.
Understand why diversity in training data matters.
Recognize limitations of AI systems (e.g., perspective issues).
Through interactive machine teaching combined with gamification principles, YVES turns passive users into active AI "teachers."

## Author

Anna Papanakli
