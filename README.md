# YVES – Playfully Teaching AI Principles through Interactive Machine Teaching

## 📖 Overview

This project started in June 2022 and was completed in February 2023

## 📖 Overview

**YVES** is a playful, tangible AI teaching prototype developed during my Bachelor's thesis at Ludwig-Maximilians-Universität (LMU) Munich under the Human-Centered Ubiquitous Media group.  
It teaches non-experts — especially young users — fundamental principles of Artificial Intelligence (AI) and Machine Learning (ML) through hands-on interaction with an object detection system.

YVES was first evaluated at LMU's Open Lab Day and later exhibited at the Copernicus Science Centre (Warsaw, Poland), demonstrating that playful interaction can significantly improve AI understanding among the public.  
In 2023, the Ubiquitous Media group further developed the system and deployed it at the Deutsches Museum (Munich).

## About

Through a museum-style experience, YVES lets users:
- Create their own training datasets.
- Simulate AI model training.
- Test object detection performance.
- Understand why diversity and quality in data are critical for AI success.

Built with **Flask**, **TensorFlow**, and **YOLO**, YVES uses pre-trained object detection models to demonstrate how AI systems learn — and why they sometimes fail.

## 🎯 Motivation

Despite AI's growing influence in daily life, public understanding of its workings remains limited.  
YVES addresses this gap by offering an interactive, gamified machine teaching experience that transforms users into active AI "teachers" — building better awareness of training data, learning processes, and AI system limitations.


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

## Author

Anna Papanakli
