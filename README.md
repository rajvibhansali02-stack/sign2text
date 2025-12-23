# Sign Language Detection System

This project is a real-time Sign Language Detection system using
MediaPipe and Deep Learning.

## Features
- Live webcam hand sign detection
- Deep learning gesture classification
- Web-based user interface
- Backend API for model inference

## Tech Stack
- Python
- TensorFlow / Keras
- MediaPipe
- FastAPI
- HTML, CSS, JavaScript

## Project Structure
backend/
app.py
gesture_model.keras

frontend/
index.html
style.css
script.js
## How to Run

### Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn app:app --reload

---

