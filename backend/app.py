from fastapi import FastAPI, UploadFile
import tensorflow as tf
import numpy as np
import cv2

app = FastAPI()

model = tf.keras.models.load_model("gesture_model.keras")
labels = ["YES", "NO", "HELLO", "THANK YOU"]

@app.post("/predict")
async def predict(file: UploadFile):
    contents = await file.read()
    npimg = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    frame = cv2.resize(frame, (224, 224))
    frame = frame / 255.0
    frame = np.expand_dims(frame, axis=0)

    preds = model.predict(frame)
    return {
        "prediction": labels[np.argmax(preds)],
        "confidence": float(np.max(preds))
    }
