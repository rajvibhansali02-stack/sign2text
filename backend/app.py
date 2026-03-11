from flask import Flask, request, jsonify
import base64
import cv2
import numpy as np
import joblib
import mediapipe as mp
import time
from collections import deque, Counter
from groq import Groq

# =========================
# FLASK APP
# =========================
app = Flask(__name__)

# =========================
# LOAD MODEL & LABELS
# =========================
model = joblib.load("gesture_model.pkl")
labels = joblib.load("labels.pkl")

# =========================
# MEDIAPIPE SETUP
# =========================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.4,
    min_tracking_confidence=0.4
)

# =========================
# GROQ LLM
# =========================
import os
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def refine_sentence(words):
    prompt = f"""
Return ONLY the final grammatically correct English sentence.
Words: {' '.join(words)}
"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return response.choices[0].message.content.strip()

# =========================
# STATE (STABILIZATION)
# =========================
prediction_buffer = deque(maxlen=10)
final_words = []
last_added_time = time.time()
last_gesture_time = time.time()
final_sentence = ""

# =========================
# ROUTES
# =========================
@app.route("/")
def home():
    return "Backend running"

@app.route("/predict", methods=["POST"])
def predict():
    global last_added_time, last_gesture_time, final_sentence

    try:
        # 1️⃣ Receive image
        data = request.json["image"]
        encoded = data.split(",")[1]
        img_bytes = base64.b64decode(encoded)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # 2️⃣ MediaPipe processing
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        print("📥 Frame received")

        if not result.multi_hand_landmarks:
            print("❌ No hand detected")
            return jsonify({"prediction": final_sentence})

        print("✅ Hand detected")

        hand = result.multi_hand_landmarks[0]

        # 3️⃣ Extract normalized landmarks
        wrist = hand.landmark[0]
        print("📏 Landmark length:", len(landmarks))

        if len(landmarks) != 63:
            print("❌ Invalid landmark size")
            return jsonify({"prediction": final_sentence})
        landmarks = []
        for lm in hand.landmark:
            landmarks.extend([
                lm.x - wrist.x,
                lm.y - wrist.y,
                lm.z - wrist.z
            ])

        if len(landmarks) != 63:
            return jsonify({"prediction": final_sentence})

        # 4️⃣ ML prediction
        X = np.array(landmarks).reshape(1, -1)
        probs = model.predict_proba(X)[0]
        confidence = max(probs)
        pred = np.argmax(probs)
        word = labels.inverse_transform([pred])[0]

        if confidence > 0.75:
            prediction_buffer.append(word)
            last_gesture_time = time.time()

        if len(prediction_buffer) == prediction_buffer.maxlen:
            stable_word = Counter(prediction_buffer).most_common(1)[0][0]
            if time.time() - last_added_time > 1.5:
                final_words.append(stable_word)
                last_added_time = time.time()

        # 5️⃣ Generate sentence after pause
        if time.time() - last_gesture_time > 2.5 and final_words:
            final_sentence = refine_sentence(final_words)
            final_words.clear()
            prediction_buffer.clear()

        return jsonify({"prediction": final_sentence})

    except Exception as e:
        return jsonify({"error": str(e)})

# =========================
# RUN SERVER
# =========================
if __name__ == "__main__":
    app.run(debug=True)