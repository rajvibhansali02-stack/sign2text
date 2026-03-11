import cv2
import mediapipe as mp
import numpy as np
import joblib
import time
from collections import deque, Counter
import threading
import pyttsx3
from groq import Groq

# =========================
# GROQ LLM SETUP
# =========================
import os
client = Groq(api_key=os.getenv("GROQ_API_KEY"))  # <-- Use environment variable for security

def refine_sentence(words):
    prompt = f"""
You are a sentence generator.

Rules:
- Return ONLY the final corrected English sentence.
- Do NOT explain.
- Do NOT give multiple options.
- Output ONE short sentence only.

Input words: {' '.join(words)}

Final sentence:
"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return response.choices[0].message.content.strip()

# =========================
# TEXT TO SPEECH (YOUR WORKING METHOD)
# =========================
def speak_text(text):
    engine = pyttsx3.init()
    engine.setProperty("rate", 120)
    engine.say(text)
    engine.runAndWait()
    engine.stop()

# =========================
# LOAD ML MODEL
# =========================
model = joblib.load("model/gesture_model.pkl")
labels = joblib.load("model/labels.pkl")

# =========================
# MEDIAPIPE SETUP
# =========================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

# =========================
# STABILIZATION VARIABLES
# =========================
prediction_buffer = deque(maxlen=10)
stable_word = ""
final_words = []

last_added_time = time.time()
last_gesture_time = time.time()

final_sentence = ""

print("🟢 Real-Time Sign Language AI Started")
print("👉 Press 'S' to Speak | 'Q' to Quit")

# =========================
# MAIN LOOP
# =========================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    landmarks = []

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]

        # DRAW LANDMARKS
        mp_draw.draw_landmarks(
            frame,
            hand,
            mp_hands.HAND_CONNECTIONS,
            mp_styles.get_default_hand_landmarks_style(),
            mp_styles.get_default_hand_connections_style()
        )

        for lm in hand.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])

        if len(landmarks) == 63:
            data = np.array(landmarks).reshape(1, -1)

            probs = model.predict_proba(data)[0]
            confidence = max(probs)
            pred = np.argmax(probs)
            word = labels.inverse_transform([pred])[0]

            if confidence > 0.75:
                prediction_buffer.append(word)
                last_gesture_time = time.time()

            if len(prediction_buffer) == prediction_buffer.maxlen:
                stable_word = Counter(prediction_buffer).most_common(1)[0][0]

            if stable_word and time.time() - last_added_time > 1.5:
                if not final_words or final_words[-1] != stable_word:
                    final_words.append(stable_word)
                    last_added_time = time.time()

    # =========================
    # GENERATE TEXT AFTER PAUSE
    # =========================
    if time.time() - last_gesture_time > 2.5 and final_words:
        final_sentence = refine_sentence(final_words)
        final_words = []
        prediction_buffer.clear()
        stable_word = ""

    # =========================
    # DISPLAY
    # =========================
    cv2.putText(frame, f"Gesture: {stable_word}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(frame, f"Text: {final_sentence}", (30, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

    cv2.putText(frame, "Press 'S' to Speak | 'Q' to Quit", (30, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    cv2.imshow("Sign Language AI", frame)

    key = cv2.waitKey(1) & 0xFF

    # 🔊 SPEAK WHEN USER PRESSES 'S'
    if key == ord('s') and final_sentence:
        threading.Thread(
            target=speak_text,
            args=(final_sentence,),
            daemon=True
        ).start()

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
