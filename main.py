from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import cv2
import mediapipe as mp
import numpy as np
import joblib
import time
from collections import deque, Counter
import threading
import pyttsx3
from groq import Groq
import os

# FASTAPI 
app = FastAPI(title="Sign Language AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# GROQ LLM 
client = Groq(api_key=os.getenv("GROQ_API_KEY"))  # ⚠️ Use environment variable for security

def refine_sentence(words):
    joined = " ".join(words)

    prompt = f"""
You translate sign language gestures into a natural short sentence.

Rules:
- Base the sentence ONLY on the given words.
- You may add very small common words (I, you, is, are, this, it, how, are, please) to make it sound natural.
- Do NOT introduce new topics, objects, or actions.
- Do NOT create stories.
- Keep it short and conversational.
- Return ONE sentence only. No explanation.

Examples:
Words: hello you
Sentence: Hello, how are you?

Words: love
Sentence: I love this.

Words: i love
Sentence: I love you.

Words: please stop
Sentence: Please stop it immediately.

Words: thank you
Sentence: Thank you.

Words: help please
Sentence: Please help me.

Now translate:

Words: {joined}
Sentence:
"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    text = response.choices[0].message.content.strip()

    if len(text.split()) <= 1:
        return joined.capitalize() + "."

    return text

# TEXT TO SPEECH
def speak_text(text):
    engine = pyttsx3.init()
    engine.setProperty("rate", 120)
    engine.say(text)
    engine.runAndWait()
    engine.stop()


# LOAD ML MODEL
 
model = joblib.load("model/gesture_model.pkl")
labels = joblib.load("model/labels.pkl")

# MEDIAPIPE 
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
 
prediction_buffer = deque(maxlen=10)
stable_word = ""
final_words = []
last_added_time = time.time()
last_gesture_time = time.time()
final_sentence = ""


@app.get("/")
def root():
    return {"status": "Sign Language AI Backend Running"}

@app.get("/health")
def health():
    return {"ok": True}


# PREDICT
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    global stable_word, final_words, last_added_time, last_gesture_time, final_sentence

    try:
        image_bytes = await file.read()
        np_img = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        print("🖐 Hand detected:", bool(result.multi_hand_landmarks))

        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            landmarks = []

            for lm in hand.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            print("📏 Landmarks length:", len(landmarks))

            if len(landmarks) == 63:
                data = np.array(landmarks).reshape(1, -1)
                probs = model.predict_proba(data)[0]
                confidence = float(max(probs))
                pred = int(np.argmax(probs))
                word = labels.inverse_transform([pred])[0]

                print(f"🤖 Pred: {word} | Conf: {confidence:.2f}")

                if confidence > 0.4:  # lowered threshold
                    stable_word = word
                    last_gesture_time = time.time()

                if len(prediction_buffer) == prediction_buffer.maxlen:
                    stable_word = Counter(prediction_buffer).most_common(1)[0][0]

                if stable_word and time.time() - last_added_time > 1.2:
                    if not final_words or final_words[-1] != stable_word:
                        final_words.append(stable_word)
                        last_added_time = time.time()

        # generate sentence
        if time.time() - last_gesture_time > 2 and final_words:
            final_sentence = refine_sentence(final_words)
            final_words = []
            prediction_buffer.clear()
            stable_word = ""


        return {
            "gesture": stable_word,
            "sentence": final_sentence
        }

    except Exception as e:
        print("❌ ERROR:", e)
        return {"gesture": "", "sentence": ""}

# SPEAK
@app.post("/speak")
def speak(data: dict):
    text = data.get("text", "")
    if text:
        threading.Thread(target=speak_text, args=(text,), daemon=True).start()
        return {"status": "speaking"}
    return {"status": "no text"}
