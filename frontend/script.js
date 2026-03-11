const video = document.getElementById("video");
const enableBtn = document.getElementById("enableBtn");
const stopBtn = document.getElementById("stopBtn");
const predictionText = document.getElementById("predictionText");

let stream = null;
let intervalId = null;
let lastSpoken = "";

const API_URL = "http://127.0.0.1:8000";

// check backend
window.onload = async () => {
  try {
    const res = await fetch(API_URL + "/health");
    if (!res.ok) throw new Error();
    console.log("✅ Backend connected");
  } catch {
    predictionText.innerText = "❌ Backend not running";
  }
};

// ===============================
// SEND FRAME
// ===============================
async function sendFrameToBackend() {
  if (!video.srcObject || video.videoWidth === 0) return;

  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;

  const ctx = canvas.getContext("2d");
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  canvas.toBlob(async (blob) => {
    const formData = new FormData();
    formData.append("file", blob, "frame.jpg");

    try {
      const response = await fetch(API_URL + "/predict", {
        method: "POST",
        body: formData
      });

      if (!response.ok) throw new Error();

      const data = await response.json();

      predictionText.innerText =
        `Gesture: ${data.gesture || "-"}\nSentence: ${data.sentence || "-"}`;

      if (data.sentence && data.sentence.trim() !== "") {
        speakSentence(data.sentence);
      }

    } catch (err) {
      console.error(err);
      predictionText.innerText = "❌ Backend not responding";
    }
  }, "image/jpeg");
}

// ===============================
// SPEAK
// ===============================
async function speakSentence(text) {
  if (text === lastSpoken) return;
  lastSpoken = text;

  try {
    await fetch(API_URL + "/speak", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text })
    });
  } catch (err) {
    console.error("Speak error:", err);
  }
}

// ===============================
// ENABLE CAMERA
// ===============================
enableBtn.addEventListener("click", async () => {
  if (stream) return;

  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
    predictionText.innerText = "📷 Camera enabled...";

    intervalId = setInterval(sendFrameToBackend, 700);
  } catch {
    predictionText.innerText = "❌ Camera access denied";
  }
});

// ===============================
// STOP CAMERA
// ===============================
stopBtn.addEventListener("click", () => {
  if (stream) {
    stream.getTracks().forEach(track => track.stop());
    stream = null;
    video.srcObject = null;
  }

  if (intervalId) {
    clearInterval(intervalId);
    intervalId = null;
  }

  predictionText.innerText = "⛔ Camera stopped";
});
