const video = document.getElementById("video");

navigator.mediaDevices.getUserMedia({ video: true })
  .then(stream => video.srcObject = stream);

async function sendFrame() {
  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext("2d").drawImage(video, 0, 0);

  const blob = await new Promise(resolve =>
    canvas.toBlob(resolve, "image/jpeg")
  );

  const formData = new FormData();
  formData.append("file", blob);

  const res = await fetch("https://your-backend.onrender.com/predict", {
    method: "POST",
    body: formData
  });

  const data = await res.json();
  document.querySelector(".highlight").innerText = data.prediction;
}

setInterval(sendFrame, 1500);
