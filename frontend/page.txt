const API = "";

const tabs = document.querySelectorAll(".tab");
const panels = {
  live: document.getElementById("panel-live"),
  upload: document.getElementById("panel-upload"),
  ai: document.getElementById("panel-ai"),
};

tabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    tabs.forEach((t) => t.classList.remove("active"));
    tab.classList.add("active");
    Object.values(panels).forEach((p) => p.classList.remove("active"));
    panels[tab.dataset.tab].classList.add("active");
    if (tab.dataset.tab !== "live") stopLive();
  });
});

// --- Live camera ---
const liveSource = document.getElementById("live-source");
const liveMode = document.getElementById("live-mode");
const liveStart = document.getElementById("live-start");
const liveStop = document.getElementById("live-stop");
const serverStream = document.getElementById("server-stream");
const browserCanvas = document.getElementById("browser-canvas");
const browserVideo = document.getElementById("browser-video");
const livePlaceholder = document.getElementById("live-placeholder");
const liveStatus = document.getElementById("live-status");

let browserLoop = null;
let browserStream = null;

function setLiveStatus(msg) {
  liveStatus.textContent = msg;
}

function stopLive() {
  serverStream.hidden = true;
  serverStream.removeAttribute("src");
  browserCanvas.hidden = true;
  browserVideo.hidden = true;
  livePlaceholder.hidden = false;

  if (browserLoop) {
    cancelAnimationFrame(browserLoop);
    browserLoop = null;
  }
  if (browserStream) {
    browserStream.getTracks().forEach((t) => t.stop());
    browserStream = null;
  }

  liveStart.disabled = false;
  liveStop.disabled = true;
  setLiveStatus("Stopped");
}

async function startServerStream() {
  const mode = liveMode.value;
  serverStream.src = `${API}/api/video/${mode}?t=${Date.now()}`;
  serverStream.hidden = false;
  browserCanvas.hidden = true;
  livePlaceholder.hidden = true;
  setLiveStatus(`Streaming from PC webcam — ${mode}`);
}

async function processBrowserFrame() {
  const ctx = browserCanvas.getContext("2d");
  const w = browserVideo.videoWidth;
  const h = browserVideo.videoHeight;
  if (!w || !h) {
    browserLoop = requestAnimationFrame(processBrowserFrame);
    return;
  }

  browserCanvas.width = w;
  browserCanvas.height = h;
  ctx.drawImage(browserVideo, 0, 0, w, h);

  browserCanvas.toBlob(
    async (blob) => {
      if (!blob) {
        browserLoop = requestAnimationFrame(processBrowserFrame);
        return;
      }
      const mode = liveMode.value === "original" ? "enhanced" : liveMode.value;
      if (liveMode.value === "original") {
        browserLoop = requestAnimationFrame(processBrowserFrame);
        return;
      }

      const form = new FormData();
      form.append("file", blob, "frame.jpg");
      try {
        const res = await fetch(`${API}/api/process-frame?mode=${mode}`, {
          method: "POST",
          body: form,
        });
        if (res.ok) {
          const buf = await res.arrayBuffer();
          const url = URL.createObjectURL(new Blob([buf], { type: "image/jpeg" }));
          const img = new Image();
          img.onload = () => {
            ctx.drawImage(img, 0, 0, w, h);
            URL.revokeObjectURL(url);
          };
          img.src = url;
          const count = res.headers.get("X-Region-Count");
          if (count != null) setLiveStatus(`Browser webcam — regions: ${count}`);
        }
      } catch {
        setLiveStatus("Error sending frame to server");
      }
      browserLoop = requestAnimationFrame(processBrowserFrame);
    },
    "image/jpeg",
    0.75
  );
}

async function startBrowserStream() {
  browserStream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: "user" },
    audio: false,
  });
  browserVideo.srcObject = browserStream;
  await browserVideo.play();

  serverStream.hidden = true;
  browserCanvas.hidden = false;
  livePlaceholder.hidden = true;

  if (liveMode.value === "original") {
    setLiveStatus("Browser webcam — original (processed on device only)");
    const drawOriginal = () => {
      const ctx = browserCanvas.getContext("2d");
      const w = browserVideo.videoWidth;
      const h = browserVideo.videoHeight;
      if (w && h) {
        browserCanvas.width = w;
        browserCanvas.height = h;
        ctx.drawImage(browserVideo, 0, 0, w, h);
      }
      browserLoop = requestAnimationFrame(drawOriginal);
    };
    drawOriginal();
  } else {
    setLiveStatus("Browser webcam — sending frames to OpenCV…");
    processBrowserFrame();
  }
}

liveStart.addEventListener("click", async () => {
  liveStart.disabled = true;
  liveStop.disabled = false;
  try {
    if (liveSource.value === "server") {
      await startServerStream();
    } else {
      await startBrowserStream();
    }
  } catch (err) {
    setLiveStatus(err.message || "Could not start camera");
    stopLive();
  }
});

liveStop.addEventListener("click", stopLive);

liveMode.addEventListener("change", () => {
  if (liveStop.disabled) return;
  stopLive();
  liveStart.disabled = false;
  liveStop.disabled = true;
  setLiveStatus("Mode changed — click Start again");
});

liveSource.addEventListener("change", () => {
  if (!liveStop.disabled) stopLive();
});

// --- Upload ---
const fileInput = document.getElementById("file-input");
const uploadAnalyze = document.getElementById("upload-analyze");
const imgOriginal = document.getElementById("img-original");
const imgEnhanced = document.getElementById("img-enhanced");
const imgAnnotated = document.getElementById("img-annotated");
const reportText = document.getElementById("report-text");

fileInput.addEventListener("change", () => {
  uploadAnalyze.disabled = !fileInput.files?.length;
});

uploadAnalyze.addEventListener("click", async () => {
  const file = fileInput.files?.[0];
  if (!file) return;

  uploadAnalyze.disabled = true;
  reportText.textContent = "Analyzing…";

  const form = new FormData();
  form.append("file", file);

  try {
    const res = await fetch(`${API}/api/analyze`, { method: "POST", body: form });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Analyze failed");

    imgOriginal.src = `data:image/jpeg;base64,${data.original_jpeg_base64}`;
    imgEnhanced.src = `data:image/jpeg;base64,${data.enhanced_jpeg_base64}`;
    imgAnnotated.src = `data:image/jpeg;base64,${data.annotated_jpeg_base64}`;

    const lines = [
      `Size: ${data.image_size.width} × ${data.image_size.height} px`,
      `Regions found: ${data.region_count}`,
      "",
    ];
    data.regions.forEach((r) => {
      const b = r.bbox;
      lines.push(
        `Region ${r.id}: area ${r.area_pixels} px — box (${b.x}, ${b.y}) ${b.width}×${b.height}`
      );
    });
    reportText.textContent = lines.join("\n");
  } catch (err) {
    reportText.textContent = String(err.message || err);
  } finally {
    uploadAnalyze.disabled = false;
  }
});

// --- AI + Grad-CAM ---
const aiFileInput = document.getElementById("ai-file-input");
const aiPredictBtn = document.getElementById("ai-predict");
const aiModelHint = document.getElementById("ai-model-hint");
const aiOriginal = document.getElementById("ai-original");
const aiGradcam = document.getElementById("ai-gradcam");
const aiReport = document.getElementById("ai-report");

async function refreshAiStatus() {
  try {
    const res = await fetch(`${API}/api/ai/status`);
    const data = await res.json();
    aiModelHint.textContent = data.message || (data.ready ? "AI ready" : "AI not ready");
    aiModelHint.style.color = data.ready ? "#3dcea3" : "#f0c674";
  } catch {
    aiModelHint.textContent = "Could not reach server for AI status.";
  }
}
refreshAiStatus();

aiFileInput.addEventListener("change", () => {
  aiPredictBtn.disabled = !aiFileInput.files?.length;
});

aiPredictBtn.addEventListener("click", async () => {
  const file = aiFileInput.files?.[0];
  if (!file) return;

  aiPredictBtn.disabled = true;
  aiReport.textContent = "Running deep learning inference…";

  const form = new FormData();
  form.append("file", file);

  try {
    const res = await fetch(`${API}/api/ai/predict`, { method: "POST", body: form });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "AI predict failed");

    aiOriginal.src = `data:image/jpeg;base64,${data.original_jpeg_base64}`;
    aiGradcam.src = `data:image/jpeg;base64,${data.gradcam_jpeg_base64}`;

    const lines = [
      `Prediction: ${data.label}`,
      `Confidence: ${(data.confidence * 100).toFixed(1)}%`,
      `Method: ${data.method}`,
      "",
      "Class probabilities:",
    ];
    Object.entries(data.probabilities).forEach(([k, v]) => {
      lines.push(`  ${k}: ${(v * 100).toFixed(1)}%`);
    });
    aiReport.textContent = lines.join("\n");
  } catch (err) {
    aiReport.textContent = String(err.message || err);
  } finally {
    aiPredictBtn.disabled = false;
  }
});

