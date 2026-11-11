"""FastAPI backend: live webcam streams + upload analysis."""

from __future__ import annotations

import asyncio
import base64
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator, Literal

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from processing import analyze_bgr, decode_upload, enhance_bgr, encode_jpeg

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FRONTEND_DIR = ROOT / "frontend"

_camera: cv2.VideoCapture | None = None
_camera_lock = asyncio.Lock()


def _get_camera() -> cv2.VideoCapture:
    global _camera
    if _camera is None or not _camera.isOpened():
        _camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not _camera.isOpened():
            _camera = cv2.VideoCapture(0)
    return _camera


def _release_camera() -> None:
    global _camera
    if _camera is not None:
        _camera.release()
        _camera = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    yield
    _release_camera()


app = FastAPI(title="Medical Image Analyzer", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _mjpeg_generator(mode: Literal["original", "enhanced", "analyzed"]) -> AsyncGenerator[bytes, None]:
    cap = _get_camera()
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if mode == "original":
            out = frame
        elif mode == "enhanced":
            out = enhance_bgr(frame)
        else:
            out, _ = analyze_bgr(frame)

        jpeg = encode_jpeg(out, quality=80)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
        )


@app.get("/api/health")
def health():
    from ml.inference import model_available

    return {
        "status": "ok",
        "disclaimer": "Educational demo only — not for medical diagnosis.",
        "ai_model_ready": model_available(),
    }


@app.get("/api/ai/status")
def ai_status():
    from ml.inference import model_available

    if model_available():
        return {"ready": True, "message": "Deep learning model loaded."}
    return {
        "ready": False,
        "message": "Train weights first: python ml/train_demo.py (from project root)",
    }


@app.post("/api/ai/predict")
async def ai_predict(file: UploadFile = File(...)):
    from ml.inference import model_available, predict_with_gradcam

    if not model_available():
        raise HTTPException(
            status_code=503,
            detail="AI model not found. From project root run: python ml/train_demo.py",
        )

    raw = await file.read()
    try:
        original = decode_upload(raw)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        result = predict_with_gradcam(original)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    def b64(img: bytes) -> str:
        return base64.b64encode(img).decode("ascii")

    return {
        "label": result.label,
        "confidence": round(result.confidence, 4),
        "probabilities": result.probabilities,
        "original_jpeg_base64": b64(encode_jpeg(original)),
        "gradcam_jpeg_base64": b64(encode_jpeg(result.gradcam_overlay_bgr)),
        "method": "ResNet18 transfer learning + Grad-CAM (demo weights)",
    }


@app.get("/api/video/{mode}")
async def video_stream(mode: Literal["original", "enhanced", "analyzed"]):
    async with _camera_lock:
        cap = _get_camera()
        if not cap.isOpened():
            raise HTTPException(
                status_code=503,
                detail="Webcam not available. Check permissions and that no other app is using it.",
            )

    return StreamingResponse(
        _mjpeg_generator(mode),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.post("/api/analyze")
async def analyze_upload(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file (PNG or JPEG).")

    raw = await file.read()
    if len(raw) > 15 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image too large (max 15 MB).")

    try:
        original = decode_upload(raw)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    enhanced = enhance_bgr(original)
    annotated, regions = analyze_bgr(original)

    def b64(img: bytes) -> str:
        return base64.b64encode(img).decode("ascii")

    return {
        "regions": regions,
        "region_count": len(regions),
        "image_size": {"width": int(original.shape[1]), "height": int(original.shape[0])},
        "original_jpeg_base64": b64(encode_jpeg(original)),
        "enhanced_jpeg_base64": b64(encode_jpeg(enhanced)),
        "annotated_jpeg_base64": b64(encode_jpeg(annotated)),
    }


@app.post("/api/process-frame")
async def process_frame(file: UploadFile = File(...), mode: Literal["enhanced", "analyzed"] = "enhanced"):
    """Browser webcam: send a JPEG frame, get processed JPEG back."""
    raw = await file.read()
    try:
        frame = decode_upload(raw)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if mode == "analyzed":
        out, regions = analyze_bgr(frame)
        return Response(
            content=encode_jpeg(out),
            media_type="image/jpeg",
            headers={"X-Region-Count": str(len(regions))},
        )
    return Response(content=encode_jpeg(enhance_bgr(frame)), media_type="image/jpeg")


if FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
