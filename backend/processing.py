"""OpenCV pipelines for enhancement and simple region detection."""

from __future__ import annotations

import cv2
import numpy as np


def enhance_bgr(frame: np.ndarray) -> np.ndarray:
    """
    Medical-style contrast boost for demos: CLAHE, percentile stretch, mild sharpen.
    Makes Original vs Enhanced clearly different even on high-contrast samples.
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    lo, hi = np.percentile(l, (2, 98))
    if hi > lo:
        l = np.clip((l.astype(np.float32) - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)

    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    enhanced = cv2.bilateralFilter(enhanced, d=5, sigmaColor=75, sigmaSpace=75)

    blur = cv2.GaussianBlur(enhanced, (0, 0), 1.2)
    enhanced = cv2.addWeighted(enhanced, 1.35, blur, -0.35, 0)
    return enhanced


def analyze_bgr(frame: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    """
    Enhance, segment bright/dark regions, draw overlays.
    Returns annotated BGR image and region stats (pixel coordinates).
    """
    enhanced = enhance_bgr(frame)
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output = enhanced.copy()
    regions: list[dict] = []
    min_area = max(500, (frame.shape[0] * frame.shape[1]) // 2000)

    for idx, contour in enumerate(contours, start=1):
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.drawContours(output, [contour], -1, (255, 120, 0), 1)
        regions.append(
            {
                "id": idx,
                "area_pixels": int(area),
                "bbox": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)},
            }
        )

    cv2.putText(
        output,
        f"Regions: {len(regions)} (educational demo)",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return output, regions


def decode_upload(content: bytes) -> np.ndarray:
    arr = np.frombuffer(content, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image. Use PNG or JPEG.")
    return image


def encode_jpeg(frame: np.ndarray, quality: int = 85) -> bytes:
    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("Failed to encode JPEG")
    return buffer.tobytes()
