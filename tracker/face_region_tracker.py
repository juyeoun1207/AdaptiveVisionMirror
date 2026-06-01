import cv2
import mediapipe as mp
import numpy as np
import os
import time
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'face_landmarker.task')

ZOOM_MIN = 1.0
ZOOM_MAX = 5.0

_EYE_IDX = [33, 133, 159, 145, 263, 362, 386, 374, 70, 300]
_NOSE_IDX = [1, 2, 4, 98, 327, 168, 197]
_MOUTH_IDX = [61, 291, 0, 17, 13, 14]
_JAW_IDX = [152, 148, 176, 149, 150, 136, 172, 58, 377, 400, 378, 379, 365, 397, 288]

REGIONS = {
    ord("e"): {"name": "Eyes", "indices": _EYE_IDX, "padding": 0.55, "color": (0, 255, 255)},
    ord("n"): {"name": "Nose", "indices": _NOSE_IDX, "padding": 0.35, "color": (0, 255, 0)},
    ord("m"): {"name": "Mouth", "indices": _MOUTH_IDX, "padding": 0.55, "color": (0, 100, 255)},
    ord("j"): {"name": "Jaw", "indices": _JAW_IDX, "padding": 0.45, "color": (200, 100, 255)},
}


def get_region_name(region_key):
    if region_key in REGIONS:
        return REGIONS[region_key]["name"]
    return "None"


def _get_bbox(landmarks, indices, h, w, padding):
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
    pw, ph = int((x2 - x1) * padding), int((y2 - y1) * padding)
    return (
        max(0, x1 - pw),
        max(0, y1 - ph),
        min(w, x2 + pw),
        min(h, y2 + ph),
    )


def apply_zoom(crop, zoom_factor):
    if zoom_factor <= 1.0 or crop.size == 0:
        return crop

    h, w = crop.shape[:2]
    scale = 1.0 / zoom_factor
    cx, cy = w // 2, h // 2
    half_w = max(1, int(w * scale / 2))
    half_h = max(1, int(h * scale / 2))
    return crop[
        max(0, cy - half_h):min(h, cy + half_h),
        max(0, cx - half_w):min(w, cx + half_w),
    ]


def fit_to_canvas(image, canvas_size):
    target_w, target_h = canvas_size
    if image is None or image.size == 0:
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)

    return cv2.resize(image, (target_w, target_h))


def make_zoom_panel(crop, zoom_factor, size=(520, 360)):
    zoomed = apply_zoom(crop, zoom_factor) if crop is not None else None
    panel = fit_to_canvas(zoomed, size)

    panel_h, panel_w = panel.shape[:2]
    bar_w = int((zoom_factor - ZOOM_MIN) / (ZOOM_MAX - ZOOM_MIN) * panel_w)
    bar_w = int(np.clip(bar_w, 0, panel_w))
    cv2.rectangle(panel, (0, panel_h - 10), (bar_w, panel_h), (255, 220, 0), -1)
    cv2.putText(panel, f"x{zoom_factor:.2f}", (8, panel_h - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 220, 0), 1)
    return panel


def overlay_panel(display, panel, center, border_color=(255, 220, 0)):
    if center is None or panel is None or panel.size == 0:
        return

    h, w = display.shape[:2]
    ph, pw = panel.shape[:2]
    cx, cy = center
    x = int(np.clip(cx - pw // 2, 0, max(0, w - pw)))
    y = int(np.clip(cy - ph // 2, 0, max(0, h - ph)))

    display[y:y + ph, x:x + pw] = panel
    cv2.rectangle(display, (x, y), (x + pw, y + ph), border_color, 2)


class FaceRegionTracker:
    def __init__(self):
        with open(_MODEL_PATH, 'rb') as f:
            model_data = f.read()
        base_options = mp_python.BaseOptions(model_asset_buffer=model_data)
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            running_mode=mp_vision.RunningMode.VIDEO,
        )
        self.detector = mp_vision.FaceLandmarker.create_from_options(options)

    def get_region_crop(self, frame, region_key):
        if region_key not in REGIONS:
            return None, None

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(time.time() * 1000)
        result = self.detector.detect_for_video(mp_image, timestamp_ms)
        if not result.face_landmarks:
            return None, None

        landmarks = result.face_landmarks[0]
        region = REGIONS[region_key]
        bbox = _get_bbox(landmarks, region["indices"], h, w, region["padding"])
        x1, y1, x2, y2 = bbox
        return bbox, frame[y1:y2, x1:x2].copy()

    def release(self):
        self.detector.close()
