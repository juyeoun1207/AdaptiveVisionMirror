import os
import sys

import cv2
import numpy as np


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRACKER_DIR = os.path.join(BASE_DIR, "tracker")
ZOOM_DIR = os.path.join(BASE_DIR, "zoom")
sys.path.insert(0, TRACKER_DIR)
sys.path.insert(0, ZOOM_DIR)

from face_region_tracker import (  # noqa: E402
    REGIONS,
    ZOOM_MAX,
    ZOOM_MIN,
    FaceRegionTracker,
    get_region_name,
    make_zoom_panel,
    overlay_panel,
)
from gaze_controller import GazeController  # noqa: E402
from gesture_zoom_tracker import GestureZoomTracker  # noqa: E402
from hand_tracker import HandTracker  # noqa: E402


SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720

HAND_PIP_SIZE = 240
HAND_CROP_RADIUS = 60
FACE_PANEL_DEFAULT_SIZE = (420, 300)
FACE_PANEL_MIN_SIZE = (280, 220)
FACE_PANEL_MAX_SIZE = (620, 420)
FACE_PANEL_BORDER = (255, 100, 0)
FACE_BBOX_SMOOTHING = 0.18
FIXED_PIP_POS = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)


def clamp_zoom(zoom_factor):
    return max(ZOOM_MIN, min(ZOOM_MAX, zoom_factor))


def open_camera():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, SCREEN_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, SCREEN_HEIGHT)
    return cap


class SmoothPanelSizer:
    def get_size(self, crop):
        if crop is None or crop.size == 0:
            return FACE_PANEL_DEFAULT_SIZE

        h, w = crop.shape[:2]
        aspect = w / h if h else 1.0
        min_w, min_h = FACE_PANEL_MIN_SIZE
        max_w, max_h = FACE_PANEL_MAX_SIZE

        base_area = FACE_PANEL_DEFAULT_SIZE[0] * FACE_PANEL_DEFAULT_SIZE[1]
        target_w = float(np.sqrt(base_area * aspect))
        target_h = target_w / max(aspect, 0.01)

        if target_w < min_w:
            target_w = min_w
            target_h = target_w / max(aspect, 0.01)
        if target_h < min_h:
            target_h = min_h
            target_w = target_h * aspect
        if target_w > max_w:
            target_w = max_w
            target_h = target_w / max(aspect, 0.01)
        if target_h > max_h:
            target_h = max_h
            target_w = target_h * aspect

        target_w = max(1, int(round(target_w)))
        target_h = max(1, int(round(target_h)))
        return target_w, target_h


class SmoothRegionCropper:
    def __init__(self):
        self._region_key = None
        self._bbox = None

    def update(self, frame, bbox, region_key):
        if bbox is None:
            return None

        target = np.array(bbox, dtype=np.float32)
        if self._bbox is None or region_key != self._region_key:
            self._region_key = region_key
            self._bbox = target
        else:
            self._bbox = self._bbox + (target - self._bbox) * FACE_BBOX_SMOOTHING

        h, w = frame.shape[:2]
        x1, y1, x2, y2 = np.rint(self._bbox).astype(int)
        x1 = int(np.clip(x1, 0, w - 1))
        y1 = int(np.clip(y1, 0, h - 1))
        x2 = int(np.clip(x2, x1 + 1, w))
        y2 = int(np.clip(y2, y1 + 1, h))
        return frame[y1:y2, x1:x2].copy()


def draw_calibration_target(display, cevent):
    if cevent is None:
        return

    cx = int(np.clip(cevent.point[0], 0, SCREEN_WIDTH - 1))
    cy = int(np.clip(cevent.point[1], 0, SCREEN_HEIGHT - 1))

    cv2.circle(display, (cx, cy), 30, (255, 0, 0), 3)
    cv2.circle(display, (cx, cy), 15, (0, 255, 0), 2)
    cv2.circle(display, (cx, cy), 5, (255, 255, 0), -1)
    cv2.putText(display, "Look at the circle. Press C when ready.",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


def draw_status(display, text, y=None, color=(0, 255, 255), scale=0.6):
    if y is None:
        y = SCREEN_HEIGHT - 20
    cv2.putText(display, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)


def draw_hand_pip_from_point(display, source_point, pip_pos, zoom_factor=1.0):
    if source_point is None or pip_pos is None:
        return

    h, w = display.shape[:2]
    sx, sy = source_point
    crop_radius = max(12, int(HAND_CROP_RADIUS / zoom_factor))
    x1 = max(0, sx - crop_radius)
    y1 = max(0, sy - crop_radius)
    x2 = min(w, sx + crop_radius)
    y2 = min(h, sy + crop_radius)
    crop = display[y1:y2, x1:x2].copy()
    if crop.size == 0:
        return

    pip_img = cv2.resize(crop, (HAND_PIP_SIZE, HAND_PIP_SIZE))
    px, py = pip_pos
    bx = max(0, min(w - HAND_PIP_SIZE, px - HAND_PIP_SIZE // 2))
    by = max(0, min(h - HAND_PIP_SIZE, py - HAND_PIP_SIZE // 2))
    display[by:by + HAND_PIP_SIZE, bx:bx + HAND_PIP_SIZE] = pip_img
    cv2.rectangle(display, (bx, by), (bx + HAND_PIP_SIZE, by + HAND_PIP_SIZE), (255, 100, 0), 3)


def draw_region_bbox(display, bbox, region_key):
    if bbox is None or region_key not in REGIONS:
        return

    x1, y1, x2, y2 = bbox
    region = REGIONS[region_key]
    cv2.rectangle(display, (x1, y1), (x2, y2), region["color"], 2)
    cv2.putText(display, region["name"], (x1, max(y1 - 8, 14)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, region["color"], 2)


def handle_common_keys(key, gaze_controller=None):
    if key == ord("q"):
        return True
    if gaze_controller is not None and key == ord("c"):
        gaze_controller.toggle_calibrate()
    if gaze_controller is not None and key == ord("t"):
        gaze_controller.toggle_pip_pin()
    return False


def run_hand_tracker_test():
    cap = open_camera()
    tracker = HandTracker()

    window_name = "Hand Tracker Test"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, SCREEN_WIDTH, SCREEN_HEIGHT)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = cv2.flip(frame, 1)
        hand_pos = tracker.get_hand_position(display)

        if hand_pos is not None:
            cv2.circle(display, hand_pos, 10, (0, 255, 0), -1)
            cv2.putText(display, f"Hand {hand_pos}", hand_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        if tracker.index_tip is not None:
            cv2.circle(display, tracker.index_tip, 7, (255, 255, 0), -1)

        draw_status(display, "Hand Tracker Test | Q: quit")
        cv2.imshow(window_name, display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def run_gaze_controller_test():
    cap = open_camera()
    gaze_controller = GazeController(SCREEN_WIDTH, SCREEN_HEIGHT)

    window_name = "Gaze Controller Test"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, SCREEN_WIDTH, SCREEN_HEIGHT)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = cv2.flip(frame, 1)
        gaze_pos, pip_pos, cevent = gaze_controller.get_gaze_position(frame)

        if gaze_controller.calibrate:
            draw_calibration_target(display, cevent)
        if gaze_pos is not None:
            cv2.circle(display, gaze_pos, 10, (0, 0, 255), -1)
        if pip_pos is not None:
            cv2.rectangle(display, (pip_pos[0] - 50, pip_pos[1] - 50),
                          (pip_pos[0] + 50, pip_pos[1] + 50), (255, 100, 0), 3)

        mode_status = "Calibrating (C: stop)" if gaze_controller.calibrate else "Tracking (C: calibrate)"
        pin_status = "PINNED (T: unpin)" if gaze_controller.pip_pinned else "UNPINNED (T: pin)"
        draw_status(display, f"{mode_status} | {pin_status} | Q: quit")
        cv2.imshow(window_name, display)

        if handle_common_keys(cv2.waitKey(1) & 0xFF, gaze_controller):
            break

    cap.release()
    cv2.destroyAllWindows()


def run_hand_magnifier(use_gaze=False):
    cap = open_camera()
    hand_tracker = HandTracker()
    gesture_tracker = GestureZoomTracker(max_hands=2)
    gaze_controller = GazeController(SCREEN_WIDTH, SCREEN_HEIGHT) if use_gaze else None
    zoom_factor = 1.0

    window_name = "Hand Magnifier + Gaze PIP" if use_gaze else "Hand Magnifier"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, SCREEN_WIDTH, SCREEN_HEIGHT)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = cv2.flip(frame, 1)
        hand_pos = hand_tracker.get_hand_position(display)
        index_tip = hand_tracker.index_tip

        if use_gaze:
            gaze_pos, pip_pos, cevent = gaze_controller.get_gaze_position(frame)
        else:
            gaze_pos, pip_pos, cevent = None, FIXED_PIP_POS, None

        gdata = gesture_tracker.process_frame(display, ignore_point=index_tip)
        if gdata is not None:
            zoom_factor = clamp_zoom(zoom_factor + gdata["zoom_delta"])
            cv2.circle(display, gdata["index_tip"], 8, (255, 220, 0), -1)

        if use_gaze and gaze_controller.calibrate:
            draw_calibration_target(display, cevent)
        draw_hand_pip_from_point(display, index_tip, pip_pos, zoom_factor)

        if gaze_pos is not None:
            cv2.circle(display, gaze_pos, 10, (0, 0, 255), -1)
        if hand_pos is not None:
            cv2.circle(display, hand_pos, 10, (0, 255, 0), -1)
        if index_tip is not None:
            cv2.circle(display, index_tip, 8, (255, 255, 0), -1)

        if use_gaze:
            mode_status = "Calibrating (C: stop)" if gaze_controller.calibrate else "Tracking (C: calibrate)"
            pin_status = "PINNED (T: unpin)" if gaze_controller.pip_pinned else "UNPINNED (T: pin)"
            status = f"{mode_status} | {pin_status} | Q: quit"
        else:
            status = "Hand Magnifier | Q: quit"
        draw_status(display, f"{status} | second hand pinch zoom x{zoom_factor:.2f}")

        cv2.imshow(window_name, display)
        key = cv2.waitKey(1) & 0xFF
        if handle_common_keys(key, gaze_controller):
            break
        if key == ord("z"):
            zoom_factor = 1.0

    gesture_tracker.release()
    cap.release()
    cv2.destroyAllWindows()


def run_face_region_zoom(use_gaze=False):
    cap = open_camera()
    face_tracker = FaceRegionTracker()
    gesture_tracker = GestureZoomTracker()
    gaze_controller = GazeController(SCREEN_WIDTH, SCREEN_HEIGHT) if use_gaze else None
    panel_sizer = SmoothPanelSizer()
    region_cropper = SmoothRegionCropper()

    active_key = None
    zoom_factor = 1.0
    window_name = "Face Region + Gaze Panel" if use_gaze else "Face Region Zoom"

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, SCREEN_WIDTH, SCREEN_HEIGHT)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = cv2.flip(frame, 1)

        if use_gaze:
            gaze_pos, pip_pos, cevent = gaze_controller.get_gaze_position(frame)
            if gaze_controller.calibrate:
                draw_calibration_target(display, cevent)
            if gaze_pos is not None:
                cv2.circle(display, gaze_pos, 10, (0, 0, 255), -1)
        else:
            gaze_pos, pip_pos = None, FIXED_PIP_POS

        bbox, crop = face_tracker.get_region_crop(display, active_key)
        draw_region_bbox(display, bbox, active_key)

        gdata = gesture_tracker.process_frame(display)
        if gdata is not None:
            zoom_factor = clamp_zoom(zoom_factor + gdata["zoom_delta"])
            cv2.circle(display, gdata["index_tip"], 8, (255, 220, 0), -1)

        if active_key in REGIONS:
            smooth_crop = region_cropper.update(display, bbox, active_key)
            panel_size = panel_sizer.get_size(smooth_crop)
            if smooth_crop is not None and smooth_crop.size > 0:
                panel = make_zoom_panel(smooth_crop, zoom_factor, panel_size)
            else:
                panel = np.zeros((panel_size[1], panel_size[0], 3), dtype=np.uint8)
                cv2.putText(panel, "No face region", (max(20, panel_size[0] // 4), panel_size[1] // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 220), 2)

            overlay_panel(display, panel, pip_pos, FACE_PANEL_BORDER)

        region_name = get_region_name(active_key)
        mode = "Gaze panel" if use_gaze else "Screen panel"
        draw_status(display,
                    f"{mode} | [{region_name}] E eyes N nose M mouth J jaw | Z reset | Q quit",
                    scale=0.5)
        draw_status(display, f"gesture zoom x{zoom_factor:.2f}", y=40, color=(255, 220, 0), scale=0.65)

        cv2.imshow(window_name, display)

        key = cv2.waitKey(1) & 0xFF
        if handle_common_keys(key, gaze_controller):
            break
        if key in REGIONS:
            active_key = key
            zoom_factor = 1.0
        if key == ord("z"):
            zoom_factor = 1.0

    gesture_tracker.release()
    face_tracker.release()
    cap.release()
    cv2.destroyAllWindows()


MODES = {
    ord("1"): ("Hand Tracker Test", run_hand_tracker_test),
    ord("2"): ("Gaze Controller Test", run_gaze_controller_test),
    ord("3"): ("Hand Magnifier", lambda: run_hand_magnifier(use_gaze=False)),
    ord("4"): ("Hand Magnifier + Gaze PIP", lambda: run_hand_magnifier(use_gaze=True)),
    ord("5"): ("Face Region Zoom", lambda: run_face_region_zoom(use_gaze=False)),
    ord("6"): ("Face Region + Gaze Panel", lambda: run_face_region_zoom(use_gaze=True)),
}

MENU_ITEMS = [
    ("1", "Hand Tracker Test", (0, 255, 0)),
    ("2", "Gaze Controller Test", (0, 100, 255)),
    ("3", "Hand Magnifier", (255, 220, 0)),
    ("4", "Hand Magnifier + Gaze PIP", (255, 150, 255)),
    ("5", "Face Region Zoom", (0, 255, 255)),
    ("6", "Face Region + Gaze Panel", (200, 100, 255)),
]


def show_menu(cap):
    while True:
        ret, frame = cap.read()
        if not ret:
            return None

        display = cv2.flip(frame, 1)
        panel = display.copy()
        cv2.rectangle(panel, (250, 120), (1030, 640), (0, 0, 0), -1)
        cv2.addWeighted(panel, 0.55, display, 0.45, 0, display)

        cv2.putText(display, "AdaptiveVisionMirror", (310, 190),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
        cv2.line(display, (310, 215), (990, 215), (180, 180, 180), 1)

        for i, (key, label, color) in enumerate(MENU_ITEMS):
            y = 270 + i * 55
            cv2.putText(display, f"[{key}]  {label}", (340, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.78, color, 2)

        cv2.putText(display, "[Q]  Quit", (340, 600),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (150, 150, 150), 1)

        cv2.imshow("AdaptiveVisionMirror", display)
        key = cv2.waitKey(1) & 0xFF
        if key in MODES:
            return key
        if key == ord("q"):
            return None


if __name__ == "__main__":
    while True:
        menu_cap = open_camera()
        cv2.namedWindow("AdaptiveVisionMirror", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("AdaptiveVisionMirror", SCREEN_WIDTH, SCREEN_HEIGHT)

        choice = show_menu(menu_cap)
        menu_cap.release()
        cv2.destroyAllWindows()

        if choice is None:
            break

        MODES[choice][1]()
