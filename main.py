import sys
import os

TRACKER_DIR = os.path.join(os.path.dirname(__file__), "tracker")
sys.path.insert(0, TRACKER_DIR)
os.chdir(TRACKER_DIR)

import cv2
import numpy as np


SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720

PIP_SIZE = 240
CROP_RADIUS = 60


def draw_calibration_target(display, cevent, screen_width, screen_height):
    if cevent is None:
        return

    cx = int(np.clip(cevent.point[0], 0, screen_width - 1))
    cy = int(np.clip(cevent.point[1], 0, screen_height - 1))

    cv2.circle(display, (cx, cy), 30, (255, 0, 0), 3)
    cv2.circle(display, (cx, cy), 15, (0, 255, 0), 2)
    cv2.circle(display, (cx, cy), 5, (255, 255, 0), -1)

    cv2.putText(display,
                "Look at the circle. Press C when ready.",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2)


def draw_pip_from_point(display, source_point, pip_pos):
    h, w = display.shape[:2]

    if source_point is None or pip_pos is None:
        return

    sx, sy = source_point

    x1 = max(0, sx - CROP_RADIUS)
    y1 = max(0, sy - CROP_RADIUS)
    x2 = min(w, sx + CROP_RADIUS)
    y2 = min(h, sy + CROP_RADIUS)

    crop = display[y1:y2, x1:x2].copy()

    if crop.size == 0:
        return

    pip_img = cv2.resize(crop, (PIP_SIZE, PIP_SIZE))

    px, py = pip_pos
    bx = max(0, min(w - PIP_SIZE, px - PIP_SIZE // 2))
    by = max(0, min(h - PIP_SIZE, py - PIP_SIZE // 2))

    display[by:by + PIP_SIZE, bx:bx + PIP_SIZE] = pip_img
    cv2.rectangle(display,
                  (bx, by),
                  (bx + PIP_SIZE, by + PIP_SIZE),
                  (255, 100, 0),
                  3)


def run_hand_magnifier(use_gaze=True):
    from hand_tracker import HandTracker

    if use_gaze:
        from gaze_controller import GazeController
        gaze_controller = GazeController(SCREEN_WIDTH, SCREEN_HEIGHT)
    else:
        gaze_controller = None

    hand_tracker = HandTracker()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, SCREEN_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, SCREEN_HEIGHT)

    window_name = "Hand Magnifier + Gaze PIP" if use_gaze else "Hand Magnifier"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, SCREEN_WIDTH, SCREEN_HEIGHT)

    fixed_pip_pos = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)

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
            gaze_pos, pip_pos, cevent = None, fixed_pip_pos, None

        if use_gaze and gaze_controller.calibrate:
            draw_calibration_target(display, cevent, SCREEN_WIDTH, SCREEN_HEIGHT)

        draw_pip_from_point(display, index_tip, pip_pos)

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
            status = "Hand magnifier | Q: quit"

        cv2.putText(display,
                    status,
                    (10, SCREEN_HEIGHT - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    2)

        cv2.imshow(window_name, display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if use_gaze and key == ord("c"):
            gaze_controller.toggle_calibrate()

        if use_gaze and key == ord("t"):
            gaze_controller.toggle_pip_pin()

    cap.release()
    cv2.destroyAllWindows()


def run_hand_tracker_test():
    from hand_tracker import HandTracker

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, SCREEN_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, SCREEN_HEIGHT)

    tracker = HandTracker()

    cv2.namedWindow("Hand Tracker Test", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Hand Tracker Test", SCREEN_WIDTH, SCREEN_HEIGHT)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = cv2.flip(frame, 1)
        pos = tracker.get_hand_position(display)

        if pos:
            cv2.circle(display, pos, 10, (0, 255, 0), -1)
            cv2.putText(display, f"Hand {pos}", pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if tracker.index_tip:
            cv2.circle(display, tracker.index_tip, 7, (255, 255, 0), -1)

        cv2.putText(display,
                    "Hand Tracker Test | Q: quit",
                    (10, SCREEN_HEIGHT - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2)

        cv2.imshow("Hand Tracker Test", display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def run_gaze_controller_test():
    from gaze_controller import GazeController

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, SCREEN_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, SCREEN_HEIGHT)

    controller = GazeController(SCREEN_WIDTH, SCREEN_HEIGHT)

    cv2.namedWindow("Gaze Controller Test", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Gaze Controller Test", SCREEN_WIDTH, SCREEN_HEIGHT)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = cv2.flip(frame, 1)

        gaze_pos, pip_pos, cevent = controller.get_gaze_position(frame)

        if controller.calibrate:
            draw_calibration_target(display, cevent, SCREEN_WIDTH, SCREEN_HEIGHT)

        if gaze_pos:
            cv2.circle(display, gaze_pos, 10, (0, 0, 255), -1)

        if pip_pos:
            pip_size = 100
            px, py = pip_pos
            cv2.rectangle(display,
                          (px - pip_size // 2, py - pip_size // 2),
                          (px + pip_size // 2, py + pip_size // 2),
                          (255, 100, 0),
                          3)

        mode_status = "Calibrating (C: stop)" if controller.calibrate else "Tracking (C: calibrate)"
        pin_status = "PINNED (T: unpin)" if controller.pip_pinned else "UNPINNED (T: pin)"
        status = f"{mode_status} | {pin_status} | Q: quit"

        cv2.putText(display,
                    status,
                    (10, SCREEN_HEIGHT - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    2)

        cv2.imshow("Gaze Controller Test", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord("c"):
            controller.toggle_calibrate()

        if key == ord("t"):
            controller.toggle_pip_pin()

    cap.release()
    cv2.destroyAllWindows()


MODES = {
    ord("1"): ("Hand Tracker Test", run_hand_tracker_test),
    ord("2"): ("Gaze Controller Test", run_gaze_controller_test),
    ord("3"): ("Hand Magnifier", lambda: run_hand_magnifier(use_gaze=False)),
    ord("4"): ("Hand Magnifier + Gaze PIP", lambda: run_hand_magnifier(use_gaze=True)),
}

MENU_ITEMS = [
    ("1", "Hand Tracker Test", (0, 255, 0)),
    ("2", "Gaze Controller Test", (0, 100, 255)),
    ("3", "Hand Magnifier", (255, 220, 0)),
    ("4", "Hand Magnifier + Gaze PIP", (255, 150, 255)),
]


def show_menu(cap):
    while True:
        ret, frame = cap.read()
        if not ret:
            return None

        display = cv2.flip(frame, 1)

        panel = display.copy()
        cv2.rectangle(panel, (250, 160), (1030, 590), (0, 0, 0), -1)
        cv2.addWeighted(panel, 0.55, display, 0.45, 0, display)

        cv2.putText(display,
                    "AdaptiveVisionMirror",
                    (310, 230),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.1,
                    (255, 255, 255),
                    2)

        cv2.line(display, (310, 250), (990, 250), (180, 180, 180), 1)

        for i, (key, label, color) in enumerate(MENU_ITEMS):
            y = 310 + i * 65
            cv2.putText(display,
                        f"[{key}]  {label}",
                        (340, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.82,
                        color,
                        2)

        cv2.putText(display,
                    "[Q]  Quit",
                    (340, 550),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (150, 150, 150),
                    1)

        cv2.imshow("AdaptiveVisionMirror", display)

        key = cv2.waitKey(1) & 0xFF

        if key in MODES:
            return key

        if key == ord("q"):
            return None


if __name__ == "__main__":
    while True:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, SCREEN_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, SCREEN_HEIGHT)

        cv2.namedWindow("AdaptiveVisionMirror", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("AdaptiveVisionMirror", SCREEN_WIDTH, SCREEN_HEIGHT)

        choice = show_menu(cap)

        cap.release()
        cv2.destroyAllWindows()

        if choice is None:
            break

        MODES[choice][1]()