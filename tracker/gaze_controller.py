import cv2
import numpy as np
from collections import deque
from eyeGestures import EyeGestures_v3


class GazeController:
    def __init__(self, screen_width=1280, screen_height=720,
                 smooth_window=5, pip_smooth_window=15):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.gestures = EyeGestures_v3()

        # 처음에는 캘리브레이션 모드로 시작
        self.calibrate = True

        self.gaze_history = deque(maxlen=smooth_window)
        self.pip_history = deque(maxlen=pip_smooth_window)

        self.prev_x = screen_width // 2
        self.prev_y = screen_height // 2
        self.pip_x = screen_width // 2
        self.pip_y = screen_height // 2

        self.pip_pinned = False
        self.pinned_pip_pos = (self.pip_x, self.pip_y)

    def toggle_calibrate(self):
        self.calibrate = not self.calibrate

    def toggle_pip_pin(self):
        self.pip_pinned = not self.pip_pinned
        if self.pip_pinned:
            self.pinned_pip_pos = (self.pip_x, self.pip_y)

    def get_gaze_position(self, frame):
        try:
            event, cevent = self.gestures.step(
                frame,
                self.calibrate,
                self.screen_width,
                self.screen_height,
                context="main"
            )
        except Exception:
            if self.pip_pinned:
                return (self.prev_x, self.prev_y), self.pinned_pip_pos, None
            return (self.prev_x, self.prev_y), (self.pip_x, self.pip_y), None

        if event is not None:
            x = int(np.clip(event.point[0], 0, self.screen_width - 1))
            y = int(np.clip(event.point[1], 0, self.screen_height - 1))

            self.gaze_history.append((x, y))
            self.prev_x = int(np.mean([p[0] for p in self.gaze_history]))
            self.prev_y = int(np.mean([p[1] for p in self.gaze_history]))

            if not self.pip_pinned:
                self.pip_history.append((x, y))
                new_pip_x = int(np.mean([p[0] for p in self.pip_history]))
                new_pip_y = int(np.mean([p[1] for p in self.pip_history]))

                if abs(new_pip_x - self.pip_x) > 20 or abs(new_pip_y - self.pip_y) > 20:
                    self.pip_x = new_pip_x
                    self.pip_y = new_pip_y

        if self.pip_pinned:
            return (self.prev_x, self.prev_y), self.pinned_pip_pos, cevent

        return (self.prev_x, self.prev_y), (self.pip_x, self.pip_y), cevent


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    screen_width, screen_height = 1280, 720
    tracker = GazeController(screen_width, screen_height)

    cv2.namedWindow("Gaze Controller Test", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Gaze Controller Test", screen_width, screen_height)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = cv2.flip(frame, 1)

        gaze_pos, pip_pos, cevent = tracker.get_gaze_position(frame)

        # 캘리브레이션 타겟 표시
        if cevent is not None and tracker.calibrate:
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

        # 현재 추정 gaze 위치
        cv2.circle(display, gaze_pos, 10, (0, 0, 255), -1)

        # PIP 위치 표시
        pip_size = 100
        px, py = pip_pos

        cv2.rectangle(display,
                      (px - pip_size // 2, py - pip_size // 2),
                      (px + pip_size // 2, py + pip_size // 2),
                      (255, 100, 0),
                      3)

        mode_status = "Calibrating (C: stop)" if tracker.calibrate else "Tracking (C: calibrate)"
        pin_status = "PINNED (T: unpin)" if tracker.pip_pinned else "UNPINNED (T: pin)"
        status = f"{mode_status} | {pin_status} | Q: quit"

        cv2.putText(display,
                    status,
                    (10, screen_height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    2)

        cv2.imshow("Gaze Controller Test", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord("c"):
            tracker.toggle_calibrate()

        if key == ord("t"):
            tracker.toggle_pip_pin()

    cap.release()
    cv2.destroyAllWindows()