import cv2
import numpy as np
import os
from collections import deque
from eyeGestures import EyeGestures_v3

CALIB_SAVE_PATH = "gaze_calibration.pkl"

class GazeTracker:
    def __init__(self, screen_width=1280, screen_height=720,
                 smooth_window=5, pip_smooth_window=15):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.gestures = EyeGestures_v3()
        self.calibrate = True

        self.gaze_history = deque(maxlen=smooth_window)
        self.pip_history = deque(maxlen=pip_smooth_window)

        self.prev_x = screen_width // 2
        self.prev_y = screen_height // 2
        self.pip_x = screen_width // 2
        self.pip_y = screen_height // 2

        # 저장된 캘리브레이션 있으면 불러오기
        if os.path.exists(CALIB_SAVE_PATH):
            try:
                with open(CALIB_SAVE_PATH, "rb") as f:
                    data = f.read()
                self.gestures.loadModel(data, "main")
                print(f"캘리브레이션 불러옴: {CALIB_SAVE_PATH}")
            except Exception as e:
                print(f"캘리브레이션 불러오기 실패: {e}")

    def set_calibrate(self, value: bool):
        self.calibrate = value

    def save_calibration(self):
        try:
            data = self.gestures.saveModel("main")
            if data:
                with open(CALIB_SAVE_PATH, "wb") as f:
                    f.write(data)
                print(f"캘리브레이션 저장됨: {CALIB_SAVE_PATH}")
            else:
                print("저장할 데이터 없음")
        except Exception as e:
            print(f"저장 실패: {e}")

    def get_gaze_position(self, frame):
        try:
            event, cevent = self.gestures.step(
                frame, self.calibrate,
                self.screen_width, self.screen_height,
                context="main"
            )
        except Exception:
            return (self.prev_x, self.prev_y), (self.pip_x, self.pip_y), None

        if event is not None:
            x = int(np.clip(event.point[0], 0, self.screen_width - 1))
            y = int(np.clip(event.point[1], 0, self.screen_height - 1))

            self.gaze_history.append((x, y))
            self.prev_x = int(np.mean([p[0] for p in self.gaze_history]))
            self.prev_y = int(np.mean([p[1] for p in self.gaze_history]))

            self.pip_history.append((x, y))
            new_pip_x = int(np.mean([p[0] for p in self.pip_history]))
            new_pip_y = int(np.mean([p[1] for p in self.pip_history]))

            if abs(new_pip_x - self.pip_x) > 20 or abs(new_pip_y - self.pip_y) > 20:
                self.pip_x = new_pip_x
                self.pip_y = new_pip_y

        return (self.prev_x, self.prev_y), (self.pip_x, self.pip_y), cevent


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    screen_width, screen_height = 1280, 720
    tracker = GazeTracker(screen_width, screen_height)

    cv2.namedWindow("Gaze Tracker Test", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Gaze Tracker Test", screen_width, screen_height)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = cv2.flip(frame, 1)
        gaze_pos, pip_pos, cevent = tracker.get_gaze_position(frame)

        if cevent is not None:
            cx = int(np.clip(cevent.point[0], 0, screen_width - 1))
            cy = int(np.clip(cevent.point[1], 0, screen_height - 1))
            cv2.circle(display, (cx, cy), 30, (255, 0, 0), 3)
            cv2.circle(display, (cx, cy), 15, (0, 255, 0), 2)
            cv2.circle(display, (cx, cy), 5, (255, 255, 0), -1)
            cv2.putText(display, "Look at the blue circle! (S: save | C: toggle | Q: quit)",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.circle(display, gaze_pos, 10, (0, 0, 255), -1)

        pip_size = 100
        px, py = pip_pos
        cv2.rectangle(display,
                      (px - pip_size//2, py - pip_size//2),
                      (px + pip_size//2, py + pip_size//2),
                      (255, 100, 0), 3)

        status = "Calibrating (C: stop)" if tracker.calibrate else "Tracking (C: calibrate)"
        cv2.putText(display, status,
                    (10, screen_height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Gaze Tracker Test", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("c"):
            tracker.set_calibrate(not tracker.calibrate)
        if key == ord("s"):
            tracker.save_calibration()  # 수동 저장

    cap.release()
    cv2.destroyAllWindows()