import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.components import containers

class HandTracker:
    def __init__(self, smooth_window=5):
        self.history = deque(maxlen=smooth_window)
        self.latest_pos = None

        base_options = mp_python.BaseOptions(
            model_asset_path='hand_landmarker.task'
        )
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            running_mode=mp_vision.RunningMode.VIDEO
        )
        self.detector = mp_vision.HandLandmarker.create_from_options(options)
        self.timestamp = 0

    def get_finger_position(self, frame):
        
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        self.timestamp += 1
        result = self.detector.detect_for_video(mp_image, self.timestamp)

        if not result.hand_landmarks:
            self.history.clear()
            return None

        # 손 전체 랜드마크 평균
        landmarks = result.hand_landmarks[0]
        raw_x = int(np.mean([lm.x for lm in landmarks]) * w)
        raw_y = int(np.mean([lm.y for lm in landmarks]) * h)

        # Moving Average 스무딩
        self.history.append((raw_x, raw_y))
        smooth_x = int(np.mean([p[0] for p in self.history]))
        smooth_y = int(np.mean([p[1] for p in self.history]))

        return (smooth_x, smooth_y)


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    tracker = HandTracker()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        pos = tracker.get_finger_position(frame)
        if pos:
            cv2.circle(frame, pos, 10, (0, 255, 0), -1)
            cv2.putText(frame, f"{pos}", pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Vision Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()