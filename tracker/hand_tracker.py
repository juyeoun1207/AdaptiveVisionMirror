import cv2
import mediapipe as mp
import numpy as np
import os
import time
from collections import deque
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.components import containers

# MediaPipe가 Windows에서 절대경로를 잘못 조합하는 버그가 있어서 파일을 직접 읽어서 넘김
_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hand_landmarker.task')

class HandTracker:
    def __init__(self, smooth_window=5):
        self.history = deque(maxlen=smooth_window)  # deq 이용해서 최근 좌표 5개만 저장 (스무딩용!!!)
        self.latest_pos = None
        self.index_tip = None

        with open(_MODEL_PATH, 'rb') as f:
            model_data = f.read()
        base_options = mp_python.BaseOptions(
            model_asset_buffer=model_data
        )
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,  # 최대 손 1개만 인식 (일단은 처음 인식된 손을 쭉 인식)
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            running_mode=mp_vision.RunningMode.VIDEO
        )
        self.detector = mp_vision.HandLandmarker.create_from_options(options)

    def get_hand_position(self, frame):
        
        h, w = frame.shape[:2]  # 참고로 채널은 BGR을 의미하긴 함 (h, w, c)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timestamp_ms = int(time.time() * 1000)
        result = self.detector.detect_for_video(mp_image, timestamp_ms)

        if not result.hand_landmarks:
            self.history.clear()
            self.index_tip = None
            return None

        # 엄지(1~4)와 검지(5~8) 랜드마크 평균
        landmarks = result.hand_landmarks[0]
        THUMB_IDX = [4]
        INDEX_IDX = [8]
        selected = [landmarks[i] for i in THUMB_IDX + INDEX_IDX]
        raw_x = int(np.mean([lm.x for lm in selected]) * w)
        raw_y = int(np.mean([lm.y for lm in selected]) * h)

        self.index_tip = (int(landmarks[8].x * w), int(landmarks[8].y * h))

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

    cv2.namedWindow("Vision Test", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Vision Test", 1280, 720)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = cv2.flip(frame, 1)
        pos = tracker.get_hand_position(display)
        if pos:
            cv2.circle(display, pos, 10, (0, 255, 0), -1)  #원 그릴 이미지, 원의 중심좌표, 원의 반지름, BGR, 원을 채워서 그림
            cv2.putText(display, f"{pos}", pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2) #텍스트 그릴 이미지, 표시할 문자열, 텍스트 시작 위치, 글꼴, 글자크기, 흰색, 글자 두께

        cv2.imshow("Vision Test", display) # 창에 메시지 띄움
        if cv2.waitKey(1) & 0xFF == ord('q'):  # 1ms 동안 키 입력 기다림 -> q 눌리면 종료 
            break

    cap.release() # 종료하면 -> 카메라 해제 
    cv2.destroyAllWindows() # OpenCV로 만든 모든 창 닫기 
