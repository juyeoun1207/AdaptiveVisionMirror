"""
gesture_zoom_tracker.py
-----------------------
역할: 엄지(landmark 4) ↔ 검지(landmark 8) 핀치 거리를 측정하여
      zoom_delta를 반환한다.

반환값 형식:
    {
        "index_tip": (x, y),       # 검지 끝 화면 좌표 (픽셀)
        "pinch_distance": 83.2,    # 현재 프레임 핀치 거리 (픽셀)
        "zoom_delta": +0.03        # 직전 프레임 대비 확대/축소 변화량
    }

Role 3(UI)에서 zoom_delta를 누적해 배율에 반영하면 됨.
"""

import math
import mediapipe as mp
import cv2


# ── 상수 ──────────────────────────────────────────────────────────────
THUMB_TIP  = 4   # 엄지 끝 랜드마크 인덱스
INDEX_TIP  = 8   # 검지 끝 랜드마크 인덱스

# zoom_delta 감도 조절: 픽셀 거리 변화 1px → delta 몇 배로 변환할지
SENSITIVITY = 0.005

# 노이즈 억제: 거리 변화가 이 픽셀 이하면 delta = 0 으로 처리
DEAD_ZONE_PX = 4.0


class GestureZoomTracker:
    """
    MediaPipe Hands를 사용해 핀치 제스처를 추적하고
    zoom_delta를 계산한다.
    """

    def __init__(self, max_hands: int = 1, detection_confidence: float = 0.7):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=0.6,
        )
        self._prev_distance: float | None = None  # 직전 프레임 핀치 거리

    # ── 내부 헬퍼 ─────────────────────────────────────────────────────

    @staticmethod
    def _euclidean(p1: tuple, p2: tuple) -> float:
        """두 (x, y) 픽셀 좌표 사이 유클리드 거리."""
        return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

    def _calc_zoom_delta(self, current_dist: float) -> float:
        """
        직전 프레임과 현재 프레임의 핀치 거리 차이로 zoom_delta 산출.
        - 양수 → 손가락 벌림 → 확대
        - 음수 → 손가락 좁힘 → 축소
        """
        if self._prev_distance is None:
            # 첫 프레임은 비교 대상이 없으므로 delta = 0
            self._prev_distance = current_dist
            return 0.0

        diff = current_dist - self._prev_distance
        self._prev_distance = current_dist

        # dead zone: 미세 떨림 무시
        if abs(diff) < DEAD_ZONE_PX:
            return 0.0

        return round(diff * SENSITIVITY, 4)

    # ── 공개 API ──────────────────────────────────────────────────────

    def process_frame(self, frame_bgr) -> dict | None:
        """
        BGR 프레임 하나를 입력받아 핀치 정보를 반환한다.

        Parameters
        ----------
        frame_bgr : np.ndarray
            cv2.VideoCapture 에서 읽은 BGR 프레임

        Returns
        -------
        dict | None
            손이 감지된 경우 → {"index_tip", "pinch_distance", "zoom_delta"}
            손이 없는 경우  → None  (prev_distance 초기화)
        """
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)

        if not result.multi_hand_landmarks:
            self._prev_distance = None  # 손 사라지면 기준점 리셋
            return None

        # 첫 번째 손만 사용
        landmarks = result.multi_hand_landmarks[0].landmark

        thumb_tip  = (int(landmarks[THUMB_TIP].x * w),
                      int(landmarks[THUMB_TIP].y * h))
        index_tip  = (int(landmarks[INDEX_TIP].x * w),
                      int(landmarks[INDEX_TIP].y * h))

        pinch_dist = self._euclidean(thumb_tip, index_tip)
        zoom_delta = self._calc_zoom_delta(pinch_dist)

        return {
            "index_tip"     : index_tip,
            "pinch_distance": round(pinch_dist, 2),
            "zoom_delta"    : zoom_delta,
        }

    def release(self):
        """MediaPipe 리소스 해제."""
        self.hands.close()


# ── 단독 실행 (디버그용) ───────────────────────────────────────────────
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    tracker = GestureZoomTracker()
    mp_draw = mp.solutions.drawing_utils

    print("핀치 제스처를 해보세요 (q: 종료)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        data = tracker.process_frame(frame)

        if data:
            ix, iy = data["index_tip"]
            dist   = data["pinch_distance"]
            delta  = data["zoom_delta"]

            # 검지 끝 표시
            cv2.circle(frame, (ix, iy), 8, (0, 255, 0), -1)

            # 정보 오버레이
            cv2.putText(frame, f"pinch_distance: {dist:.1f}px",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"zoom_delta:     {delta:+.4f}",
                        (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 100) if delta >= 0 else (0, 100, 255), 2)

            print(data)  # Role 3에 넘길 dict 확인용

        cv2.imshow("GestureZoomTracker", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    tracker.release()
    cap.release()
    cv2.destroyAllWindows()