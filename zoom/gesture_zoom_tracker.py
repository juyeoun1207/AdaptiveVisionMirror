import math

import cv2
import mediapipe as mp


THUMB_TIP = 4
INDEX_TIP = 8

# Distance change of 1px becomes this much zoom delta.
SENSITIVITY = 0.015

# Ignore tiny finger jitter.
DEAD_ZONE_PX = 0.5

# Stroke-style pinch thresholds.
# - Start near and spread fingers to zoom in.
# - Start far and pinch fingers together to zoom out.
# Returning to the start position after a stroke is ignored, so zoom stays fixed.
PINCH_START_PX = 30.0  # 적당히 오므려도 줌 인 준비 완료
PINCH_END_PX = 90.0

MODE_IDLE = "idle"
MODE_ZOOM_IN = "zoom_in"
MODE_ZOOM_OUT = "zoom_out"
MODE_REARM_CLOSE = "rearm_close"
MODE_REARM_OPEN = "rearm_open"


class GestureZoomTracker:
    # Track thumb/index pinch strokes and return an incremental zoom delta.

    def __init__(self, max_hands: int = 2, detection_confidence: float = 0.7):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=0.6,
        )
        self._prev_distance: float | None = None
        self._mode = MODE_IDLE

    @staticmethod
    def _euclidean(p1: tuple, p2: tuple) -> float:
        """Euclidean distance between two (x, y) pixel points."""
        return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

    def _reset_stroke(self):
        self._prev_distance = None
        self._mode = MODE_IDLE

    def _calc_zoom_delta(self, current_dist: float) -> float:
        if self._mode == MODE_IDLE:
            if current_dist <= PINCH_START_PX:
                self._mode = MODE_ZOOM_IN
                self._prev_distance = current_dist
            elif current_dist >= PINCH_END_PX:
                self._mode = MODE_ZOOM_OUT
                self._prev_distance = current_dist
            return 0.0

        if self._mode == MODE_REARM_CLOSE:
            if current_dist <= PINCH_START_PX:
                self._mode = MODE_ZOOM_IN
                self._prev_distance = current_dist
            return 0.0

        if self._mode == MODE_REARM_OPEN:
            if current_dist >= PINCH_END_PX:
                self._mode = MODE_ZOOM_OUT
                self._prev_distance = current_dist
            return 0.0

        if self._prev_distance is None:
            self._prev_distance = current_dist
            return 0.0

        diff = current_dist - self._prev_distance
        self._prev_distance = current_dist

        if self._mode == MODE_ZOOM_IN:
            if diff <= 0:
                return 0.0

        if self._mode == MODE_ZOOM_OUT:
            if diff >= 0:
                return 0.0

        if abs(diff) < DEAD_ZONE_PX:
            return 0.0

        zoom_delta = round(diff * SENSITIVITY, 4)

        if self._mode == MODE_ZOOM_IN and current_dist >= PINCH_END_PX:
            self._mode = MODE_REARM_CLOSE
            self._prev_distance = None
        elif self._mode == MODE_ZOOM_OUT and current_dist <= PINCH_START_PX:
            self._mode = MODE_REARM_OPEN
            self._prev_distance = None

        return zoom_delta

    def _pick_landmarks(self, hand_landmarks, w, h, ignore_point=None, ignore_radius_px=120):
        if ignore_point is None:
            return hand_landmarks[0].landmark

        ix, iy = ignore_point
        fallback = None
        fallback_dist = -1

        for hand in hand_landmarks:
            landmarks = hand.landmark
            thumb_tip = (
                int(landmarks[THUMB_TIP].x * w),
                int(landmarks[THUMB_TIP].y * h),
            )
            index_tip = (
                int(landmarks[INDEX_TIP].x * w),
                int(landmarks[INDEX_TIP].y * h),
            )
            pinch_center = (
                (thumb_tip[0] + index_tip[0]) // 2,
                (thumb_tip[1] + index_tip[1]) // 2,
            )
            dist_from_ignored = self._euclidean(pinch_center, (ix, iy))

            if dist_from_ignored > fallback_dist:
                fallback = landmarks
                fallback_dist = dist_from_ignored
            if dist_from_ignored >= ignore_radius_px:
                return landmarks

        if fallback_dist < ignore_radius_px:
            self._reset_stroke()
            return None
        return fallback

    def process_frame(self, frame_bgr, ignore_point=None) -> dict | None:
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)

        if not result.multi_hand_landmarks:
            self._reset_stroke()
            return None

        landmarks = self._pick_landmarks(result.multi_hand_landmarks, w, h, ignore_point)
        if landmarks is None:
            return None

        thumb_tip = (
            int(landmarks[THUMB_TIP].x * w),
            int(landmarks[THUMB_TIP].y * h),
        )
        index_tip = (
            int(landmarks[INDEX_TIP].x * w),
            int(landmarks[INDEX_TIP].y * h),
        )

        pinch_dist = self._euclidean(thumb_tip, index_tip)
        zoom_delta = self._calc_zoom_delta(pinch_dist)

        return {
            "index_tip": index_tip,
            "pinch_distance": round(pinch_dist, 2),
            "zoom_delta": zoom_delta,
            "zoom_mode": self._mode,
        }

    def release(self):
        self.hands.close()


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    tracker = GestureZoomTracker()

    print("Move thumb/index pinch strokes. Press q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        data = tracker.process_frame(frame)

        if data:
            ix, iy = data["index_tip"]
            dist = data["pinch_distance"]
            delta = data["zoom_delta"]
            mode = data["zoom_mode"]

            cv2.circle(frame, (ix, iy), 8, (0, 255, 0), -1)
            cv2.putText(
                frame,
                f"pinch_distance: {dist:.1f}px",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                f"zoom_delta: {delta:+.4f}  mode: {mode}",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 100) if delta >= 0 else (0, 100, 255),
                2,
            )

            print(data)

        cv2.imshow("GestureZoomTracker", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    tracker.release()
    cap.release()
    cv2.destroyAllWindows()
