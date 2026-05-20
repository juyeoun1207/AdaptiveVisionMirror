import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh

SCREEN_W, SCREEN_H = 1280, 720
HALF_W = SCREEN_W // 2  # 카메라 / 줌 패널 각각 640px

# 각 부위 랜드마크 인덱스 (Face Mesh 468점 기준)
_EYE_IDX   = [33, 133, 159, 145, 263, 362, 386, 374, 70, 300]  # 양쪽 눈 + 눈썹 끝
_NOSE_IDX  = [1, 2, 4, 98, 327, 168, 197]                       # 코 끝 · 콧날 · 콧대
_MOUTH_IDX = [61, 291, 0, 17, 13, 14]                           # 입 꼭짓점 · 윗입술 · 아랫입술

REGIONS = {
    ord('e'): {'name': 'Eyes',  'indices': _EYE_IDX,   'padding': 0.55, 'color': (0, 255, 255)},
    ord('n'): {'name': 'Nose',  'indices': _NOSE_IDX,  'padding': 0.65, 'color': (0, 255, 0)},
    ord('m'): {'name': 'Mouth', 'indices': _MOUTH_IDX, 'padding': 0.55, 'color': (0, 100, 255)},
}


def _get_bbox(landmarks, indices, h, w, padding):
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
    pw, ph = int((x2 - x1) * padding), int((y2 - y1) * padding)
    return (max(0, x1 - pw), max(0, y1 - ph),
            min(w, x2 + pw), min(h, y2 + ph))


class FaceRegionZoom:
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def get_region_crop(self, frame, region_key):
        """
        frame: BGR 프레임 (display 기준, 이미 flip된 상태)
        region_key: ord('e') / ord('n') / ord('m')
        반환: (bbox=(x1,y1,x2,y2), crop) 또는 (None, None)
        """
        if region_key not in REGIONS:
            return None, None

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return None, None

        lm = results.multi_face_landmarks[0].landmark
        region = REGIONS[region_key]
        bbox = _get_bbox(lm, region['indices'], h, w, region['padding'])
        x1, y1, x2, y2 = bbox
        crop = frame[y1:y2, x1:x2].copy()
        return bbox, crop


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, SCREEN_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, SCREEN_H)

    zoomer = FaceRegionZoom()
    active_key = None  # 현재 선택된 부위 (None = 선택 없음)

    cv2.namedWindow('Face Region Zoom', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Face Region Zoom', SCREEN_W, SCREEN_H)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = cv2.flip(frame, 1)
        h, w = display.shape[:2]

        bbox, crop = zoomer.get_region_crop(display, active_key)

        # ── 왼쪽 패널: 카메라 (절반 너비로 축소) ──────────────────────
        left = cv2.resize(display, (HALF_W, SCREEN_H))

        if bbox is not None:
            region = REGIONS[active_key]
            x1, y1, x2, y2 = bbox
            scale = HALF_W / w
            lx1, lx2 = int(x1 * scale), int(x2 * scale)
            cv2.rectangle(left, (lx1, y1), (lx2, y2), region['color'], 2)
            cv2.putText(left, region['name'], (lx1, max(y1 - 8, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, region['color'], 2)

        region_name = REGIONS[active_key]['name'] if active_key in REGIONS else 'None'
        cv2.putText(left, f"Region: {region_name}  |  E: Eyes  N: Nose  M: Mouth  Q: Quit",
                    (8, SCREEN_H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # ── 오른쪽 패널: 확대 or 안내 ────────────────────────────────
        if crop is not None and crop.size > 0:
            right = cv2.resize(crop, (HALF_W, SCREEN_H))
        else:
            right = np.zeros((SCREEN_H, HALF_W, 3), dtype=np.uint8)
            lines = (
                [('얼굴이 감지되지 않습니다', (0, 0, 200))]
                if active_key in REGIONS
                else [
                    ('부위를 선택하세요', (200, 200, 200)),
                    ('E : 눈',           (0, 255, 255)),
                    ('N : 코',           (0, 255, 0)),
                    ('M : 입',           (0, 100, 255)),
                    ('Q : 종료',         (150, 150, 150)),
                ]
            )
            for i, (text, color) in enumerate(lines):
                cv2.putText(right, text, (30, 100 + i * 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

        canvas = np.hstack([left, right])
        cv2.imshow('Face Region Zoom', canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key in REGIONS:
            active_key = key

    cap.release()
    cv2.destroyAllWindows()
