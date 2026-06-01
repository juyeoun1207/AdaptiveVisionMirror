"""
mediapipe 0.10.30+ 호환성 shim
mp.solutions API가 제거되어 eyeGestures 라이브러리가 동작하지 않는 문제를 해결.
FaceMesh를 Tasks API로 재구현해 mp.solutions.face_mesh 네임스페이스에 주입한다.
main.py 최상단에서 import mediapipe_compat 으로 로드할 것.
"""
import os
import time
import types

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

if hasattr(mp, 'solutions'):
    pass  # 이미 solutions가 있으면 패치 불필요
else:
    _MODEL_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'tracker', 'face_landmarker.task'
    )

    # mediapipe 소스 기준 face mesh 연결 상수
    _FACEMESH_LEFT_EYE = frozenset([
        (263, 249), (249, 390), (390, 373), (373, 374),
        (374, 380), (380, 381), (381, 382), (382, 362),
        (263, 466), (466, 388), (388, 387), (387, 386),
        (386, 385), (385, 384), (384, 398), (398, 362),
    ])
    _FACEMESH_RIGHT_EYE = frozenset([
        (33, 7),   (7, 163),  (163, 144), (144, 145),
        (145, 153),(153, 154),(154, 155), (155, 133),
        (33, 246), (246, 161),(161, 160), (160, 159),
        (159, 158),(158, 157),(157, 173), (173, 133),
    ])

    class _FaceMeshResult:
        def __init__(self, landmark_lists):
            self.multi_face_landmarks = landmark_lists  # None 또는 [_LandmarkList, ...]

    class _LandmarkList:
        def __init__(self, landmarks):
            self.landmark = landmarks  # list of NormalizedLandmark (x, y, z 속성 보유)

    class _FaceMesh:
        """mp.solutions.face_mesh.FaceMesh 호환 래퍼 (Tasks API 내부 사용)"""

        def __init__(self, refine_landmarks=True, static_image_mode=False,
                     min_detection_confidence=0.5, min_tracking_confidence=0.5,
                     max_num_faces=1):
            with open(_MODEL_PATH, 'rb') as f:
                model_data = f.read()
            running_mode = (mp_vision.RunningMode.IMAGE
                            if static_image_mode else mp_vision.RunningMode.VIDEO)
            options = mp_vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_buffer=model_data),
                num_faces=max_num_faces,
                min_face_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
                running_mode=running_mode,
            )
            self._detector = mp_vision.FaceLandmarker.create_from_options(options)
            self._static_mode = static_image_mode

        def process(self, rgb_image):
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
            if self._static_mode:
                result = self._detector.detect(mp_image)
            else:
                result = self._detector.detect_for_video(mp_image, int(time.time() * 1000))

            if not result.face_landmarks:
                return _FaceMeshResult(None)
            return _FaceMeshResult([_LandmarkList(lms) for lms in result.face_landmarks])

        def close(self):
            self._detector.close()

        def __enter__(self):
            return self

        def __exit__(self, *_):
            self.close()

    # mp.solutions.face_mesh 네임스페이스 주입
    _face_mesh_mod = types.ModuleType('mediapipe.solutions.face_mesh')
    _face_mesh_mod.FACEMESH_LEFT_EYE  = _FACEMESH_LEFT_EYE
    _face_mesh_mod.FACEMESH_RIGHT_EYE = _FACEMESH_RIGHT_EYE
    _face_mesh_mod.FaceMesh           = _FaceMesh

    _solutions_mod = types.ModuleType('mediapipe.solutions')
    _solutions_mod.face_mesh = _face_mesh_mod

    mp.solutions = _solutions_mod


# ── eyeGestures 3.x + numpy 2.x 호환 패치 ──────────────────────────────
# eyeGestures 3.2.4의 getLandmarks()가 numpy 2.0에서 제거된
# "스칼라 슬롯에 shape(1,) 배열 대입" 방식을 사용해 ValueError 발생.
# key_points[-1,0] = head_offset[:,0]  →  head_offset[0,0] 으로 수정.
import numpy as _np
import cv2 as _cv2

def _getLandmarks_patched(self, frame):
    frame = _cv2.cvtColor(frame, _cv2.COLOR_BGR2RGB)
    frame = _cv2.flip(frame, 1)

    self.face.process(frame, self.finder.find(frame))

    face_landmarks  = self.face.getLandmarks()
    l_eye           = self.face.getLeftEye()
    r_eye           = self.face.getRightEye()
    l_eye_landmarks = l_eye.getLandmarks()
    r_eye_landmarks = r_eye.getLandmarks()
    blink           = l_eye.getBlink() and r_eye.getBlink()

    x_offset = _np.min(face_landmarks[:, 0])
    y_offset = _np.min(face_landmarks[:, 1])
    x_width  = _np.max(face_landmarks[:, 0]) - x_offset
    y_width  = _np.max(face_landmarks[:, 1]) - y_offset

    head_offset = _np.zeros((1, 2))
    scale_x = scale_y = 1
    if _np.array_equal(self.starting_head_position, _np.zeros((1, 2))):
        self.starting_head_position = _np.array([[x_offset, y_offset]])
        self.starting_size          = _np.array([[x_width,  y_width]])
    else:
        head_offset = _np.array([[x_offset, y_offset]]) - self.starting_head_position
        scale_x     = self.starting_size[0, 0] / x_width
        scale_y     = self.starting_size[0, 1] / y_width

    key_points = _np.concatenate((
        l_eye_landmarks, r_eye_landmarks,
        _np.array([[scale_x, scale_y]]), head_offset,
    ))
    key_points[:, 0] = key_points[:, 0] - head_offset[0, 0]
    key_points[:, 1] = key_points[:, 1] - head_offset[0, 1]
    key_points[:, 0] = key_points[:, 0] * scale_x
    key_points[:, 1] = key_points[:, 1] * scale_y

    # numpy 2.x fix: 스칼라 슬롯에 배열이 아닌 스칼라 대입
    key_points[-1, 0] = float(head_offset[0, 0])
    key_points[-1, 1] = float(head_offset[0, 1])

    subframe = frame[
        int(y_offset): int(y_offset + y_width),
        int(x_offset): int(x_offset + x_width),
    ]
    return key_points, blink, subframe

try:
    from eyeGestures import EyeGestures_v3 as _EyeGestures_v3
    _EyeGestures_v3.getLandmarks = _getLandmarks_patched
except ImportError:
    pass  # eyeGestures 없으면 패치 불필요
