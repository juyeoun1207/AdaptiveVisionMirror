# Adaptive Vision Mirror — UI 총괄 (`mirror_ui.py`)

> 저시력자를 위한 전체화면 웹캠 스마트 거울 앱  
> **PyQt5 + OpenCV** 기반, Windows 11 / Python 3.x

---

## 목차

1. [개요](#개요)
2. [실행 방법](#실행-방법)
3. [화면 구성](#화면-구성)
4. [주요 기능](#주요-기능)
5. [클래스 구조](#클래스-구조)
6. [팀 연동 API](#팀-연동-api)
7. [키보드 단축키](#키보드-단축키)
8. [모드 상수](#모드-상수)

---

## 개요

`mirror_ui.py`는 스마트 거울의 **UI 레이어 전체**를 담당합니다.  
카메라 피드 표시, 돋보기 박스, 사이드바 버튼, 상태 바를 구성하고  
**Role 2(Vision Engineer)** 및 **Voice 팀**이 연동할 공개 API를 제공합니다.

---

## 실행 방법

```bash
pip install PyQt5 opencv-python numpy
python mirror_ui.py
```

---

## 화면 구성

```
┌─────────────────────────────────────────────────────────┐
│  [사용자 뱃지]  [모드]          [자막]          [배율]   │  ← 상태 바
├─────────────────────────────────────────────────────────┤
│                                        ┌──────────────┐ │
│   웹캠 피드 (전체 화면, 거울 반전)       │  사이드바    │ │
│                                        │  🔍 기본모드 │ │
│   ┌──────────┐                         │  ✋ 손 추적  │ │
│   │ ZOOM BOX │  ← 드래그 / 고정 가능   │  👁 시선추적 │ │
│   └──────────┘                         │  🎙 마이크   │ │
│                                        │  🔲 돋보기   │ │
│                                        │  🔳 크기 +   │ │
│                                        │  ▪ 크기 -   │ │
│                                        └──────────────┘ │
├─────────────────────────────────────────────────────────┤
│  [눈 확대 E]  [코 확대 N]  [입 확대 M]  [줌 리셋 Z]  [화면 고정 T] │  ← 버튼 바
└─────────────────────────────────────────────────────────┘
```

---

## 주요 기능

### 사용자 선택 화면
- 앱 시작 시 전체화면 반투명 오버레이로 사용자 카드 2장 표시
- 카드 클릭 → 카메라 피드로 전환
- **카메라는 선택 화면이 표시되는 동안 백그라운드에서 워밍업** → 선택 즉시 피드 표시
- 사용자 추가: `UserSelectOverlay._USERS` 리스트에 `("User N", "이름")` 추가

### 웹캠 피드
- 거울 모드 (좌우 반전)
- `KeepAspectRatioByExpanding` — 검은 letterbox 없이 화면 전체를 꽉 채움
- `CaptureThread`: 최신 프레임만 보관하는 Lock 기반 폴링 구조 (드래그 끊김 방지)

### 돋보기 박스 (ZOOM BOX)
- 마우스로 자유롭게 드래그
- 더블클릭 또는 `T` 키로 위치 고정/해제
- 박스 크기 조절: 150px ~ 500px (50px 단위, 사이드바 버튼)
- 돋보기 켜기/끄기 토글
- 드래그 중 즉시 내용 갱신 (타이머 틱 대기 없음)
- 배율: 마우스 휠 또는 `update_tracking_data()` API로 제어 (1.0x ~ 5.0x)

### 성능 최적화
| 항목 | 내용 |
|------|------|
| 프레임 렌더링 | `QTimer(33ms)` 폴링 — signal emit 방식의 이벤트 큐 누적 제거 |
| 색상 변환 | `cvtColor` 1회만 수행, RGB 프레임 캐시 재사용 |
| 메인 피드 스케일링 | `FastTransformation` 사용 |
| 드래그 즉시 반응 | `position_changed` 시그널로 타이머 독립적 줌 갱신 |
| 줌 크롭 | 슬라이딩 윈도우 방식 — 경계 클램핑 시 배율 뛰는 현상 방지 |

---

## 클래스 구조

```
SmartMirrorApp (QMainWindow)
├── FeedContainer (QWidget)
│   ├── _feed_label (QLabel)          — 웹캠 피드 전체 화면
│   ├── zoom_box: DraggableLabel      — 돋보기 박스
│   └── sidebar: SideBar              — 우측 버튼 사이드바
├── CaptureThread (QThread)           — 웹캠 캡처 (백그라운드)
└── UserSelectOverlay (QWidget)       — 시작 시 사용자 선택 오버레이
    └── UserCard (QWidget) × 2
```

### `CaptureThread`
웹캠 프레임을 최대 속도로 읽어 `threading.Lock`으로 최신 프레임 1장만 보관.  
메인 스레드는 `QTimer(33ms)`로 `latest_frame()`을 폴링해 렌더링.

### `DraggableLabel`
돋보기 박스. 드래그/고정/크기 조절 지원.  
`position_changed = pyqtSignal()` — 드래그 시 즉시 줌 갱신 트리거.

### `SideBar`
화면 우측 부유 버튼 바. 모드 전환 및 돋보기 제어 시그널 방출.

---

## 팀 연동 API

### Role 2 — Vision Engineer

```python
# 스무딩된 추적 좌표와 배율을 UI에 주입
window.update_tracking_data(
    x_norm:    float,   # 수평 정규화 좌표 [0.0 ~ 1.0]
    y_norm:    float,   # 수직 정규화 좌표  [0.0 ~ 1.0]
    zoom_scale: float   # 배율 [1.0 ~ 5.0]
)

# 현재 UI 모드 조회
mode = window.get_current_mode()  # → str (모드 상수 참고)
```

> ⚠️ **반드시 GUI(메인) 스레드에서 호출할 것.**  
> 별도 스레드에서 호출 시 `QMetaObject.invokeMethod` 사용:
> ```python
> QMetaObject.invokeMethod(window, "update_tracking_data",
>     Qt.QueuedConnection,
>     Q_ARG(float, x), Q_ARG(float, y), Q_ARG(float, z))
> ```

**모드 전환 버튼 연동** (사이드바 시그널에 추가 connect):
```python
window._feed.sidebar.mode_hand.connect(your_hand_tracking_start)
window._feed.sidebar.mode_gaze.connect(your_gaze_tracking_start)
```

**눈/코/입 확대 모드 연동** (`mirror_ui.py` 내 TODO 슬롯):
```python
def _on_eye_zoom(self):
    self._set_mode(MODE_EYE)
    # 여기에 vision_controller.set_roi("eye") 연결

def _on_nose_zoom(self):
    self._set_mode(MODE_NOSE)
    # 여기에 vision_controller.set_roi("nose") 연결

def _on_mouth_zoom(self):
    self._set_mode(MODE_MOUTH)
    # 여기에 vision_controller.set_roi("mouth") 연결
```

---

### Voice Team

```python
# 음성 인식 결과를 상단 자막 슬롯에 표시
window.update_subtitle(text: str)
```

---

## 키보드 단축키

| 키 | 동작 |
|----|------|
| `E` | 눈 확대 모드 |
| `N` | 코 확대 모드 |
| `M` | 입 확대 모드 |
| `Z` | 줌 1.0x 리셋 |
| `T` | 돋보기 박스 고정 / 해제 |
| `Esc` | 앱 종료 |
| 마우스 휠 | 배율 조절 (1.0x ~ 5.0x) |

---

## 모드 상수

| 상수 | 값 | 설명 |
|------|----|------|
| `MODE_IDLE` | `"대기 중"` | 사용자 선택 전 초기 상태 |
| `MODE_TRACKING` | `"손가락 추적"` | 기본 추적 모드 |
| `MODE_EYE` | `"눈 확대"` | Role 2: 눈 ROI 추적 |
| `MODE_NOSE` | `"코 확대"` | Role 2: 코 ROI 추적 |
| `MODE_MOUTH` | `"입 확대"` | Role 2: 입 ROI 추적 |
| `MODE_PINNED` | `"얼굴 부위 고정"` | 줌 박스 위치 고정 |
| `MODE_HAND` | `"손 추적"` | Role 2: 손 제스처 추적 |
| `MODE_GAZE` | `"시선 추적"` | Role 2: 시선 추적 |
