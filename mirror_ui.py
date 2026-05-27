#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adaptive Vision Mirror — 총괄 UI
main.py

╔══════════════════════════════════════════════════════════════════╗
║  ROLE 2 (Vision Engineer) 인터페이스 계약                        ║
║  window.update_tracking_data(                                    ║
║      x_norm   : float,   # 스무딩된 수평 좌표 [0.0 – 1.0]        ║
║      y_norm   : float,   # 스무딩된 수직 좌표  [0.0 – 1.0]        ║
║      zoom_scale: float   # 제스처 배율 [1.0 – 5.0]               ║
║  )                                                               ║
║  window.get_current_mode() → str   # 현재 UI 모드 조회            ║
║                                                                  ║
║  VOICE TEAM 인터페이스                                            ║
║  window.update_subtitle(text: str) # 자막 안내창 갱신              ║
╚══════════════════════════════════════════════════════════════════╝
"""

import sys
import threading
from typing import Optional

import cv2
import numpy as np

from PyQt5.QtCore import Qt, QPoint, QRectF, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


# ─────────────────────────────────────────────────────────────────────
#  모드 상수
# ─────────────────────────────────────────────────────────────────────

MODE_IDLE     = "대기 중"
MODE_TRACKING = "손가락 추적"
MODE_EYE      = "눈 확대"       # Role 2: 눈 ROI 추적 시작
MODE_NOSE     = "코 확대"       # Role 2: 코 ROI 추적 시작
MODE_MOUTH    = "입 확대"       # Role 2: 입 ROI 추적 시작
MODE_PINNED   = "얼굴 부위 고정"
MODE_HAND     = "손 추적"       # Role 2: 손 제스처 추적 시작
MODE_GAZE     = "시선 추적"     # Role 2: 시선 추적 시작


# ─────────────────────────────────────────────────────────────────────
#  웹캠 캡처 스레드
# ─────────────────────────────────────────────────────────────────────

class CaptureThread(QThread):
    """
    웹캠 프레임을 최대 속도로 읽어 최신 프레임만 보관한다.
    메인 스레드는 QTimer로 원하는 주기마다 latest_frame()을 폴링한다.
    시그널 emit 방식은 이벤트 큐 누적으로 드래그가 끊기는 원인이 된다.
    """

    def __init__(self, camera_index: int = 0, parent=None):
        super().__init__(parent)
        self._camera_index = camera_index
        self._running = False
        self._lock  = threading.Lock()
        self._frame: Optional[np.ndarray] = None

    def latest_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._frame

    def run(self):
        cap = cv2.VideoCapture(self._camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)
        self._running = True
        while self._running:
            ret, frame = cap.read()
            if ret:
                flipped = cv2.flip(frame, 1)
                with self._lock:
                    self._frame = flipped
            else:
                self.msleep(1)          # ret=False 시 CPU 스핀 방지
        cap.release()

    def stop(self):
        self._running = False
        self.wait()


# ─────────────────────────────────────────────────────────────────────
#  돋보기 박스 (DraggableLabel)
# ─────────────────────────────────────────────────────────────────────

class DraggableLabel(QLabel):
    """
    300×300 고정 비율 돋보기 박스.

    동작 규칙
    ---------
    • 기본(Free) 상태   : 마우스 드래그로 이동, 초록 테두리
    • 고정(Pinned) 상태 : 위치 잠금, 빨간 테두리, "[고정]" 배너 오버레이
                          영상은 고정 상태에서도 계속 실시간 갱신됨
    • 더블클릭 or 키 T  → 상태 토글
    """

    SIZE = 400

    position_changed = pyqtSignal()

    _STYLE_FREE = """
        border: 3px solid #00FF00;
        background: rgba(0, 0, 0, 180);
        color: #00FF00;
        font-size: 14px;
        font-weight: bold;
    """
    _STYLE_PINNED = """
        border: 4px solid #FF0000;
        background: rgba(0, 0, 0, 180);
        color: #FF0000;
        font-size: 14px;
        font-weight: bold;
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._size: int = self.SIZE
        self.setFixedSize(self._size, self._size)
        self.setAlignment(Qt.AlignCenter)
        self._pinned: bool = False
        self._drag_origin: Optional[QPoint] = None
        self._pos_origin:  Optional[QPoint] = None
        self._apply_style()

    # ── 공개 API ─────────────────────────────────────────────────────

    def is_pinned(self) -> bool:
        return self._pinned

    def toggle_pin(self):
        self._pinned = not self._pinned
        self._apply_style()

    def set_pinned(self, state: bool):
        if self._pinned != state:
            self._pinned = state
            self._apply_style()

    @property
    def current_size(self) -> int:
        return self._size

    def resize_box(self, new_size: int):
        self._size = new_size
        self.setFixedSize(new_size, new_size)
        if self.parent():
            pw = self.parent().width()
            ph = self.parent().height()
            self.move(
                max(0, min(self.x(), pw - new_size)),
                max(0, min(self.y(), ph - new_size)),
            )

    def set_frame(self, pixmap: QPixmap):
        """고정 여부와 무관하게 영상 갱신 — 위치만 잠긴다."""
        self.setPixmap(
            pixmap.scaled(
                self._size, self._size,
                Qt.KeepAspectRatioByExpanding,
                Qt.SmoothTransformation,
            )
        )
        # setPixmap이 paintEvent를 트리거하므로 고정 오버레이도 자동 재렌더링됨

    # ── 내부 ─────────────────────────────────────────────────────────

    def _apply_style(self):
        if self._pinned:
            self.setStyleSheet(self._STYLE_PINNED)
            # setText 호출 금지 — pixmap(마지막 프레임)을 그대로 유지
            self.update()
        else:
            self.setStyleSheet(self._STYLE_FREE)
            self.clear()
            self.setText("ZOOM AREA")
            self.update()

    def paintEvent(self, event):
        super().paintEvent(event)   # pixmap / 텍스트 먼저 그림
        if not self._pinned:
            return
        # 마지막 프레임 위에 "[고정]" 배너를 반투명 오버레이로 그림
        p = QPainter(self)
        p.fillRect(0, 0, self.width(), 26, QColor(180, 0, 0, 210))
        p.setPen(QColor("#FFFFFF"))
        p.setFont(QFont("Arial", 12, QFont.Bold))
        p.drawText(6, 18, "[고정]  ZOOM AREA")
        p.end()

    # ── 이벤트 ───────────────────────────────────────────────────────

    def mouseDoubleClickEvent(self, event):
        self.toggle_pin()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and not self._pinned:
            self._drag_origin = event.globalPos()
            self._pos_origin  = self.pos()

    def mouseMoveEvent(self, event):
        if self._pinned or self._drag_origin is None or self._pos_origin is None:
            return
        delta   = event.globalPos() - self._drag_origin
        new_pos = self._pos_origin + delta
        if self.parent():
            pw = self.parent().width()
            ph = self.parent().height()
            x  = max(0, min(new_pos.x(), pw - self._size))
            y  = max(0, min(new_pos.y(), ph - self._size))
            self.move(x, y)
            self.position_changed.emit()

    def mouseReleaseEvent(self, event):
        self._drag_origin = None
        self._pos_origin  = None

    def wheelEvent(self, event):
        # 휠 이벤트를 소비하지 않고 부모(SmartMirrorApp)로 전파
        event.ignore()


# ─────────────────────────────────────────────────────────────────────
#  사용자 선택 카드
# ─────────────────────────────────────────────────────────────────────

class UserCard(QWidget):
    """클릭 가능한 사용자 선택 카드."""

    clicked = pyqtSignal(str)   # 선택된 사용자 이름

    def __init__(self, user_num: str, user_name: str, parent=None):
        super().__init__(parent)
        self._user_num  = user_num
        self._user_name = user_name
        self._hovered   = False
        self.setFixedSize(280, 360)
        self.setCursor(Qt.PointingHandCursor)
        self._build()

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(20)
        layout.setContentsMargins(30, 40, 30, 40)

        initial = self._user_name[0] if self._user_name else "?"
        icon_lbl = QLabel(initial)
        icon_lbl.setAlignment(Qt.AlignCenter)
        icon_lbl.setFixedSize(90, 90)
        icon_lbl.setFont(QFont("Arial", 36, QFont.Bold))
        icon_lbl.setStyleSheet(
            "background:#4A90D9; color:#FFFFFF; border-radius:45px;"
        )
        layout.addWidget(icon_lbl, alignment=Qt.AlignHCenter)

        num_lbl = QLabel(self._user_num)
        num_lbl.setAlignment(Qt.AlignCenter)
        num_lbl.setFont(QFont("Arial", 15))
        num_lbl.setStyleSheet("color:#AAAAAA; background:transparent;")
        layout.addWidget(num_lbl)

        name_lbl = QLabel(self._user_name)
        name_lbl.setAlignment(Qt.AlignCenter)
        name_lbl.setFont(QFont("Arial", 28, QFont.Bold))
        name_lbl.setStyleSheet("color:#FFFFFF; background:transparent;")
        layout.addWidget(name_lbl)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        if self._hovered:
            fill   = QColor(255, 255, 255, 55)
            border = QColor(255, 255, 255, 190)
            bw = 3
        else:
            fill   = QColor(255, 255, 255, 18)
            border = QColor(255, 255, 255, 55)
            bw = 2
        rect = QRectF(bw / 2, bw / 2, self.width() - bw, self.height() - bw)
        p.setBrush(fill)
        p.setPen(QPen(border, bw))
        p.drawRoundedRect(rect, 20, 20)
        p.end()

    def enterEvent(self, event):
        self._hovered = True
        self.update()

    def leaveEvent(self, event):
        self._hovered = False
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self._user_name)


# ─────────────────────────────────────────────────────────────────────
#  사용자 선택 오버레이
# ─────────────────────────────────────────────────────────────────────

class UserSelectOverlay(QWidget):
    """
    앱 시작 시 전체 화면을 덮는 사용자 선택 오버레이.
    카드를 클릭하면 `finished(user_name)` 시그널을 방출하고 사라진다.
    """

    finished = pyqtSignal(str)

    # 사용자 추가 시 ("User N", "이름") 형태로 항목 추가
    _USERS = [
        ("User 1", "김지훈"),
        ("User 2", "박수정"),
    ]

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setGeometry(parent.rect())
        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        self._build_ui()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setAlignment(Qt.AlignCenter)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        title = QLabel("사용자를 선택하세요")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Arial", 28, QFont.Bold))
        title.setStyleSheet("color:#FFFFFF; background:transparent;")
        outer.addWidget(title)
        outer.addSpacing(48)

        row = QWidget()
        h = QHBoxLayout(row)
        h.setAlignment(Qt.AlignCenter)
        h.setSpacing(60)
        h.setContentsMargins(0, 0, 0, 0)
        for user_num, user_name in self._USERS:
            card = UserCard(user_num, user_name)
            card.clicked.connect(self._select)
            h.addWidget(card)
        outer.addWidget(row)

    def _select(self, user_name: str):
        self.hide()
        self.finished.emit(user_name)

    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(0, 0, 0, 215))
        p.end()


# ─────────────────────────────────────────────────────────────────────
#  우측 사이드바
# ─────────────────────────────────────────────────────────────────────

class SideBar(QWidget):
    """화면 우측 가장자리에 부유하는 반투명 세로 버튼 바."""

    WIDTH = 140

    mode_basic      = pyqtSignal()
    mode_hand       = pyqtSignal()
    mode_gaze       = pyqtSignal()
    toggle_zoom     = pyqtSignal()
    zoom_size_plus  = pyqtSignal()
    zoom_size_minus = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(self.WIDTH)
        self._mic_on      = True
        self._zoom_visible = True
        self._build()

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
        layout.setContentsMargins(8, 24, 8, 24)
        layout.setSpacing(14)

        self._btn_basic     = self._make_btn("🔍\nManual")
        self._btn_hand      = self._make_btn("✋\nHand Tracking")
        self._btn_gaze      = self._make_btn("👁\nGaze Tracking")
        self._btn_mic       = self._make_btn("🎙\nMIC ON")
        self._btn_zoom_tog  = self._make_btn("🔲\nTurnOff\n ZoomBox")
        self._btn_zoom_plus = self._make_btn("+", 50)
        self._btn_zoom_minus= self._make_btn("-\n    \n \n",70)

        self._btn_basic.clicked.connect(self.mode_basic)
        self._btn_hand.clicked.connect(self.mode_hand)
        self._btn_gaze.clicked.connect(self.mode_gaze)
        self._btn_mic.clicked.connect(self._toggle_mic)
        self._btn_zoom_tog.clicked.connect(self._toggle_zoom_visibility)
        self._btn_zoom_plus.clicked.connect(self.zoom_size_plus)
        self._btn_zoom_minus.clicked.connect(self.zoom_size_minus)

        for btn in (
            self._btn_basic, self._btn_hand, self._btn_gaze, self._btn_mic,
            self._btn_zoom_tog, self._btn_zoom_plus, self._btn_zoom_minus,
        ):
            layout.addWidget(btn)
        layout.addStretch()

    @staticmethod
    def _make_btn(label: str, font_size = 12) -> QPushButton:
        btn = QPushButton(label)
        btn.setFixedSize(130, 100)
        btn.setFont(QFont("Segoe UI", font_size))
        btn.setStyleSheet("""
            QPushButton {
                background: rgba(0, 0, 0, 120);
                color: #FFFFFF;
                border: 1px solid rgba(255, 255, 255, 70);
                border-radius: 12px;
                text-align: center;      /* 가로 중앙 정렬 */
                padding-bottom: 10px;
            }
            QPushButton:hover   { background: rgba(255, 255, 255, 55); }
            QPushButton:pressed { background: rgba(255, 255, 255, 100); }
        """)
        return btn

    def _toggle_mic(self):
        self._mic_on = not self._mic_on
        self._btn_mic.setText(
            "🎙\n마이크 ON" if self._mic_on else "🔇\n마이크 OFF"
        )
        main_window = self.window()
        main_window._mic_on = self._mic_on  # main.py의 음성 스레드가 이 변수를 봅니다.
        
        if self._mic_on:
            main_window.update_subtitle("🎙️ [Voice ON] Microphone is active.")
        else:
            main_window.update_subtitle("🔇 [Voice OFF] Microphone is muted.")

    def _toggle_zoom_visibility(self):
        self._zoom_visible = not self._zoom_visible
        self._btn_zoom_tog.setText(
            "🔲\n돋보기 끄기" if self._zoom_visible else "🔲\n돋보기 켜기"
        )
        self.toggle_zoom.emit()

    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(0, 0, 0, 80))
        p.end()


# ─────────────────────────────────────────────────────────────────────
#  웹캠 피드 컨테이너 (feed_label + zoom_box를 함께 수용)
# ─────────────────────────────────────────────────────────────────────

class FeedContainer(QWidget):
    """
    창 크기에 맞춰 자동으로 늘어나는 피드 영역.
    내부에 웹캠 QLabel과 부유하는 DraggableLabel(zoom_box)을 담는다.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background:black;")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # 전체 화면을 채우는 웹캠 피드 레이블
        self._feed_label = QLabel(self)
        self._feed_label.setAlignment(Qt.AlignCenter)
        self._feed_label.setStyleSheet("background:black;")

        # 부유하는 돋보기 박스
        self.zoom_box = DraggableLabel(self)
        self.zoom_box.move(40, 40)
        self.zoom_box.raise_()

        # 우측 사이드바
        self.sidebar = SideBar(self)
        self.sidebar.raise_()

    def resizeEvent(self, event):
        self._feed_label.setGeometry(self.rect())
        self.sidebar.setFixedHeight(self.height())
        self.sidebar.move(self.width() - SideBar.WIDTH, 0)
        super().resizeEvent(event)

    def set_frame(self, pixmap: QPixmap):
        """스케일된 웹캠 프레임을 피드 레이블에 표시."""
        lw, lh = self._feed_label.width(), self._feed_label.height()
        if lw > 0 and lh > 0:
            self._feed_label.setPixmap(
                pixmap.scaled(lw, lh, Qt.KeepAspectRatioByExpanding, Qt.FastTransformation)
            )


# ─────────────────────────────────────────────────────────────────────
#  메인 윈도우
# ─────────────────────────────────────────────────────────────────────

class SmartMirrorApp(QMainWindow):
    """
    ┌──────────────────────────────────────────────────────────────────┐
    │  [모드] …   │         자막 (Voice 팀 전용)          │  배율: x.xx │  ← 상태 바
    ├──────────────────────────────────────────────────────────────────┤
    │                                                                  │
    │                      웹캠 피드 (전체 화면)                        │
    │         ╔══════════╗                                             │
    │         ║ ZOOM BOX ║  ← DraggableLabel (드래그/고정)              │
    │         ╚══════════╝                                             │
    │                                                                  │
    ├──────────────────────────────────────────────────────────────────┤
    │  [눈 확대 E]  [코 확대 N]  [입 확대 M]  [줌 리셋 Z]  [화면 고정 T] │  ← 버튼 바
    └──────────────────────────────────────────────────────────────────┘

    ═══════════════════════════════════════════════════════════════════
     ROLE 2 (Vision Engineer) 공개 API
    ───────────────────────────────────────────────────────────────────
     update_tracking_data(x_norm, y_norm, zoom_scale)
     get_current_mode() → str

     VOICE TEAM 공개 API
    ───────────────────────────────────────────────────────────────────
     update_subtitle(text: str)
    ═══════════════════════════════════════════════════════════════════
    """

    def __init__(self):
        super().__init__()
        self._mode:         str   = MODE_IDLE
        self._zoom_scale:   float = 1.0
        self._track_x:      float = 0.5
        self._track_y:      float = 0.5
        self._current_user: str   = ""
        self._last_rgb:     Optional[np.ndarray] = None

        self._build_ui()
        self._connect_sidebar()
        self._start_capture()
        QTimer.singleShot(0, self._show_user_select)

    # ── UI 구성 ────────────────────────────────────────────────────────

    def _build_ui(self):
        self.setWindowTitle("Adaptive Vision Mirror")
        self.showFullScreen()

        root = QWidget()
        root.setStyleSheet("background:black;")
        self.setCentralWidget(root)

        vbox = QVBoxLayout(root)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(0)

        vbox.addWidget(self._make_status_bar())

        self._feed = FeedContainer()
        vbox.addWidget(self._feed, stretch=1)

        vbox.addWidget(self._make_button_bar())

    def _make_status_bar(self) -> QWidget:
        """상단 상태 바: 모드(좌) | 자막(중) | 배율(우)"""
        bar = QWidget()
        bar.setFixedHeight(58)
        bar.setStyleSheet("background:rgba(0,0,0,215);")

        h = QHBoxLayout(bar)
        h.setContentsMargins(14, 4, 14, 4)
        h.setSpacing(10)

        bold16 = QFont("Arial", 16, QFont.Bold)

        # 좌측 최우선 — 선택된 사용자 이름 뱃지
        self._user_label = QLabel("")
        self._user_label.setFont(QFont("Arial", 15, QFont.Bold))
        self._user_label.setStyleSheet(
            "color:#FFFFFF; background:#2A6FDB;"
            "border-radius:8px; padding:2px 14px;"
        )
        self._user_label.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        self._user_label.hide()

        # 좌측 — 현재 모드
        self._mode_label = QLabel(f"[모드] {self._mode}")
        self._mode_label.setFont(bold16)
        self._mode_label.setStyleSheet("color:#00FF88; background:transparent;")
        self._mode_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        # 중앙 — Voice 팀 자막 슬롯 (투명 점선 박스)
        self._subtitle_label = QLabel()
        self._subtitle_label.setFont(QFont("Arial", 15))
        self._subtitle_label.setStyleSheet(
            "color:#FFFFFF; background:transparent;"
            "border:1px dashed rgba(255,255,255,75);"
            "border-radius:6px; padding:2px 16px;"
        )
        self._subtitle_label.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        self._subtitle_label.setMinimumWidth(300)

        # 우측 — 현재 배율
        self._zoom_label = QLabel(f"배율: {self._zoom_scale:.1f}x")
        self._zoom_label.setFont(bold16)
        self._zoom_label.setStyleSheet("color:#FFDD00; background:transparent;")
        self._zoom_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        h.addWidget(self._user_label)
        h.addWidget(self._mode_label,     stretch=2)
        h.addWidget(self._subtitle_label, stretch=3)
        h.addWidget(self._zoom_label,     stretch=2)
        return bar

    def _make_button_bar(self) -> QWidget:
        """하단 버튼 바 — 기능별 색상 구분, 저시력자용 고대비"""
        bar = QWidget()
        bar.setFixedHeight(88)
        bar.setStyleSheet(
            "background: rgba(8, 10, 18, 235);"
            "border-top: 1px solid rgba(255,255,255,22);"
        )

        h = QHBoxLayout(bar)
        h.setContentsMargins(24, 12, 24, 12)
        h.setSpacing(12)

        button_config = [
            ("눈 확대",   "E", self._on_eye_zoom,   "#1D6FE8", "#3B82F6"),
            ("코 확대",   "N", self._on_nose_zoom,  "#7B3FE4", "#9B6FEA"),
            ("입 확대",   "M", self._on_mouth_zoom, "#0F9E85", "#14B8A6"),
            ("줌 리셋",   "Z", self._on_zoom_reset, "#3D5068", "#526480"),
            ("화면 고정", "T", self._on_toggle_pin, "#C97A10", "#D9901A"),
        ]
        for label, key, slot, color, hover in button_config:
            h.addWidget(self._make_btn(f"{label}  [{key}]", slot, color, hover))
        return bar

    @staticmethod
    def _make_btn(label: str, slot, color: str, hover: str) -> QPushButton:
        btn = QPushButton(label)
        btn.setFixedHeight(60)
        btn.setMinimumWidth(138)
        btn.setFont(QFont("Segoe UI", 14, QFont.Bold))
        btn.setStyleSheet(f"""
            QPushButton {{
                background: {color};
                color: #FFFFFF;
                border: none;
                border-radius: 14px;
                padding: 4px 18px;
            }}
            QPushButton:hover {{
                background: {hover};
                border: 2px solid rgba(255, 255, 255, 150);
            }}
            QPushButton:pressed {{
                background: {color};
                border: 2px solid rgba(255, 255, 255, 60);
                padding-top: 6px;
            }}
        """)
        btn.clicked.connect(slot)
        return btn

    # ── 사이드바 연결 ─────────────────────────────────────────────────

    def _connect_sidebar(self):
        sb = self._feed.sidebar
        sb.mode_basic.connect(lambda: self._set_mode(MODE_TRACKING))
        # Role 2: 아래 두 시그널에 추가로 connect해 추적 로직 시작/전환
        sb.mode_hand.connect(lambda: self._set_mode(MODE_HAND))
        sb.mode_gaze.connect(lambda: self._set_mode(MODE_GAZE))
        sb.toggle_zoom.connect(self._on_toggle_zoom_box)
        sb.zoom_size_plus.connect(self._on_zoom_size_plus)
        sb.zoom_size_minus.connect(self._on_zoom_size_minus)

    # ── 사용자 선택 오버레이 ──────────────────────────────────────────

    def _show_user_select(self):
        parent = self.centralWidget()
        self._overlay = UserSelectOverlay(parent)
        self._overlay.setGeometry(parent.rect())
        self._overlay.finished.connect(self._on_user_selected)
        self._overlay.raise_()
        self._overlay.show()

    def _on_user_selected(self, user_name: str):
        self._current_user = user_name
        self._user_label.setText(user_name)
        self._user_label.show()
        self._set_mode(MODE_TRACKING)
        self._begin_render()

    # ── 웹캠 캡처 ─────────────────────────────────────────────────────

    def _start_capture(self):
        self._capture = CaptureThread(camera_index=0)
        self._capture.start()
        self._feed.zoom_box.position_changed.connect(self._update_zoom_box)
        # render 타이머는 사용자 선택 후 _begin_render()에서 시작

    def _begin_render(self):
        self._render_timer = QTimer(self)
        self._render_timer.timeout.connect(self._render_frame)
        self._render_timer.start(33)

    def _render_frame(self):
        frame = self._capture.latest_frame()
        if frame is not None:
            self._on_frame(frame)

    def _on_frame(self, frame: np.ndarray):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._last_rgb = rgb
        h, w, ch = rgb.shape
        qt_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self._feed.set_frame(QPixmap.fromImage(qt_img))
        self._update_zoom_box()

    def _update_zoom_box(self):
        """
        줌 박스의 실제 화면 위치를 카메라 좌표로 역변환해 크롭한다.
        캐시된 RGB 프레임을 사용하므로 cvtColor 중복 없음.
        드래그 중에도 position_changed 시그널로 즉시 호출된다.
        """
        if self._last_rgb is None:
            return
        frame = self._last_rgb  # already RGB

        fw = self._feed.width()
        fh = self._feed.height()
        if fw == 0 or fh == 0:
            return

        box    = self._feed.zoom_box
        bsz    = box.current_size
        box_cx = box.x() + bsz // 2
        box_cy = box.y() + bsz // 2

        fh_cam, fw_cam = frame.shape[:2]
        scale      = max(fw / fw_cam, fh / fh_cam)   
        rendered_w = int(fw_cam * scale)
        rendered_h = int(fh_cam * scale)
        offset_x   = (rendered_w - fw) // 2          
        offset_y   = (rendered_h - fh) // 2          

        # 💡 무엇을 잘라서 돋보기 안에 보여줄지 결정 (렌즈 중심점 계산)
        if getattr(self, "_use_face_crop", False):
            # 1순위: 눈/코/입 확대 타겟 버튼이 활성화되어 있을 때
            cam_x = int(self._face_crop_x * fw_cam)
            cam_y = int(self._face_crop_y * fh_cam)
            
        elif box.is_pinned() and (self._mode == MODE_GAZE or getattr(self, "_prev_mode_before_pin", None) == MODE_GAZE):
            # 2순위: 시선 추적 고정 -> 렌즈(비추는 상)만 그 자리에 영구 박제!
            cam_x = int(getattr(self, "_lens_x", self._track_x) * fw_cam)
            cam_y = int(getattr(self, "_lens_y", self._track_y) * fh_cam)

        elif box.is_pinned() and self._mode == MODE_HAND:
            # 💡 [신규 3순위] 손 추적 고정 -> 박스는 고정되고, 상(렌즈)은 손을 따라갑니다!
            cam_x = int(self._track_x * fw_cam)
            cam_y = int(self._track_y * fh_cam)
            
        else:
            # 4순위: 일반 미고정 상태 및 타 모드 고정 상태 
            # -> 물리적인 일반 돋보기처럼 박스가 놓여있는 위치의 해상도를 크롭함!
            cam_x = int((box_cx + offset_x) / scale)
            cam_y = int((box_cy + offset_y) / scale)

        cam_x = max(0, min(cam_x, fw_cam - 1))
        cam_y = max(0, min(cam_y, fh_cam - 1))

        r_at_1x = (bsz / 2) / scale
        r = max(15, int(r_at_1x / max(1.0, self._zoom_scale)))

        x1, x2 = cam_x - r, cam_x + r
        y1, y2 = cam_y - r, cam_y + r
        if x1 < 0:       x1, x2 = 0, min(fw_cam, 2 * r)
        elif x2 > fw_cam: x1, x2 = max(0, fw_cam - 2 * r), fw_cam
        if y1 < 0:       y1, y2 = 0, min(fh_cam, 2 * r)
        elif y2 > fh_cam: y1, y2 = max(0, fh_cam - 2 * r), fh_cam
        crop = np.ascontiguousarray(frame[y1:y2, x1:x2])
        if crop.size == 0:
            return

        qi = QImage(crop.data, crop.shape[1], crop.shape[0], crop.shape[1] * 3, QImage.Format_RGB888)
        box.set_frame(QPixmap.fromImage(qi))

    # ── 키보드 / 휠 ────────────────────────────────────────────────────

    def keyPressEvent(self, event):
        mapping = {
            Qt.Key_E:      self._on_eye_zoom,
            Qt.Key_N:      self._on_nose_zoom,
            Qt.Key_M:      self._on_mouth_zoom,
            Qt.Key_Z:      self._on_zoom_reset,
            Qt.Key_T:      self._on_toggle_pin,
            Qt.Key_Escape: self.close,
        }
        handler = mapping.get(event.key())
        if handler:
            handler()
        else:
            super().keyPressEvent(event)

    def wheelEvent(self, event):
        """마우스 휠로 배율 조절: 위 → 확대, 아래 → 축소.
        delta 크기에 비례 처리 — 트랙패드 고해상도 이벤트에서 급등 방지.
        표준 1노치(120)당 0.1 변화, 최대 단일 이벤트 ±0.5 제한.
        """
        delta = event.angleDelta().y()
        if delta == 0:
            return
        step = max(-0.5, min(0.5, (delta / 120.0) * 0.1))
        self._set_zoom(round(max(1.0, min(5.0, self._zoom_scale + step)), 2))
        event.accept()

    # ── 내부 헬퍼 ─────────────────────────────────────────────────────

    def _set_mode(self, mode: str):
        self._mode = mode
        self._mode_label.setText(f"[모드] {mode}")

    def _set_zoom(self, scale: float):
        self._zoom_scale = scale
        self._zoom_label.setText(f"배율: {scale:.1f}x")

    # ── 버튼 / 키 핸들러 스켈레톤 (Role 2가 내부 로직을 채운다) ──────────

    def _on_eye_zoom(self):
        """눈 확대 모드 전환. Role 2: 눈 랜드마크 ROI 추적을 여기에 연결."""
        self._set_mode(MODE_EYE)
        # TODO (Role 2): e.g. vision_controller.set_roi("eye")

    def _on_nose_zoom(self):
        """코 확대 모드 전환. Role 2: 코 랜드마크 ROI 추적을 여기에 연결."""
        self._set_mode(MODE_NOSE)
        # TODO (Role 2): e.g. vision_controller.set_roi("nose")

    def _on_mouth_zoom(self):
        """입 확대 모드 전환. Role 2: 입 랜드마크 ROI 추적을 여기에 연결."""
        self._set_mode(MODE_MOUTH)
        # TODO (Role 2): e.g. vision_controller.set_roi("mouth")

    def _on_zoom_reset(self):
        """배율 1.0x 리셋 및 기본 추적 모드 복귀."""
        self._set_zoom(1.0)
        self._set_mode(MODE_TRACKING)
        # TODO (Role 2): e.g. vision_controller.reset_zoom()

    def _on_toggle_pin(self):
        """돋보기 박스 고정 / 해제 토글."""
        box = self._feed.zoom_box
        box.toggle_pin()
        
        if box.is_pinned():
            # 💡 고정하는 순간의 현재 위치를 렌즈 좌표로 영구 박제!
            self._lens_x = self._track_x
            self._lens_y = self._track_y
            
            # 🔥 [핵심] 시선 추적 모드일 때는 MODE_PINNED로 모드를 바꾸지 않고 유지합니다!
            # 그래야 main.py가 데이터를 계속 보내주고 박스가 시선을 따라 날아다닙니다.
            if self._mode == MODE_GAZE:
                self.update_subtitle("📌 [시선 고정] 상은 고정되고, 박스는 시선을 따라 움직입니다.")
            elif self._mode == MODE_HAND:
                self.update_subtitle("📌 [손 고정] 박스는 고정되고, 상이 손을 따라 이동합니다.")
            else:
                # 손 추적 등 다른 모드일 때는 원래 명세대로 고정 모드로 전환하여 통째로 묶어버립니다.
                self._prev_mode_before_pin = self._mode  # 해제 시 복귀용 백업
                self._set_mode(MODE_PINNED)
                self.update_subtitle("📌 [전체 고정] 돋보기 박스와 상이 모두 고정됩니다.")
        else:
            # 고정 해제 시 원래 상태로 복귀
            if getattr(self, "_mode", None) == MODE_PINNED and hasattr(self, "_prev_mode_before_pin"):
                self._set_mode(self._prev_mode_before_pin)
            elif self._mode in [MODE_GAZE, MODE_HAND]:
                pass
            else:
                self._set_mode(MODE_TRACKING)
            self.update_subtitle("📌 고정 해제됨 — 다시 정상 추적합니다.")

    def _on_toggle_zoom_box(self):
        box = self._feed.zoom_box
        if box.isVisible():
            box.hide()
        else:
            box.show()
            self._update_zoom_box()

    def _on_zoom_size_plus(self):
        box = self._feed.zoom_box
        new_size = min(500, box.current_size + 50)
        box.resize_box(new_size)
        self._update_zoom_box()

    def _on_zoom_size_minus(self):
        box = self._feed.zoom_box
        new_size = max(150, box.current_size - 50)
        box.resize_box(new_size)
        self._update_zoom_box()

    # ═════════════════════════════════════════════════════════════════
    #  ROLE 2 (Vision Engineer) 공개 인터페이스
    # ═════════════════════════════════════════════════════════════════

    def update_tracking_data(
        self,
        x_norm:     float,
        y_norm:     float,
        zoom_scale: float,
    ) -> None:
        """
        스무딩된 좌표와 제스처 배율을 UI에 주입한다.

        Parameters
        ----------
        x_norm     : float  수평 정규화 좌표 [0.0, 1.0]
        y_norm     : float  수직 정규화 좌표  [0.0, 1.0]
        zoom_scale : float  배율 [1.0, 5.0]

        고정(Pinned) 상태가 아닐 때만 zoom_box 위치가 이동된다.

        ⚠️ 반드시 GUI(메인) 스레드에서 호출할 것.
           별도 스레드에서 호출 시:
           QMetaObject.invokeMethod(window, "update_tracking_data",
               Qt.QueuedConnection,
               Q_ARG(float, x), Q_ARG(float, y), Q_ARG(float, z))
        """
        self._track_x = float(x_norm)
        self._track_y = float(y_norm)
        self._set_zoom(float(zoom_scale))

        is_pinned = self._feed.zoom_box.is_pinned()

        # 💡 [핵심] AI가 돋보기 박스 위치를 제어하도록 허용된 '자동 모드' 명단
        auto_modes = [MODE_GAZE, MODE_HAND, MODE_EYE, MODE_NOSE, MODE_MOUTH]

        # :기본 모드(MODE_TRACKING)면 AI는 박스를 절대 건드리지 않고 조용히 퇴장 (오직 마우스만 허용)
        if self._mode not in auto_modes:
            return
            
        # 고정(Pinned) 상태일 때, 시선 모드가 아니라면 박스 이동 정지
        if is_pinned and (self._mode != MODE_GAZE):
            return

        # 위 두 개의 방어막을 무사히 통과했을 때만 AI 좌표로 박스를 움직입니다.
        fw   = self._feed.width()
        fh   = self._feed.height()
        sz   = self._feed.zoom_box.current_size
        half = sz // 2
        bx   = max(0, min(int(self._track_x * fw) - half, fw - sz))
        by   = max(0, min(int(self._track_y * fh) - half, fh - sz))
        self._feed.zoom_box.move(bx, by)

    def get_current_mode(self) -> str:
        """현재 활성 모드 문자열 반환."""
        return self._mode

    # ═════════════════════════════════════════════════════════════════
    #  VOICE TEAM 공개 인터페이스
    # ═════════════════════════════════════════════════════════════════

    def update_subtitle(self, text: str) -> None:
        """음성 인식 결과를 중앙 자막 안내창에 표시한다."""
        self._subtitle_label.setText(text)

    # ── 정리 ─────────────────────────────────────────────────────────

    def closeEvent(self, event):
        if hasattr(self, "_render_timer"):
            self._render_timer.stop()
        if hasattr(self, "_capture"):
            self._capture.stop()
        super().closeEvent(event)


# ─────────────────────────────────────────────────────────────────────
#  엔트리 포인트
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setApplicationName("Adaptive Vision Mirror")
    window = SmartMirrorApp()
    sys.exit(app.exec_())
