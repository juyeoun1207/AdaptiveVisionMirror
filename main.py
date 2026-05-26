import sys
import threading
import queue
import time
import cv2

# PyQt5 모듈
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer, QMetaObject, Qt, Q_ARG

# --- 팀원들의 모듈 가져오기 ---
# Role 1 (UI)
from mirror_ui import SmartMirrorApp, MODE_HAND, MODE_GAZE, MODE_MOUTH,MODE_TRACKING, MODE_EYE, MODE_NOSE
# Role 2 (Vision)
from tracker.hand_tracker import HandTracker
from tracker.gaze_controller import GazeController
from zoom.gesture_zoom_tracker import GestureZoomTracker
from tracker.face_region_tracker import FaceRegionTracker
# Role 3 (Voice)
from voice_listener import listen_command

# 전역 작업 큐 및 상태
event_queue = queue.Queue()
system_running = True

shared_state = {
    "track_x": 0.5, "track_y": 0.5, "zoom": 1.0,
    "use_face": False, "face_x": 0.5, "face_y": 0.5 ,"gesture_zoom": None
}

# =====================================================================
# 🎙️ Role 3: 음성 인식 스레드 (Producer)
# =====================================================================
def run_voice_thread():
    print("[Voice Thread] 🎙️ 백그라운드 음성 인식 가동!")
    while system_running:
        #  UI의 마이크 스위치가 켜져 있을 때만(True) 귀를 엽니다!
        if getattr(window, "_mic_on", True):
            command = listen_command()
            if command and command != "UNKNOWN":
                event_queue.put({"target": "SYSTEM", "action": command})
        else:
            # 마이크가 꺼져있으면 시스템에 과부하를 주지 않도록 0.5초씩 푹 쉽니다.
            time.sleep(0.5)

# =====================================================================
# 👁️ Role 2: 통합 비전 스레드
# =====================================================================
def run_vision_thread(window):
    print("[Vision Thread] 📷 사각지대 관찰용 하이브리드 엔진 가동!")
    
    hand_tracker = HandTracker()
    gaze_tracker = GazeController(screen_width=1280, screen_height=720) # (더 이상 좌표계산엔 안 씀)
    face_tracker = FaceRegionTracker()
    gesture_tracker = GestureZoomTracker()
    
    while system_running:
        frame = window._capture.latest_frame()
        if frame is None:
            time.sleep(0.01)
            continue
            
        current_mode = window.get_current_mode()
        h, w = frame.shape[:2]
        
        try:
            gesture_data = gesture_tracker.process_frame(frame)
            
            # 손이 인식되었고 데이터가 넘어왔다면
            if gesture_data:
                delta = gesture_data["zoom_delta"]
                
                # 손을 가만히 있는 게 아니라 실제로 줌(Pinch) 제스처를 해서 변화량이 생겼을 때만!
                if delta != 0.0:
                    current_zoom = window._zoom_scale
                    # 기존 배율에 변화량을 더하고, 최소 1.0배 ~ 최대 5.0배 사이로 가둬줍니다.
                    new_zoom = max(1.0, min(5.0, current_zoom + delta))
                    shared_state["gesture_zoom"] = new_zoom
            # 💡 [핵심] 느린 gaze_tracker는 버리고, 무조건 빠르고 정확한 '코(Nose)' 좌표를 박스 이동에 씁니다!
            bbox_nose, _ = face_tracker.get_region_crop(frame, ord('n'))
           
            # ==========================================
            # 🔀 현재 모드에 따른 동작 분기점
            # ==========================================
            
            # [A] 얼굴 부위(사각지대) 모드: 렌즈는 특정 부위에 박제, 박스는 고개(코)를 따라감!
            if current_mode in [MODE_EYE, MODE_NOSE, MODE_MOUTH]:
                region_key = ord('e') if current_mode == MODE_EYE else ord('n') if current_mode == MODE_NOSE else ord('m')
                if region_key == ord('e'):
                    subtitle = "EYE Magnifying"
                elif region_key == ord('n'):
                    subtitle = "NOSE Magnifying"
                elif region_key == ord('m'):
                    subtitle = "MOUTH Magnifying"
                
                bbox, _ = face_tracker.get_region_crop(frame, region_key)
                
                if bbox:
                    # 렌즈 고정 좌표
                    shared_state["face_x"] = ((bbox[0] + bbox[2]) / 2) / w
                    shared_state["face_y"] = ((bbox[1] + bbox[3]) / 2) / h
                    shared_state["use_face"] = True
                
                if bbox_nose:
           
                    shared_state["track_x"] = ((bbox_nose[0] + bbox_nose[2]) / 2) / w
                    shared_state["track_y"] = ((bbox_nose[1] + bbox_nose[3]) / 2) / h
                    
                window.update_subtitle(subtitle)

            # [B] 일반 시선 추적 모드 (헤드 트래킹 대체 모드)
            elif current_mode == MODE_GAZE:
                shared_state["use_face"] = False 
                
                if bbox_nose:
                    nx = ((bbox_nose[0] + bbox_nose[2]) / w)
                    nx = ((bbox_nose[0] + bbox_nose[2]) / 2) / w
                    ny = ((bbox_nose[1] + bbox_nose[3]) / 2) / h
                    
                    # 💡 수정됨: '1.0 -' 거울 반전 수식 제거 및 고개 감도 2.0배 증폭!
                    shared_state["track_x"] = max(0.0, min(1.0, 0.5 + (nx - 0.5) * 2.0))
                    shared_state["track_y"] = max(0.0, min(1.0, 0.5 + (ny - 0.5) * 2.0))
                    
                window.update_subtitle("👁️ Gaze Tracking Mode")

            # [C] 손 추적 모드
            elif current_mode == MODE_HAND:
                shared_state["use_face"] = False
                pos = hand_tracker.get_hand_position(frame)
                if pos:
                    shared_state["track_x"] = pos[0] / w
                    shared_state["track_y"] = pos[1] / h
                window.update_subtitle("✋ Hand Tracking Mode")

            # [D] 기본 모드 (아무것도 안 함!)
            else:
                shared_state["use_face"] = False
                window.update_subtitle("Manual Mode")
                # 💡 기본 모드일 때는 track_x, track_y를 건드리지 않습니다.

        except Exception as e:
            print(f"[Vision Error] {e}")
            
        time.sleep(0.033)


# =====================================================================
# 🖥️ 메인 큐 폴러 (UI Update Loop를 대신함)
# =====================================================================
def poll_event_queue(window):
    try:
        while not event_queue.empty():
            packet = event_queue.get_nowait()
            action = packet["action"]
            
            # --- 1. 시스템 및 줌 제어 ---
            if action == "EXIT":
                window.update_subtitle("🎙️ Shutting down the system.")
                global system_running
                system_running = False
                window.close()
                
            elif action == "ZOOM_IN":
                window.update_subtitle("🎙️ Zooming in.")
                window._set_zoom(min(5.0, window._zoom_scale + 1.0))
                
            elif action == "ZOOM_OUT":
                window.update_subtitle("🎙️ Zooming out.")
                window._set_zoom(max(1.0, window._zoom_scale - 1.0))
                
            elif action == "RESET":
                window.update_subtitle("🎙️ Resetting zoom and mode.")
                window._on_zoom_reset()

            # --- 2. 추적 모드 전환 ---
            elif action == "MODE_BASIC":
                window.update_subtitle("🎙️ Switched to Manual Mode.")
                window._set_mode(MODE_TRACKING)
                
            elif action == "MODE_HAND":
                window.update_subtitle("🎙️ Hand Tracking Mode ON.")
                window._set_mode(MODE_HAND)
                
            elif action == "MODE_GAZE":
                window.update_subtitle("🎙️ Gaze Tracking Mode ON.")
                window._set_mode(MODE_GAZE)

            # --- 3. 사각지대 (얼굴 부위) 관찰 모드 ---
            elif action == "MODE_EYE":
                window.update_subtitle("🎙️ EYE Magnifying Mode ON.")
                window._on_eye_zoom()
                
            elif action == "MODE_NOSE":
                window.update_subtitle("🎙️ NOSE Magnifying Mode ON.")
                window._on_nose_zoom()
                
            elif action == "MODE_MOUTH":
                window.update_subtitle("🎙️ MOUTH Magnifying Mode ON.")
                window._on_mouth_zoom()

            # --- 4. 화면 고정 ---
            elif action == "TOGGLE_PIN":
                window._on_toggle_pin()
                
    except queue.Empty:
        pass

    # 💡 UI 엔진에게 카메라 렌즈(크롭) 좌표를 직접 주입!
    if shared_state["gesture_zoom"] is not None:
        window._set_zoom(shared_state["gesture_zoom"])
        shared_state["gesture_zoom"] = None  # UI에 적용했으니 다시 비워둠 (1회성 소비)


    new_gesture_zoom = shared_state.get("gesture_zoom")
    if new_gesture_zoom is not None:
        window._set_zoom(new_gesture_zoom)
        shared_state["gesture_zoom"] = None  # UI에 적용했으니 다시 비워둠

    # 기존 코드 유지
    window._use_face_crop = shared_state.get("use_face", False)
    window._face_crop_x = shared_state.get("face_x", 0.5)
    window._face_crop_y = shared_state.get("face_y", 0.5)

    window.update_tracking_data(
        shared_state.get("track_x", 0.5), 
        shared_state.get("track_y", 0.5), 
        window._zoom_scale  # UI가 가진 배율을 그대로 넘겨줌
    )
    

# =====================================================================
# 🚀 시스템 부팅
# =====================================================================
if __name__ == "__main__":
    print("=== ✨ PPIYOUNG 스마트 거울 엔진 통합 부팅 ===")
    
    # 1. PyQt5 앱 및 UI 윈도우 생성 (메인 스레드)
    app = QApplication(sys.argv)
    window = SmartMirrorApp()
    
    # 2. 보이스 스레드 발사
    voice_t = threading.Thread(target=run_voice_thread, daemon=True)
    voice_t.start()
    
    # 3. 비전 스레드 발사 (window 객체를 넘겨주어 프레임을 읽게 함)
    vision_t = threading.Thread(target=run_vision_thread, args=(window,), daemon=True)
    vision_t.start()
    
    # 4. 큐 폴링용 QTimer 가동 (유니티의 Update() 역할)
    timer = QTimer()
    timer.timeout.connect(lambda: poll_event_queue(window))
    timer.start(16)  # 16ms 마다 실행 (약 60FPS)
    
    # 5. UI 무한 루프 시작 (여기서 코드가 멈추고 GUI가 돌아감)
    sys.exit(app.exec_())