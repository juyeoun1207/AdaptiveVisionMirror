# # ui_engine.py (Role 1이 작업할 UI 모듈의 뼈대)
# from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel
# from PyQt5.QtCore import QTimer
# import sys

# class SmartMirrorUI(QMainWindow):
#     def __init__(self, shared_state, event_queue):
#         super().__init__()
#         self.shared_state = shared_state
#         self.event_queue = event_queue
        
#         self.initUI()
        
#         # 💡 [핵심] 유니티의 Update() 역할을 하는 QTimer 세팅!
#         self.update_timer = QTimer(self)
#         self.update_timer.timeout.connect(self.game_loop) # 매 틱마다 실행할 함수
#         self.update_timer.start(16) # 16ms 마다 실행 (약 60FPS)

#     def initUI(self):
#         # 화면 크기 설정, 버튼, 거울 화면 레이아웃 등 Role 1이 예쁘게 짤 공간
#         self.cursor_label = QLabel("🖐", self) # 가상의 마우스 커서
#         self.showFullScreen() # 거울이니까 전체화면

#     def game_loop(self):
#         """메인 스레드에서 1초에 60번씩 도는 진짜 통합 루프"""
        
#         # 1. 아키텍트의 큐(Queue) 확인 (음성/제스처 이벤트 처리)
#         self.check_events()
        
#         # 2. 전역 상태(shared_state)를 읽어서 화면 갱신
#         self.render_frame()

#     def check_events(self):
#         """큐를 폴링(Polling)해서 UI 이벤트를 처리하는 곳"""
#         try:
#             # 아키텍트님이 만든 큐에서 패킷을 빼옵니다
#             packet = self.event_queue.get_nowait()
            
#             if packet["target"] == "UI":
#                 if packet["action"] == "ZOOM_IN":
#                     print("UI: 줌 인 애니메이션 실행!")
#                     # self.do_zoom_animation()
                    
#         except:
#             pass # 큐가 비어있으면 그냥 넘어감

#     def render_frame(self):
#         """상태(Mode)에 따라 화면을 다르게 그리는 분기점"""
#         mode = self.shared_state["system_mode"]
        
#         if mode == "HAND":
#             # 손 추적 모드일 때만 커서를 화면에 그림!
#             hx = self.shared_state["hand_x"]
#             hy = self.shared_state["hand_y"]
#             self.cursor_label.move(hx, hy) 
#             self.cursor_label.show()
#         else:
#             self.cursor_label.hide()
            
#     # 💡 물리적인 터치 모니터일 경우 (실제 손가락 터치)
#     def mousePressEvent(self, event):
#         print(f"화면 터치됨! 좌표: {event.x()}, {event.y()}")
#         # 물리적 터치 시의 확대 로직..

# if __name__ == "__main__":
#     # 1. 통신망 생성
#     event_queue = queue.Queue()
#     shared_state = {"hand_x": 0, "hand_y": 0, "system_mode": "DEFAULT"}

#     # 2. 보이스 스레드 가동 (파이프 연결)
#     voice_thread = threading.Thread(target=run_voice_thread, args=(event_queue,))
#     voice_thread.start()

#     # 3. 비전 스레드 가동 (파이프 연결)
#     # vision_thread = threading.Thread(target=run_vision_thread, args=(event_queue, shared_state))
#     # vision_thread.start()

#     # 4. UI 엔진 가동 (파이프 연결 및 메인 스레드 권한 양도!)
#     app = QApplication(sys.argv)
#     ui = SmartMirrorUI(shared_state, event_queue)
#     sys.exit(app.exec_()) # ⬅️ 기존 while 문을 대체하는 UI 무한 루프

import threading 
import queue #뮤텍스가 돼있는 작업큐 용
import time

from voice_listener import listen_command #음성 인식 모듈

event_queue = queue.Queue()
system_running = True

# 큐에 들어갈 '명령 패킷'의 구조 (예시)
# command_packet = {
#     "target": "UI",          # 누가 이 명령을 처리해야 하는가? (UI, VISION, SYSTEM 등)
#     "action": "ZOOM_IN",     # 구체적으로 뭘 해야 하는가?
#     "data": 1.5              # 추가로 전달할 값 (줌 배율, 좌표 등 / 없으면 None)
# }

shared_state = {
    "hand_x": 0.0,
    "hand_y": 0.0,
    "system_mode": "DEFAULT"  # 기본값. 나중에 "HAND", "EYE" 등으로 바뀝니다.
}

def run_voice_thread():
    """음성 스레드: 듣기만 하고, 실행은 큐에 던지고 빠집니다 (Producer)"""
    print("[음성 스레드] 가동!")
    while True:
        command = listen_command()
        
        if command == "EXIT":
            # 구조체(딕셔너리)를 큐에 밀어 넣음
            event_queue.put({"target": "SYSTEM", "action": "EXIT"})
        elif command == "ZOOM_IN":
            event_queue.put({"target": "UI", "action": "ZOOM_IN", "data": None})

def ui_update_loop():
    # 1. 큐 확인 (상태 변경 명령이 들어왔는지 확인)
    try:
        packet = event_queue.get_nowait()
        
        if packet["target"] == "SYSTEM" and packet["action"] == "CHANGE_MODE":
            new_mode = packet["data"]
            shared_state["system_mode"] = new_mode # 상태 업데이트!
            print(f"🔄 시스템 모드가 [{new_mode}] 모드로 변경되었습니다!")
            
    except queue.Empty:
        pass
        
    # =======================================================
    # 2. 질문자님이 말씀하신 바로 그 "분기(Branch)" 영역!
    # =======================================================
    current_mode = shared_state["system_mode"]
    
    if current_mode == "DEFAULT":
        # 아무것도 안 함 (또는 UI 엔진에게 마우스 커서를 숨기라고 지시)
        # ui_engine.hide_cursor()
        pass
        
    elif current_mode == "HAND":
        # 손 추적 모드일 때만 좌표를 읽어와서 커서를 그립니다.
        current_x = shared_state["hand_x"]
        current_y = shared_state["hand_y"]
        # ui_engine.draw_cursor(current_x, current_y)
        
    elif current_mode == "EYE":
        # 시선 추적 모드일 때의 로직...
        pass

if __name__ == "__main__":
    print("=== ✨ PPIYOUNG 스마트 거울 엔진 부팅 ===")
    
    # 1. 음성 인식 스레드 생성 및 발사
    voice_thread = threading.Thread(target=run_voice_thread, daemon=True)
    voice_thread.start()
    
    # 2. 메인 스레드 (나중에 UI 렌더링이 돌아갈 메인 루프)
    print("[메인 스레드] 🖥️ 거울 UI 렌더링 루프 가동 시작...\n")
    
    try:
        # system_running이 True인 동안 무한 반복 (거울 화면 유지)
        while system_running:
            # 💡 1. 큐에 밀린 명령이 있는지 확인하고 처리합니다.
            ui_update_loop() 
            
            # 💡 2. CPU가 폭주하는 것을 막기 위해 0.1초(10FPS) 정도만 쉬어줍니다.
            # (나중에 PyQt5가 들어오면 이 sleep은 빼고 PyQt5의 타이머에 맡기게 됩니다.)
            time.sleep(0.1) 
            
    except KeyboardInterrupt:
        print("\n[시스템 제어] 🛑 강제 종료(Ctrl+C)가 감지되었습니다.")
        system_running = False
        
    print("\n=== ✨ 스마트 거울 엔진이 안전하게 종료되었습니다. ===")