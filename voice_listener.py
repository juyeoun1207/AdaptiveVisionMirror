import speech_recognition as sr

COMMAND_MAP = {
    "MODE_EYE": [
        "눈확대", "눈보여줘", "아이모드", "eyemode", "zoomeyes", 
        "showmyeyes", "focusoneyes", "trackeyes", "showeyes", "magnifyeyes"
    ],
    "MODE_NOSE": [
        "소확대", "코확대", "코보여줘", "노즈모드", "nosemode", "zoomnose", 
        "showmynose", "focusonnose", "tracknose", "shownose", "magnifynose"
    ],
    "MODE_MOUTH": [
        "입확대", "턱확대", "마우스모드", "입보여줘", "턱보여줘", "mouthmode", 
        "zoommouth", "showmymouth", "focusonmouth", "trackmouth", "showmouth", 
        "magnifymouth", "jawmode", "chinmode"
    ],

    "MODE_BASIC": [
        "기본모드", "수동모드", "일반모드", "basicmode", "manualmode", 
        "normalmode", "standardmode", "disabletracking", "trackoff"
    ],
    "MODE_HAND": [
        "손추적", "핸드모드", "손가락모드", "handtracking", "handmode", 
        "usehands", "trackhands", "trackmyhands", "handcontrol", "gesturemode"
    ],
    "MODE_GAZE": [
        "시선추적", "헤드트래킹", "고개추적", "시선모드", "eyegazemodeon", 
        "activateeyegazemode", "gazetracking", "gazemode", "headtracking", 
        "headmode", "eyetracking", "trackmyhead", "lookmode"
    ],
    
    "TOGGLE_PIN": [
        "화면고정", "고정해제", "멈춰", "고정해", "화면멈춰", "pin", "unpin", 
        "freeze", "stop", "holdit", "lock", "unlock", "lockscreen", 
        "pinview", "freezeview", "stay"
    ],
    "EXIT": [
        "종료", "거울꺼", "시스템종료", "exit", "quit", "turnoff", 
        "shutdown", "close", "stopmirror", "bye", "goodbye", "poweroff"
    ],

    "ZOOM_IN": [
        "확대", "커져", "크게", "zoomin", "enlarge", "magnify", "bigger", 
        "makeitbigger", "closer", "getcloser", "scaleup", "zoommore"
    ],
    "ZOOM_OUT": [
        "축소", "작아져", "작게", "zoomout", "shrink", "smaller", 
        "makeitsmaller", "farther", "getfarther", "scaledown", "backout"
    ],
    "RESET": [
        "원래대로", "초기화", "리셋", "줌리셋", "reset", "normal", 
        "default", "original", "backtonormal", "resetzoom", "defaultmode", "clear"
    ]
}

recognizer = sr.Recognizer()
is_calibrated = False  # 💡 최초 1회 실행을 체크하기 위한 스위치

def listen_command():
    global is_calibrated
    
    try:
        with sr.Microphone() as source:
            # 💡 [핵심] 음성 스레드가 최초로 진입할 때만 마이크 소음을 파악합니다.
            if not is_calibrated:
                print("\n[Voice] 🎙️ 시스템 부팅 중... 마이크 환경 설정 중 (최초 1회)")
                recognizer.adjust_for_ambient_noise(source, duration=2)
                is_calibrated = True
                print("[Voice] ✅ 마이크 설정 완료! 이제 언제든 명령을 내리세요.")
            
            # 💡 timeout=1: 1초마다 귀를 닫았다가 다시 엽니다. (시스템이 멈추지 않고 부드럽게 돌아감)
            # 1초 동안 말이 없으면 아래 WaitTimeoutError 로 빠져서 조용히 넘어갑니다.
            audio = recognizer.listen(source, timeout=1, phrase_time_limit=3)
            
            text = recognizer.recognize_google(audio, language='ko-KR')
            print(f"\n[Voice] 🗣️ Detected: '{text}'")
            
            return parse_voice_command(text)
            
    except sr.WaitTimeoutError:
        # 1초 동안 아무 말도 안 했을 때 (정상적인 상시 대기 상태이므로 로그 없이 무시)
        return None
    except sr.UnknownValueError:
        # 잡음이나 숨소리를 잘못 들었을 때
        return None
    except sr.RequestError as e:
        print(f"[Voice] 🔌 API Network Error: {e}")
        return None
    except Exception as e:
        return None

def parse_voice_command(text):
    if not text:
        return None
        
    clean_text = text.lower().replace(" ", "") 
    
    for command_type, keywords in COMMAND_MAP.items():
        if any(keyword in clean_text for keyword in keywords):
            print(f"[Voice] 🎯 Action Triggered: {command_type}")
            return command_type
            
    return "UNKNOWN"


# 이 파일만 단독으로 실행했을 때 테스트하기 위한 로직
# if __name__ == "__main__":
#     print("=== 음성 인식 단독 테스트 시작 ===")
#     while True:
#         result = listen_command()
        
#         # 💡 테스트 로직 수정: 파서가 리턴하는 "상수"를 기준으로 종료 확인
#         if result == "EXIT":
#             print("테스트를 종료합니다.")
#             break
#         elif result:
#             print(f"👉 시스템 전달 명령: {result}")