import speech_recognition as sr

COMMAND_MAP = {
    "ZOOM_IN": ["확대", "커져", "zoomin", "enlarge"],
    "ZOOM_OUT": ["축소", "작아져", "zoomout", "shrink"],
    "MANUAL_ZOOM_IN_ON" : ["수동확대모드켜줘", "activatemanualmagifymode"],
    "MANUAL_ZOOM_IN_ON" : ["수동확대모드꺼줘", "deactivatemanualmagifymode"],
    "EYE_GAZE_MODE_ON" : ["시선추적모드켜줘","시선추적켜줘" "eyegazemodeon", "activateeyegazemode"],
    "EYE_GAZE_MODE_OFF" : ["시선추적모드꺼줘","시선추적꺼줘" "eyegazemodeoff", "deactivateeyegazemode"],
    "RESET": ["원래대로", "초기화", "리셋","꺼줘", "reset"],
    "EXIT": ["종료", "거울꺼", "exit", "quit", "turnoff"]
}

def listen_command():
    """마이크로 음성을 듣고 텍스트로 변환해주는 함수"""
    r = sr.Recognizer()
    
    # 마이크 켜기
    with sr.Microphone() as source:
        print("\n[음성 인식] 🎙️ 주변 소음을 파악 중입니다... (1초 대기)")
        r.adjust_for_ambient_noise(source, duration=1) # 소음 보정
        
        print("[음성 인식] 🟢 이제 말씀해 주세요! (한국어/영어 모두 가능)")
        try:
            # 사용자가 말할 때까지 기다림 (최대 5초 대기, 한 문장 최대 3초)
            audio = r.listen(source, timeout=5, phrase_time_limit=3)
            
            # 구글 STT 엔진을 이용해 변환 (ko-KR로 설정해도 명확한 영어는 영어로 인식됨)
            text = r.recognize_google(audio, language='ko-KR')
            print(f"[음성 인식] 🗣️ 인식된 명령: '{text}'")
            
            return parse_voice_command(text)
            
        except sr.WaitTimeoutError:
            print("[음성 인식] ⏱️ 아무 말씀도 안 하셨네요.")
            return None
        except sr.UnknownValueError:
            print("[음성 인식] ❌ 무슨 말인지 정확히 듣지 못했습니다.")
            return None
        except sr.RequestError as e:
            print(f"[음성 인식] 🔌 인터넷 연결 에러: {e}")
            return None

def parse_voice_command(text):
    if not text:
        return None
        
    clean_text = text.lower().replace(" ", "") 
    
    for command_type, keywords in COMMAND_MAP.items():
        # any(): keywords 리스트 안의 단어 중 하나라도 clean_text에 있으면 True 반환
        if any(keyword in clean_text for keyword in keywords):
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