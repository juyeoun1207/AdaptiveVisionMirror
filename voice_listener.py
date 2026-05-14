import speech_recognition as sr

def listen_command():
    """마이크로 음성을 듣고 텍스트로 변환해주는 함수"""
    r = sr.Recognizer()
    
    # 마이크 켜기
    with sr.Microphone() as source:
        print("\n[음성 인식] 🎙️ 주변 소음을 파악 중입니다... (1초 대기)")
        r.adjust_for_ambient_noise(source, duration=1) # 소음 보정
        
        print("[음성 인식] 🟢 이제 말씀해 주세요! (예: '확대', '초기화')")
        try:
            # 사용자가 말할 때까지 기다림 (최대 5초 대기, 한 문장 최대 3초)
            audio = r.listen(source, timeout=5, phrase_time_limit=3)
            
            # 구글 STT 엔진을 이용해 한국어로 변환
            text = r.recognize_google(audio, language='ko-KR')
            print(f"[음성 인식] 🗣️ 인식된 명령: '{text}'")
            return text
            
        except sr.WaitTimeoutError:
            print("[음성 인식] ⏱️ 아무 말씀도 안 하셨네요.")
            return None
        except sr.UnknownValueError:
            print("[음성 인식] ❌ 무슨 말인지 정확히 듣지 못했습니다.")
            return None
        except sr.RequestError as e:
            print(f"[음성 인식] 🔌 인터넷 연결 에러: {e}")
            return None

# 이 파일만 단독으로 실행했을 때 테스트하기 위한 로직
if __name__ == "__main__":
    print("=== 음성 인식 단독 테스트 시작 ===")
    while True:
        result = listen_command()
        if result == "종료":
            print("테스트를 종료합니다.")
            break