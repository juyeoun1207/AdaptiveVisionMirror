import cv2
import numpy as np
from eyeGestures import EyeGestures_v3

gestures = EyeGestures_v3()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

screen_width = 1280
screen_height = 720
prev_x, prev_y = screen_width//2, screen_height//2

cv2.namedWindow("Gaze Mirror", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Gaze Mirror", screen_width, screen_height)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display = cv2.flip(frame, 1)  # 화면 표시용만 flip

    try:
        event, cevent = gestures.step(
            frame,  # flip 안 한 원본 넘기기
            True, screen_width, screen_height,
            context="main"
        )
    except Exception as e:
        print(f"error: {e}")
        cv2.imshow("Gaze Mirror", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    if cevent is not None:
        cx = int(np.clip(cevent.point[0], 0, screen_width - 1))
        cy = int(np.clip(cevent.point[1], 0, screen_height - 1))
        cv2.circle(display, (cx, cy), 30, (255, 0, 0), 3)
        cv2.circle(display, (cx, cy), 15, (0, 255, 0), 2)
        cv2.circle(display, (cx, cy), 5, (255, 255, 0), -1)
        cv2.putText(display, "Look at the blue circle!",
                    (screen_width//2 - 200, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    if event is not None:
        x = int(np.clip(event.point[0], 0, screen_width - 1))
        y = int(np.clip(event.point[1], 0, screen_height - 1))
        prev_x, prev_y = x, y

    cv2.circle(display, (prev_x, prev_y), 12, (0, 0, 255), -1)
    cv2.circle(display, (prev_x, prev_y), 14, (255, 255, 255), 2)

    cv2.putText(display, "Calibrating... (Q: quit)",
                (10, screen_height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Gaze Mirror", display)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()