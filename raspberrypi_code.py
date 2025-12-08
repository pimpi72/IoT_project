import cv2
import numpy as np
import time
import socket
import struct
import pickle

# ---------------------------
# 설정값
# ---------------------------
BLUR_SIZE = (7, 7)
THRESHOLD_SENSITIVITY = 50
MIN_CONTOUR_AREA = 2000
FRAME_WAIT_TIME = 0.05
PORT = 8486

# ---------------------------
# 카메라 초기화
# ---------------------------
def initialize_camera():
    try:
        from picamera2 import Picamera2
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)})
        picam2.configure(config)
        picam2.start()
        time.sleep(1)
        print("[INFO] Raspberry Pi camera initialized.")
        return picam2, True
    except ImportError:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise Exception("ERROR: Cannot open webcam.")
        time.sleep(1)
        print("[INFO] Webcam initialized.")
        return cap, False

# ---------------------------
# 프레임 가져오기
# ---------------------------
def capture_frame(camera, is_picam):
    if is_picam:
        frame = camera.capture_array()  # RGB 그대로 반환
        return frame
    else:
        ret, frame = camera.read()
        if not ret:
            raise Exception("ERROR: Failed to capture frame.")
        return frame

# ---------------------------
# 움직임 분석
# ---------------------------
def get_threshold_frame(current_frame, previous_frame):
    current_blurred = cv2.GaussianBlur(current_frame, BLUR_SIZE, 0)
    previous_blurred = cv2.GaussianBlur(previous_frame, BLUR_SIZE, 0)

    gray_current = cv2.cvtColor(current_blurred, cv2.COLOR_RGB2GRAY)
    gray_previous = cv2.cvtColor(previous_blurred, cv2.COLOR_RGB2GRAY)

    diff = cv2.absdiff(gray_previous, gray_current)
    _, thresh = cv2.threshold(diff, THRESHOLD_SENSITIVITY, 255, cv2.THRESH_BINARY)
    thresh = cv2.dilate(thresh, None, iterations=2)
    return thresh

def detect_movement_value(thresh):
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) >= MIN_CONTOUR_AREA:
            return 1
    return 0

# ---------------------------
# 소켓 서버 초기화
# ---------------------------
def initialize_server():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('', PORT))
    server_socket.listen(1)
    print(f"[INFO] Server listening on port {PORT}")
    conn, addr = server_socket.accept()
    print(f"[INFO] Connected by {addr}")
    return conn

# ---------------------------
# 메인 실행
# ---------------------------
def main():
    camera, is_picam = initialize_camera()
    previous_frame = capture_frame(camera, is_picam)
    conn = initialize_server()

    try:
        while True:
            current_frame = capture_frame(camera, is_picam)
            thresh = get_threshold_frame(current_frame, previous_frame)
            movement = detect_movement_value(thresh)

            # RGB 그대로 JPEG 압축
            _, jpeg = cv2.imencode('.jpg', current_frame)
            data = pickle.dumps((jpeg, movement))
            size = struct.pack(">L", len(data))
            conn.sendall(size + data)

            previous_frame = current_frame.copy()
            time.sleep(FRAME_WAIT_TIME)

    except KeyboardInterrupt:
        print("\n[INFO] Stopped.")
    finally:
        conn.close()
        if is_picam:
            camera.stop()
        else:
            camera.release()

if __name__ == "__main__":
    main()
