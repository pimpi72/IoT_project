from flask import Flask, Response, render_template
import cv2
import socket
import struct
import pickle
import mediapipe as mp
import numpy as np
import time
import threading

app = Flask(__name__)

# ---------------------------
# Raspberry Pi 서버 설정
# ---------------------------
RASPBERRY_IP = "172.20.9.83"
PORT = 8486

# ---------------------------
# Mediapipe 초기화
# ---------------------------
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# ---------------------------
# 낙상 상태 플래그 (웹에서 읽을 값)
# ---------------------------
fall_flag = False   # ★ 추가됨

# ---------------------------
# 각도 계산 함수
# ---------------------------
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

# ---------------------------
# 낙상 판정 함수
# ---------------------------
def is_fall_condition_met(keypoints):
    try:
        ls = keypoints[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        rs = keypoints[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        lh = keypoints[mp_pose.PoseLandmark.LEFT_HIP.value]
        rh = keypoints[mp_pose.PoseLandmark.RIGHT_HIP.value]
        lk = keypoints[mp_pose.PoseLandmark.LEFT_KNEE.value]
        rk = keypoints[mp_pose.PoseLandmark.RIGHT_KNEE.value]

        # 관절 가시성 체크
        if min(ls.visibility, rs.visibility, lh.visibility, rh.visibility, lk.visibility, rk.visibility) < 0.5:
            return False

        shoulder = [(ls.x + rs.x) / 2, (ls.y + rs.y) / 2]
        hip = [(lh.x + rh.x) / 2, (lh.y + rh.y) / 2]
        knee = [(lk.x + rk.x) / 2, (lk.y + rk.y) / 2]

        angle = calculate_angle(shoulder, hip, knee)
        return 30 <= angle <= 150

    except:
        return False

# ---------------------------
# 전역 변수
# ---------------------------
latest_frame = None  # Flask가 스트리밍할 최신 프레임

def socket_receiver():
    """라즈베리파이로부터 프레임을 받아 처리 후 latest_frame과 fall_flag 업데이트"""
    global latest_frame, fall_flag  # ★ fall_flag 업데이트 위해 추가

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((RASPBERRY_IP, PORT))

    data = b""
    payload_size = struct.calcsize(">L")

    fall_detected = False
    fall_start = None
    required_duration = 0.2

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

        while True:
            # ---- 메시지 크기 수신 ----
            while len(data) < payload_size:
                packet = client_socket.recv(4096)
                if not packet:
                    return
                data += packet

            packed_size = data[:payload_size]
            data = data[payload_size:]
            msg_size = struct.unpack(">L", packed_size)[0]

            # ---- 실제 프레임 ----
            while len(data) < msg_size:
                packet = client_socket.recv(4096)
                if not packet:
                    return
                data += packet

            frame_data = data[:msg_size]
            data = data[msg_size:]

            jpeg, movement = pickle.loads(frame_data)
            frame = cv2.imdecode(jpeg, cv2.IMREAD_COLOR)

            # ---------------------------
            # Fall Detection
            # ---------------------------
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)
            now = time.time()

            if results.pose_landmarks:
                keypoints = results.pose_landmarks.landmark
                fall_cond = is_fall_condition_met(keypoints)

                if fall_cond:
                    if not fall_detected:
                        fall_detected = True
                        fall_start = now
                    else:
                        elapsed = now - fall_start
                        if elapsed >= required_duration:
                            fall_flag = True    # ★ 웹에 "낙상 발생" 신호 보냄
                            cv2.putText(frame, 'Fall Detected', (50, 50),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    fall_detected = False
                    fall_start = None
                    fall_flag = False  # ★ 정상 상태로 복귀

                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # 웹 스트리밍용 최신 프레임 업데이트
            latest_frame = frame


# ---------------------------
# Flask MJPEG 스트리밍
# ---------------------------
def generate_frames():
    global latest_frame
    while True:
        if latest_frame is None:
            continue

        _, buffer = cv2.imencode('.jpg', latest_frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# ---------------------------
# ★ 낙상 상태 확인 API (HTML에서 fetch로 호출)
# ---------------------------
@app.route("/fall-status")
def fall_status():
    global fall_flag
    return {"fall": fall_flag}


# ---------------------------
# 라우트
# ---------------------------
@app.route('/')
def home():
    return render_template('cameraPage.html')

@app.route('/camera')
def camera():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# ---------------------------
# 메인
# ---------------------------
if __name__ == "__main__":
    threading.Thread(target=socket_receiver, daemon=True).start()
    app.run(host="0.0.0.0", port=5000, debug=False)
