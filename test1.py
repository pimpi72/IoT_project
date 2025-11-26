#pip install picamera2 numpy opencv-python #라즈베리파이에서 이 부분을 실행해서 picamera2를 설치 ㄱㄱ
#pip install mediapipe opencv-python #일반 pc에서는 이 부분을 실행해서 mediapipe 설치 ㄱ
import cv2
import mediapipe as mp
import numpy as np
import math
import time
# 🚨 Picamera2 라이브러리 추가
from picamera2 import Picamera2

# Mediapipe 초기화
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# 각도 계산 함수 (변경 없음)
def calculate_angle(a, b, c):
    """
    세 점 a, b, c의 각도를 계산합니다.
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    # 벡터 계산
    ba = a - b
    bc = c - b
    
    # 코사인 유사도 계산
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(cosine_angle)
    
    return np.degrees(angle)

# 쓰러짐 판정 함수
def is_fall_condition_met(keypoints):
    """
    mediapipe가 감지한 keypoints를 이용해 쓰러짐 여부를 판별합니다.
    (기존 로직 유지)
    """
    try:
        # Get keypoints for shoulders, hips, and knees
        left_shoulder = keypoints[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = keypoints[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_hip = keypoints[mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = keypoints[mp_pose.PoseLandmark.RIGHT_HIP.value]
        left_knee = keypoints[mp_pose.PoseLandmark.LEFT_KNEE.value]
        right_knee = keypoints[mp_pose.PoseLandmark.RIGHT_KNEE.value]

        # Ensure keypoints are within the frame (visibility > 0.5)
        if (left_shoulder.visibility < 0.5 or right_shoulder.visibility < 0.5 or
                left_hip.visibility < 0.5 or right_hip.visibility < 0.5 or
                left_knee.visibility < 0.5 or right_knee.visibility < 0.5):
            return False

        # Calculate the midpoint of shoulders and hips
        shoulder = [(left_shoulder.x + right_shoulder.x) / 2, (left_shoulder.y + right_shoulder.y) / 2]
        hip = [(left_hip.x + right_hip.x) / 2, (left_hip.y + right_hip.y) / 2]
        knee = [(left_knee.x + right_knee.x) / 2, (left_knee.y + right_knee.y) / 2]

        # Calculate the angle between shoulder-hip and hip-knee lines
        angle_shoulder_hip_knee = calculate_angle(shoulder, hip, knee)
        
        # 기존 로직 유지: 30도 이하 또는 150도 이상일 때 '쓰러짐'으로 간주
        if 30 < angle_shoulder_hip_knee < 150:
             return False
        else:
             return True
             
    except:
        return False

#실시간으로 영상 처리하는 부분
picam2 = Picamera2()
# 카메라 설정: BGR(OpenCV 기본 형식)로 캡처하도록 설정
picam2.configure(picam2.create_preview_configuration(main={"format": 'BGR', "size": (640, 480)}))
picam2.start()


# 포즈 객체 초기화
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    # 쓰러짐 상태 추적 변수
    fall_detected = False
    fall_alert_triggered = False
    fall_start_time = None
    required_duration = 0.2  # 초 단위

    while True: # 무한 루프 (실시간 스트리밍)
        # 🚨 Picamera2에서 최신 프레임을 캡처하고 NumPy 배열로 변환합니다.
        frame = picam2.capture_array()
        
        if frame is None:
            print("프레임을 불러올 수 없습니다.")
            break

        # 이미지 좌우 반전 (필요에 따라 주석 처리 가능)
        image = cv2.flip(frame, 1) 
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            # Mediapipe 랜드마크 추출
            keypoints = results.pose_landmarks.landmark

            # 각도 판정
            fall_condition = is_fall_condition_met(keypoints)

            current_time = time.time()

            if fall_condition:
                if not fall_detected:
                    # 첫 쓰러짐 조건 만족
                    fall_start_time = current_time
                    fall_detected = True
                else:
                    # 지속 시간 확인
                    elapsed_time = current_time - fall_start_time
                    if elapsed_time >= required_duration:
                        # 지정된 시간 이상 쓰러짐 상태 유지 -> 알람 트리거
                        fall_alert_triggered = True
            else:
                # 쓰러짐 조건 불만족
                fall_detected = False
                fall_start_time = None
                fall_alert_triggered = False

            # 
            if fall_alert_triggered:
                cv2.putText(image, '🚨 쓰러졌습니다!!!🚨', (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            # 랜드마크 그리기
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # 화면에 이미지 표시
        cv2.imshow('Fall Detection', image)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# 🚨 자원 해제: Picamera2 중지 및 창 닫기
picam2.stop()
cv2.destroyAllWindows()