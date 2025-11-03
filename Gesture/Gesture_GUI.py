import cv2
import mediapipe as mp
import numpy as np
from joblib import load
import tensorflow as tf
import time
import math
import threading
import os,sys
def resource_path(rel_path):
    """Get absolute path for PyInstaller or normal run"""
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, rel_path)
    return os.path.join(os.path.abspath("."), rel_path)

m_p = resource_path("Gesture/model/hand_gesture_model.tflite")
s_c = resource_path("Gesture/model/hand_scaler.pkl")
def start_visualizer():
    interpreter = tf.lite.Interpreter(model_path=m_p)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    scaler = load(s_c)

    gesture_labels = {
        0: "Pointing",
        1: "Open Hand",
        2: "Closed Fist",
        3: "Right Click",
        4: "Left Click",
    }

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.6, min_tracking_confidence=0.6)

    def predict_gesture(landmarks):
        coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
        coords -= coords[0]
        X = coords.flatten().reshape(1, -1)
        X_scaled = scaler.transform(X).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], X_scaled)
        interpreter.invoke()
        out = interpreter.get_tensor(output_details[0]['index'])[0]
        return gesture_labels.get(int(np.argmax(out)), "Unknown")

    def distance(a, b): return math.hypot(a[0] - b[0], a[1] - b[1])

    def compute_percentage(hand_landmarks, w, h):
        coords = [(lm.x * w, lm.y * h) for lm in hand_landmarks.landmark]
        thumb, index, middle, ring, pinky = coords[4], coords[8], coords[12], coords[16], coords[20]
        thumb_index = distance(thumb, index)
        mid_ring = distance(middle, ring)
        ring_pinky = distance(ring, pinky)
        if mid_ring > 55 or ring_pinky > 70:
            return None
        return np.clip((thumb_index - 50) / 150 * 100, 0, 100)

    def draw_ring(frame, cx, cy, pct, label, color=(255, 255, 180)):
        r = 45
        end = int(360 * pct / 100)
        cv2.circle(frame, (cx, cy), r + 6, (0, 128, 255), 1)
        cv2.ellipse(frame, (cx, cy), (r, r), 0, -90, -90 + end, color, 3)
        cv2.putText(frame, f"{int(pct)}%", (cx - 25, cy + 10),cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, label, (cx - 40, cy - 60),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    def draw_box(frame, hand_landmarks, w, h, color=(0, 255, 255)):
        x_min = min([lm.x for lm in hand_landmarks.landmark]) * w
        y_min = min([lm.y for lm in hand_landmarks.landmark]) * h
        x_max = max([lm.x for lm in hand_landmarks.landmark]) * w
        y_max = max([lm.y for lm in hand_landmarks.landmark]) * h
        cv2.rectangle(frame, (int(x_min)-10, int(y_min)-10),(int(x_max)+10, int(y_max)+10), color, 2)
        return int((x_min + x_max) / 2), int((y_min + y_max) / 2)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 30)
    p_time = 0
    gesture_data = {"Left": {"text": "None", "time": 0},"Right": {"text": "None", "time": 0}}
    gesture_timeout = 2.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        c_time = time.time()
        fps = 1 / (c_time - p_time) if p_time else 0
        p_time = c_time

        left_pct, right_pct = None, None

        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,mp_drawing.DrawingSpec(color=(0,255,255), thickness=2, circle_radius=2),mp_drawing.DrawingSpec(color=(255,0,255), thickness=2))
                gesture = predict_gesture(hand_landmarks)
                label = handedness.classification[0].label
                gesture_data[label]["text"] = gesture
                gesture_data[label]["time"] = time.time()
                pct = compute_percentage(hand_landmarks, w, h)
                if label == "Left": left_pct = pct
                else: right_pct = pct
                cx, cy = draw_box(frame, hand_landmarks, w, h)
                color = (0, 255, 255) if label == "Left" else (255, 255, 180)
                cv2.putText(frame, f"{label}: {gesture}", (cx - 70, cy - 70),cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        for label in ["Left", "Right"]:
            if time.time() - gesture_data[label]["time"] < gesture_timeout:
                txt = gesture_data[label]["text"]
                y = 60 if label == "Left" else 100
                cv2.putText(frame, f"{label}: {txt}", (20, y),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.putText(frame, f"FPS: {int(fps)}", (w - 120, 40),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        if left_pct is not None:
            draw_ring(frame, 80, h // 2, left_pct, "LEFT")
        if right_pct is not None:
            draw_ring(frame, w - 80, h // 2, right_pct, "RIGHT")

        cv2.imshow("Gesture Visualizer (press ESC to exit)", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()