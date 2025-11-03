import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import os

SAVE_PATH = "Gesture/Handdata/hand_data.csv"
MIN_SAMPLES_PER_ID = 1500
MAX_FRAMES_PER_SESSION = 1000

headers = ["ID"]
for name in [
    "WRIST", "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_MCP", "INDEX_PIP", "INDEX_DIP", "INDEX_TIP",
    "MIDDLE_MCP", "MIDDLE_PIP", "MIDDLE_DIP", "MIDDLE_TIP",
    "RING_MCP", "RING_PIP", "RING_DIP", "RING_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP"
]:
    headers += [f"{name}_x", f"{name}_y", f"{name}_z"]

df = pd.DataFrame(columns=headers)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)
recording = False
frame_count = 0
temp_frames = []
session_counts = {i: 0 for i in range(10)}

def load_existing_counts(path):
    if not os.path.exists(path):
        return {i: 0 for i in range(10)}
    data = pd.read_csv(path)
    if "ID" not in data.columns:
        return {i: 0 for i in range(10)}
    counts = data["ID"].value_counts().to_dict()
    for i in range(10):
        counts.setdefault(i, 0)
    return counts

existing_counts = load_existing_counts(SAVE_PATH)

print("Press 'S' to start/stop recording, 0–9 to save with label, ESC to quit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    h, w, _ = frame.shape

    row = [np.nan] * len(headers)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for i, lm in enumerate(hand_landmarks.landmark):
                row[i * 3 + 1] = lm.x
                row[i * 3 + 2] = lm.y
                row[i * 3 + 3] = lm.z
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.putText(frame, f"Recording: {'ON' if recording else 'OFF'}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if recording else (0, 0, 255), 2)

    cv2.putText(frame, f"Frames: {frame_count}/{MAX_FRAMES_PER_SESSION}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    y_offset = 90
    for i in range(10):
        total = existing_counts.get(i, 0)
        added = session_counts.get(i, 0)
        total_display = total + added
        color = (0, 255, 0) if total_display >= MIN_SAMPLES_PER_ID else (0, 0, 255)
        cv2.putText(frame, f"ID {i}: total={total_display}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        y_offset += 25

    cv2.imshow("Dual 3D Hand Data Collector", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        recording = not recording
        frame_count = 0
        if recording:
            print("Started recording new session...")
        else:
            print("Stopped recording.")

    elif key in range(ord('0'), ord('9') + 1):
        if temp_frames:
            label = int(chr(key))
            for row in temp_frames:
                row[0] = label
                df.loc[len(df)] = row
            session_counts[label] += len(temp_frames)
            print(f"Saved {len(temp_frames)} frames with label {label}")
            temp_frames.clear()

    elif key == 27:
        print("Exiting...")
        break

    if recording and not np.isnan(row[1]):
        temp_frames.append(row)
        frame_count += 1

        if frame_count >= MAX_FRAMES_PER_SESSION:
            recording = False
            print(f"Auto-stopped after {MAX_FRAMES_PER_SESSION} frames.")
            frame_count = 0

cap.release()
cv2.destroyAllWindows()

if not df.empty:
    if os.path.exists(SAVE_PATH):
        old_df = pd.read_csv(SAVE_PATH)
        df = pd.concat([old_df, df], ignore_index=True)
        print(f"Appended new data to existing file.")
    else:
        print(f"Created new file with {len(df)} rows.")

    df.to_csv(SAVE_PATH, index=False)
    print("Data saved to", SAVE_PATH)
else:
    print("No data recorded.")
