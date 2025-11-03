import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_face_detection = mp.solutions.face_detection

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands()
face = mp_face.FaceMesh()
pose = mp_pose.Pose()
holistic = mp_holistic.Holistic()
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    hand_results = hands.process(rgb)
    face_results = face.process(rgb)
    pose_results = pose.process(rgb)
    holistic_results = holistic.process(rgb)
    face_detect_results = face_detection.process(rgb)

    if holistic_results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, holistic_results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            mp_drawing_styles.get_default_pose_landmarks_style()
        )

    if holistic_results.face_landmarks:
        mp_drawing.draw_landmarks(
            frame, holistic_results.face_landmarks,
            mp_holistic.FACEMESH_TESSELATION,
            mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1)
        )

    if holistic_results.left_hand_landmarks:
        mp_drawing.draw_landmarks(frame, holistic_results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if holistic_results.right_hand_landmarks:
        mp_drawing.draw_landmarks(frame, holistic_results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    if face_detect_results.detections:
        for detection in face_detect_results.detections:
            mp_drawing.draw_detection(frame, detection)

    cv2.imshow("MediaPipe All-in-One (Updated)", frame)
    if cv2.waitKey(1) & 0xFF == 27:  
        break

cap.release()
cv2.destroyAllWindows()
