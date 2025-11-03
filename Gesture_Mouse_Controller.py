import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import tensorflow as tf
from joblib import load
import time
import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple
from queue import Queue, Empty
from threading import Thread, Event, Lock
from concurrent.futures import ThreadPoolExecutor
import screen_brightness_control as sbc
from VolumeController import SystemVolume as sv
import logging
import os, sys

def resource_path(rel_path):
    """Get absolute path for PyInstaller or normal run"""
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, rel_path)
    return os.path.join(os.path.abspath("."), rel_path)

m_p = resource_path("Gesture/model/hand_gesture_model.tflite")
s_c = resource_path("Gesture/model/hand_scaler.pkl")

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

@dataclass
class Config:
    model_path: str = m_p
    scaler_path: str = s_c
    gesture_labels: Dict[int, str] = field(default_factory=lambda: {
        0: "pointing",
        1: "open hand",
        2: "closed hand",
        3: "right click",
        4: "left click",
        5: "maximize",
        6: "minimize",
        7: "next_w",
        8: "prev_w"
    })
    cam_index: int = 0
    cam_width: int = 640
    cam_height: int = 480
    cam_fps: int = 30
    default_mode: str = "trackpad"
    center_x: float = 0.5
    center_y: float = 0.5
    dead_zone: float = 0.08
    mouse_speed: float = 1200.0
    smoothing_factor: float = 0.25
    move_threshold_px: float = 2.0
    mouse_jump_threshold: int = 100
    scroll_speed: float = 12.0
    max_scroll: int = 1000
    brightness_hand_label: str = "Left"
    volume_hand_label: str = "Right"
    max_queue_size: int = 4
    inference_workers: int = 1
    inference_rate_hz: float = 15
    frame_skip_ratio: int = 1
    window_action_debounce_s: float = 1.0
    global_action_cooldown_s: float = 1.0

CONFIG = Config()

pyautogui.FAILSAFE = False
SCREEN_W, SCREEN_H = pyautogui.size()

def now() -> float:
    return time.time()

def clamp(v, a, b):
    return max(a, min(b, v))

class GestureModel:
    def __init__(self, model_path: str, scaler_path: str, gesture_map: Dict[int, str]):
        logging.info("Initializing gesture model...")
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.gesture_map = gesture_map
        self.scaler = load(self.scaler_path)
        self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        logging.info("Model & scaler loaded.")

    def predict(self, hand_landmarks) -> str:
        coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
        if coords.shape[0] != 21:
            return "Unknown"
        coords -= coords[0]
        X = coords.flatten().reshape(1, -1)
        X_scaled = self.scaler.transform(X).astype(np.float32)
        self.interpreter.set_tensor(self.input_details[0]['index'], X_scaled)
        self.interpreter.invoke()
        out = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        gesture_id = int(np.argmax(out))
        return self.gesture_map.get(gesture_id, "Unknown")

class GestureActions:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.last_window_action_time = 0.0
        self.click_start_time: Optional[float] = None
        self.click_in_progress = False
        self.grab_state = False
        self.last_gesture = None
        self.lock = Lock()
        self.last_action_time: Dict[str, float] = {}
        self.global_cooldown = cfg.global_action_cooldown_s
        self.exempt_keywords = [
            "brightness", "volume",
            "scroll", "click", "grab", "release", "mouse"
        ]

    def _is_exempt(self, gesture: str) -> bool:
        g = gesture.lower()
        return any(k in g for k in self.exempt_keywords)

    def _can_trigger(self, gesture: str) -> bool:
        if self._is_exempt(gesture):
            return True
        t = now()
        g = gesture.lower()
        last = self.last_action_time.get(g, 0.0)
        if (t - last) >= self.global_cooldown:
            self.last_action_time[g] = t
            return True
        return False

    def handle_click(self, gesture: str):
        if not self._can_trigger(gesture):
            return
        with self.lock:
            if gesture == "right click":
                pyautogui.click(button='right')
                self._reset_click()
                return
            if gesture == "left click":
                t = now()
                if not self.click_in_progress:
                    self.click_in_progress = True
                    self.click_start_time = t
                else:
                    if (t - self.click_start_time) > 2.5:
                        pyautogui.doubleClick(button='left')
                        self._reset_click()
                return
            if self.click_in_progress:
                duration = now() - (self.click_start_time or 0)
                if duration < 2.0:
                    pyautogui.click(button='left')
                self._reset_click()

    def _reset_click(self):
        self.click_in_progress = False
        self.click_start_time = None

    def update_grab(self, gesture: str) -> bool:
        with self.lock:
            self.grab_state = (gesture == "closed hand")
            self.last_gesture = gesture
            return self.grab_state

    def perform_scroll(self, grabbed: bool, hand_landmarks) -> None:
        if not grabbed:
            return
        if not self._can_trigger("scroll"):
            return
        avg_x = np.mean([lm.x for lm in hand_landmarks.landmark])
        avg_y = np.mean([lm.y for lm in hand_landmarks.landmark])
        offset_x = avg_x - self.cfg.center_x
        offset_y = avg_y - self.cfg.center_y
        if abs(offset_x) < self.cfg.dead_zone and abs(offset_y) < self.cfg.dead_zone:
            return
        distance = math.sqrt(offset_x * offset_x + offset_y * offset_y)
        strength = clamp(distance * self.cfg.scroll_speed * 1000, -self.cfg.max_scroll, self.cfg.max_scroll)
        scroll_y = int(clamp(-offset_y * strength, -self.cfg.max_scroll, self.cfg.max_scroll))
        scroll_x = int(clamp(-offset_x * strength, -self.cfg.max_scroll, self.cfg.max_scroll))
        if abs(scroll_y) >= 1:
            pyautogui.scroll(scroll_y)
        if abs(scroll_x) >= 1:
            try:
                pyautogui.hscroll(scroll_x)
            except Exception:
                pass

    def adjust_brightness_volume(self, gesture: str, hand_landmarks, handedness) -> None:
        if gesture != "open hand":
            return
        coords = [(lm.x * SCREEN_W, lm.y * SCREEN_H) for lm in hand_landmarks.landmark]
        thumb, index, middle, ring, pinky = coords[4], coords[8], coords[12], coords[16], coords[20]
        def d(a,b): return math.hypot(a[0]-b[0], a[1]-b[1])
        thumb_index_dist = d(thumb, index)
        middle_ring_dist = d(middle, ring)
        ring_pinky_dist = d(ring, pinky)
        pct = clamp((thumb_index_dist - 50) / (200 - 50), 0.0, 1.0) * 100.0
        adjust_mode = (middle_ring_dist < 50 and ring_pinky_dist < 65)
        if not adjust_mode:
            return
        hand_label = handedness.classification[0].label
        if hand_label == self.cfg.brightness_hand_label:
            try:
                sbc.set_brightness(pct)
            except Exception as e:
                logging.debug("Brightness set failed: %s", e)
        elif hand_label == self.cfg.volume_hand_label:
            try:
                sv.set(pct)
            except Exception as e:
                logging.debug("Volume set failed: %s", e)


class GestureController:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model = GestureModel(cfg.model_path, cfg.scaler_path, cfg.gesture_labels)
        self.actions = GestureActions(cfg)
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hand_detector = self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
        self.frame_queue: Queue = Queue(maxsize=cfg.max_queue_size)
        self.stop_event = Event()
        self.camera_thread: Optional[Thread] = None
        self.inference_executor = ThreadPoolExecutor(max_workers=cfg.inference_workers)
        self.last_inference_time = 0.0
        self.cursor_x, self.cursor_y = pyautogui.position()
        self.prev_cursor_x, self.prev_cursor_y = self.cursor_x, self.cursor_y

    def _camera_worker(self):
        cap = cv2.VideoCapture(self.cfg.cam_index, cv2.CAP_DSHOW if hasattr(cv2, 'CAP_DSHOW') else 0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.cam_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.cam_height)
        cap.set(cv2.CAP_PROP_FPS, self.cfg.cam_fps)
        logging.info("Camera thread started.")
        frame_index = 0
        while not self.stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                logging.warning("Camera read failed; stopping camera thread.")
                break
            frame = cv2.flip(frame, 1)
            frame_index += 1
            if self.cfg.frame_skip_ratio > 1 and (frame_index % self.cfg.frame_skip_ratio != 0):
                pass
            try:
                if self.frame_queue.full():
                    try:
                        _ = self.frame_queue.get_nowait()
                    except Empty:
                        pass
                self.frame_queue.put_nowait(frame)
            except Exception:
                pass
        cap.release()
        logging.info("Camera thread exiting.")

    def _inference_task(self, frame) -> Tuple[Any, Any]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hand_detector.process(rgb)
        return results, frame

    def run(self, show):
        self.camera_thread = Thread(target=self._camera_worker, daemon=True)
        self.camera_thread.start()
        logging.info("GestureController started.")
        try:
            while True:
                try:
                    frame = self.frame_queue.get(timeout=0.5)
                except Empty:
                    if self.stop_event.is_set():
                        break
                    continue
                current_t = now()
                min_interval = 1.0 / max(0.1, self.cfg.inference_rate_hz)
                run_inference = (current_t - self.last_inference_time) >= min_interval
                if run_inference:
                    future = self.inference_executor.submit(self._inference_task, frame)
                    self.last_inference_time = current_t
                    try:
                        results, infer_frame = future.result(timeout=0.5)
                    except Exception:
                        continue
                    out_frame = self._process_frame(infer_frame, results)
                else:
                    out_frame = frame
                if show:
                    cv2.imshow("Gesture Mouse Controller", out_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('m'):
                    self.cfg.default_mode = "joystick" if self.cfg.default_mode == "trackpad" else "trackpad"
                    logging.info("Switched to mode: %s", self.cfg.default_mode)
                elif key == 27:
                    logging.info("ESC pressed, exiting.")
                    break
        finally:
            self.stop()

    def _process_frame(self, frame, results):
        if results and results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                gesture = self.model.predict(hand_landmarks)
                grabbed = self.actions.update_grab(gesture)
                if self.cfg.default_mode == "joystick":
                    self._handle_joystick(hand_landmarks, gesture)
                else:
                    self._handle_trackpad(hand_landmarks, gesture)
                self.actions.perform_scroll(grabbed, hand_landmarks)
                self.actions.adjust_brightness_volume(gesture, hand_landmarks, handedness)
                self.actions.handle_click(gesture)
                cv2.putText(frame, f"{gesture} | Grab={grabbed}", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        return frame

    def _handle_joystick(self, hand_landmarks, gesture):
        index = hand_landmarks.landmark[8]
        offset_x = index.x - self.cfg.center_x
        offset_y = index.y - self.cfg.center_y

        if gesture in ["pointing", "right click", "left click"]:
            if abs(offset_x) > self.cfg.dead_zone or abs(offset_y) > self.cfg.dead_zone:
                move_x = offset_x * self.cfg.mouse_speed
                move_y = offset_y * self.cfg.mouse_speed

                # Smooth interpolation (LERP)
                target_x = clamp(self.cursor_x + move_x, 0, SCREEN_W - 1)
                target_y = clamp(self.cursor_y + move_y, 0, SCREEN_H - 1)
                self.cursor_x += (target_x - self.cursor_x) * self.cfg.smoothing_factor
                self.cursor_y += (target_y - self.cursor_y) * self.cfg.smoothing_factor

                # Only move if significant change
                if (abs(self.cursor_x - self.prev_cursor_x) >= self.cfg.move_threshold_px or
                    abs(self.cursor_y - self.prev_cursor_y) >= self.cfg.move_threshold_px):
                    pyautogui.moveTo(self.cursor_x, self.cursor_y, duration=0)
                    self.prev_cursor_x, self.prev_cursor_y = self.cursor_x, self.cursor_y


    def _handle_trackpad(self, hand_landmarks, gesture):
        if gesture in ["pointing", "right click", "left click"]:
            index = hand_landmarks.landmark[8]
            target_x = index.x * SCREEN_W
            target_y = index.y * SCREEN_H

            # Skip big jumps (when hand re-enters frame)
            if abs(target_x - self.prev_cursor_x) > self.cfg.mouse_jump_threshold or abs(target_y - self.prev_cursor_y) > self.cfg.mouse_jump_threshold:
                self.prev_cursor_x, self.prev_cursor_y = target_x, target_y
                return

            # Smooth interpolation (LERP)
            self.cursor_x += (target_x - self.cursor_x) * self.cfg.smoothing_factor
            self.cursor_y += (target_y - self.cursor_y) * self.cfg.smoothing_factor

            # Move only if actual motion
            if (abs(self.cursor_x - self.prev_cursor_x) >= self.cfg.move_threshold_px or
                abs(self.cursor_y - self.prev_cursor_y) >= self.cfg.move_threshold_px):
                pyautogui.moveTo(self.cursor_x, self.cursor_y, duration=0)
                self.prev_cursor_x, self.prev_cursor_y = self.cursor_x, self.cursor_y


    def stop(self):
        logging.info("Stopping GestureController...")
        self.stop_event.set()
        if self.camera_thread and self.camera_thread.is_alive():
            self.camera_thread.join(timeout=1.0)
        self.inference_executor.shutdown(wait=False)
        try:
            if self.hand_detector:
                self.hand_detector.close()
        except Exception:
            pass
        cv2.destroyAllWindows()
        logging.info("Stopped.")

def main():
    controller = GestureController(CONFIG)
    try:
        controller.run(False)
    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt received.")
    finally:
        controller.stop()

if __name__ == "__main__":
    main()
