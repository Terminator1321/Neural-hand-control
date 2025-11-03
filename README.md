# Neural Hand Control

Neural Hand Control is an AI-based gesture recognition system that lets you control your computer using hand movements.
It allows you to move the mouse, perform clicks, scroll, and adjust brightness or system volume — all through intuitive hand gestures.

It includes a real-time gesture visualizer and a pretrained TensorFlow Lite model for accurate recognition and fast performance.

Download the `.exe` version and usage guide here:
[https://rohan-das.great-site.net/Projects/Docs/tut_app.html](https://rohan-das.great-site.net/Projects/Docs/tut_app.html)

---

## Features

* Real-time hand gesture tracking using MediaPipe
* AI-powered gesture classification with TensorFlow Lite
* Control mouse movement, clicks, and scrolling with hand gestures
* Adjust brightness and volume using left and right hand respectively
* Two control modes: Trackpad and Joystick
* Multithreaded execution for smooth, lag-free performance
* Visualizer tool to see how gestures are interpreted by AI
* Centralized configuration system using a Python dataclass

---

## Overview

Neural Hand Control combines computer vision, machine learning, and system automation to enable touchless computer control.
It uses MediaPipe for real-time hand tracking and a custom TensorFlow Lite model for gesture classification.

The system recognizes gestures such as pointing, open hand, closed fist, and click actions, mapping them to cursor control and system commands.

It integrates:

* PyAutoGUI for mouse movement, clicking, and scrolling
* Screen Brightness Control for display brightness adjustment
* Pycaw (COMTypes) for controlling system volume

This allows full gesture-based computer interaction, including joystick-style navigation and real-time multitasking.

---

## Main Modules

* **Gesture_GUI.py** – Visualizer for seeing real-time gesture recognition.
* **Gesture_Mouse_Controller.py** – Main controller managing gesture detection, mouse movement, brightness, and volume control.
* **CONFIG Dataclass** – Central configuration file storing runtime parameters such as camera settings, gesture mappings, and control thresholds.

---

## Folder Structure

```
NeuralHandControl/
│
├── Gesture/
│   ├── Handdata/
│   │   └── hand_data.csv
│   │
│   ├── model/
│   │   ├── hand_gesture_model.h5
│   │   ├── hand_gesture_model.tflite
│   │   └── hand_scaler.pkl
│   │
│   ├── Gesture_GUI.py
│   ├── Gesture_Mouse_Controller.py
│   ├── hand_point
│   ├── train.ipynb
│   │
│   ├── main.py
│   ├── NeuralHandControl.spec
│   ├── r.txt
│   ├── test.py
│   └── VolumeController.py
│
└── (Generated files & build outputs)
```

---

## Folder Highlights

* **Handdata/** – Contains the training dataset (`hand_data.csv`).
* **model/** – Stores trained gesture models and scaler files.
* **Gesture_GUI.py** – Visualizer for live gesture detection.
* **Gesture_Mouse_Controller.py** – Core controller for gesture recognition and action mapping.
* **train.ipynb** – Jupyter notebook for model training and testing.
* **VolumeController.py** – Adjusts system audio via Pycaw.
* **main.py** – Primary application entry point.

---

## How Gesture_Mouse_Controller.py Works

`Gesture_Mouse_Controller.py` acts as the central engine of Neural Hand Control.
It integrates MediaPipe, TensorFlow Lite, and PyAutoGUI to recognize gestures and trigger real-time actions such as mouse control, scrolling, and brightness or volume adjustment.

### Core Components

**1. GestureModel**

* Loads the TensorFlow Lite model and scaler.
* Extracts and normalizes 21 hand landmarks using MediaPipe.
* Classifies gestures like pointing, open hand, closed hand, and clicks.

**2. GestureActions**

* Executes actions based on the recognized gesture.
* Handles clicks (left, double, right), scrolling, brightness, and volume changes.
* Uses cooldowns and locks to prevent rapid or duplicate actions.
* Left hand adjusts brightness; right hand adjusts volume.

**3. GestureController**

* Orchestrates the entire workflow.
* Captures video frames using OpenCV.
* Processes hand landmarks and gesture inference in parallel threads.
* Passes results to GestureActions for execution.
* Supports:

  * **Trackpad Mode:** Direct cursor control from hand position.
  * **Joystick Mode:** Relative cursor control (toggle with “M”).

---

### Runtime Flow

Camera Feed → MediaPipe → TensorFlow Lite Model → Gesture Classification → GestureActions → System Actions
(Mouse, Scroll, Brightness, Volume)

The system uses multithreading for real-time responsiveness and smooth motion.

---

### Technical Highlights

* **Hand Tracking:** MediaPipe for 3D hand landmark detection
* **Gesture Classification:** TensorFlow Lite for efficient inference
* **Mouse Control:** PyAutoGUI for cursor and click automation
* **Brightness Control:** ScreenBrightnessControl for system brightness
* **Volume Control:** Pycaw (COMTypes) for system volume control
* **Concurrency:** ThreadPoolExecutor and Queue for asynchronous processing

---

### Exit Controls

* Press **M** – Toggle between joystick and trackpad modes
* Press **ESC** – Stop and exit the controller safely

---

## Technologies Combined

This project brings together multiple advanced technologies:

* **Computer Vision:** MediaPipe and OpenCV for hand tracking
* **Machine Learning:** TensorFlow Lite and Scikit-learn for gesture recognition
* **System Automation:** PyAutoGUI, Pycaw, and ScreenBrightnessControl for controlling the OS
* **Multithreading:** Python’s ThreadPoolExecutor for parallel real-time processing
* **Configuration:** Python Dataclasses for centralized configuration and tuning

---

## Setup & Run

Follow these steps to set up and run Neural Hand Control from source:

### 1. Clone the Repository

```bash
git clone https://github.com/Terminator1321/Neural-hand-control.git
cd Neural-hand-control/Gesture
```

### 2. Install Dependencies

Make sure you have Python 3.8+ installed, then run:

```bash
pip install -r r.txt
```

### 3. Run the Application

To start the main gesture controller:

```bash
python Gesture_Mouse_Controller.py
```

To view the visualizer (for AI gesture detection preview):

```bash
python Gesture_GUI.py
```

### 4. Optional: Run the Prebuilt Executable

If you don’t want to set up Python, download the `.exe` file directly from:
[https://rohan-das.great-site.net/Projects/Docs/tut_app.html](https://rohan-das.great-site.net/Projects/Docs/tut_app.html)

### 5. Controls

* Move hand to control the cursor
* Close fist to scroll
* Left hand open → Adjust brightness
* Right hand open → Adjust volume
* Press **M** → Switch control mode
* Press **ESC** → Exit

