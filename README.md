# 🖐️ Gesture-Controlled Virtual Mouse

A real-time computer vision application that enables touchless control of a computer using hand gestures. The system leverages MediaPipe Hand Landmarks and OpenCV to track hand movements and translate them into mouse actions such as cursor movement, clicking, scrolling, and screenshot capture.

## 🚀 Features

- Real-time hand tracking using MediaPipe
- Cursor control through index finger movement
- Click detection using thumb–index finger pinch gestures
- Scroll up/down using finger position analysis
- Screenshot capture through custom gestures
- Motion smoothing for stable cursor movement
- Visual landmark rendering for real-time feedback

## 🛠️ Tech Stack

- Python
- OpenCV
- MediaPipe
- PyAutoGUI

## 📌 System Architecture

```text
Webcam Feed
     ↓
Frame Capture (OpenCV)
     ↓
Hand Detection & Landmark Estimation (MediaPipe)
     ↓
Gesture Recognition Logic
     ↓
Mouse & System Actions (PyAutoGUI)

