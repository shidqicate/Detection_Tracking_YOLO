# YOLO Drone Object Tracking & Following with DJI Tello
## _The Last Markdown Editor, Ever_

[![N|Solid](https://cldup.com/dTxpPi9lDf.thumb.png)](https://nodesource.com/products/nsolid)

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://github.com/shidqicate)

This project implements an autonomous drone control system using the **DJI Tello**, equipped with real-time object tracking and following capabilities. The system utilizes the **YOLO (You Only Look Once)** model optimized with *TensorRT* for fast and efficient inference. The drone can detect trained objects, track their movements, and automatically adjust its position to stay within the frame. This project is a perfect example of a practical application of computer vision and robotics.

## 🎯 Features
- 🎯 Real-time object detection using YOLOv5s optimized with TensorRT
- 🧠 Custom object tracking logic for smooth movement and target lock
- 🤖 Autonomous drone control via djitellopy (Python SDK for Tello)
- 🧩 Integrated with OpenCV for video streaming and visual debugging
- ⚡ Lightweight and runs efficiently on Jetson Nano

## 🛠 ️Tech Stack
---
Before running this project, make sure your system has:

- [YOLO Model](https://docs.ultralytics.com/)
- [TensorRT](https://developer.nvidia.com/tensorrt)
- Python 3
- OpenCV for Python
- [DJITelloPy](https://github.com/damiafuentes/DJITelloPy)
- PyTorch


## 📸 How it works
---
- A YOLO model detects an object (e.g., person) in real-time.
- The script calculates the object's position relative to the center of the frame.
- The DJI Tello adjusts its yaw, forward/backward, and up/down movement to keep the object centered.
- A video feed displays bounding boxes and debug info.

## Installation
---
How to Installation:

- Download & Install [YOLO Requirements](https://docs.ultralytics.com/)
- Download a pretrained model like  [Yolov8](https://docs.ultralytics.com/models/yolov8/)
- See [DjiTelloPy Documentation](https://djitellopy.readthedocs.io/en/latest/tello/)
- Connect your computer to the Tello drone's Wi-Fi network.
- Run the main Python script:
 ```python3 track_final.py```
