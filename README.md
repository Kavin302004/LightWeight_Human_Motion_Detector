### Real-Time Human  Motion Detection using Lightweight YOLO Model

#### Overview

This project demonstrates real-time human  motion detection using a lightweight YOLO (You Only Look Once) object detection model with OpenCV in Python. The emphasis is on low computational requirements suitable for integration with security cameras.
![image](https://github.com/Kavin302004/LightWeight_Human_Motion_Detector/assets/140266232/11b1a5de-7f70-4e11-8510-aec15657b3c4)


#### Features

- **Human Detection**: Utilizes a lightweight YOLO model to detect humans in video frames captured from a camera.
- **Motion Detection**: Detects motion by comparing each frame with an initial background frame using image differencing techniques.
- **Visualization**: Draws bounding boxes around detected humans and displays the processed frames in real-time using OpenCV.

#### Requirements

- Python 3.x
- OpenCV
- YOLOv3 Tiny (weights and configuration)
- COCO Names file

#### How to Use

1. **Setup**:
   - Install Python dependencies (`opencv-python`, `numpy`, etc.).
   - Download YOLOv3 Tiny weights (`yolov3-tiny.weights`) and configuration (`yolov3-tiny1.cfg`).
   - Download COCO names file (`coco.names`).

2. **Execution**:
   - Run the Python script (`python your_script.py`).
   - Ensure the camera is correctly connected and providing input.

3. **Interact**:
   - The script detects and highlights humans in real-time video feed.
   - Motion detection triggers upon significant changes in the frame compared to the initial background.

#### Notes

- Customize parameters (`confidence threshold`, `motion sensitivity`) in the script for optimal performance based on your environment.
- This lightweight YOLO model is designed for low computational requirements, making it suitable for integration with security cameras.

#### Future Enhancements

- Integration with cloud services for remote monitoring.
- Incorporation of additional lightweight models or optimizations for further reducing computational load.
