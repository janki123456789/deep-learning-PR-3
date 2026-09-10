# 🚀 Computer Vision PR-3 — Classical Image Processing to Real-Time Deep Learning Detection

> **Red & White Skill Education | Deep Learning | Practical/Project PR-3**  
> **OpenCV 5.0 + YuNet + YOLOv8 + NumPy + Matplotlib**

[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5.0.0-green?logo=opencv)](https://opencv.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-red)](https://github.com/ultralytics/ultralytics)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](https://jupyter.org/)

---

## 📌 Project Overview

This project demonstrates a complete **Computer Vision pipeline**, starting from classical image-processing techniques and progressing to modern deep-learning-based detection.

The notebook covers:

- 🧹 Morphological Operations
- 🔲 Bitwise Operations
- 📊 Image Histograms
- 🙂 YuNet Face Detection
- 🎯 YOLOv8 Object Detection
- 🔗 Integrated Real-Time Detection Pipeline
- ⚡ FPS Benchmarking and Final Comparison

The overall workflow is:

**Image Processing → Masking & Histograms → Face Detection → Object Detection → Integrated Real-Time Pipeline**

---

## 🎯 Objectives

The main objectives of this project are to:

1. Understand how classical morphological operations can clean and modify image regions.
2. Apply bitwise operations for masking and image compositing.
3. Analyse grayscale and colour histograms.
4. Understand brightness and contrast adjustment using `alpha` and `beta`.
5. Detect human faces using the pretrained **YuNet** detector.
6. Detect multiple object classes using pretrained **YOLOv8n**.
7. Compare confidence and IoU threshold behaviour.
8. Build a combined YuNet + YOLO real-time webcam pipeline.
9. Compare the computational performance of face detection, YOLO, and the combined pipeline.

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python 3.x | Programming language |
| OpenCV 5.0.0 | Image processing and face detection |
| NumPy | Numerical and matrix operations |
| Matplotlib | Image and result visualization |
| Ultralytics YOLOv8 | Object detection |
| Jupyter Notebook | Development environment |
| YuNet ONNX model | Face detection |

---

## 📦 Dataset & Model Sources

### Static Images

The project uses OpenCV/sample images and additional images stored inside the `images/` folder.

Examples used in the notebook include:

- `apple.jpg`
- `lena.jpg`
- `coins.png`
- `leaves.jpg`
- `person.jpg`
- `chair.jpg`
- `solid.png`

### YuNet Face Detector

Model:

`face_detection_yunet_2023mar.onnx`

YuNet is loaded using:

```python
cv2.FaceDetectorYN_create()
```

Official source:

- [OpenCV Zoo](https://github.com/opencv/opencv_zoo)

### YOLOv8

Model:

```python
YOLO("yolov8n.pt")
```

`yolov8n.pt` is the lightweight YOLOv8 Nano model pretrained on the **80-class COCO dataset**.

Official source:

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)

### OpenCV Sample Data

- [OpenCV Sample Data](https://github.com/opencv/opencv/tree/master/samples/data)

---

📚 Topics Covered

1️⃣ Morphological Operations

- Erosio
- Dilation
- Opening
- Closing
- Kernel shape and size comparison

2️⃣ Bitwise & Histogram

- AND, OR, XOR, NOT
- Image masking
- Grayscale & RGB histograms
- Brightness and contrast adjustment using Alpha & Beta

3️⃣ YuNet Face Detection

- Face detection using YuNet
- Confidence threshold comparison
- Facial landmarks
- Real-time webcam detection
- Privacy blur

4️⃣ YOLOv8 Object Detection

- YOLOv8n object detection
- Static image detection
- Real-time webcam detection
- Confidence & IoU tuning
- Per-class detection summary

5️⃣ Integrated Pipeline

YuNet + YOLOv8 are combined for real-time face and object detection.

📊 Final Comparison

| Technique | Main Use |
|---|---|
| Morphology | Noise removal & image processing |
| Bitwise / Histogram | Masking & image analysis |
| YuNet | Face detection |
| YOLOv8n | Multi-object detection |

---

# 📊 Project Screenshots

> Add your three meaningful project screenshots in the `screenshots/` folder using the filenames below.

### 🧹 Screenshot 1 — Classical Image Processing

![Classical Image Processing](screenshots/screenshot1.png)

*Morphology, bitwise operations and histogram analysis.*

### 🙂 Screenshot 2 — YuNet Face Detection

![YuNet Face Detection](screenshots/screenshot2.png)

*YuNet face detection with bounding boxes, landmarks and confidence scores.*

### 🎯 Screenshot 3 — YOLO / Integrated Pipeline

![YOLO Integrated Pipeline](screenshots/screenshot3.png)

*YOLO object detection and integrated real-time detection results.*

---

# 📋 Requirements

```text
opencv-python==5.0.0.*
opencv-contrib-python==5.0.0.*
ultralytics
numpy
matplotlib
```

---

# 💡 Reflection

For a real-time monitoring system, a good accuracy-to-compute trade-off is achieved by combining **light classical preprocessing** with **YuNet for faces** and **YOLOv8n for objects**.

For a **low-power edge device**, keeping YOLOv8n and using a moderate YuNet threshold around `0.7` helps maintain real-time performance.

For a **cloud server**, a larger model such as **YOLOv8s** can be considered when higher detection accuracy is more important than maximum FPS.

---

# 🏁 Conclusion

This project demonstrates how classical computer vision and deep learning can work together in a practical real-time pipeline.

**Morphology → Bitwise & Histograms → YuNet → YOLOv8 → Integrated Real-Time Detection**

Classical techniques provide fast and inexpensive image analysis and preprocessing, while YuNet and YOLOv8 provide robust face and multi-object detection. The final integrated pipeline demonstrates a practical foundation for applications such as **smart surveillance, retail analytics, privacy-aware monitoring and real-time computer vision**.

---

## 👩‍💻 Project

Janki Dholariya