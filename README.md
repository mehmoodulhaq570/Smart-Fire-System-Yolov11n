# YOLO V11n - Fire Detection

This repository contains the implementation and resources for the `YOLO V11n` object detection model, specifically trained for fire detection. Built for efficient and accurate real-time object detection, this project provides a robust solution for various computer vision tasks.

---

## 🚀 Features

* **High-performance object detection**: Leveraging the latest advancements in the YOLO series for fast and accurate detections.
* **Customizable training**: Easily train your own `YOLO V11n` model on custom datasets.
* **Real-time inference**: Optimized for real-time applications on various hardware.
* **Easy to use**: Straightforward setup for both training and inference.

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/mehmoodulhaq570/Yolov11n
cd Yolov11n
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows
```

### 3. Install dependencies

```bash
pip install torch torchvision ultralytics numpy opencv-python pillow
# or if a requirements.txt is available:
# pip install -r requirements.txt
```

---

## 📥 Dataset Setup

A custom dataset was curated to enable robust fire and smoke detection in real-world conditions.

### Dataset Details

* **Dataset size**: 32,603 labeled images
* **Classes**: Fire (class 0) and Smoke (class 1)
* **Composition**: 9,940 images contain smoke (some also contain both fire and smoke)
* **Split**: 26,379 training images, 4,394 validation images
* **Annotations**: Bounding boxes in YOLO format
* **Scenario coverage**: Includes lighter flames, reflections, varying illumination, and different smoke densities

> **Note**: The current model is trained for **fire detection only**.

### Download from Kaggle

1. Get Kaggle API credentials and save your `kaggle.json` in `~/.kaggle/`.
2. Download the dataset:

```bash
kaggle datasets download -d mehmoodulhaq570/fire-dataset
```

3. Unzip the dataset:

```bash
unzip -r fire-dataset.zip
```

### Example `data.yaml`

```yaml
path: ../datasets/fire-dataset  # dataset root dir
train: images/train             # train images (relative to 'path')
val: images/val                 # val images (relative to 'path')

nc: 1                           # number of classes (fire only)
names: ['fire']                 # class names
```

---

## 🏋️‍♂️ Trained Models

Models were trained using **Lightning AI** on **T4 GPU instances**:

| Model Name       | Epochs | Notes               |
| ---------------- | ------ | ------------------- |
| fire-detection-1 | 15     | Results and best.pt |
| fire-detection-2 | 50     |                     |
| fire-detection-3 | 100    |                     |
| fire-detection-4 | 200    |                     |

---

## 🏃‍♀️ Usage

### Inference

Run inference on images, videos, or webcam:

```bash
# Image
python yolov11.py --weights weights/fire-detection-4/best.pt --source path/to/image.jpg --conf 0.25 --imgsz 640

# Video
python yolov11.py --weights weights/fire-detection-4/best.pt --source path/to/video.mp4 --conf 0.25 --imgsz 640

# Webcam
python yolov11.py --weights weights/fire-detection-4/best.pt --source 0
```

### Training

Train on your own dataset:

```bash
python yolov11.py --imgsz 640 --batch 16 --epochs 100 --data data.yaml --weights weights/yolov11n.pt --name yolov11n_custom
```

> Replace `yolov11.py` with `yolov11-2.py` if you want to use a different configuration.

---

## 🖼️ Examples

![example\_detection\_1](assets/example1.jpg)
*Fire detection in low-light conditions.*

![example\_detection\_2](assets/example2.mp4)
*Smoke and fire detection in real-world scenarios.*

---

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## ⚡ Quickstart Commands

```bash
# Clone repo
git clone https://github.com/mehmoodulhaq570/Yolov11n
cd Yolov11n

# Setup environment
python -m venv venv
source venv/bin/activate
pip install torch torchvision ultralytics numpy opencv-python pillow

# Download dataset
kaggle datasets download -d mehmoodulhaq570/fire-dataset
unzip -r fire-dataset.zip

# Train or run inference
python yolov11.py --weights weights/fire-detection-4/best.pt --source path/to/image.jpg --conf 0.25 --imgsz 640
```
