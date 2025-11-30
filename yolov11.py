import os
from ultralytics import YOLO
import json

# -----------------------------
# 1️⃣ Dataset and project paths
# -----------------------------
dataset_path = "Fire Dataset for YOLOv11 divided"
data_yaml = os.path.join(dataset_path, "data.yaml")  # path to your data.yaml
project_name = "fire-detection-3"                    # folder where outputs are saved
experiment_name = "yolo11n-640-fire-only"           # subfolder for this run

os.makedirs(os.path.join(project_name, experiment_name), exist_ok=True)
metrics_file = os.path.join(project_name, experiment_name, "metrics.json")

# -----------------------------
# 2️⃣ Load YOLOv11n model
# -----------------------------
# Make sure yolov11n.pt is downloaded and available in 'model/' folder
# Example: !wget https://github.com/ultralytics/assets/releases/download/v8.1.0/yolo11n.pt -P model/
model = YOLO("model/yolo11n.pt")

# -----------------------------
# 3️⃣ Train the model
# -----------------------------
# Parameters tuned for better accuracy:
# - imgsz=640 → larger image for better feature detection
# - epochs=50 → longer training
# - batch=16 → adjust depending on GPU memory
# - verbose=True → print metrics during training
history = model.train(
    data=data_yaml,
    epochs=100,
    imgsz=640,
    batch=20,
    task='detect',
    project=project_name,
    name=experiment_name,
    verbose=True
)

# -----------------------------
# 4️⃣ Save metrics to a JSON file
# -----------------------------
# history.history contains per-epoch metrics
# Example keys: metrics/precision, metrics/recall, metrics/mAP_50, metrics/mAP_50_95
metrics_data = {}

for key, values in history.history.items():
    metrics_data[key] = values  # all epochs

with open(metrics_file, "w") as f:
    json.dump(metrics_data, f, indent=4)

print(f"All metrics saved to {metrics_file}")

# -----------------------------
# 5️⃣ Export trained model to ONNX format
# -----------------------------
onnx_file = os.path.join(project_name, experiment_name, "yolo11n-fire.onnx")
model.export(format="onnx", weights_only=False)
print(f"Trained YOLOv11n model exported to ONNX: {onnx_file}")

# -----------------------------
# 6️⃣ Print final epoch metrics
# -----------------------------
print("Final epoch metrics:")
for key, values in history.history.items():
    print(f"{key}: {values[-1]}")



