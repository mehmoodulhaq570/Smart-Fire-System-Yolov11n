import os
from ultralytics import YOLO
import json

# -----------------------------
# 1️⃣ Dataset and project paths
# -----------------------------
dataset_path = "Fire Dataset for YOLOv11 divided"
data_yaml = os.path.join(dataset_path, "data.yaml")  # path to your data.yaml
project_name = "fire-detection-4"
experiment_name = "yolo11n-896-fire-optimized"

os.makedirs(os.path.join(project_name, experiment_name), exist_ok=True)
metrics_file = os.path.join(project_name, experiment_name, "metrics.json")

# -----------------------------
# 2️⃣ Load YOLOv11n model
# -----------------------------
model = YOLO("model/yolo11n.pt")

# -----------------------------
# 3️⃣ Training parameters
# -----------------------------
history = model.train(
    data=data_yaml,
    epochs=200,                 # longer training
    imgsz=896,                  # larger image size
    batch=32,                   # larger batch
    task='detect',
    project=project_name,
    name=experiment_name,
    verbose=True,
    lr0=0.005,                  # reduced initial learning rate
    save=True,                  # save checkpoints
    save_period=1,              # save every epoch
    patience=20                 # stop if no improvement in 30 epochs
)

# -----------------------------
# 4️⃣ Save metrics to JSON
# -----------------------------
metrics_data = {k: v for k, v in history.history.items()}
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
