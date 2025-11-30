import os
import cv2
import numpy as np

def detect_possible_fire(img):
    """
    Very simple heuristic to detect potential fire regions.
    YOLO labels should exist if fire pixels are detected.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Fire-like color range (orange/red)
    lower_fire = np.array([0, 80, 80])
    upper_fire = np.array([35, 255, 255])

    mask = cv2.inRange(hsv, lower_fire, upper_fire)
    fire_pixels = cv2.countNonZero(mask)

    return fire_pixels > 500  # threshold of fire presence


def validate_yolo_fire_dataset(dataset_path, num_classes=1):
    images_dir = os.path.join(dataset_path, "images")
    labels_dir = os.path.join(dataset_path, "labels")

    issues = []

    for root, dirs, files in os.walk(images_dir):
        for img_file in files:
            if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img_path = os.path.join(root, img_file)
            label_path = os.path.join(labels_dir, os.path.splitext(img_file)[0] + ".txt")

            # -----------------------------
            # 1️⃣ Check corrupted image
            # -----------------------------
            img = cv2.imread(img_path)
            if img is None:
                issues.append(f"[CORRUPT IMAGE] {img_path}")
                continue

            # -----------------------------
            # 2️⃣ Fire check (optional)
            # -----------------------------
            fire_present = detect_possible_fire(img)

            # -----------------------------
            # 3️⃣ Label file check
            # -----------------------------
            if not os.path.exists(label_path):
                if fire_present:
                    issues.append(f"[MISSING LABEL BUT FIRE VISIBLE] {img_path}")
                continue

            with open(label_path, "r") as f:
                lines = [l.strip() for l in f.readlines()]

            # -----------------------------
            # 4️⃣ Allow empty labels IF no fire
            # -----------------------------
            if len(lines) == 0:
                if fire_present:
                    issues.append(f"[NO LABELS BUT FIRE VISIBLE] {label_path}")
                continue

            # -----------------------------
            # 5️⃣ Validate YOLO annotations
            # -----------------------------
            for line in lines:
                parts = line.split()

                if len(parts) != 5:
                    issues.append(f"[BAD FORMAT] {label_path} → '{line}'")
                    continue

                cls, x, y, w, h = parts

                if not cls.isdigit():
                    issues.append(f"[INVALID CLASS] {label_path} → '{line}'")
                    continue

                cls = int(cls)

                if cls >= num_classes:
                    issues.append(f"[CLASS OUT OF RANGE] {label_path} → '{line}'")

                try:
                    x = float(x); y = float(y)
                    w = float(w); h = float(h)
                    if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                        issues.append(f"[INVALID BBOX VALUES] {label_path} → '{line}'")
                except:
                    issues.append(f"[BAD NUMBERS] {label_path} → '{line}'")

    # -----------------------------
    # 6️⃣ Final output
    # -----------------------------
    if len(issues) == 0:
        print("✅ Dataset is clean! No issues found.")
    else:
        print("⚠️ Issues found:")
        for issue in issues:
            print(issue)

    return issues


# -----------------------
# Run the checker
# -----------------------
dataset_path = "Fire Dataset for YOLOv11 divided"
validate_yolo_fire_dataset(dataset_path, num_classes=1)
