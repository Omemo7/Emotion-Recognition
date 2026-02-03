# ============================
# Face-crop dataset builder (MediaPipe Tasks FaceDetector)
# Input structure:
#   DATASET_ROOT/
#     train/<class>/*.jpg
#     test/<class>/*.jpg
#
# Output structure:
#   OUTPUT_ROOT/
#     train/<class>/*.jpg
#     test/<class>/*.jpg
#
# Creates cropped-face images using MediaPipe BlazeFace short-range model.
# ============================

import os
import shutil
from pathlib import Path
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ---------- 1) Download the face detector model (if missing) ----------
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
MODEL_PATH = "blaze_face_short_range.tflite"

def ensure_model(model_url=MODEL_URL, model_path=MODEL_PATH):
    if not os.path.exists(model_path):
        print(f"Downloading model -> {model_path}")
        urllib.request.urlretrieve(model_url, model_path)
        print("Download complete.")
    size = os.path.getsize(model_path)
    if size < 50_000:
        raise RuntimeError(f"Model file seems too small ({size} bytes). The download likely failed.")
    return model_path


# ---------- 2) Create detector ONCE ----------
def create_face_detector(model_path, min_conf=0.5):
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceDetectorOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        min_detection_confidence=min_conf
    )
    return vision.FaceDetector.create_from_options(options)


# ---------- 3) Detect + crop helper ----------
def detect_and_crop_face_rgb(
    face_detector,
    img_bgr: np.ndarray,
    expand: float = 0.25,
    choose: str = "largest",   # "largest" or "best"
    target_size=None           # None recommended to avoid double-resize
):
    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    result = face_detector.detect(mp_image)

    if not result.detections:
        return None, None, None

    if choose == "best":
        det = max(result.detections, key=lambda d: d.categories[0].score)
    else:
        det = max(result.detections, key=lambda d: d.bounding_box.width * d.bounding_box.height)

    bbox = det.bounding_box
    score = float(det.categories[0].score)

    x, y = int(bbox.origin_x), int(bbox.origin_y)
    bw, bh = int(bbox.width), int(bbox.height)

    dx, dy = int(bw * expand), int(bh * expand)

    x1 = max(0, x - dx)
    y1 = max(0, y - dy)
    x2 = min(w, x + bw + dx)
    y2 = min(h, y + bh + dy)

    face = img_rgb[y1:y2, x1:x2]
    if face.size == 0:
        return None, None, None

    if target_size is not None:
        tw, th = target_size
        face = cv2.resize(face, (tw, th), interpolation=cv2.INTER_AREA)

    return face, (x1, y1, x2, y2), score


# ---------- 4) Build cropped dataset (NON-SPLIT -> NON-SPLIT) ----------
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def build_cropped_face_dataset_from_single_root(
    dataset_root: str,
    output_root: str,
    min_conf=0.5,
    expand=0.25,
    choose="largest",
    keep_failed=False,         # if True: copy original when no face detected
    resize_to=None,            # keep None; let TF resize once later
    verbose_every=200
):
    """
    Input structure (NOT split):
      dataset_root/
        classA/
        classB/
        ...

    Output structure (NOT split):
      output_root/
        classA/
        classB/
        ...
    """
    dataset_root = Path(dataset_root)
    output_root = Path(output_root)

    model_path = ensure_model()
    face_detector = create_face_detector(model_path, min_conf=min_conf)

    total = 0
    kept = 0
    failed = 0
    skipped = 0

    if not dataset_root.exists():
        raise FileNotFoundError(f"dataset_root not found: {dataset_root}")

    class_dirs = sorted([p for p in dataset_root.iterdir() if p.is_dir()])
    if not class_dirs:
        raise ValueError(f"No class folders found under: {dataset_root}")

    for class_dir in class_dirs:
        out_class = output_root / class_dir.name
        out_class.mkdir(parents=True, exist_ok=True)

        images = [p for p in class_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
        print(f"[{class_dir.name}] images: {len(images)}")

        for i, img_path in enumerate(images, start=1):
            total += 1

            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                failed += 1
                continue

            face_rgb, bbox, score = detect_and_crop_face_rgb(
                face_detector=face_detector,
                img_bgr=img_bgr,
                expand=expand,
                choose=choose,
                target_size=resize_to
            )

            out_path = out_class / img_path.name

            if face_rgb is None:
                failed += 1
                if keep_failed:
                    shutil.copy2(img_path, out_path)
                    kept += 1
                else:
                    skipped += 1
                continue

            face_bgr = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR)
            ok = cv2.imwrite(str(out_path), face_bgr)
            if ok:
                kept += 1
            else:
                failed += 1

            if verbose_every and (i % verbose_every == 0):
                print(f"  ...processed {i}/{len(images)} in {class_dir.name}")

    # Close detector resources
    try:
        face_detector.close()
    except Exception:
        pass

    print("\n===== Done =====")
    print("Input root :", dataset_root)
    print("Output root:", output_root)
    print("Total seen :", total)
    print("Saved      :", kept)
    print("Failed     :", failed)
    print("Skipped    :", skipped, "(when keep_failed=False and no face detected)")
    print("Keep_failed:", keep_failed)
    print("Resize_to  :", resize_to, "(recommend None; resize in TF)")
    return str(output_root)


# ---------- Example usage ----------
DATASET_ROOT = "my clean data"           # contains class folders directly
OUTPUT_ROOT  = "my_clean_cropped_face_dataset"     # will be created

build_cropped_face_dataset_from_single_root(
    dataset_root=DATASET_ROOT,
    output_root=OUTPUT_ROOT,
    min_conf=0.5,
    expand=0.25,
    choose="largest",
    keep_failed=False,
    resize_to=None,
    verbose_every=200
)