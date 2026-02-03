import random
from pathlib import Path


def augment_train_folder_by_class(
    train_root: str | Path,
    out_root: str | Path,
    class_targets: dict[str, int],
    seed: int = 42,
    exts=(".jpg", ".jpeg", ".png", ".webp"),
    size=(256, 256),
    copy_originals: bool = True,
):
    """
    - train_root: OUTPUT_ROOT/train
    - out_root: OUTPUT_ROOT/train_balanced_aug
    - class_targets: e.g. {"happy": 230, "sad": 230, ...}
    - Creates for each class: copies originals (optional) + generates enough augmented images to reach target.
    """

    import cv2
    import numpy as np
    import albumentations as A

    train_root = Path(train_root)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    random.seed(seed)
    np.random.seed(seed)

    IMG_EXTS = set(e.lower() for e in exts)

    # ---------- Aug pipelines ----------
    # Minority: stronger but still "emotion-safe" (avoid extreme warps)
    minority_aug = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.18, contrast_limit=0.18, p=0.8),
        A.HueSaturationValue(hue_shift_limit=6, sat_shift_limit=10, val_shift_limit=8, p=0.35),
        A.GaussianBlur(blur_limit=(3, 5), p=0.25),
    ])

    # Majority (happy): light only (optional) – keep distribution stable
    majority_aug = A.Compose([
        A.HorizontalFlip(p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.10, contrast_limit=0.10, p=0.4),
        A.GaussianBlur(blur_limit=(3, 3), p=0.10),
    ])

    def read_rgb(path: Path):
        img_bgr = cv2.imread(str(path))
        if img_bgr is None:
            return None
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        if size is not None:
            img_rgb = cv2.resize(img_rgb, size, interpolation=cv2.INTER_AREA)
        return img_rgb

    def write_rgb(path: Path, img_rgb):
        path.parent.mkdir(parents=True, exist_ok=True)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(path), img_bgr)

    # Determine majority class by max target OR you can explicitly set it:
    majority_class = max(class_targets, key=lambda k: class_targets[k])

    total_generated = 0

    for class_dir in train_root.iterdir():
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name
        if class_name not in class_targets:
            print(f"[SKIP] Class '{class_name}' not in class_targets.")
            continue

        target_n = int(class_targets[class_name])

        imgs = [p for p in class_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
        if not imgs:
            print(f"[WARNING] No images in '{class_name}'.")
            continue

        out_class = out_root / class_name
        out_class.mkdir(parents=True, exist_ok=True)

        # Optionally copy originals into output (so output is a full training folder)
        if copy_originals:
            for p in imgs:
                img_rgb = read_rgb(p)
                if img_rgb is None:
                    continue
                write_rgb(out_class / f"{p.stem}__orig.jpg", img_rgb)

        base_n = len(imgs)
        need = max(0, target_n - base_n)

        # choose which pipeline
        aug_pipe = majority_aug if class_name == majority_class else minority_aug

        if need == 0:
            print(f"[OK] '{class_name}': base={base_n}, target={target_n}, generated=0")
            continue

        # Generate 'need' augmented images by sampling base images with replacement
        for i in range(need):
            src = random.choice(imgs)
            img_rgb = read_rgb(src)
            if img_rgb is None:
                continue

            aug_img = aug_pipe(image=img_rgb)["image"]
            out_path = out_class / f"{src.stem}__aug{i+1}.jpg"
            write_rgb(out_path, aug_img)
            total_generated += 1

        print(f"[AUG] '{class_name}': base={base_n}, target={target_n}, generated={need}")

    print("-" * 60)
    print(f"Done. Total augmented images generated: {total_generated}")
    print(f"Output: {out_root}")




desiredCount=200
    
class_targets = {
        "happy": desiredCount,
        "sad": desiredCount,
        "angry": desiredCount,
        "fear": desiredCount,
        "surprised": desiredCount,
    }

augment_train_folder_by_class(
        train_root="my_clean_cropped_data_splitted_and_resized/train",
        out_root="my_clean_cropped_data_splitted_and_resized/train_balanced_aug",
        class_targets=class_targets,
        seed=42,
        size=(224, 224),
        copy_originals=True,
    )
