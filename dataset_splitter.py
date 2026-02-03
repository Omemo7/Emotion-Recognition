import random
from pathlib import Path
from PIL import Image

# ----------------------------
# 1) Resize + save helper
# ----------------------------
def save_resized_image(src: Path, dst: Path, size=(256, 256)):
    try:
        with Image.open(src) as img:
            img = img.convert("RGB")  # ensure 3 channels
            img = img.resize(size, Image.LANCZOS)
            dst.parent.mkdir(parents=True, exist_ok=True)
            img.save(dst)
    except Exception as e:
        print(f"[ERROR] Failed processing {src}: {e}")


# ----------------------------
# 2) Train/Val/Test split
#    val_ratio is taken from the FULL dataset (like test_ratio)
# ----------------------------
def stratified_train_val_test_split(
    source_root: str | Path,
    output_root: str | Path,
    test_ratio: float = 0.15,
    val_ratio: float = 0.15,
    seed: int = 42,
    move_files: bool = False,
    size=(256, 256),
):
    source_root = Path(source_root)
    output_root = Path(output_root)

    train_root = output_root / "train"
    val_root = output_root / "val"
    test_root = output_root / "test"

    for d in (train_root, val_root, test_root):
        d.mkdir(parents=True, exist_ok=True)

    random.seed(seed)
    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}

    total_train = total_val = total_test = 0

    for class_dir in source_root.iterdir():
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name
        images = [
            p for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMG_EXTS
        ]

        if not images:
            print(f"[WARNING] No images found in class '{class_name}', skipping.")
            continue

        random.shuffle(images)
        n_total = len(images)

        n_test = int(round(n_total * test_ratio))
        n_val = int(round(n_total * val_ratio))

        # Keep at least 1 in each split when possible (for small classes)
        if n_total >= 3:
            n_test = max(1, n_test)
            n_val = max(1, n_val)
        elif n_total == 2:
            # Can't have train+val+test; pick test=1, train=1, val=0
            n_test, n_val = 1, 0
        else:  # n_total == 1
            n_test, n_val = 0, 0

        # Ensure we don't exceed total
        if n_test + n_val >= n_total:
            # Reduce val first, then test, leaving at least 1 for train if possible
            overflow = (n_test + n_val) - (n_total - 1) if n_total > 1 else (n_test + n_val) - n_total
            while overflow > 0 and n_val > 0:
                n_val -= 1
                overflow -= 1
            while overflow > 0 and n_test > 0:
                n_test -= 1
                overflow -= 1

        test_images = images[:n_test]
        val_images = images[n_test:n_test + n_val]
        train_images = images[n_test + n_val:]

        class_train_dir = train_root / class_name
        class_val_dir = val_root / class_name
        class_test_dir = test_root / class_name
        for d in (class_train_dir, class_val_dir, class_test_dir):
            d.mkdir(parents=True, exist_ok=True)

        # Save resized copies
        for img_path in test_images:
            save_resized_image(img_path, class_test_dir / img_path.name, size=size)
            if move_files:
                img_path.unlink()

        for img_path in val_images:
            save_resized_image(img_path, class_val_dir / img_path.name, size=size)
            if move_files:
                img_path.unlink()

        for img_path in train_images:
            save_resized_image(img_path, class_train_dir / img_path.name, size=size)
            if move_files:
                img_path.unlink()

        total_train += len(train_images)
        total_val += len(val_images)
        total_test += len(test_images)

        print(
            f"Class '{class_name}': total={n_total}, "
            f"train={len(train_images)}, val={len(val_images)}, test={len(test_images)}"
        )

    total = total_train + total_val + total_test
    print("-" * 50)
    print(f"Done. Total train images: {total_train}")
    print(f"Done. Total val images:   {total_val}")
    print(f"Done. Total test images:  {total_test}")
    if total > 0:
        print(f"Train ratio: ~{total_train / total:.3f}")
        print(f"Val   ratio: ~{total_val / total:.3f}")
        print(f"Test  ratio: ~{total_test / total:.3f}")








if __name__ == "__main__":
    # Example:
    # Happy is majority (230). Make everything ~230.
    BASE_DIR = Path(__file__).resolve().parent

    SOURCE_ROOT = BASE_DIR / "my_clean_cropped_face_dataset"
    OUTPUT_ROOT = BASE_DIR / "my_clean_cropped_data_splitted_and_resized"

    stratified_train_val_test_split(
        source_root=SOURCE_ROOT,
        output_root=OUTPUT_ROOT,
        test_ratio=0.15,
        val_ratio=0.15,
        seed=42,
        move_files=False,
        size=(224, 224),
    )


