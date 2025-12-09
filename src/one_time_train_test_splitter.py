import os
import random
import shutil
from pathlib import Path

def stratified_train_test_split(
    source_root: str,
    output_root: str,
    test_ratio: float = 0.15,
    seed: int = 42,
    move_files: bool = False,
):
    source_root = Path(source_root)
    output_root = Path(output_root)

    train_root = output_root / "train"
    test_root = output_root / "test"

    train_root.mkdir(parents=True, exist_ok=True)
    test_root.mkdir(parents=True, exist_ok=True)

    random.seed(seed)

    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}

    total_train = 0
    total_test = 0

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
        if n_test == 0 and n_total > 1:
            n_test = 1

        test_images = images[:n_test]
        train_images = images[n_test:]

        class_train_dir = train_root / class_name
        class_test_dir = test_root / class_name
        class_train_dir.mkdir(parents=True, exist_ok=True)
        class_test_dir.mkdir(parents=True, exist_ok=True)

        op = shutil.move if move_files else shutil.copy2

        for img_path in test_images:
            dest_path = class_test_dir / img_path.name
            op(str(img_path), str(dest_path))

        for img_path in train_images:
            dest_path = class_train_dir / img_path.name
            op(str(img_path), str(dest_path))

        total_train += len(train_images)
        total_test += len(test_images)

        print(
            f"Class '{class_name}': total={n_total}, "
            f"train={len(train_images)}, test={len(test_images)}"
        )

    print("-" * 50)
    print(f"Done. Total train images: {total_train}")
    print(f"Total test images:  {total_test}")
    print(f"Train ratio: ~{total_train / (total_train + total_test):.3f}")
    print(f"Test  ratio: ~{total_test / (total_train + total_test):.3f}")


if __name__ == "__main__":
    # Folder where THIS file lives
    BASE_DIR = Path(__file__).resolve().parent

    # Put your original images under:  <same folder as script>/dataset
    SOURCE_ROOT = "dataset"

    # Split will be created as: <same folder as script>/dataset_split/train,test
    OUTPUT_ROOT = "dataset_split"

    stratified_train_test_split(
        source_root=SOURCE_ROOT,
        output_root=OUTPUT_ROOT,
        test_ratio=0.15,
        seed=42,
        move_files=False,
    )
