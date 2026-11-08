import random
import shutil
from training.utils.common import DATA_AUG, YOLO_DATASET, ensure_dir

random.seed(123)


def main(val_ratio=0.15, test_ratio=0.0):
    ensure_dir(YOLO_DATASET)
    for s in ["train", "val"]:
        ensure_dir(YOLO_DATASET / s / "images")
        ensure_dir(YOLO_DATASET / s / "labels")

    imgs = sorted([p for p in (DATA_AUG / "images").glob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])
    pairs = []
    for img in imgs:
        lbl = DATA_AUG / "labels" / (img.stem + ".txt")
        if lbl.exists():
            pairs.append((img, lbl))
    random.shuffle(pairs)
    n = len(pairs)
    v = int(n * val_ratio)
    val_list = pairs[:v]
    train_list = pairs[v:]

    for s, lst in [("train", train_list), ("val", val_list)]:
        for img, lbl in lst:
            shutil.copy2(img, YOLO_DATASET / s / "images" / img.name)
            shutil.copy2(lbl, YOLO_DATASET / s / "labels" / lbl.name)
    print(f"YOLO dataset ready: train={len(train_list)} val={len(val_list)} -> {YOLO_DATASET}")


if __name__ == "__main__":
    main()
