import cv2
from pathlib import Path
from tqdm import tqdm
from training.utils.common import DATA_AUG, OCR_CROPS, ensure_dir
from training.utils.yolo_to_xyxy import yolo_poly_to_bbox, yolo_rect_to_bbox


def main():
    ensure_dir(OCR_CROPS)
    imgs_dir = DATA_AUG / "images"
    lbls_dir = DATA_AUG / "labels"
    count = 0
    for img_path in tqdm(list(imgs_dir.glob("*.jpg")) + list(imgs_dir.glob("*.png")) + list(imgs_dir.glob("*.jpeg"))):
        lbl_path = lbls_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                cls = int(float(parts[0]))
                pts = parts[1:]
                if len(pts) >= 6 and len(pts) % 2 == 0:
                    coords = []
                    for i in range(0, len(pts), 2):
                        coords.append((float(pts[i]), float(pts[i + 1])))
                    x1n, y1n, x2n, y2n = yolo_poly_to_bbox(coords)
                elif len(pts) == 4:
                    cx, cy, bw, bh = float(pts[0]), float(pts[1]), float(pts[2]), float(pts[3])
                    x1n, y1n, x2n, y2n = yolo_rect_to_bbox(cx, cy, bw, bh)
                else:
                    continue
                x1 = max(0, int(x1n * w))
                y1 = max(0, int(y1n * h))
                x2 = min(w, int(x2n * w))
                y2 = min(h, int(y2n * h))
                if x2 <= x1 or y2 <= y1:
                    continue
                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                out = OCR_CROPS / f"{img_path.stem}_c{cls}_{count}.jpg"
                cv2.imwrite(str(out), crop)
                count += 1
    print(f"Saved {count} crops to {OCR_CROPS}")


if __name__ == "__main__":
    main()
