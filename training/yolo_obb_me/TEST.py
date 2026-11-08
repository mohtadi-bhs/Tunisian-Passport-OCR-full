"""Run the trained YOLO11-OBB model on one or more images.

Usage:
    python test_obb_images.py --weights best.pt --images img1.jpg img2.jpg --conf 0.25 --out results

Outputs in --out:
    <name>_annotated.jpg   image with oriented boxes, class and confidence drawn on it
    detections.csv         one row per detection: image, class, confidence, 4 corner points
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, type=Path)
    ap.add_argument("--images", required=True, nargs="+", type=Path)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.weights))
    rows = []

    for img_path in args.images:
        if not img_path.exists():
            print(f"missing: {img_path}")
            continue
        result = model.predict(str(img_path), imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
        img = cv2.imread(str(img_path))
        names = result.names

        if result.obb is None or len(result.obb) == 0:
            print(f"{img_path.name}: no detections above conf={args.conf}")
        else:
            polys = result.obb.xyxyxyxy.cpu().numpy()  # (N, 4, 2) corner points in pixels
            clss = result.obb.cls.cpu().numpy().astype(int)
            confs = result.obb.conf.cpu().numpy()
            print(f"{img_path.name}: {len(clss)} detections")
            for poly, c, p in zip(polys, clss, confs):
                pts = poly.astype(int).reshape(-1, 1, 2)
                cv2.polylines(img, [pts], True, (0, 255, 0), 2)
                label = f"{names[c]} {p:.2f}"
                x, y = pts[0, 0]
                cv2.putText(img, label, (int(x), max(int(y) - 5, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                print(f"   class={names[c]:<6} conf={p:.3f}")
                rows.append([img_path.name, names[c], round(float(p), 4), *np.round(poly, 1).flatten().tolist()])

        out_img = args.out / f"{img_path.stem}_annotated.jpg"
        cv2.imwrite(str(out_img), img)

    csv_path = args.out / "detections.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "class", "confidence", "x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"])
        w.writerows(rows)
    print(f"saved annotated images and {csv_path}")


if __name__ == "__main__":
    main()