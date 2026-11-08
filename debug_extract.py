"""Debug the passport pipeline on ONE image with the SAME code the server uses.

Run from the project root (the folder that contains app/ and models/):
    python debug_extract.py --image "C:\\Users\\User\\Downloads\\15f820f1-7e43-4a62-a712-9d113f882499.jpg"

Compare the crop margins by running twice with different output folders:
    python debug_extract.py --image "..." --out debug_nomargin --margin-x 0 --margin-y 0
    python debug_extract.py --image "..." --out debug_margin  --margin-x 0.25 --margin-y 0.05

Writes to the --out folder (default debug_out):
    annotated.jpg       the image with every detected oriented box, field name and confidence
    crops/NN_field.png  each straightened crop, numbered in detection order
    mrz_band.png        the bottom of the page that is searched for the MRZ
    report.csv          one row per detection: field, confidence, crop size, OCR text, error
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from app.ml.inference import engine
from app.ml.preprocess import Preprocessor
from app.ml.ocr import read_field
from app.domain.passport.pipeline import CLASS_TO_FIELD
from app.domain.passport import mrz as mrzmod


def show_mrz(img, out: Path):
    """Run the MRZ reader and print what it found. Saves the band it searched."""
    h = img.shape[0]
    band = img[int(h * mrzmod.MRZ_BAND_FROM):, :]
    cv2.imwrite(str(out / "mrz_band.png"), band)
    try:
        result = mrzmod.read_mrz_from_image(img)
    except Exception as e:
        print("MRZ: reader error:", str(e).splitlines()[0][:200])
        return
    if result is None:
        print("MRZ: not found (no two MRZ-like lines in the bottom of the page)")
        return
    print("MRZ OCR text:")
    for line in result["ocr_text"].splitlines():
        if line.strip():
            print("   ", line)
    print("MRZ valid:", result["valid"], "| corrected:", result["corrected"])
    print("MRZ checks:", result["checks"])
    print("MRZ fields:", result["fields"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=Path("debug_out"))
    ap.add_argument("--margin-x", type=float, default=0.25,
                    help="horizontal crop margin as a fraction of box height (0 = none)")
    ap.add_argument("--margin-y", type=float, default=0.05,
                    help="vertical crop margin as a fraction of box height (0 = none)")
    args = ap.parse_args()

    out = args.out
    (out / "crops").mkdir(parents=True, exist_ok=True)
    print(f"crop margins: x={args.margin_x} y={args.margin_y}")

    pre = Preprocessor()
    img = pre.read_image(args.image.read_bytes())
    print("image size (w x h):", img.shape[1], img.shape[0])

    dets = engine.detect(img)
    print(f"detections: {len(dets)}")

    annotated = img.copy()
    rows = []
    for i, d in enumerate(dets, 1):
        field = CLASS_TO_FIELD.get(d.class_id, f"class{d.class_id}")
        if d.poly is not None:
            pts = np.array(d.poly, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(annotated, [pts], True, (0, 255, 0), 2)
            x, y = pts[0, 0]
            cv2.putText(annotated, f"{i}:{field} {d.conf:.2f}", (int(x), max(int(y) - 5, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            crop = pre.warp_crop(img, d.poly, margin_x=args.margin_x, margin_y=args.margin_y)
        else:
            crop = pre.crop(img, d.bbox)

        fname = f"{i:02d}_{field}.png"
        cv2.imwrite(str(out / "crops" / fname), crop)

        text, error = "", ""
        try:
            text = read_field(crop, field)
        except Exception as e:
            error = str(e).splitlines()[0][:200]

        rows.append({
            "n": i,
            "field": field,
            "conf": round(d.conf, 3),
            "bbox": " ".join(f"{v:.0f}" for v in d.bbox),
            "crop_size": f"{crop.shape[1]}x{crop.shape[0]}",
            "ocr_text": text,
            "error": error,
            "crop_file": fname,
        })
        status = error if error else f"'{text}'"
        print(f"{i:02d} {field:<20} conf={d.conf:.2f} crop={crop.shape[1]}x{crop.shape[0]} {status}")

    cv2.imwrite(str(out / "annotated.jpg"), annotated)
    with open(out / "report.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["n"])
        w.writeheader()
        w.writerows(rows)

    print("--- MRZ ---")
    show_mrz(img, out)
    print(f"saved to {out.resolve()}: annotated.jpg, crops/, mrz_band.png, report.csv")


if __name__ == "__main__":
    main()