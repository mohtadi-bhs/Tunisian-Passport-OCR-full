"""Reversed pipeline test: OCR the whole page first, then assign words to YOLO field boxes.

Run from the project root (the folder that contains app/ and models/):
    python reverse_extract.py --image "C:\\Users\\User\\Downloads\\15f820f1-7e43-4a62-a712-9d113f882499.jpg" --out reverse_out

Writes to the --out folder:
    result.json   the JSON response (fields, status, errors), built with the same merge rules as the server
    words.json    every word found on the page, with its position, language and confidence
    annotated.jpg the page with each word box and the field it was assigned to
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pytesseract

from app.ml.inference import engine
from app.ml.preprocess import Preprocessor
from app.ml.ocr import _setup_tesseract, ARABIC_FIELDS
from app.domain.passport.pipeline import CLASS_TO_FIELD, PassportPipeline
from app.domain.passport.entities import OCRResult


def page_words(img, lang):
    """All words on the page with their boxes, from one Tesseract pass in the given language."""
    _setup_tesseract()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    data = pytesseract.image_to_data(gray, lang=lang, config="--psm 11",
                                     output_type=pytesseract.Output.DICT)
    words = []
    for i, text in enumerate(data["text"]):
        text = text.strip()
        if not text:
            continue
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        words.append({"text": text, "box": (x, y, x + w, y + h),
                      "centre": (x + w / 2, y + h / 2),
                      "lang": lang, "conf": float(data["conf"][i])})
    return words


def assign(words, dets):
    """Give each word to the detection whose oriented box contains its centre.

    A word is assigned to at most one field: the detection with the highest confidence that
    contains its centre. Words outside every box are kept in words.json but not used.
    """
    per_det = {i: [] for i in range(len(dets))}
    for w in words:
        best, best_conf = None, -1.0
        for i, d in enumerate(dets):
            poly = np.array(d.poly, dtype=np.float32) if d.poly is not None else None
            if poly is None:
                x1, y1, x2, y2 = d.bbox
                inside = x1 <= w["centre"][0] <= x2 and y1 <= w["centre"][1] <= y2
            else:
                inside = cv2.pointPolygonTest(poly.reshape(-1, 1, 2), w["centre"], False) >= 0
            if inside and d.conf > best_conf:
                best, best_conf = i, d.conf
        w["assigned_to"] = best
        if best is not None:
            per_det[best].append(w)
    return per_det


def field_text(words, field):
    """Join the words of one field in reading order. Arabic runs right to left."""
    if field in ARABIC_FIELDS:
        ordered = sorted(words, key=lambda w: (round(w["centre"][1] / 20), -w["centre"][0]))
    else:
        ordered = sorted(words, key=lambda w: (round(w["centre"][1] / 20), w["centre"][0]))
    return " ".join(w["text"] for w in ordered)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=Path("reverse_out"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    pre = Preprocessor()
    img = pre.read_image(args.image.read_bytes())
    dets = engine.detect(img)
    print(f"YOLO detections: {len(dets)}")

    words = page_words(img, "eng") + page_words(img, "ara")
    print(f"words found on page: {len(words)}")

    per_det = assign(words, dets)

    ocr_results = []
    report = []
    for i, d in enumerate(dets):
        field = CLASS_TO_FIELD.get(d.class_id, f"class{d.class_id}")
        text = field_text(per_det[i], field)
        ocr_results.append(OCRResult(class_id=d.class_id, text=text, conf=d.conf))
        report.append({"field": field, "conf": round(d.conf, 3), "text": text,
                       "words": [w["text"] for w in per_det[i]]})
        print(f"{field:<20} conf={d.conf:.2f} '{text}'")

    merged = PassportPipeline().merge(dets, ocr_results, mrz=None)
    result = {"status": merged.status, "data": merged.fields, "errors": merged.errors,
              "fields": report}
    (args.out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out / "words.json").write_text(json.dumps(
        [{k: v for k, v in w.items() if k != "centre"} for w in words],
        ensure_ascii=False, indent=2), encoding="utf-8")

    annotated = img.copy()
    for w in words:
        x1, y1, x2, y2 = w["box"]
        colour = (0, 255, 0) if w["assigned_to"] is not None else (0, 0, 255)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 1)
    cv2.imwrite(str(args.out / "annotated.jpg"), annotated)

    print(f"status: {merged.status}")
    print(json.dumps(merged.fields, ensure_ascii=False, indent=2))
    print(f"saved to {args.out.resolve()}: result.json, words.json, annotated.jpg")


if __name__ == "__main__":
    main()