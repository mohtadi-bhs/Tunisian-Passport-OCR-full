"""Test Surya OCR on Arabic field crops. Test only, not part of the app.

Install the release this script was written for:
    pip install surya-ocr==0.16.7

Usage:
    python surya_test.py debug_out\\crops\\11_nameArabic.png debug_out\\crops\\06_addressArabic.png
"""
import sys
from pathlib import Path

from PIL import Image

from surya.foundation import FoundationPredictor
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    paths = [Path(p) for p in sys.argv[1:]]
    images = [Image.open(p).convert("RGB") for p in paths]

    foundation = FoundationPredictor()
    recognition = RecognitionPredictor(foundation)
    detection = DetectionPredictor()

    results = recognition(images, task_names=["ocr_with_boxes"] * len(images), det_predictor=detection)

    for path, result in zip(paths, results):
        lines = [line.text for line in result.text_lines]
        print(f"{path.name}: {' | '.join(lines)!r}")
        for line in result.text_lines:
            print(f"    {line.text!r}  confidence={line.confidence:.2f}")


if __name__ == "__main__":
    main()