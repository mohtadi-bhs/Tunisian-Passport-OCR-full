import json
import torch.nn as nn
from pathlib import Path
from training.ocr.model_crnn import CRNN
from training.utils.common import OCR_CROPS, ensure_dir


def main():
    ensure_dir(OCR_CROPS)
    lbl_file = OCR_CROPS / "crops_labels.json"
    if not lbl_file.exists():
        print("Missing crops_labels.json. Create it mapping crop filename -> ground truth text.")
        return
    with open(lbl_file, encoding="utf-8") as f:
        mapping = json.load(f)
    chars = set()
    for v in mapping.values():
        for c in str(v):
            chars.add(c)
    chars = sorted(list(chars))
    char_map = {c: i + 1 for i, c in enumerate(chars)}
    char_map["<blank>"] = 0
    nclass = len(char_map)
    model = CRNN(nclass=nclass)
    print("Model created. Add your DataLoader here with CTCLoss.")


if __name__ == "__main__":
    main()
