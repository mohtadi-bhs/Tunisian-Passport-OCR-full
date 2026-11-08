import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_AUG = ROOT / "Data_aug"
TRAINING = ROOT / "training"
YOLO_DATASET = TRAINING / "yolo" / "dataset"
OCR_CROPS = TRAINING / "ocr" / "crops"
MODELS_OUT = ROOT / "models"


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_json(p: Path):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def save_json(p: Path, obj):
    ensure_dir(p.parent)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
