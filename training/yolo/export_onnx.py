from ultralytics import YOLO
from pathlib import Path
import shutil

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
MODELS = ROOT / "models"


def main(weights="runs/passport_yolo/weights/best.pt", imgsz=640):
    MODELS.mkdir(parents=True, exist_ok=True)
    model = YOLO(weights)
    model.export(format="onnx", imgsz=imgsz, simplify=True, opset=12)
    src = Path(weights).with_suffix(".onnx")
    dst = MODELS / "yolo_detector.onnx"
    if src.exists():
        shutil.copy2(src, dst)
        print(f"Exported to {dst}")
    else:
        print("ONNX not found:", src)


if __name__ == "__main__":
    main()
