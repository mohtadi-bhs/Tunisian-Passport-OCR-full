from ultralytics import YOLO
from pathlib import Path


HERE = Path(__file__).parent
DATA_YAML = HERE / "data.yaml"
RUNS_DIR = HERE / "runs"


def main(
    weights: str = "yolov8n.pt",
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 8,
    device: str = "",
    patience: int = 50,
    name: str = "passport_yolo",
    seed: int = 0,
    deterministic: bool = True,
    close_mosaic: int = 10,
):
    model = YOLO(weights)
    model.train(
        data=str(DATA_YAML),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(RUNS_DIR),
        name=name,
        patience=patience,
        seed=seed,
        deterministic=deterministic,
        close_mosaic=close_mosaic,
        save=True,
        plots=True,
    )
    print(f"Training finished. Check {RUNS_DIR / name / 'weights'}/")


if __name__ == "__main__":
    main()
