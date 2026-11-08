import numpy as np
from typing import List, Tuple
from app.ml.preprocess import Preprocessor
from app.domain.passport.entities import Detection, OCRResult
from app.domain.passport.pipeline import CLASS_TO_FIELD
from app.core.logging import get_logger
from app.core.config import settings

logger = get_logger(__name__)

DET_CONF = 0.05  # the detector's own threshold; weaker boxes are dropped by select_per_class
KEEP_CONF = 0.25  # boxes at or above this are always kept; below it, only the best box per class


def select_per_class(dets: List[Detection], keep_conf: float = KEEP_CONF) -> List[Detection]:
    """Keep every detection at or above keep_conf. For each class with none at that level,
    keep only its single best detection. Weak extra boxes therefore cannot add noisy text."""
    kept = [d for d in dets if d.conf >= keep_conf]
    strong_classes = {d.class_id for d in kept}
    best_weak = {}
    for d in dets:
        if d.conf < keep_conf and d.class_id not in strong_classes:
            if d.class_id not in best_weak or d.conf > best_weak[d.class_id].conf:
                best_weak[d.class_id] = d
    return kept + list(best_weak.values())


class InferenceEngine:
    def __init__(self):
        self.pre = Preprocessor()
        self._yolo = None

    def _load_yolo(self):
        if self._yolo is None:
            from ultralytics import YOLO
            path = settings.yolo_model_abs
            if not path.exists():
                logger.error(f"OBB model not found at {path.resolve()}")
                return None
            # task is set explicitly so the OBB output is never decoded as plain boxes
            self._yolo = YOLO(str(path), task="obb")
            logger.info(f"Loaded OBB model from {path}")
        return self._yolo

    def detect(self, img: np.ndarray) -> List[Detection]:
        try:
            model = self._load_yolo()
            if model is None:
                return []
            results = model(img, verbose=False, conf=0.05)
            dets: List[Detection] = []
            for r in results:
                obb = getattr(r, "obb", None)
                if obb is None or len(obb) == 0:
                    continue
                polys = obb.xyxyxyxy.cpu().numpy()   # (N, 4, 2) oriented corners, pixels
                boxes = obb.xyxy.cpu().numpy()       # (N, 4) axis-aligned envelope, pixels
                clss = obb.cls.cpu().numpy()
                confs = obb.conf.cpu().numpy()
                for poly, box, k, c in zip(polys, boxes, clss, confs):
                    if float(c) < DET_CONF:
                        continue
                    dets.append(Detection(
                        class_id=int(k),
                        bbox=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                        conf=float(c),
                        poly=[(float(x), float(y)) for x, y in poly],
                    ))
            selected = select_per_class(dets)
            logger.debug(f"OBB detections: {len(dets)} found, {len(selected)} kept")
            return selected
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return []

    def recognize(self, crops: List[Tuple[int, np.ndarray, float]]) -> List[OCRResult]:
        """Read each field crop with the OCR settings for its field (see app/ml/ocr.py).
        Each crop comes with its detection confidence, which the pipeline uses to choose between
        boxes of the same field."""
        from app.ml.ocr import read_field
        results: List[OCRResult] = []
        for cid, crop, conf in crops:
            field = CLASS_TO_FIELD.get(cid)
            if field is None or crop is None or crop.size == 0:
                continue
            try:
                text = read_field(crop, field)
            except Exception as e:
                logger.error(f"OCR failed for {field}: {e}")
                continue
            if text:
                results.append(OCRResult(class_id=cid, text=text, conf=conf))
        return results


engine = InferenceEngine()