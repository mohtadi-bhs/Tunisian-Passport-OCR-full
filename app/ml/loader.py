from pathlib import Path
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class ModelLoader:
    def __init__(self):
        self.yolo_sess = None
        self.ocr_sess = None
        self._loaded = False

    def load(self):
        if self._loaded:
            return
        try:
            import onnxruntime as ort  # lazy import
            yolo_path = settings.yolo_model_abs
            ocr_path = settings.ocr_model_abs
            if yolo_path.exists():
                try:
                    self.yolo_sess = ort.InferenceSession(str(yolo_path), providers=["CPUExecutionProvider"])
                    logger.info(f"Loaded YOLO model from {yolo_path}")
                except Exception as e:
                    logger.error(f"Failed to load YOLO model: {e}")
            else:
                logger.warning(f"YOLO model not found at {yolo_path}")
            if ocr_path.exists():
                try:
                    self.ocr_sess = ort.InferenceSession(str(ocr_path), providers=["CPUExecutionProvider"])
                    logger.info(f"Loaded OCR model from {ocr_path}")
                except Exception as e:
                    logger.error(f"Failed to load OCR model: {e}")
            else:
                logger.warning(f"OCR model not found at {ocr_path}")
            self._loaded = True
        except Exception as e:
            logger.error(f"onnxruntime not available or load failed: {e}")
            self._loaded = True

    def get_sessions(self):
        self.load()
        return self.yolo_sess, self.ocr_sess


loader = ModelLoader()
