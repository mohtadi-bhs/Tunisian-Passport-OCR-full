from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class Detection:
    class_id: int
    bbox: Tuple[float, float, float, float]  # axis-aligned xyxy in original image pixels
    conf: float = 0.0
    # Four corners of the oriented box in original image pixels, [(x, y) x 4].
    # Set by the OBB model; None for axis-aligned detections.
    poly: Optional[List[Tuple[float, float]]] = None


@dataclass
class OCRResult:
    class_id: int
    text: str
    conf: float = 0.0


@dataclass
class MergedPassport:
    fields: dict
    status: str  # ok|partial
    errors: List[str]