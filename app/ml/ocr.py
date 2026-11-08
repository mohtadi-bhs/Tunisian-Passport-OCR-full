import os
import re
import numpy as np
import cv2
from app.core.logging import get_logger

logger = get_logger(__name__)

TESSERACT_CMD_DEFAULT = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Arabic fields are read with the Arabic model; everything else with English.
ARABIC_FIELDS = {"nameArabic", "placeOfBirthArabic", "addressArabic", "jobArabic"}

# Short Arabic fields. Tesseract reads them from the black-and-white crop, which is what worked
# on this photo (Surya returned garbage on these binary crops).
ARABIC_BINARY = {"addressArabic", "jobArabic"}

# Names are read by Surya from the original colours at their native size. Scaling the name
# crop up stopped Surya from finding the text.
ARABIC_SURYA = {"nameArabic", "placeOfBirthArabic"}

# Latin fields Surya reads better than Tesseract on this photo (for example the date of issue,
# which Tesseract reads as 45-05-2025 and Surya reads as 15-05-2025).
LATIN_SURYA = {"dateOfIssue"}

UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DIGITS = "0123456789"

# Character whitelist passed to Tesseract. Only fields WITHOUT a space are passed this way,
# because a space inside the config string breaks the command line on some pytesseract versions.
FIELD_WHITELIST = {
    "dateOfBirth": DIGITS + "-",
    "dateOfIssue": DIGITS + "-",
    "dateOfExpiry": DIGITS + "-",
    "passportNumber": UPPER + DIGITS,
    "identityNumber": DIGITS,
    "sex": "MF",
    "nationality": UPPER,
}

# Name-like fields may contain spaces and hyphens. They are filtered in Python after OCR instead.
NAME_FIELDS = {"surname", "givenNames", "placeOfBirth", "issuingPlace"}
_NAME_CLEAN = re.compile(r"[^A-Z \-]")

# Page segmentation mode per field. Default is 7 (one line); a single character needs 10.
FIELD_PSM = {
    "sex": "10",
}

# Text height the enhanced crop is scaled to, per script. Arabic is small and faint on this
# passport, so it gets more pixels and a binary threshold.
TARGET_HEIGHT = {"arabic": 120, "latin": 64}

_tesseract_ready = False


def _setup_tesseract():
    global _tesseract_ready
    if _tesseract_ready:
        return
    import pytesseract
    cmd = os.environ.get("TESSERACT_CMD") or TESSERACT_CMD_DEFAULT
    if os.path.exists(cmd):
        pytesseract.pytesseract.tesseract_cmd = cmd
    _tesseract_ready = True


def enhance(crop_bgr: np.ndarray, field: str) -> np.ndarray:
    """Enhance one field crop before OCR. Every crop passes through here.

    1. grayscale
    2. scale so the text is a fixed height (Arabic gets more pixels)
    3. contrast: CLAHE for local contrast; stronger for Arabic
    4. Arabic only: Otsu threshold to black text on white
    5. white border (OCR engines need a margin around the text)
    """
    arabic = field in ARABIC_FIELDS
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY) if crop_bgr.ndim == 3 else crop_bgr

    target = TARGET_HEIGHT["arabic" if arabic else "latin"]
    h = gray.shape[0]
    if h != target:
        s = target / max(h, 1)
        gray = cv2.resize(gray, None, fx=s, fy=s,
                          interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)

    clip = 3.0 if arabic else 2.0
    gray = cv2.createCLAHE(clipLimit=clip, tileGridSize=(4, 4) if arabic else (8, 8)).apply(gray)

    if field in ARABIC_BINARY:
        _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    border = 16 if arabic else 12  # Latin keeps the border that the Latin fields were tuned with
    return cv2.copyMakeBorder(gray, border, border, border, border, cv2.BORDER_CONSTANT, value=255)


_surya = None  # (recognition, detection) predictors, loaded once on first use
_surya_failed = False


def _load_surya():
    """Load Surya once. Returns None if it is not installed or fails to load."""
    global _surya, _surya_failed
    if _surya is not None or _surya_failed:
        return _surya
    try:
        from surya.foundation import FoundationPredictor
        from surya.recognition import RecognitionPredictor
        from surya.detection import DetectionPredictor
        foundation = FoundationPredictor()
        _surya = (RecognitionPredictor(foundation), DetectionPredictor())
    except Exception as e:
        _surya_failed = True
        logger.error(f"Surya could not be loaded, Arabic will use Tesseract: {e}")
    return _surya


def surya_input(crop_bgr: np.ndarray, field: str) -> np.ndarray:
    """The image Surya reads: the original colours at native size, with no border.

    A white border stopped Surya's detector from finding the name on this photo, while the
    crop without a border is read correctly (confidence 0.93). So Surya gets the crop as it is.
    """
    return crop_bgr


def read_arabic_surya(crop_bgr: np.ndarray, field: str) -> str:
    """Read a field with Surya. Empty string if nothing is found."""
    from PIL import Image
    predictors = _load_surya()
    if predictors is None:
        return ""
    recognition, detection = predictors
    image = Image.fromarray(cv2.cvtColor(surya_input(crop_bgr, field), cv2.COLOR_BGR2RGB))
    results = recognition([image], task_names=["ocr_with_boxes"], det_predictor=detection)
    lines = [line.text.strip() for line in results[0].text_lines if line.text and line.text.strip()]
    return " ".join(lines)


def read_arabic_tesseract(crop_bgr: np.ndarray, field: str) -> str:
    import pytesseract
    _setup_tesseract()
    img = enhance(crop_bgr, field)
    text = pytesseract.image_to_string(img, lang="ara", config="--psm 7")
    return " ".join(str(text).split())


def read_latin_tesseract(crop_bgr: np.ndarray, field: str) -> str:
    import pytesseract
    _setup_tesseract()
    config = f"--psm {FIELD_PSM.get(field, '7')}"  # 7 = one text line, 10 = single character
    if field in FIELD_WHITELIST:
        config += f" -c tessedit_char_whitelist={FIELD_WHITELIST[field]}"
    img = enhance(crop_bgr, field)
    text = pytesseract.image_to_string(img, lang="eng", config=config)
    text = " ".join(str(text).split())
    if field in NAME_FIELDS:
        text = " ".join(_NAME_CLEAN.sub("", text).split())
    return text


def read_field(crop_bgr: np.ndarray, field: str) -> str:
    """Return the text of one field crop, or an empty string if nothing readable was found.

    Address and job: Tesseract on the black-and-white crop.
    Names: Surya on the original colours, Tesseract if Surya is unavailable or finds nothing.
    Date of issue: Surya on the original colours, Tesseract if Surya finds nothing.
    Other Latin fields: Tesseract with the per-field settings.
    """
    if field in ARABIC_BINARY:
        return read_arabic_tesseract(crop_bgr, field)
    if field in ARABIC_SURYA:
        try:
            text = read_arabic_surya(crop_bgr, field)
        except Exception as e:
            logger.error(f"Surya failed for {field}: {e}")
            text = ""
        if not text:
            text = read_arabic_tesseract(crop_bgr, field)
        return text
    if field in LATIN_SURYA:
        try:
            text = read_arabic_surya(crop_bgr, field)
        except Exception as e:
            logger.error(f"Surya failed for {field}: {e}")
            text = ""
        if not text:
            text = read_latin_tesseract(crop_bgr, field)
        return text
    return read_latin_tesseract(crop_bgr, field)