import re
import difflib
from datetime import datetime
from typing import List, Dict, Any, Optional
from app.domain.passport.entities import Detection, OCRResult, MergedPassport
from app.core.logging import get_logger

logger = get_logger(__name__)

# Map model class ids (0-14) to target fields.
# Order matches classes.txt / the class names stored in passport_obb.onnx.
CLASS_TO_FIELD = {
    0: "addressArabic",       # address_ar
    1: "dateOfBirth",         # date_of_birth
    2: "dateOfExpiry",        # date_of_expiry
    3: "dateOfIssue",         # date_of_issue
    4: "nameArabic",          # full_name_ar
    5: "givenNames",          # given_names
    6: "identityNumber",      # identity_number
    7: "issuingPlace",        # issuing_place
    8: "jobArabic",           # job_ar
    9: "nationality",         # nationality
    10: "passportNumber",     # passport_number
    11: "placeOfBirth",       # place_of_birth
    12: None,                 # place_of_birth_ar: not returned (no reliable detection)
    13: "sex",                # sex
    14: "surname",            # surname
}

# Expected format for fields that have a fixed shape. A value that does not match is
# reported as unreadable instead of being returned as if it were correct.
FORMAT_RULES = {
    "dateOfBirth": re.compile(r"^\d{2}-\d{2}-\d{4}$"),
    "dateOfIssue": re.compile(r"^\d{2}-\d{2}-\d{4}$"),
    "dateOfExpiry": re.compile(r"^\d{2}-\d{2}-\d{4}$"),
    "passportNumber": re.compile(r"^[A-Z]\d{6,8}$"),
    "identityNumber": re.compile(r"^\d{8}$"),
    "sex": re.compile(r"^[MF]$"),
    "nationality": re.compile(r"^[A-Z]{3}$"),
}

# Nationality words as printed on the page, and their three-letter MRZ codes.
NATIONALITY_WORDS = {
    "TUNISIAN": "TUN",
    "ALGERIAN": "DZA",
    "MOROCCAN": "MAR",
    "LIBYAN": "LBY",
    "EGYPTIAN": "EGY",
}

# Tunisian places printed on the passport. A visual read close to one of these is corrected to it.
KNOWN_PLACES = [
    "TUNIS", "SFAX", "SOUSSE", "JERBA", "KAIROUAN", "BIZERTE", "GABES", "MEDENINE", "NABEUL",
    "MONASTIR", "MAHDIA", "KASSERINE", "GAFSA", "BEJA", "JENDOUBA", "KEF", "SILIANA", "ZAGHOUAN",
    "TOZEUR", "KEBILI", "TATAOUINE", "SIDI BOUZID", "MANOUBA", "ARIANA", "BEN AROUS",
]
PLACE_FIELDS = ("placeOfBirth", "issuingPlace")

REQUIRED = ["passportNumber", "surname", "givenNames", "dateOfBirth"]


def _match_nationality(value: str) -> Optional[str]:
    """Map a printed nationality that is close to a known word (for example TUNYSYAN -> TUNISIAN)."""
    candidates = difflib.get_close_matches(value.upper(), list(NATIONALITY_WORDS), n=1, cutoff=0.6)
    return candidates[0] if candidates else None


def _match_place(value: str) -> Optional[str]:
    """Return the known place a visual read is close to (for example VERBAY -> JERBA), or None."""
    candidates = difflib.get_close_matches(value.upper(), KNOWN_PLACES, n=1, cutoff=0.7)
    return candidates[0] if candidates else None


ARABIC_LETTER = re.compile(r"[\u0621-\u064A]")  # Arabic letters (not digits or punctuation)
ARABIC_FIELD_NAMES = ("nameArabic", "placeOfBirthArabic", "addressArabic", "jobArabic")


def _plausible(field: str, value: str) -> bool:
    """Extra checks beyond the shape: real calendar dates, a three-letter nationality code,
    and at least two real Arabic letters in an Arabic field."""
    if field in ("dateOfBirth", "dateOfIssue", "dateOfExpiry"):
        try:
            datetime.strptime(value, "%d-%m-%Y")
            return True
        except ValueError:
            return False
    if field == "nationality":
        return bool(re.fullmatch(r"[A-Z]{3}", value))
    if field in ARABIC_FIELD_NAMES:
        return len(ARABIC_LETTER.findall(value)) >= 2
    return True


class PassportPipeline:
    def merge(self, detections: List[Detection], ocr_results: List[OCRResult],
              mrz: Optional[Dict] = None) -> MergedPassport:
        fields: Dict[str, Any] = {}
        errors: List[str] = []

        # One text per class: the reading from the most confident detection that produced text.
        # (Longest text is not a good rule: a weak, noisy box can be longer than the right one.)
        best: Dict[int, Any] = {}
        for res in ocr_results:
            txt = (res.text or "").strip()
            if not txt:
                continue
            if res.class_id not in best or res.conf > best[res.class_id][0]:
                best[res.class_id] = (res.conf, txt)
        class_to_text: Dict[int, str] = {cid: txt for cid, (_, txt) in best.items()}

        for cid, txt in class_to_text.items():
            field = CLASS_TO_FIELD.get(cid)
            if not field:
                logger.debug(f"Unmapped class_id {cid} with text present")
                continue
            fields[field] = txt

        # MRZ: when its check digits pass, its values are the most reliable. They override the
        # visual OCR, and any disagreement is reported so it can be checked.
        if mrz and mrz.get("valid"):
            for key, value in mrz["fields"].items():
                if value is None:
                    continue
                visual = fields.get(key)
                if visual and visual != value:
                    errors.append(f"{key}: MRZ {value!r} used instead of visual {visual!r}")
                fields[key] = value
            fields["mrz"] = "\n".join(mrz["lines"])
            if mrz.get("corrected"):
                c = mrz["correction"]
                errors.append(f"MRZ: corrected character {c['from']!r} to {c['to']!r} at position {c['position']}")
            for warning in mrz.get("warnings", []):
                errors.append(f"MRZ: {warning}")

        # Nationality: if the printed word is close to a known one, use that word's MRZ code.
        visual_nat = fields.get("nationality")
        if visual_nat and not re.fullmatch(r"[A-Z]{3}", visual_nat):
            match = _match_nationality(visual_nat)
            if match:
                errors.append(f"nationality: read {visual_nat!r}, matched to {match!r}")
                fields["nationality"] = NATIONALITY_WORDS[match]

        # Places: a visual read that is close to a known place is corrected to that place.
        for field in PLACE_FIELDS:
            value = fields.get(field)
            if value and value not in KNOWN_PLACES:
                match = _match_place(value)
                if match:
                    errors.append(f"{field}: read {value!r}, matched to {match!r}")
                    fields[field] = match

        # Format check: a value that does not match its expected shape is not returned.
        rejected = False
        for field in list(FORMAT_RULES) + [f for f in ARABIC_FIELD_NAMES if f not in FORMAT_RULES]:
            value = fields.get(field)
            if value is None:
                continue
            rule = FORMAT_RULES.get(field)
            if (rule and not rule.match(value)) or not _plausible(field, value):
                errors.append(f"{field}: unreadable ({value!r})")
                fields[field] = None
                rejected = True

        # Status: ok only when every required field is present and valid.
        missing = [k for k in REQUIRED if not fields.get(k)]
        if not fields:
            status = "partial"
            errors.append("No fields extracted")
        elif missing:
            status = "partial"
            errors.append("missing: " + ", ".join(missing))
        elif rejected:
            status = "partial"  # a value was read but rejected by the format check
        else:
            status = "ok"

        return MergedPassport(fields={k: v for k, v in fields.items() if v is not None},
                              status=status, errors=errors)