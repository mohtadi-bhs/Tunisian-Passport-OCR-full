"""MRZ (machine-readable zone) parser for TD3 passports: two lines of 44 characters.

Every value in the MRZ has a check digit (ICAO Doc 9303). A line whose check digits pass is
almost certainly read correctly, so the MRZ is the most reliable source of the printed fields.

Line 1:  P<ISSUER<SURNAME<<GIVEN<NAMES<<<...
Line 2:  DOCNUMBER<CD NAT YYMMDD CD SEX YYMMDD CD OPTIONAL<<CD COMPOSITE_CD
"""
import re
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple

WEIGHTS = (7, 3, 1)
LINE_LEN = 44
MRZ_CHARS = re.compile(r"[^A-Z0-9<]")

# Look-alike characters that OCR confuses. Used only for the single-character repair below.
CONFUSABLE = {
    "O": "0", "0": "O", "D": "0", "Q": "0",
    "I": "1", "1": "I", "L": "1",
    "S": "5", "5": "S",
    "B": "8", "8": "B",
    "Z": "2", "2": "Z",
    "G": "6", "6": "G",
}


class MRZError(ValueError):
    pass


def _value(ch: str) -> int:
    if ch == "<":
        return 0
    if ch.isdigit():
        return int(ch)
    if "A" <= ch <= "Z":
        return ord(ch) - ord("A") + 10
    raise MRZError(f"invalid MRZ character {ch!r}")


def check_digit(data: str) -> int:
    """ICAO check digit: weighted sum (7, 3, 1 repeating) modulo 10."""
    return sum(_value(c) * WEIGHTS[i % 3] for i, c in enumerate(data)) % 10


def _digit_ok(data: str, check: str) -> bool:
    return check.isdigit() and check_digit(data) == int(check)


def _yymmdd_to_ddmmyyyy(s: str, is_birth: bool) -> Optional[str]:
    if not re.fullmatch(r"\d{6}", s):
        return None
    yy, mm, dd = int(s[:2]), int(s[2:4]), int(s[4:6])
    if is_birth:
        # a birth year cannot be in the future: 2-digit years above the current one are 19xx
        year = 1900 + yy if yy > datetime.now().year % 100 else 2000 + yy
    else:
        year = 2000 + yy  # expiry dates are in the 2000s
    try:
        return date(year, mm, dd).strftime("%d-%m-%Y")
    except ValueError:
        return None


def _checks(line2: str) -> Dict[str, bool]:
    """All check digits of line 2. The optional-data check is valid when the field is empty."""
    optional = line2[28:42]
    opt_ok = _digit_ok(optional, line2[42]) if optional.strip("<") else line2[42] in "<0"
    composite_data = line2[0:10] + line2[13:20] + line2[21:43]
    return {
        "document": _digit_ok(line2[0:9], line2[9]),
        "birth": _digit_ok(line2[13:19], line2[19]),
        "expiry": _digit_ok(line2[21:27], line2[27]),
        "optional": opt_ok,
        "composite": _digit_ok(composite_data, line2[43]),
    }


def _parse(line1: str, line2: str) -> Dict:
    checks = _checks(line2)
    names = line1[5:44]
    surname_part, _, given_rest = names.partition("<<")
    # Given names end at the next "<<". Anything after it is filler that OCR may misread (e.g. "SSS").
    given_part = given_rest.split("<<")[0]
    fields = {
        "passportNumber": line2[0:9].replace("<", "") or None,
        "nationality": line2[10:13].replace("<", "") or None,
        "dateOfBirth": _yymmdd_to_ddmmyyyy(line2[13:19], is_birth=True),
        "sex": {"M": "M", "F": "F"}.get(line2[20]),
        "dateOfExpiry": _yymmdd_to_ddmmyyyy(line2[21:27], is_birth=False),
        "surname": surname_part.replace("<", " ").strip() or None,
        "givenNames": given_part.replace("<", " ").strip() or None,
    }
    if fields["nationality"] and not re.fullmatch(r"[A-Z]{3}", fields["nationality"]):
        fields["nationality"] = None  # OCR digits in the nationality field are not a country code
    return {"fields": fields, "checks": checks}

def _normalise(line: str) -> str:
    line = MRZ_CHARS.sub("", line.upper().replace(" ", "<"))
    if len(line) > LINE_LEN:
        line = line[:LINE_LEN]
    return line.ljust(LINE_LEN, "<")


MANDATORY = ("document", "birth", "expiry")


def _mandatory_ok(line2: str) -> bool:
    checks = _checks(line2)
    return all(checks[k] for k in MANDATORY)


def _align_line2(line2: str) -> Tuple[str, int, int]:
    """OCR often adds or drops leading characters, which shifts every field. Try small shifts and
    keep the one where the most mandatory checks pass. Returns (aligned line, shift, checks passed)."""
    candidates = [(k, line2[k:].ljust(LINE_LEN, "<")[:LINE_LEN]) for k in range(0, 4)]
    candidates += [(-k, ("<" * k + line2)[:LINE_LEN]) for k in range(1, 4)]
    best = None
    for shift, cand in candidates:
        checks = _checks(cand)
        score = sum(checks[k] for k in MANDATORY)
        key = (score, -abs(shift))
        if best is None or key > best[0]:
            best = (key, cand, shift)
    return best[1], best[2], best[0][0]


def parse_td3(line1: str, line2: str) -> Dict:
    """Parse two TD3 lines.

    1. Line 2 is aligned (small shifts for OCR-added or dropped characters).
    2. If fewer than the three mandatory checks (document number, birth date, expiry) pass, one
       misread character is repaired, but only when exactly one repair makes all three pass.
    3. The MRZ is valid when the three mandatory checks pass. The composite check covers the
       optional data field, which OCR often garbles, so it is reported as a warning only.
    """
    l1, raw2 = _normalise(line1), _normalise(line2)
    if not l1.startswith("P"):
        raise MRZError("line 1 does not start with a passport type 'P'")

    l2, shift, score = _align_line2(raw2)
    correction = None
    ambiguous = 0
    if score < len(MANDATORY):
        aligned = l2
        found = []
        for i, ch in enumerate(l2):
            for alt in CONFUSABLE.get(ch, ""):
                trial = l2[:i] + alt + l2[i + 1:]
                if _mandatory_ok(trial):
                    found.append((trial, i, alt))
        if len(found) == 1:
            l2, i, alt = found[0]
            correction = {"position": i, "from": aligned[i], "to": alt}
        else:
            ambiguous = len(found)

    result = _parse(l1, l2)
    result["valid"] = _mandatory_ok(l2)
    result["corrected"] = correction is not None
    result["alignment_shift"] = shift
    result["lines"] = [l1, l2]
    if correction:
        result["correction"] = correction
    if ambiguous:
        result["ambiguous_repairs"] = ambiguous
    result["warnings"] = [] if result["checks"]["composite"] else ["composite check failed (optional data field)"]
    return result


def find_mrz_lines(ocr_text: str) -> Optional[Tuple[str, str]]:
    """From raw OCR text of the MRZ band, return the last two lines that look like MRZ lines."""
    lines = [_normalise(l) for l in ocr_text.splitlines() if len(MRZ_CHARS.sub("", l.upper())) >= 30]
    if len(lines) < 2:
        return None
    return lines[-2], lines[-1]


MRZ_BAND_FROM = 0.72  # the MRZ is in the bottom part of the data page
MRZ_CONFIG = "--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"


def _mrz_variants(band_bgr):
    """Preprocessing variants of the MRZ band. Glare and faint print need different treatments,
    so several are tried and the first whose check digits pass is kept."""
    import cv2
    gray = cv2.cvtColor(band_bgr, cv2.COLOR_BGR2GRAY)
    variants = []
    for scale in (2.0, 3.0):
        g = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)
        otsu = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        adaptive = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 41, 15)
        for name, im in (("clahe", clahe), ("otsu", otsu), ("adaptive", adaptive)):
            variants.append((f"x{scale:g}-{name}", cv2.copyMakeBorder(im, 15, 15, 15, 15,
                                                                      cv2.BORDER_CONSTANT, value=255)))
    return variants


def read_mrz_from_image(img) -> Optional[Dict]:
    """Find the MRZ in the bottom of the page and parse it.

    Tries several preprocessing variants. Returns the first result whose check digits all pass.
    If none passes, returns the best attempt (fewest failed checks) marked invalid, so the
    debug output shows what was read. Returns None if no MRZ-like lines are found at all.
    """
    import pytesseract
    from app.ml.ocr import _setup_tesseract
    _setup_tesseract()

    h = img.shape[0]
    band = img[int(h * MRZ_BAND_FROM):, :]

    best = None
    for name, variant in _mrz_variants(band):
        text = pytesseract.image_to_string(variant, lang="eng", config=MRZ_CONFIG)
        lines = find_mrz_lines(text)
        if lines is None:
            continue
        try:
            result = parse_td3(*lines)
        except MRZError:
            continue
        result["ocr_text"] = text
        result["variant"] = name
        if result["valid"]:
            return result
        failed = sum(1 for ok in result["checks"].values() if not ok)
        if best is None or failed < best[0]:
            best = (failed, result)
    return best[1] if best else None