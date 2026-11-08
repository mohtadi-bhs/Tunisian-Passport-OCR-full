from fastapi import UploadFile, HTTPException, status
from typing import Tuple


ALLOWED_MIME = {"image/jpeg", "image/png", "image/jpg"}
ALLOWED_EXT = {".jpg", ".jpeg", ".png"}


def validate_image_upload(file: UploadFile) -> Tuple[UploadFile, bytes]:
    if not file:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No file provided")

    content_type = (file.content_type or "").lower()
    filename = (file.filename or "").lower()
    ext_ok = any(filename.endswith(e) for e in ALLOWED_EXT)
    mime_ok = content_type in ALLOWED_MIME or content_type == "image/jpg"

    if not (mime_ok or ext_ok):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only image files (jpg/jpeg/png) are allowed")

    try:
        data = file.file.read()
    except Exception:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Failed to read uploaded file")
    finally:
        try:
            file.file.seek(0)
        except Exception:
            pass

    if not data:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty file")

    return file, data
