from pydantic import BaseModel, Field
from typing import Dict, List, Literal


class PassportData(BaseModel):
    surname: str | None = None
    givenNames: str | None = None
    nameArabic: str | None = None
    passportNumber: str | None = None
    nationality: str | None = None
    dateOfBirth: str | None = None
    sex: str | None = None
    placeOfBirth: str | None = None
    dateOfIssue: str | None = None
    dateOfExpiry: str | None = None
    issuingPlace: str | None = None
    mrz: str | None = None
    addressArabic: str | None = None
    jobArabic: str | None = None
    identityNumber: str | None = None


class PassportMeta(BaseModel):
    models: List[str] = Field(default_factory=lambda: ["yolo_detector", "ocr_extractor"])
    processing_ms: int = 0


class PassportExtractResponse(BaseModel):
    status: Literal["ok", "partial"] = "partial"
    data: PassportData = Field(default_factory=PassportData)
    meta: PassportMeta = Field(default_factory=PassportMeta)
    errors: List[str] = Field(default_factory=list)