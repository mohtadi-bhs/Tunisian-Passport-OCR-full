from fastapi import APIRouter, UploadFile, File, HTTPException, status
from app.api.schemas.passport import PassportExtractResponse
from app.domain.passport.service import service
from app.core.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.post("/extract", response_model=PassportExtractResponse)
async def extract_passport(file: UploadFile = File(...)):
    try:
        result = await service.extract(file)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"router.extract failed: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")
