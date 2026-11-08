import time
from fastapi import UploadFile
from app.api.deps import validate_image_upload
from app.ml.preprocess import Preprocessor
from app.ml.inference import engine
from app.domain.passport.pipeline import PassportPipeline
from app.domain.passport.mrz import read_mrz_from_image
from app.api.schemas.passport import PassportExtractResponse, PassportData, PassportMeta
from app.core.logging import get_logger

logger = get_logger(__name__)


class PassportService:
    def __init__(self):
        self.pre = Preprocessor()
        self.pipe = PassportPipeline()

    def _crop(self, img, det):
        """Straightened crop when the OBB corners are known, otherwise the axis-aligned crop.

        identityNumber (class 6) gets a wider horizontal margin, because its box cuts off the
        last digit. Every other field keeps the zero margin that read best.
        """
        if det.poly is not None:
            if det.class_id == 6:
                return self.pre.warp_crop(img, det.poly, margin_x=0.3, margin_y=0.0)
            return self.pre.warp_crop(img, det.poly)
        return self.pre.crop(img, det.bbox)

    def _mrz(self, img):
        """MRZ result, or None. A failure here must not stop the visual extraction."""
        try:
            return read_mrz_from_image(img)
        except Exception as e:
            logger.error(f"MRZ reading failed: {e}")
            return None

    async def extract(self, file: UploadFile) -> PassportExtractResponse:
        start = time.time()
        try:
            _, data = validate_image_upload(file)
            img = self.pre.read_image(data)
            dets = engine.detect(img)
            crops = []
            for d in dets:
                try:
                    crops.append((d.class_id, self._crop(img, d), d.conf))
                except Exception as e:
                    logger.debug(f"Crop failed for det: {e}")
            ocr_res = engine.recognize(crops)
            mrz = self._mrz(img)
            merged = self.pipe.merge(dets, ocr_res, mrz)
            resp = PassportExtractResponse(
                status=merged.status if merged.status in ("ok", "partial") else "partial",
                data=PassportData(**merged.fields),
                meta=PassportMeta(processing_ms=int((time.time() - start) * 1000)),
                errors=merged.errors,
            )
            return resp
        except Exception as e:
            logger.error(f"extract failed: {e}")
            return PassportExtractResponse(
                status="partial",
                data=PassportData(),
                meta=PassportMeta(processing_ms=int((time.time() - start) * 1000)),
                errors=["Extraction failed"],
            )


service = PassportService()