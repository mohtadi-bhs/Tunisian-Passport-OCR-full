from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.logging import setup_logging
from app.core.errors import AppError, app_error_handler, http_error_handler, generic_error_handler
from app.api.routers import passport

setup_logging()

app = FastAPI(title="Passport Extractor API", version="0.1.0")

app.add_exception_handler(AppError, app_error_handler)
app.add_exception_handler(StarletteHTTPException, http_error_handler)
app.add_exception_handler(RequestValidationError, http_error_handler)
app.add_exception_handler(Exception, generic_error_handler)

app.include_router(passport.router, tags=["passport"])


@app.get("/health")
async def health():
    return {"status": "ok"}
