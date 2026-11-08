from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse
from typing import Any, Dict


class AppError(Exception):
    def __init__(self, message: str, details: Dict[str, Any] | None = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"status": "error", "message": exc.message, "details": exc.details},
    )


async def http_error_handler(request: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"status": "error", "message": exc.detail, "details": {}},
    )


async def generic_error_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"status": "error", "message": "Internal server error", "details": {}},
    )
