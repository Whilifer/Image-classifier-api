from fastapi import Request
from fastapi.responses import JSONResponse

from app.logger import logger


async def global_exception_handler(
    request: Request,
    exc: Exception
):

    logger.exception(exc)

    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error"
        }
    )