"""
Health check controller for speech service
"""

import logging
from fastapi import APIRouter, status
from fastapi.responses import JSONResponse
from datetime import datetime

from src.shared.config import get_settings

health_router = APIRouter()
logger = logging.getLogger(__name__)


@health_router.get(
    "/",
    status_code=status.HTTP_200_OK,
    summary="Service Health Check",
    description="Overall health status of the speech service"
)
async def health_check() -> JSONResponse:
    """
    **Overall Service Health Check**
    
    Returns the health status of the entire speech service including:
    - Service availability
    - Configuration status
    - Timestamp
    """
    try:
        settings = get_settings()
        
        return JSONResponse(
            content={
                "success": True,
                "status": "healthy",
                "service": settings.APP_NAME,
                "version": settings.APP_VERSION,
                "environment": settings.ENVIRONMENT,
                "timestamp": datetime.utcnow().isoformat(),
                "endpoints": {
                    "phonemization": "/api/v1/speech/phonemization",
                    "pronunciation": "/api/v1/speech/pronunciation", 
                    "recognition": "/api/v1/speech/recognize"
                }
            },
            status_code=status.HTTP_200_OK
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            content={
                "success": False,
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            },
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE
        )