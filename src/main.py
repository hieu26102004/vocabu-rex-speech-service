"""
VocabuRex Speech Service
Main FastAPI application entry point
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse

from src.presentation.controllers.health_controller import health_router
from src.presentation.controllers.phonemization_controller import phonemization_router
from src.presentation.controllers.alignment_controller import alignment_router
from src.presentation.controllers.scoring_controller import scoring_router
from src.api.controllers.asr_controller import asr_router
from src.presentation.middleware.auth_middleware import AuthMiddleware
from src.presentation.middleware.logging_middleware import LoggingMiddleware
from src.presentation.middleware.error_middleware import ErrorHandlerMiddleware
from src.shared.config import get_settings
from src.core.exceptions import SpeechServiceException

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events"""
    # Startup
    logger.info("🚀 Starting VocabuRex Speech Service...")
    
    # Skip database and Redis for local development
    logger.info("⚡ Running in local mode - no external dependencies")
    
    # Create upload directory
    settings = get_settings()
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    logger.info(f"✅ Upload directory created: {settings.UPLOAD_DIR}")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down VocabuRex Speech Service...")
    logger.info("✅ Local mode shutdown complete")


def create_app() -> FastAPI:
    """Create and configure FastAPI application"""
    settings = get_settings()
    
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description="Speech processing service for VocabuRex - Handles STT, pronunciation assessment, and voice interactions",
        docs_url="/docs" if settings.ENVIRONMENT == "development" else None,
        redoc_url="/redoc" if settings.ENVIRONMENT == "development" else None,
        lifespan=lifespan
    )
    
    # Add security middleware
    app.add_middleware(
        TrustedHostMiddleware, 
        allowed_hosts=["localhost", "127.0.0.1", "*.vocaburex.com"]
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS.split(","),
        allow_credentials=True,
        allow_methods=settings.ALLOWED_METHODS.split(","),
        allow_headers=settings.ALLOWED_HEADERS.split(","),
    )
    
    # Add custom middleware
    app.add_middleware(ErrorHandlerMiddleware)
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(AuthMiddleware)
    
    # Include routers
    app.include_router(health_router, prefix="/health", tags=["Health"])
    app.include_router(asr_router, tags=["Enhanced ASR"])
    app.include_router(phonemization_router, prefix="/api/v1", tags=["Phonemization"])
    app.include_router(alignment_router, prefix="/api/v1", tags=["Forced Alignment"])
    app.include_router(scoring_router, tags=["Comprehensive Scoring"])
    
    # Global exception handler
    @app.exception_handler(SpeechServiceException)
    async def speech_exception_handler(request: Request, exc: SpeechServiceException):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "success": False,
                "error": {
                    "code": exc.error_code,
                    "message": exc.message,
                    "details": exc.details
                },
                "timestamp": exc.timestamp,
                "path": str(request.url)
            }
        )
    
    @app.get("/", tags=["Root"])
    async def root():
        """Root endpoint - Service information"""
        return {
            "service": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "status": "healthy",
            "endpoints": {
                "health": "/health",
                "docs": "/docs",
                "speech": "/api/v1/speech"
            }
        }
    
    return app


# Create the FastAPI app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    
    settings = get_settings()
    
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
        access_log=True
    )