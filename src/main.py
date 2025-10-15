"""
VocabuRex Speech Service
Main FastAPI application entry point
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.presentation.controllers.health_controller import health_router
from src.presentation.controllers.phonemization_controller import phonemization_router
from src.presentation.controllers.alignment_controller import alignment_router
from src.presentation.controllers.scoring_controller import scoring_router
from src.api.controllers.asr_controller import asr_router
# Middleware imports removed for local development
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
    
    # Initialize and preload Whisper model with GPU detection
    logger.info("🔄 Initializing Whisper service with GPU detection...")
    try:
        import torch
        from src.infrastructure.services.enhanced_whisper_service import EnhancedWhisperASRService
        
        # Debug GPU availability
        cuda_available = torch.cuda.is_available()
        device_count = torch.cuda.device_count()
        logger.info(f"🔍 GPU Detection - CUDA Available: {cuda_available}, Device count: {device_count}")
        
        # Force GPU if available
        device = "cuda" if cuda_available else "cpu"
        logger.info(f"🎯 Selected device: {device}")
        
        if cuda_available:
            try:
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                logger.info(f"� GPU detected: {gpu_name}")
                logger.info(f"💾 GPU memory: {gpu_memory:.1f} GB")
            except Exception as e:
                logger.warning(f"⚠️ GPU info error (but still using GPU): {e}")
        else:
            logger.info("💻 Using CPU - no CUDA GPU detected")
        
        # Initialize service with explicit device
        logger.info(f"🔧 Initializing EnhancedWhisperASRService with device: {device}")
        whisper_service = EnhancedWhisperASRService(device=device)
        logger.info(f"✅ Whisper service initialized with device: {whisper_service.device}")
        
        # Update global service instance
        from src.api import controllers
        controllers.asr_controller._whisper_service = whisper_service
        
        # Preload and warm up small model
        logger.info("🔄 Preloading Whisper model (small)...")
        success = await whisper_service.preload_model("small")
        if success:
            logger.info("🚀 Whisper model preloaded and warmed up successfully!")
        else:
            logger.warning("⚠️  Model preload failed but service will continue")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Whisper service: {e}")
        logger.info("⚠️  Service will be initialized on first request")
    
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
        docs_url="/docs",  # Always enable docs for development and testing
        redoc_url="/redoc",  # Always enable redoc for development and testing
        lifespan=lifespan
    )
    
    # Add completely open CORS for local development
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # All security middleware disabled for local development
    
    # Include routers
    app.include_router(health_router, prefix="/health", tags=["Health"])
    app.include_router(asr_router, tags=["Enhanced ASR"])  # Public API with /api/v1 prefix
    
    # Internal API endpoints (no prefix for microservice calls)
    from src.api.controllers.internal_asr_controller import internal_asr_router
    app.include_router(internal_asr_router, tags=["Internal ASR"])
    
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
                "public_api": "/api/v1/asr",
                "internal_api": "/speech"
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