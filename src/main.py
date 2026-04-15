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
from src.presentation.controllers.tts_controller import tts_router
# STT-related routers imported lazily based on SERVICE_MODE to avoid
# importing openai-whisper when running in TTS-only mode
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
    settings = get_settings()
    service_mode = settings.SERVICE_MODE.lower()
    logger.info(f"🚀 Starting VocabuRex Speech Service (MODE: {service_mode})...")
    
    # Skip database and Redis for local development
    logger.info("⚡ Running in local mode - no external dependencies")
    
    # Create upload directory
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    logger.info(f"✅ Upload directory created: {settings.UPLOAD_DIR}")
    
    # GPU detection (shared)
    import torch
    cuda_available = torch.cuda.is_available()
    device = "cuda" if cuda_available else "cpu"
    logger.info(f"🎯 Device: {device} (CUDA: {cuda_available})")
    
    if cuda_available:
        try:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"🎮 GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        except Exception as e:
            logger.warning(f"⚠️ GPU info error: {e}")
    
    # ── Initialize STT (Whisper) ──────────────────────────
    if service_mode in ("stt", "both"):
        logger.info("🔄 Initializing Whisper STT service...")
        try:
            from src.infrastructure.services.enhanced_whisper_service import EnhancedWhisperASRService
            
            whisper_service = EnhancedWhisperASRService(device=device)
            logger.info(f"✅ Whisper service initialized (device: {whisper_service.device})")
            
            # Update global service instances
            from src.api import controllers
            controllers.asr_controller._whisper_service = whisper_service
            
            from src.presentation.controllers import stt_controller
            stt_controller._whisper_service = whisper_service
            
            # Preload model
            logger.info("🔄 Preloading Whisper model (small)...")
            success = await whisper_service.preload_model("small")
            if success:
                logger.info("🚀 Whisper STT ready!")
            else:
                logger.warning("⚠️ Model preload failed but service will continue")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Whisper service: {e}")
            logger.info("⚠️ STT will be initialized on first request")
    else:
        logger.info("⏭️ Skipping STT initialization (mode={service_mode})")
    
    # ── Initialize TTS (VibeVoice) ────────────────────────
    if service_mode in ("tts", "both"):
        logger.info("🔄 Initializing VibeVoice TTS service...")
        try:
            from src.infrastructure.services.vibevoice_tts_service import VibeVoiceTTSService
            from src.presentation.controllers import tts_controller
            
            tts_service = VibeVoiceTTSService(device=device)
            success = await tts_service.preload_model()
            
            if success:
                tts_controller._tts_service = tts_service
                logger.info("🚀 VibeVoice TTS ready!")
            else:
                logger.warning("⚠️ TTS model preload failed")
        except Exception as e:
            logger.error(f"❌ Failed to initialize TTS service: {e}")
            logger.info("⚠️ TTS will be unavailable")
    else:
        logger.info("⏭️ Skipping TTS initialization (mode={service_mode})")
    
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
    
    # Include routers based on SERVICE_MODE
    settings = get_settings()
    service_mode = settings.SERVICE_MODE.lower()
    
    app.include_router(health_router, prefix="/health", tags=["Health"])
    
    # STT-related routers (lazy import to avoid loading whisper in TTS-only mode)
    if service_mode in ("stt", "both"):
        from src.api.controllers.asr_controller import asr_router
        from src.presentation.controllers.stt_controller import stt_router
        from src.presentation.controllers.phonemization_controller import phonemization_router
        from src.presentation.controllers.alignment_controller import alignment_router
        from src.presentation.controllers.scoring_controller import scoring_router
        
        app.include_router(asr_router, tags=["Enhanced ASR"])
        from src.api.controllers.internal_asr_controller import internal_asr_router
        app.include_router(internal_asr_router, tags=["Internal ASR"])
        app.include_router(stt_router, tags=["Simple STT"])
        app.include_router(phonemization_router, prefix="/api/v1", tags=["Phonemization"])
        app.include_router(alignment_router, prefix="/api/v1", tags=["Forced Alignment"])
        app.include_router(scoring_router, tags=["Comprehensive Scoring"])
    
    # TTS-related routers
    if service_mode in ("tts", "both"):
        app.include_router(tts_router, tags=["Text-to-Speech"])
    
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