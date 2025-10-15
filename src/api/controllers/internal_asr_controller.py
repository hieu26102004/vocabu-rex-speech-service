"""
Internal ASR API controllers for service-to-service communication
Simplified endpoints without /api/v1 prefix for internal microservice calls
"""

import logging
import tempfile
import os
from typing import Optional
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status
from fastapi.responses import JSONResponse

from src.application.dtos.asr_dto import (
    ASRRequest, ASRResponseDTO, TranscriptionOnlyResponseDTO
)
from src.application.use_cases.asr_use_cases import (
    TranscribeAudioUseCase, TranscriptionOnlyUseCase
)
from src.core.exceptions import (
    SpeechServiceException, AudioProcessingError, ValidationError,
    audio_file_not_found_error, audio_format_unsupported_error,
    validation_required_field_error
)


# Internal Router - with speech/ prefix for microservice calls
internal_asr_router = APIRouter(
    prefix="/speech",
    tags=["Internal ASR API"],
    responses={
        500: {"description": "Internal server error"},
        400: {"description": "Bad request"},
        404: {"description": "Not found"}
    }
)

# Logger
logger = logging.getLogger(__name__)


class InternalASRController:
    """Controller for internal ASR operations"""
    
    def __init__(
        self,
        transcribe_use_case: TranscribeAudioUseCase,
        transcription_only_use_case: TranscriptionOnlyUseCase
    ):
        self.transcribe_use_case = transcribe_use_case
        self.transcription_only_use_case = transcription_only_use_case


# Use the same service instances from asr_controller
from src.api.controllers.asr_controller import whisper_service
from src.application.use_cases.asr_use_cases import TranscribeAudioUseCase, TranscriptionOnlyUseCase

def get_whisper_service_by_model(model_size: str):
    """
    Trả về instance EnhancedWhisperASRService với model_size tương ứng
    """
    from src.infrastructure.services.enhanced_whisper_service import EnhancedWhisperASRService
    device = whisper_service.device if whisper_service else "cpu"
    # Model đã được tải sẵn ở models/whisper
    return EnhancedWhisperASRService(device=device, models_directory="models/whisper")

def get_internal_asr_controller(model_size: str = "base") -> InternalASRController:
    """Get internal ASR controller with initialized services for specific model"""
    service = get_whisper_service_by_model(model_size)
    transcribe_use_case = TranscribeAudioUseCase(
        asr_service=service,
        pronunciation_analyzer=service,
        fluency_analyzer=service,
        logger=logger
    )
    transcription_only_use_case = TranscriptionOnlyUseCase(service, logger)
    return InternalASRController(
        transcribe_use_case=transcribe_use_case,
        transcription_only_use_case=transcription_only_use_case
    )



@internal_asr_router.post(
    "/transcribe",
    response_model=ASRResponseDTO,
    summary="Internal: Transcribe audio with pronunciation analysis",
    description="Internal endpoint for microservice calls - transcribe audio with detailed analysis"
)
async def internal_transcribe_audio(
    audio_file: UploadFile = File(..., description="Audio file (WAV, MP3, M4A, FLAC, OGG)"),
    reference_text: Optional[str] = Form(None, description="Reference text for pronunciation comparison"),
    language: str = Form(default="english", description="Language for transcription"),
    model_size: str = Form(default="medium", description="Whisper model size"),
) -> ASRResponseDTO:
    """
    Internal transcribe endpoint for service-to-service calls
    """
    temp_file_path = None
    
    try:
        # Validate file upload
        if not audio_file.filename:
            raise validation_required_field_error("audio_file")
        
        # Check file format
        supported_formats = ['.wav', '.mp3', '.m4a', '.flac', '.ogg']
        file_extension = Path(audio_file.filename).suffix.lower()
        if file_extension not in supported_formats:
            raise audio_format_unsupported_error(audio_file.filename, file_extension)
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(
            delete=False, 
            suffix=file_extension,
            dir=tempfile.gettempdir()
        ) as temp_file:
            content = await audio_file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # Create request DTO
        request = ASRRequest(
            reference_text=reference_text,
            language=language,
            model_size=model_size,
            include_phonemes=True,
            include_timing=True,
            include_confidence=True,
            compare_pronunciation=reference_text is not None,
            analyze_fluency=True
        )
        # Khởi tạo controller với model_size tương ứng
        controller = get_internal_asr_controller(model_size)
        # Execute transcription
        result = await controller.transcribe_use_case.execute(temp_file_path, request)
        
        # Update file path in response to original filename
        result.audio_file_path = audio_file.filename
        
        return result
        
    except ValidationError as e:
        logger.error(f"Validation error in internal_transcribe_audio: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=e.to_dict()
        )
    except AudioProcessingError as e:
        logger.error(f"Audio processing error in internal_transcribe_audio: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=e.to_dict()
        )
    except Exception as e:
        logger.error(f"Unexpected error in internal_transcribe_audio: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error_type": "InternalServerError",
                "message": "Internal server error occurred",
                "details": {"original_error": str(e)}
            }
        )
    finally:
        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup temporary file {temp_file_path}: {cleanup_error}")


@internal_asr_router.post(
    "/transcribe-only",
    response_model=TranscriptionOnlyResponseDTO,
    summary="Internal: Transcribe audio without analysis",
    description="Internal endpoint - transcribe audio without pronunciation analysis"
)
async def internal_transcribe_only(
    audio_file: UploadFile = File(..., description="Audio file (WAV, MP3, M4A, FLAC, OGG)"),
    language: str = Form(default="english", description="Language for transcription"),
    model_size: str = Form(default="medium", description="Whisper model size"),
) -> TranscriptionOnlyResponseDTO:
    """
    Internal transcribe-only endpoint for service-to-service calls
    """
    temp_file_path = None
    
    try:
        # Validate file upload
        if not audio_file.filename:
            raise validation_required_field_error("audio_file")
        
        # Check file format
        supported_formats = ['.wav', '.mp3', '.m4a', '.flac', '.ogg']
        file_extension = Path(audio_file.filename).suffix.lower()
        if file_extension not in supported_formats:
            raise audio_format_unsupported_error(audio_file.filename, file_extension)
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(
            delete=False, 
            suffix=file_extension,
            dir=tempfile.gettempdir()
        ) as temp_file:
            content = await audio_file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # Khởi tạo controller với model_size tương ứng
        controller = get_internal_asr_controller(model_size)
        # Execute transcription
        result = await controller.transcription_only_use_case.execute(
            temp_file_path, language, model_size
        )
        
        # Update file path in response to original filename
        result.audio_file_path = audio_file.filename
        
        return result
        
    except ValidationError as e:
        logger.error(f"Validation error in internal_transcribe_only: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=e.to_dict()
        )
    except AudioProcessingError as e:
        logger.error(f"Audio processing error in internal_transcribe_only: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=e.to_dict()
        )
    except Exception as e:
        logger.error(f"Unexpected error in internal_transcribe_only: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error_type": "InternalServerError",
                "message": "Internal server error occurred",
                "details": {"original_error": str(e)}
            }
        )
    finally:
        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup temporary file {temp_file_path}: {cleanup_error}")


@internal_asr_router.get(
    "/health",
    summary="Internal: Health check",
    description="Internal health check for service monitoring"
)
async def internal_health_check() -> dict:
    """
    Internal health check endpoint for microservices
    """
    try:
        return {
            "status": "healthy",
            "service": "Speech Service",
            "version": "1.0.0",
            "device": whisper_service.device if whisper_service else "unknown",
            "endpoints": [
                "POST /transcribe",
                "POST /transcribe-only", 
                "GET /health"
            ]
        }
    except Exception as e:
        logger.error(f"Internal health check failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "status": "unhealthy",
                "error": str(e)
            }
        )


# Include router function
def include_internal_asr_routes(app):
    """Include internal ASR routes in the main FastAPI application"""
    app.include_router(internal_asr_router)