"""
REST API controllers for Enhanced ASR (Step 3) operations
"""

import logging
import tempfile
import os
from typing import List, Optional
from pathlib import Path
import torch

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status
from fastapi.responses import JSONResponse

from src.application.dtos.asr_dto import (
    ASRRequest, ASRResponseDTO, BatchASRRequest, BatchASRResponseDTO,
    TranscriptionOnlyResponseDTO, ASRValidationRequest, ASRValidationResponseDTO,
    ModelInfoResponseDTO, SupportedLanguagesASRResponseDTO, ASRConfigDTO,
    ASRErrorDTO
)
from src.application.use_cases.asr_use_cases import (
    TranscribeAudioUseCase, BatchTranscribeAudioUseCase, 
    TranscriptionOnlyUseCase, ValidateAudioForASRUseCase
)
from src.core.exceptions import (
    SpeechServiceException, AudioProcessingError, ValidationError,
    audio_file_not_found_error, audio_format_unsupported_error,
    validation_required_field_error
)


# Router
asr_router = APIRouter(
    prefix="/api/v1/asr",
    tags=["Enhanced ASR (Step 3)"],
    responses={
        500: {"description": "Internal server error"},
        400: {"description": "Bad request"},
        404: {"description": "Not found"}
    }
)

# Logger
logger = logging.getLogger(__name__)


class ASRController:
    """Controller for ASR operations"""
    
    def __init__(
        self,
        transcribe_use_case: TranscribeAudioUseCase,
        batch_transcribe_use_case: BatchTranscribeAudioUseCase,
        transcription_only_use_case: TranscriptionOnlyUseCase,
        validate_audio_use_case: ValidateAudioForASRUseCase
    ):
        self.transcribe_use_case = transcribe_use_case
        self.batch_transcribe_use_case = batch_transcribe_use_case
        self.transcription_only_use_case = transcription_only_use_case
        self.validate_audio_use_case = validate_audio_use_case


# Dependency injection implementation
async def get_asr_controller() -> ASRController:
    """Get ASR controller instance with real services"""
    import logging
    from src.infrastructure.services.enhanced_whisper_service import EnhancedWhisperASRService
    from src.application.use_cases.asr_use_cases import (
        TranscribeAudioUseCase, BatchTranscribeAudioUseCase,
        TranscriptionOnlyUseCase, ValidateAudioForASRUseCase
    )
    
    # Initialize logger
    logger = logging.getLogger(__name__)
    
    # Initialize Whisper service with GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"🎯 Initializing Whisper service with device: {device}")
    if torch.cuda.is_available():
        logger.info(f"🚀 GPU detected: {torch.cuda.get_device_name(0)}")
        logger.info(f"💾 GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        logger.info("💻 Using CPU - no CUDA GPU detected")
    
    whisper_service = EnhancedWhisperASRService(device=device)
    
    # Real pronunciation and fluency analyzers using Whisper service directly
    # The EnhancedWhisperASRService already includes pronunciation and fluency analysis
    pronunciation_analyzer = whisper_service  # Use Whisper service directly
    fluency_analyzer = whisper_service  # Use Whisper service directly
    
    # Initialize use cases with all required dependencies
    transcribe_use_case = TranscribeAudioUseCase(
        asr_service=whisper_service,
        pronunciation_analyzer=pronunciation_analyzer,
        fluency_analyzer=fluency_analyzer,
        logger=logger
    )
    batch_transcribe_use_case = BatchTranscribeAudioUseCase(whisper_service, logger)
    transcription_only_use_case = TranscriptionOnlyUseCase(whisper_service, logger)
    validate_audio_use_case = ValidateAudioForASRUseCase(logger)
    
    # Create controller
    return ASRController(
        transcribe_use_case=transcribe_use_case,
        batch_transcribe_use_case=batch_transcribe_use_case,
        transcription_only_use_case=transcription_only_use_case,
        validate_audio_use_case=validate_audio_use_case
    )


@asr_router.post(
    "/transcribe",
    response_model=ASRResponseDTO,
    summary="Transcribe audio with pronunciation analysis",
    description="Upload audio file and get transcription with detailed pronunciation assessment"
)
async def transcribe_audio(
    audio_file: UploadFile = File(..., description="Audio file (WAV, MP3, M4A, FLAC, OGG)"),
    reference_text: Optional[str] = Form(None, description="Reference text for pronunciation comparison"),
    language: str = Form(default="english", description="Language for transcription"),
    model_size: str = Form(default="base", description="Whisper model size"),
    include_phonemes: bool = Form(default=True, description="Include phoneme-level analysis"),
    include_timing: bool = Form(default=True, description="Include word/phoneme timing"),
    include_confidence: bool = Form(default=True, description="Include confidence scores"),
    compare_pronunciation: bool = Form(default=True, description="Compare with reference pronunciation"),
    analyze_fluency: bool = Form(default=True, description="Analyze speech fluency"),
    controller: ASRController = Depends(get_asr_controller),
) -> ASRResponseDTO:
    """
    Transcribe uploaded audio file with Enhanced ASR and pronunciation analysis
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
            include_phonemes=include_phonemes,
            include_timing=include_timing,
            include_confidence=include_confidence,
            compare_pronunciation=compare_pronunciation and reference_text is not None,
            analyze_fluency=analyze_fluency
        )
        
        # Execute transcription
        result = await controller.transcribe_use_case.execute(temp_file_path, request)
        
        # Update file path in response to original filename
        result.audio_file_path = audio_file.filename
        
        return result
        
    except ValidationError as e:
        logger.error(f"Validation error in transcribe_audio: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=e.to_dict()
        )
    except AudioProcessingError as e:
        logger.error(f"Audio processing error in transcribe_audio: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=e.to_dict()
        )
    except Exception as e:
        logger.error(f"Unexpected error in transcribe_audio: {str(e)}")
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


@asr_router.post(
    "/transcribe-only",
    response_model=TranscriptionOnlyResponseDTO,
    summary="Transcribe audio without reference text",
    description="Upload audio file and get transcription only (no pronunciation analysis)"
)
async def transcribe_only(
    audio_file: UploadFile = File(..., description="Audio file (WAV, MP3, M4A, FLAC, OGG)"),
    language: str = Form(default="english", description="Language for transcription"),
    model_size: str = Form(default="base", description="Whisper model size"),
    controller: ASRController = Depends(get_asr_controller),
) -> TranscriptionOnlyResponseDTO:
    """
    Transcribe uploaded audio file without pronunciation analysis
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
        
        # Execute transcription
        result = await controller.transcription_only_use_case.execute(
            temp_file_path, language, model_size
        )
        
        # Update file path in response to original filename
        result.audio_file_path = audio_file.filename
        
        return result
        
    except ValidationError as e:
        logger.error(f"Validation error in transcribe_only: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=e.to_dict()
        )
    except AudioProcessingError as e:
        logger.error(f"Audio processing error in transcribe_only: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=e.to_dict()
        )
    except Exception as e:
        logger.error(f"Unexpected error in transcribe_only: {str(e)}")
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


@asr_router.post(
    "/batch-transcribe",
    response_model=BatchASRResponseDTO,
    summary="Batch transcribe multiple audio files",
    description="Upload multiple audio files for batch transcription and analysis"
)
async def batch_transcribe_audio(
    audio_files: List[UploadFile] = File(..., description="List of audio files"),
    reference_texts: Optional[List[str]] = Form(None, description="List of reference texts (optional)"),
    language: str = Form(default="english", description="Language for all files"),
    model_size: str = Form(default="base", description="Whisper model size"),
    parallel_processing: bool = Form(default=True, description="Process files in parallel"),
    controller: ASRController = Depends(get_asr_controller),
) -> BatchASRResponseDTO:
    """
    Batch transcribe multiple audio files
    """
    temp_file_paths = []
    
    try:
        # Validate batch size
        if len(audio_files) > 50:
            raise ValidationError("Too many files. Maximum 50 files per batch.")
        
        if len(audio_files) == 0:
            raise validation_required_field_error("audio_files")
        
        # Validate reference texts count if provided
        if reference_texts and len(reference_texts) != len(audio_files):
            raise ValidationError("Reference texts count must match audio files count")
        
        # Save all files temporarily
        for i, audio_file in enumerate(audio_files):
            if not audio_file.filename:
                raise ValidationError(f"Audio file at index {i} has no filename")
            
            file_extension = Path(audio_file.filename).suffix.lower()
            supported_formats = ['.wav', '.mp3', '.m4a', '.flac', '.ogg']
            if file_extension not in supported_formats:
                raise audio_format_unsupported_error(audio_file.filename, file_extension)
            
            with tempfile.NamedTemporaryFile(
                delete=False, 
                suffix=file_extension,
                dir=tempfile.gettempdir()
            ) as temp_file:
                content = await audio_file.read()
                temp_file.write(content)
                temp_file_paths.append(temp_file.name)
        
        # Create batch request
        batch_request = BatchASRRequest(
            audio_files=[f.filename for f in audio_files],
            reference_texts=reference_texts,
            language=language,
            model_size=model_size,
            parallel_processing=parallel_processing
        )
        
        # Execute batch transcription
        result = await controller.batch_transcribe_use_case.execute(
            temp_file_paths, batch_request
        )
        
        # Update file paths in results to original filenames
        for i, file_result in enumerate(result.results):
            if i < len(audio_files):
                file_result.audio_file_path = audio_files[i].filename
        
        return result
        
    except ValidationError as e:
        logger.error(f"Validation error in batch_transcribe_audio: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=e.to_dict()
        )
    except AudioProcessingError as e:
        logger.error(f"Audio processing error in batch_transcribe_audio: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=e.to_dict()
        )
    except Exception as e:
        logger.error(f"Unexpected error in batch_transcribe_audio: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error_type": "InternalServerError",
                "message": "Internal server error occurred",
                "details": {"original_error": str(e)}
            }
        )
    finally:
        # Clean up all temporary files
        for temp_path in temp_file_paths:
            if os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except Exception as cleanup_error:
                    logger.warning(f"Failed to cleanup temporary file {temp_path}: {cleanup_error}")


@asr_router.post(
    "/validate-audio",
    response_model=ASRValidationResponseDTO,
    summary="Validate audio file for ASR processing",
    description="Check if audio file is suitable for ASR processing"
)
async def validate_audio_file(
    audio_file: UploadFile = File(..., description="Audio file to validate"),
    check_audio_quality: bool = Form(default=True, description="Check audio quality"),
    check_duration: bool = Form(default=True, description="Check audio duration"),
    check_format: bool = Form(default=True, description="Check audio format"),
    whisper_compatibility: bool = Form(default=True, description="Check Whisper compatibility"),
    controller: ASRController = Depends(get_asr_controller),
) -> ASRValidationResponseDTO:
    """
    Validate audio file for ASR processing
    """
    temp_file_path = None
    
    try:
        if not audio_file.filename:
            raise validation_required_field_error("audio_file")
        
        # Save uploaded file temporarily for validation
        file_extension = Path(audio_file.filename).suffix.lower()
        with tempfile.NamedTemporaryFile(
            delete=False, 
            suffix=file_extension,
            dir=tempfile.gettempdir()
        ) as temp_file:
            content = await audio_file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # Create validation request
        validation_request = ASRValidationRequest(
            check_audio_quality=check_audio_quality,
            check_duration=check_duration,
            check_format=check_format,
            whisper_compatibility=whisper_compatibility
        )
        
        # Execute validation
        result = await controller.validate_audio_use_case.execute(
            temp_file_path, validation_request
        )
        
        # Update file path in response to original filename
        result.audio_file_path = audio_file.filename
        
        return result
        
    except ValidationError as e:
        logger.error(f"Validation error in validate_audio_file: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=e.to_dict()
        )
    except Exception as e:
        logger.error(f"Unexpected error in validate_audio_file: {str(e)}")
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


@asr_router.get(
    "/models",
    response_model=List[ModelInfoResponseDTO],
    summary="Get available Whisper models",
    description="List all available Whisper models with their information"
)
async def get_available_models(
) -> List[ModelInfoResponseDTO]:
    """
    Get list of available Whisper models
    """
    try:
        # Static list of Whisper models
        models = [
            ModelInfoResponseDTO(
                name="tiny",
                type="whisper",
                parameters="39M",
                size="~39 MB",
                speed="~32x realtime",
                download_status="available",
                accuracy_description="Fast but lower accuracy, good for quick transcription"
            ),
            ModelInfoResponseDTO(
                name="base",
                type="whisper",
                parameters="74M",
                size="~142 MB",
                speed="~16x realtime",
                download_status="available",
                accuracy_description="Balanced speed and accuracy, recommended for most use cases"
            ),
            ModelInfoResponseDTO(
                name="small",
                type="whisper",
                parameters="244M",
                size="~466 MB",
                speed="~6x realtime",
                download_status="available",
                accuracy_description="Good accuracy with reasonable speed"
            ),
            ModelInfoResponseDTO(
                name="medium",
                type="whisper",
                parameters="769M",
                size="~1.42 GB",
                speed="~2x realtime",
                download_status="available",
                accuracy_description="High accuracy, slower processing"
            ),
            ModelInfoResponseDTO(
                name="large",
                type="whisper",
                parameters="1550M",
                size="~2.87 GB",
                speed="~1x realtime",
                download_status="available",
                accuracy_description="Highest accuracy, slowest processing"
            ),
            ModelInfoResponseDTO(
                name="large-v2",
                type="whisper",
                parameters="1550M",
                size="~2.87 GB",
                speed="~1x realtime",
                download_status="available",
                accuracy_description="Improved version of large model"
            ),
            ModelInfoResponseDTO(
                name="large-v3",
                type="whisper",
                parameters="1550M",
                size="~2.87 GB",
                speed="~1x realtime",
                download_status="available",
                accuracy_description="Latest version with best accuracy"
            )
        ]
        
        return models
        
    except Exception as e:
        logger.error(f"Error getting available models: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error_type": "InternalServerError",
                "message": "Failed to retrieve model information"
            }
        )


@asr_router.get(
    "/languages",
    response_model=SupportedLanguagesASRResponseDTO,
    summary="Get supported languages",
    description="List all languages supported by the ASR service"
)
async def get_supported_languages(
) -> SupportedLanguagesASRResponseDTO:
    """
    Get list of supported languages for ASR
    """
    try:
        # Languages supported by Whisper
        whisper_languages = [
            "english", "chinese", "german", "spanish", "russian", "korean",
            "french", "japanese", "portuguese", "turkish", "polish", "catalan",
            "dutch", "arabic", "swedish", "italian", "indonesian", "hindi",
            "finnish", "vietnamese", "hebrew", "ukrainian", "greek", "malay",
            "czech", "romanian", "danish", "hungarian", "tamil", "norwegian",
            "thai", "urdu", "croatian", "bulgarian", "lithuanian", "latin",
            "maori", "malayalam", "welsh", "slovak", "telugu", "persian",
            "latvian", "bengali", "serbian", "azerbaijani", "slovenian",
            "kannada", "estonian", "macedonian", "breton", "basque", "icelandic",
            "armenian", "nepali", "mongolian", "bosnian", "kazakh", "albanian",
            "swahili", "galician", "marathi", "punjabi", "sinhala", "khmer",
            "shona", "yoruba", "somali", "afrikaans", "occitan", "georgian",
            "belarusian", "tajik", "sindhi", "gujarati", "amharic", "yiddish",
            "lao", "uzbek", "faroese", "haitian creole", "pashto", "turkmen",
            "nynorsk", "maltese", "sanskrit", "luxembourgish", "myanmar",
            "tibetan", "tagalog", "malagasy", "assamese", "tatar", "hawaiian",
            "lingala", "hausa", "bashkir", "javanese", "sundanese"
        ]
        
        # Additional supported languages for the service
        supported_languages = [
            "english", "en-us", "en-uk", "spanish", "french", "german", 
            "italian", "portuguese", "russian", "chinese", "japanese", "korean"
        ]
        
        return SupportedLanguagesASRResponseDTO(
            supported_languages=supported_languages,
            total_languages=len(supported_languages),
            default_language="english",
            whisper_languages=whisper_languages
        )
        
    except Exception as e:
        logger.error(f"Error getting supported languages: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error_type": "InternalServerError",
                "message": "Failed to retrieve language information"
            }
        )


@asr_router.get(
    "/config",
    response_model=ASRConfigDTO,
    summary="Get ASR service configuration",
    description="Get current configuration settings for the ASR service"
)
async def get_asr_config(
) -> ASRConfigDTO:
    """
    Get current ASR service configuration
    """
    try:
        return ASRConfigDTO(
            default_model_size="base",
            default_language="english",
            max_audio_duration=300.0,  # 5 minutes
            min_audio_duration=0.5,    # 500ms
            confidence_threshold=0.7,
            enable_phoneme_analysis=True,
            enable_fluency_analysis=True,
            batch_size_limit=50
        )
        
    except Exception as e:
        logger.error(f"Error getting ASR config: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error_type": "InternalServerError",
                "message": "Failed to retrieve configuration"
            }
        )


@asr_router.get(
    "/health",
    summary="Health check for ASR service",
    description="Check if the ASR service is healthy and operational"
)
async def health_check() -> dict:
    """
    Health check endpoint for ASR service
    """
    try:
        return {
            "status": "healthy",
            "service": "Enhanced ASR Service (Step 3)",
            "version": "1.0.0",
            "features": [
                "Whisper-based transcription",
                "Phoneme-level analysis", 
                "Pronunciation comparison",
                "Fluency assessment",
                "Batch processing"
            ]
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "status": "unhealthy",
                "error": str(e)
            }
        )


# Error handlers removed - handled by global middleware


# Include router in main app
def include_asr_routes(app):
    """Include ASR routes in the main FastAPI application"""
    app.include_router(asr_router)
