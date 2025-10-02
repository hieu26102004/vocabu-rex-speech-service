"""
FastAPI controllers for forced alignment endpoints
Handles audio upload, alignment processing, and result retrieval
"""

import tempfile
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

from fastapi import (
    APIRouter, 
    Depends, 
    HTTPException, 
    status, 
    UploadFile, 
    File, 
    Form,
    BackgroundTasks,
    Query
)
from fastapi.responses import FileResponse

from src.application.usecases.alignment_usecases import AlignmentUseCases
from src.application.dtos.alignment_dto import (
    AlignmentRequest,
    BatchAlignmentRequest,
    PhonemeAlignmentRequest,
    AlignmentValidationRequest,
    AlignmentResponseDTO,
    BatchAlignmentResponseDTO,
    AudioValidationResponseDTO,
    SupportedLanguagesResponseDTO,
    AvailableModelsResponseDTO,
    ModelInfoResponseDTO,
    ExportResponseDTO,
    AlignmentProgressDTO
)

logger = logging.getLogger(__name__)

# Create router
alignment_router = APIRouter(
    prefix="/alignment",
    tags=["forced-alignment"],
    responses={
        404: {"description": "Not found"},
        422: {"description": "Validation error"},
        500: {"description": "Internal server error"}
    }
)


# Dependency injection
async def get_alignment_use_cases() -> AlignmentUseCases:
    """
    Dependency provider for alignment use cases
    This would be configured with proper DI container in production
    """
    # TODO: Replace with proper dependency injection
    from src.infrastructure.services.mfa_alignment_service import MFAAlignmentService, MFAModelManager
    from src.infrastructure.utils.audio_processor import AudioProcessor
    
    alignment_service = MFAAlignmentService()
    model_manager = MFAModelManager()
    audio_processor = AudioProcessor()
    
    return AlignmentUseCases(
        alignment_service=alignment_service,
        model_manager=model_manager,
        audio_processor=audio_processor
    )


@alignment_router.post(
    "/align",
    response_model=AlignmentResponseDTO,
    summary="Perform forced alignment on audio and text",
    description="Upload audio file and text to get precise phoneme-level timing alignment"
)
async def align_audio_with_text(
    background_tasks: BackgroundTasks,
    audio_file: UploadFile = File(..., description="Audio file (WAV, MP3, FLAC)"),
    text: str = Form(..., description="Text to align with audio", min_length=1, max_length=5000),
    language: str = Form(default="english", description="Language code"),
    model_name: Optional[str] = Form(default=None, description="Specific alignment model"),
    include_phonemes: bool = Form(default=True, description="Include phoneme-level alignment"),
    include_confidence: bool = Form(default=True, description="Include confidence scores"),
    preprocess_audio: bool = Form(default=True, description="Apply audio preprocessing"),
    use_cases: AlignmentUseCases = Depends(get_alignment_use_cases)
):
    """
    Perform forced alignment between audio and text
    
    - **audio_file**: Audio file to process (supported formats: WAV, MP3, FLAC)
    - **text**: Text content to align with the audio
    - **language**: Language code (e.g., "english", "en-us")
    - **model_name**: Specific alignment model to use (optional)
    - **include_phonemes**: Whether to include phoneme-level timing
    - **include_confidence**: Whether to include confidence scores
    - **preprocess_audio**: Whether to apply audio preprocessing
    """
    try:
        # Validate file type
        if not audio_file.content_type or not audio_file.content_type.startswith(('audio/', 'application/')):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid file type. Please upload an audio file."
            )
        
        # Save uploaded file to temporary location
        temp_dir = Path(tempfile.gettempdir()) / "vocabu_rex_uploads"
        temp_dir.mkdir(exist_ok=True)
        
        temp_file_path = temp_dir / f"upload_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{audio_file.filename}"
        
        with open(temp_file_path, "wb") as buffer:
            content = await audio_file.read()
            buffer.write(content)
        
        # Create alignment request
        alignment_request = AlignmentRequest(
            text=text,
            language=language,
            model_name=model_name,
            include_phonemes=include_phonemes,
            include_confidence=include_confidence,
            preprocess_audio=preprocess_audio
        )
        
        # Perform alignment
        result = await use_cases.align_audio_with_text(str(temp_file_path), alignment_request)
        
        # Schedule cleanup of temporary file
        background_tasks.add_task(cleanup_temp_file, temp_file_path)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Alignment endpoint failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Alignment processing failed: {str(e)}"
        )


@alignment_router.post(
    "/batch-align",
    response_model=BatchAlignmentResponseDTO,
    summary="Perform batch forced alignment",
    description="Process multiple audio-text pairs in a single request"
)
async def batch_align_audio_with_text(
    request: BatchAlignmentRequest,
    use_cases: AlignmentUseCases = Depends(get_alignment_use_cases)
):
    """
    Perform batch forced alignment on multiple audio-text pairs
    
    - **items**: List of audio file paths and corresponding text
    - **language**: Language code for all items
    - **model_name**: Alignment model to use for all items
    - **parallel_processing**: Whether to process items in parallel
    """
    try:
        result = await use_cases.batch_align(request)
        return result
        
    except Exception as e:
        logger.error(f"Batch alignment endpoint failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch alignment processing failed: {str(e)}"
        )


@alignment_router.post(
    "/align-phonemes",
    response_model=AlignmentResponseDTO,
    summary="Align audio with phoneme sequence",
    description="Perform forced alignment using pre-computed phoneme sequence"
)
async def align_audio_with_phonemes(
    background_tasks: BackgroundTasks,
    audio_file: UploadFile = File(..., description="Audio file to align"),
    phonemes: str = Form(..., description="Comma-separated phoneme sequence (ARPAbet format)"),
    language: str = Form(default="english", description="Language code"),
    use_cases: AlignmentUseCases = Depends(get_alignment_use_cases)
):
    """
    Align audio with pre-computed phoneme sequence
    
    - **audio_file**: Audio file to process
    - **phonemes**: Comma-separated phoneme sequence in ARPAbet format
    - **language**: Language code
    """
    try:
        # Validate and save audio file
        temp_dir = Path(tempfile.gettempdir()) / "vocabu_rex_uploads"
        temp_dir.mkdir(exist_ok=True)
        
        temp_file_path = temp_dir / f"phoneme_upload_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{audio_file.filename}"
        
        with open(temp_file_path, "wb") as buffer:
            content = await audio_file.read()
            buffer.write(content)
        
        # Parse phonemes
        phoneme_list = [p.strip().upper() for p in phonemes.split(',') if p.strip()]
        
        if not phoneme_list:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No valid phonemes provided"
            )
        
        # Create phoneme alignment request
        phoneme_request = PhonemeAlignmentRequest(
            phonemes=phoneme_list,
            language=language
        )
        
        # Perform alignment
        result = await use_cases.align_with_phonemes(str(temp_file_path), phoneme_request)
        
        # Schedule cleanup
        background_tasks.add_task(cleanup_temp_file, temp_file_path)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Phoneme alignment endpoint failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Phoneme alignment processing failed: {str(e)}"
        )


@alignment_router.post(
    "/validate-audio",
    response_model=AudioValidationResponseDTO,
    summary="Validate audio file for alignment",
    description="Check if audio file is suitable for forced alignment processing"
)
async def validate_audio_for_alignment(
    background_tasks: BackgroundTasks,
    audio_file: UploadFile = File(..., description="Audio file to validate"),
    check_format: bool = Form(default=True, description="Validate audio format"),
    check_duration: bool = Form(default=True, description="Validate audio duration"),
    check_quality: bool = Form(default=True, description="Assess audio quality"),
    extract_features: bool = Form(default=False, description="Extract audio features"),
    use_cases: AlignmentUseCases = Depends(get_alignment_use_cases)
):
    """
    Validate audio file for alignment suitability
    
    - **audio_file**: Audio file to validate
    - **check_format**: Whether to validate audio format
    - **check_duration**: Whether to validate audio duration  
    - **check_quality**: Whether to assess audio quality
    - **extract_features**: Whether to extract detailed audio features
    """
    try:
        # Save uploaded file temporarily
        temp_dir = Path(tempfile.gettempdir()) / "vocabu_rex_validation"
        temp_dir.mkdir(exist_ok=True)
        
        temp_file_path = temp_dir / f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{audio_file.filename}"
        
        with open(temp_file_path, "wb") as buffer:
            content = await audio_file.read()
            buffer.write(content)
        
        # Create validation request
        validation_request = AlignmentValidationRequest(
            check_format=check_format,
            check_duration=check_duration,
            check_quality=check_quality,
            extract_features=extract_features
        )
        
        # Validate audio
        result = await use_cases.validate_audio_file(str(temp_file_path), validation_request)
        
        # Schedule cleanup
        background_tasks.add_task(cleanup_temp_file, temp_file_path)
        
        return result
        
    except Exception as e:
        logger.error(f"Audio validation endpoint failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Audio validation failed: {str(e)}"
        )


@alignment_router.get(
    "/languages",
    response_model=SupportedLanguagesResponseDTO,
    summary="Get supported languages",
    description="List all languages supported for forced alignment"
)
async def get_supported_languages(
    use_cases: AlignmentUseCases = Depends(get_alignment_use_cases)
):
    """
    Get list of all supported languages for forced alignment
    """
    try:
        result = await use_cases.get_supported_languages()
        return result
        
    except Exception as e:
        logger.error(f"Get languages endpoint failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve supported languages: {str(e)}"
        )


@alignment_router.get(
    "/models/{language}",
    response_model=AvailableModelsResponseDTO,
    summary="Get available models for language",
    description="List all available alignment models for a specific language"
)
async def get_available_models_for_language(
    language: str,
    use_cases: AlignmentUseCases = Depends(get_alignment_use_cases)
):
    """
    Get available alignment models for a specific language
    
    - **language**: Language code to get models for
    """
    try:
        result = await use_cases.get_available_models(language)
        return result
        
    except Exception as e:
        logger.error(f"Get models endpoint failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve models for {language}: {str(e)}"
        )


@alignment_router.post(
    "/export/{format}",
    response_model=ExportResponseDTO,
    summary="Export alignment result",
    description="Export alignment result to TextGrid or JSON format"
)
async def export_alignment_result(
    format: str,
    alignment_data: AlignmentResponseDTO,
    output_filename: Optional[str] = Query(None, description="Custom output filename"),
    use_cases: AlignmentUseCases = Depends(get_alignment_use_cases)
):
    """
    Export alignment result to file
    
    - **format**: Export format ("json" or "textgrid")
    - **alignment_data**: Alignment result to export
    - **output_filename**: Custom filename for output
    """
    try:
        # Validate format
        if format.lower() not in ["json", "textgrid"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid export format. Supported formats: json, textgrid"
            )
        
        # Create output path
        export_dir = Path(tempfile.gettempdir()) / "vocabu_rex_exports"
        export_dir.mkdir(exist_ok=True)
        
        if output_filename:
            output_path = export_dir / output_filename
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = export_dir / f"alignment_export_{timestamp}.{format.lower()}"
        
        # Perform export
        result = await use_cases.export_alignment_result(
            alignment_data, str(output_path), format
        )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Export endpoint failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Export failed: {str(e)}"
        )


@alignment_router.get(
    "/download/{file_path:path}",
    summary="Download exported file",
    description="Download exported alignment result file"
)
async def download_exported_file(file_path: str):
    """
    Download exported alignment file
    
    - **file_path**: Path to exported file
    """
    try:
        full_path = Path(tempfile.gettempdir()) / "vocabu_rex_exports" / file_path
        
        if not full_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="File not found"
            )
        
        return FileResponse(
            path=str(full_path),
            filename=full_path.name,
            media_type='application/octet-stream'
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Download endpoint failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Download failed: {str(e)}"
        )


# Model management endpoints

@alignment_router.post(
    "/models/download",
    response_model=ModelInfoResponseDTO,
    summary="Download alignment model",
    description="Download and install alignment model for specific language"
)
async def download_alignment_model(
    language: str = Form(..., description="Language code"),
    model_name: str = Form(..., description="Model name to download"),
    use_cases: AlignmentUseCases = Depends(get_alignment_use_cases)
):
    """
    Download and install alignment model
    
    - **language**: Language code
    - **model_name**: Name of model to download
    """
    try:
        result = await use_cases.download_model(language, model_name)
        return result
        
    except Exception as e:
        logger.error(f"Model download endpoint failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model download failed: {str(e)}"
        )


@alignment_router.get(
    "/models/installed",
    response_model=Dict[str, List[ModelInfoResponseDTO]],
    summary="List installed models",
    description="Get list of all installed alignment models"
)
async def list_installed_alignment_models(
    use_cases: AlignmentUseCases = Depends(get_alignment_use_cases)
):
    """
    List all installed alignment models
    """
    try:
        result = await use_cases.list_installed_models()
        return result
        
    except Exception as e:
        logger.error(f"List models endpoint failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list installed models: {str(e)}"
        )


# Health check endpoint

@alignment_router.get(
    "/health",
    summary="Alignment service health check",
    description="Check if alignment service is running properly"
)
async def alignment_service_health_check(
    use_cases: AlignmentUseCases = Depends(get_alignment_use_cases)
):
    """
    Health check for alignment service
    """
    try:
        # Basic health check - try to get supported languages
        languages = await use_cases.get_supported_languages()
        
        return {
            "status": "healthy",
            "service": "forced-alignment",
            "timestamp": datetime.now().isoformat(),
            "supported_languages_count": languages.total_languages,
            "version": "1.0.0"
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Alignment service unhealthy: {str(e)}"
        )


# Utility functions

async def cleanup_temp_file(file_path: Path):
    """Clean up temporary file"""
    try:
        if file_path.exists():
            file_path.unlink()
            logger.info(f"Cleaned up temporary file: {file_path}")
    except Exception as e:
        logger.warning(f"Failed to clean up temporary file {file_path}: {e}")


# Error handlers removed - handled by global middleware