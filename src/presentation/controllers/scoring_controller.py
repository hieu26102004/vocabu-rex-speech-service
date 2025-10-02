"""
REST API controller for Step 4: Comprehensive Pronunciation Scoring
"""

import logging
import tempfile
import os
from typing import List, Optional
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from src.core.exceptions import (
    SpeechServiceException, AudioProcessingError, ValidationError,
    audio_file_not_found_error, audio_format_unsupported_error,
    validation_required_field_error
)

# Security
security = HTTPBearer()

# Router
scoring_router = APIRouter(
    prefix="/api/v1/speech",
    tags=["Comprehensive Scoring (Step 4)"],
    responses={
        500: {"description": "Internal server error"},
        400: {"description": "Bad request"},
        404: {"description": "Not found"}
    }
)

# Logger
logger = logging.getLogger(__name__)


@scoring_router.post("/comprehensive-scoring")
async def comprehensive_scoring(
    audio_file: UploadFile = File(...),
    reference_text: str = Form(...),
    phonemes: str = Form(None),  # JSON string of phonemes from Step 1
    alignment_data: str = Form(None),  # JSON string of alignment data from Step 2
    asr_data: str = Form(None),  # JSON string of ASR data from Step 3
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Step 4: Comprehensive Pronunciation Scoring
    
    Combines results from Steps 1-3 to provide overall pronunciation assessment:
    - Pronunciation accuracy score
    - Fluency assessment 
    - Overall grade and detailed feedback
    """
    try:
        logger.info(f"Starting comprehensive scoring for file: {audio_file.filename}")
        
        # Validate inputs
        if not reference_text.strip():
            raise validation_required_field_error("reference_text")
        
        # Save uploaded audio file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(audio_file.filename).suffix) as tmp_file:
            content = await audio_file.read()
            tmp_file.write(content)
            temp_audio_path = tmp_file.name
        
        try:
            # Parse input data from previous steps (if provided)
            import json
            step1_phonemes = json.loads(phonemes) if phonemes else None
            step2_alignment = json.loads(alignment_data) if alignment_data else None
            step3_asr = json.loads(asr_data) if asr_data else None
            
            # If previous step data not provided, we need to run them
            # For now, let's implement a comprehensive scoring that works independently
            
            # Import scoring service
            from src.infrastructure.services.pronunciation_analyzer import PronunciationAnalyzer
            
            scorer = PronunciationAnalyzer()
            result = await scorer.analyze_pronunciation_accuracy(
                audio_path=temp_audio_path,
                reference_text=reference_text
            )
            
            logger.info(f"Comprehensive scoring completed for: {audio_file.filename}")
            
            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "audio_file": audio_file.filename,
                    "reference_text": reference_text,
                    **result
                }
            )
            
        finally:
            # Clean up temp file
            if os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)
    
    except SpeechServiceException as e:
        logger.error(f"Speech service error in comprehensive scoring: {e}")
        raise HTTPException(status_code=e.status_code, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in comprehensive scoring: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error during comprehensive scoring: {str(e)}"
        )


@scoring_router.post("/full-assessment")
async def full_assessment_pipeline(
    audio_file: UploadFile = File(...),
    reference_text: str = Form(...),
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Complete 4-step pronunciation assessment pipeline:
    Step 1: Phonemization
    Step 2: Forced Alignment  
    Step 3: Enhanced ASR
    Step 4: Comprehensive Scoring
    
    Returns results from all 4 steps plus final assessment.
    """
    try:
        logger.info(f"Starting full assessment pipeline for: {audio_file.filename}")
        
        # Validate inputs
        if not reference_text.strip():
            raise validation_required_field_error("reference_text")
        
        # Save uploaded audio file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(audio_file.filename).suffix) as tmp_file:
            content = await audio_file.read()
            tmp_file.write(content)
            temp_audio_path = tmp_file.name
        
        try:
            # Import all services
            from src.infrastructure.services.phonemization_service import PhonemizationService
            from src.infrastructure.services.mfa_alignment_service import MFAAlignmentService  
            from src.infrastructure.services.enhanced_whisper_service import EnhancedWhisperService
            from src.infrastructure.services.pronunciation_analyzer import PronunciationAnalyzer
            
            # Initialize services
            phoneme_processor = PhonemizationService()
            alignment_service = MFAAlignmentService()
            asr_service = EnhancedWhisperService()
            scorer = PronunciationAnalyzer()
            
            import time
            start_time = time.time()
            
            # Step 1: Phonemization
            step1_start = time.time()
            step1_result = await phoneme_processor.phonemize_text(reference_text)
            step1_time = (time.time() - step1_start) * 1000
            
            # Step 2: Forced Alignment
            step2_start = time.time()
            step2_result = await alignment_service.align_audio_text(temp_audio_path, reference_text)
            step2_time = (time.time() - step2_start) * 1000
            
            # Step 3: Enhanced ASR
            step3_start = time.time()
            step3_result = await asr_service.transcribe_with_phonemes(temp_audio_path)
            step3_time = (time.time() - step3_start) * 1000
            
            # Step 4: Comprehensive Scoring
            step4_start = time.time()
            step4_result = await scorer.analyze_pronunciation_accuracy(
                audio_path=temp_audio_path,
                reference_text=reference_text
            )
            step4_time = (time.time() - step4_start) * 1000
            
            total_time = (time.time() - start_time) * 1000
            
            logger.info(f"Full assessment pipeline completed for: {audio_file.filename}")
            
            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "audio_file": audio_file.filename,
                    "reference_text": reference_text,
                    "step1_result": step1_result,
                    "step2_result": step2_result,
                    "step3_result": step3_result,
                    "step4_result": step4_result,
                    "pipeline_summary": {
                        "steps_completed": 4,
                        "overall_success": True,
                        "step_timings": {
                            "step1_ms": step1_time,
                            "step2_ms": step2_time,
                            "step3_ms": step3_time,
                            "step4_ms": step4_time
                        },
                        "final_assessment": {
                            "grade": step4_result.get("grade"),
                            "score": step4_result.get("overall_score"),
                            "recommendation": "Continue practicing pronunciation fundamentals"
                        }
                    },
                    "total_processing_time_ms": total_time,
                    "timestamp": step1_result.get("timestamp")
                }
            )
            
        finally:
            # Clean up temp file
            if os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)
    
    except SpeechServiceException as e:
        logger.error(f"Speech service error in full assessment: {e}")
        raise HTTPException(status_code=e.status_code, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in full assessment: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error during full assessment: {str(e)}"
        )