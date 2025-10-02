"""
VocabuRex Speech Service - Simple FastAPI App
4-Step Pronunciation Assessment Pipeline:
1. Phonemization (Text → Phonemes)
2. Forced Alignment (Audio + Text → Timing) 
3. Enhanced ASR (Audio → Transcription + Analysis)
4. Comprehensive Scoring (Combined Assessment)
"""

import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Simple configuration
class SimpleConfig:
    APP_NAME = "VocabuRex Speech Service"
    APP_VERSION = "1.0.0"
    ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
    DEBUG = os.getenv("DEBUG", "true").lower() == "true"
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 3005))
    
    # Directories
    UPLOAD_DIR = "uploads"
    TEMP_DIR = "temp"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("🚀 Starting VocabuRex Speech Service...")
    
    # Create directories
    os.makedirs(SimpleConfig.UPLOAD_DIR, exist_ok=True)
    os.makedirs(SimpleConfig.TEMP_DIR, exist_ok=True)
    logger.info("✅ Directories created")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down VocabuRex Speech Service...")

# Create FastAPI app
app = FastAPI(
    title=SimpleConfig.APP_NAME,
    description="4-Step Pronunciation Assessment Pipeline",
    version=SimpleConfig.APP_VERSION,
    debug=SimpleConfig.DEBUG,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Gateway will handle CORS properly
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": SimpleConfig.APP_NAME,
        "version": SimpleConfig.APP_VERSION,
        "environment": SimpleConfig.ENVIRONMENT,
        "features": [
            "Step 1: Phonemization",
            "Step 2: Forced Alignment", 
            "Step 3: Enhanced ASR",
            "Step 4: Comprehensive Scoring"
        ]
    }

# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "VocabuRex Speech Service API",
        "version": SimpleConfig.APP_VERSION,
        "docs": "/docs",
        "health": "/health"
    }

# Step 1: Phonemization
@app.post("/api/v1/phonemization/phonemize", tags=["Step 1: Phonemization"])
async def phonemize_text(request: dict):
    """
    Step 1: Convert text to phonemes
    """
    try:
        text = request.get("text", "")
        language = request.get("language", "english")
        
        # Mock implementation for now
        phonemes = []
        words = text.lower().split()
        
        # Simple phoneme mapping (mock)
        phoneme_map = {
            'hello': ['h', 'ɛ', 'l', 'oʊ'],
            'world': ['w', 'ɜr', 'l', 'd'],
            'the': ['ð', 'ə'],
            'quick': ['k', 'w', 'ɪ', 'k'],
            'brown': ['b', 'r', 'aʊ', 'n'],
        }
        
        for word in words:
            if word in phoneme_map:
                phonemes.extend(phoneme_map[word])
            else:
                # Generate phonemes based on letters (very basic)
                for char in word:
                    if char.isalpha():
                        phonemes.append(char)
        
        return {
            "success": True,
            "step": 1,
            "step_name": "Phonemization",
            "input_text": text,
            "language": language,
            "phonemes": phonemes,
            "phoneme_count": len(phonemes),
            "word_count": len(words)
        }
        
    except Exception as e:
        logger.error(f"Phonemization failed: {str(e)}")
        return {
            "success": False,
            "step": 1,
            "error": str(e)
        }

# Step 2: Forced Alignment
@app.post("/api/v1/alignment/align", tags=["Step 2: Forced Alignment"])
async def forced_alignment(request: dict):
    """
    Step 2: Align audio with text to get timing
    """
    try:
        text = request.get("text", "")
        # audio_file would be processed here
        
        # Mock alignment result
        words = text.lower().split()
        alignment = {
            "words": [],
            "phonemes": [],
            "total_duration": len(words) * 0.6
        }
        
        current_time = 0.0
        for word in words:
            word_duration = 0.5 + (len(word) * 0.05)
            alignment["words"].append({
                "word": word,
                "start_time": current_time,
                "end_time": current_time + word_duration,
                "confidence": 0.85
            })
            current_time += word_duration + 0.1
        
        return {
            "success": True,
            "step": 2,
            "step_name": "Forced Alignment",
            "input_text": text,
            "alignment": alignment,
            "alignment_quality": 0.85
        }
        
    except Exception as e:
        logger.error(f"Forced alignment failed: {str(e)}")
        return {
            "success": False,
            "step": 2,
            "error": str(e)
        }

# Step 3: Enhanced ASR
@app.post("/api/v1/asr/transcribe", tags=["Step 3: Enhanced ASR"])
async def enhanced_asr(request: dict):
    """
    Step 3: Transcribe audio with pronunciation analysis
    """
    try:
        reference_text = request.get("reference_text", "")
        # audio_file would be processed here
        
        # Mock ASR result
        transcribed_text = reference_text  # Mock: assume perfect transcription
        
        return {
            "success": True,
            "step": 3,
            "step_name": "Enhanced ASR",
            "reference_text": reference_text,
            "transcribed_text": transcribed_text,
            "pronunciation_score": 85.5,
            "fluency_score": 78.2,
            "accuracy_score": 82.1,
            "overall_confidence": 0.87,
            "word_error_rate": 0.05,
            "processing_time_ms": 1250
        }
        
    except Exception as e:
        logger.error(f"Enhanced ASR failed: {str(e)}")
        return {
            "success": False,
            "step": 3,
            "error": str(e)
        }

# Step 4: Comprehensive Scoring
@app.post("/api/v1/assessment/comprehensive", tags=["Step 4: Comprehensive Scoring"])
async def comprehensive_assessment(request: dict):
    """
    Step 4: Combine all steps for final assessment
    """
    try:
        text = request.get("text", "")
        
        # This would combine results from Steps 1, 2, 3
        # For now, mock the final assessment
        
        final_assessment = {
            "overall_score": 82.6,
            "pronunciation_grade": "B+",
            "fluency_grade": "B",
            "accuracy_grade": "B+",
            "step_results": {
                "step_1_phonemization": "completed",
                "step_2_alignment": "completed", 
                "step_3_asr": "completed",
                "step_4_scoring": "completed"
            },
            "detailed_feedback": {
                "strengths": [
                    "Clear pronunciation of most phonemes",
                    "Good overall fluency"
                ],
                "improvements": [
                    "Work on /θ/ and /ð/ sounds",
                    "Reduce pause frequency"
                ],
                "recommendations": [
                    "Practice with tongue twisters",
                    "Focus on connected speech"
                ]
            },
            "next_level_suggestions": [
                "Advanced prosody practice",
                "Stress pattern exercises"
            ]
        }
        
        return {
            "success": True,
            "step": 4,
            "step_name": "Comprehensive Scoring",
            "input_text": text,
            "assessment": final_assessment,
            "pipeline_complete": True
        }
        
    except Exception as e:
        logger.error(f"Comprehensive assessment failed: {str(e)}")
        return {
            "success": False,
            "step": 4,
            "error": str(e)
        }

# Full Pipeline Endpoint
@app.post("/api/v1/pipeline/full-assessment", tags=["Full Pipeline"])
async def full_pronunciation_pipeline(request: dict):
    """
    Run complete 4-step pronunciation assessment pipeline
    """
    try:
        text = request.get("text", "")
        
        # Step 1: Phonemization
        step1_result = await phonemize_text({"text": text})
        
        # Step 2: Forced Alignment
        step2_result = await forced_alignment({"text": text})
        
        # Step 3: Enhanced ASR
        step3_result = await enhanced_asr({"reference_text": text})
        
        # Step 4: Comprehensive Scoring
        step4_result = await comprehensive_assessment({"text": text})
        
        return {
            "success": True,
            "pipeline": "4-Step Pronunciation Assessment",
            "input_text": text,
            "results": {
                "step_1_phonemization": step1_result,
                "step_2_alignment": step2_result,
                "step_3_asr": step3_result,
                "step_4_assessment": step4_result
            },
            "pipeline_summary": {
                "all_steps_completed": True,
                "final_score": step4_result.get("assessment", {}).get("overall_score", 0),
                "final_grade": step4_result.get("assessment", {}).get("pronunciation_grade", "N/A")
            }
        }
        
    except Exception as e:
        logger.error(f"Full pipeline failed: {str(e)}")
        return {
            "success": False,
            "pipeline": "4-Step Pronunciation Assessment", 
            "error": str(e)
        }

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "message": str(exc) if SimpleConfig.DEBUG else "An error occurred"
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=SimpleConfig.HOST,
        port=SimpleConfig.PORT,
        reload=SimpleConfig.DEBUG
    )