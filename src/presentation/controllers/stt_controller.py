"""
STT Controller (Lightweight)
Simple Speech-to-Text endpoint optimized for voice call (low-latency, no pronunciation scoring).
"""

import logging
import tempfile
import os
from typing import Optional

from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

stt_router = APIRouter(prefix="/stt", tags=["Speech-to-Text"])

# Global whisper service instance (set during app startup)
_whisper_service = None


class STTResponse(BaseModel):
    """Simple STT response for voice call."""
    text: str
    confidence: float
    language: str
    duration_seconds: float


@stt_router.post("/transcribe", response_model=STTResponse)
async def transcribe_audio(
    audio_file: UploadFile = File(..., description="Audio file (WAV, MP3, M4A, etc.)"),
    language: Optional[str] = Form(default="en", description="Language code"),
):
    """
    Simple speech-to-text transcription optimized for voice call.
    No pronunciation scoring — just fast, accurate transcription.
    """
    if _whisper_service is None:
        raise HTTPException(
            status_code=503,
            detail="STT service not available. Whisper model may still be loading."
        )

    tmp_path = None
    try:
        # Save uploaded audio to temp file
        content = await audio_file.read()
        suffix = os.path.splitext(audio_file.filename or "audio.wav")[1] or ".wav"
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        # Use whisper service for transcription (fast mode)
        result = await _whisper_service.transcribe_simple(
            audio_path=tmp_path,
            language=language,
        )

        return STTResponse(
            text=result.get("text", "").strip(),
            confidence=result.get("confidence", 0.0),
            language=result.get("language", language),
            duration_seconds=result.get("duration", 0.0),
        )

    except Exception as e:
        logger.error(f"STT transcription error: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


@stt_router.get("/status")
async def stt_status():
    """Get STT service status."""
    if _whisper_service is None:
        return {
            "service": "Whisper STT",
            "loaded": False,
            "device": "unknown",
        }
    return {
        "service": "Whisper STT",
        "loaded": True,
        "device": getattr(_whisper_service, "device", "unknown"),
        "model": getattr(_whisper_service, "model_name", "unknown"),
    }
