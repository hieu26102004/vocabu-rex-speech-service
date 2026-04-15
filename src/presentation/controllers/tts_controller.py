"""
TTS Controller
Text-to-Speech endpoints for the VibeVoice service.
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

tts_router = APIRouter(prefix="/tts", tags=["Text-to-Speech"])

# Global TTS service instance (set during app startup)
_tts_service = None


class TTSSynthesizeRequest(BaseModel):
    """Request body for TTS synthesis."""
    text: str = Field(..., min_length=1, max_length=2000, description="Text to synthesize")
    voice_style: Optional[str] = Field(default="friendly", description="Voice style")


class TTSStatusResponse(BaseModel):
    """TTS service status."""
    service: str
    loaded: bool
    device: str
    model: str
    sample_rate: int


@tts_router.post("/synthesize", response_class=Response)
async def synthesize_speech(request: TTSSynthesizeRequest):
    """
    Synthesize speech from text. Returns complete WAV audio.
    """
    if _tts_service is None or not _tts_service.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="TTS service not available. Model may still be loading."
        )

    try:
        audio_bytes = await _tts_service.synthesize(request.text)
        return Response(
            content=audio_bytes,
            media_type="audio/wav",
            headers={
                "Content-Disposition": "inline; filename=speech.wav",
                "Content-Length": str(len(audio_bytes)),
            }
        )
    except Exception as e:
        logger.error(f"TTS synthesis error: {e}")
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {str(e)}")


@tts_router.post("/stream")
async def synthesize_stream(request: TTSSynthesizeRequest):
    """
    Synthesize speech from text with streaming response.
    Returns raw PCM audio chunks (16-bit, mono) as they are generated.
    Useful for low-latency playback.
    """
    if _tts_service is None or not _tts_service.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="TTS service not available. Model may still be loading."
        )

    async def audio_generator():
        try:
            async for chunk in _tts_service.synthesize_stream(request.text):
                yield chunk
        except Exception as e:
            logger.error(f"TTS streaming error: {e}")

    return StreamingResponse(
        audio_generator(),
        media_type="audio/pcm",
        headers={
            "X-Sample-Rate": str(_tts_service.sample_rate),
            "X-Channels": "1",
            "X-Bits-Per-Sample": "16",
        }
    )


@tts_router.get("/status", response_model=TTSStatusResponse)
async def tts_status():
    """Get TTS service status."""
    if _tts_service is None:
        return TTSStatusResponse(
            service="VibeVoice TTS",
            loaded=False,
            device="unknown",
            model="not initialized",
            sample_rate=24000,
        )
    return TTSStatusResponse(**_tts_service.status)


@tts_router.get("/voices")
async def list_voices():
    """List available voice styles."""
    return {
        "voices": [
            {"id": "friendly", "name": "Friendly (Rex)", "description": "Warm and encouraging voice for learning"},
            {"id": "professional", "name": "Professional", "description": "Clear and articulate"},
            {"id": "casual", "name": "Casual", "description": "Relaxed conversational tone"},
        ],
        "default": "friendly"
    }
