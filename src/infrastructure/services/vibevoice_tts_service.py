"""
VibeVoice TTS Service
Text-to-Speech using Microsoft VibeVoice-Realtime-0.5B model.
Designed for low-latency streaming synthesis on GPU.
"""

import io
import logging
import struct
import asyncio
from typing import AsyncGenerator, Optional

import numpy as np
import torch

from src.shared.config import get_settings

logger = logging.getLogger(__name__)


class VibeVoiceTTSService:
    """TTS service using VibeVoice-Realtime-0.5B for low-latency voice synthesis."""

    def __init__(self, device: str = "auto"):
        self.settings = get_settings()
        self.model = None
        self.processor = None
        self.is_loaded = False
        self.sample_rate = getattr(self.settings, 'TTS_SAMPLE_RATE', 24000)

        # Determine device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"VibeVoice TTS Service initialized (device: {self.device})")

    async def preload_model(self) -> bool:
        """Preload the VibeVoice model into memory."""
        try:
            logger.info(f"Loading VibeVoice model: {self.settings.TTS_MODEL_NAME}")
            loop = asyncio.get_event_loop()
            success = await loop.run_in_executor(None, self._load_model_sync)
            if success:
                logger.info("✅ VibeVoice model loaded successfully!")
                self.is_loaded = True
            return success
        except Exception as e:
            logger.error(f"❌ Failed to load VibeVoice model: {e}")
            return False

    def _load_model_sync(self) -> bool:
        """Synchronous model loading."""
        try:
            try:
                from vibevoice.processor.vibevoice_streaming_processor import VibeVoiceStreamingProcessor
                from vibevoice.modular.modeling_vibevoice_streaming_inference import VibeVoiceStreamingForConditionalGenerationInference
            except ImportError:
                logger.warning("vibevoice library not found. Please install it.")
                return False

            model_name = self.settings.TTS_MODEL_NAME
            cache_dir = self.settings.TTS_MODEL_PATH

            logger.info(f"Downloading/loading model from: {model_name}")

            self.processor = VibeVoiceStreamingProcessor.from_pretrained(
                model_name,
                cache_dir=cache_dir,
            )

            model_dtype = torch.float16 if self.device == "cuda" else torch.float32

            self.model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                torch_dtype=model_dtype,
            ).to(self.device)

            self.model.eval()
            self.model.set_ddpm_inference_steps(num_steps=5)
            
            # Load default voice prompt (fallback to any if english is missing)
            import glob
            voice_dir = "c:/TLCN/VibeVoice/demo/voices/streaming_model"
            en_voices = glob.glob(f"{voice_dir}/en*.pt")
            all_voices = glob.glob(f"{voice_dir}/*.pt")
            voice_sample = en_voices[0] if en_voices else (all_voices[0] if all_voices else None)
            
            if voice_sample:
                target_device = self.device if self.device != "cpu" else "cpu"
                self.default_prompt = torch.load(voice_sample, map_location=target_device, weights_only=False)

                # Ensure all floating point tensors match the model's dtype
                def recursive_cast(obj):
                    if isinstance(obj, torch.Tensor):
                        return obj.to(dtype=model_dtype) if obj.is_floating_point() else obj
                    elif isinstance(obj, dict):
                        for k, v in list(obj.items()):
                            obj[k] = recursive_cast(v)
                        return obj
                    elif isinstance(obj, list):
                        for i in range(len(obj)):
                            obj[i] = recursive_cast(obj[i])
                        return obj
                    elif isinstance(obj, tuple):
                        try:
                            return type(obj)(*(recursive_cast(v) for v in obj))
                        except Exception:
                            return tuple(recursive_cast(v) for v in obj)
                    elif hasattr(obj, '__dict__'):
                        for k, v in list(obj.__dict__.items()):
                            setattr(obj, k, recursive_cast(v))
                        return obj
                    return obj

                self.default_prompt = recursive_cast(self.default_prompt)
            else:
                self.default_prompt = None
                logger.warning("No voice prompt found. VibeVoice Streaming requires a voice prompt.")

            logger.info(f"Model loaded on {self.device}")
            self.is_loaded = True

            # Warmup with a short text
            self._synthesize_sync("Hello")
            logger.info("Model warmed up successfully")

            return True
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            logger.info("Consider installing: pip install sentencepiece tiktoken")
            return False

    def _synthesize_sync(self, text: str) -> Optional[np.ndarray]:
        """Synchronous text-to-speech synthesis. Returns audio as numpy array."""
        if not self.is_loaded or self.model is None:
            raise RuntimeError("TTS model not loaded. Call preload_model() first.")

        try:
            # Truncate text if too long
            if len(text) > self.settings.TTS_MAX_TEXT_LENGTH:
                text = text[:self.settings.TTS_MAX_TEXT_LENGTH]

            with torch.no_grad():
                import copy
                cached_prompt = copy.deepcopy(self.default_prompt) if hasattr(self, 'default_prompt') and self.default_prompt else None
                if not cached_prompt:
                    raise RuntimeError("No voice prompt loaded. VibeVoice requires a voice sample for generation.")

                inputs = self.processor.process_input_with_cached_prompt(
                    text=text,
                    cached_prompt=cached_prompt,
                    padding=True,
                    return_tensors="pt",
                    return_attention_mask=True,
                )
                
                # Move tensors to target device
                target_device = self.device if self.device != "cpu" else "cpu"
                for k, v in inputs.items():
                    if torch.is_tensor(v):
                        inputs[k] = v.to(target_device)

                output = self.model.generate(
                    **inputs,
                    max_new_tokens=None,
                    cfg_scale=1.5,
                    tokenizer=self.processor.tokenizer,
                    generation_config={'do_sample': False},
                    all_prefilled_outputs=cached_prompt,
                )
                
                if output.speech_outputs and output.speech_outputs[0] is not None:
                    audio_data = output.speech_outputs[0].cpu().numpy().flatten()
                    return audio_data
                else:
                    return None

        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            raise

    async def synthesize(self, text: str) -> bytes:
        """Text to speech - returns complete WAV audio bytes."""
        loop = asyncio.get_event_loop()
        audio_data = await loop.run_in_executor(None, self._synthesize_sync, text)

        if audio_data is None:
            raise RuntimeError("Synthesis returned no audio data")

        return self._numpy_to_wav_bytes(audio_data)

    async def synthesize_stream(self, text: str) -> AsyncGenerator[bytes, None]:
        """
        Text to speech with streaming - yields audio chunks for low-latency playback.
        Splits text into sentences and synthesizes each chunk.
        """
        sentences = self._split_into_sentences(text)

        for sentence in sentences:
            if not sentence.strip():
                continue
            try:
                loop = asyncio.get_event_loop()
                audio_data = await loop.run_in_executor(
                    None, self._synthesize_sync, sentence
                )
                if audio_data is not None:
                    # Ensure it is properly clipped to prevent overflow crackling
                    if np.issubdtype(audio_data.dtype, np.floating):
                        audio_data = np.clip(audio_data, -1.0, 1.0)
                        
                    # Yield raw PCM bytes (no WAV header) for streaming
                    pcm_bytes = (audio_data * 32767).astype(np.int16).tobytes()
                    yield pcm_bytes
            except Exception as e:
                logger.error(f"Streaming synthesis error for chunk: {e}")
                continue

    def _numpy_to_wav_bytes(self, audio_data: np.ndarray) -> bytes:
        """Convert numpy audio array to WAV bytes."""
        # Normalize floating point to int16
        if np.issubdtype(audio_data.dtype, np.floating):
            audio_data = np.clip(audio_data, -1.0, 1.0)
            audio_data = (audio_data * 32767).astype(np.int16)

        buffer = io.BytesIO()
        num_samples = len(audio_data)
        data_size = num_samples * 2  # 16-bit = 2 bytes per sample

        # Write WAV header
        buffer.write(b'RIFF')
        buffer.write(struct.pack('<I', 36 + data_size))
        buffer.write(b'WAVE')
        buffer.write(b'fmt ')
        buffer.write(struct.pack('<I', 16))  # Chunk size
        buffer.write(struct.pack('<H', 1))   # PCM format
        buffer.write(struct.pack('<H', 1))   # Mono
        buffer.write(struct.pack('<I', self.sample_rate))
        buffer.write(struct.pack('<I', self.sample_rate * 2))  # Byte rate
        buffer.write(struct.pack('<H', 2))   # Block align
        buffer.write(struct.pack('<H', 16))  # Bits per sample
        buffer.write(b'data')
        buffer.write(struct.pack('<I', data_size))
        buffer.write(audio_data.tobytes())

        return buffer.getvalue()

    def _split_into_sentences(self, text: str) -> list:
        """Split text into sentences for streaming synthesis."""
        import re
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        # If no sentence boundaries found, split on commas or just return as-is
        if len(sentences) == 1 and len(text) > 100:
            sentences = re.split(r'(?<=,)\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    @property
    def status(self) -> dict:
        """Get service status."""
        return {
            "service": "VibeVoice TTS",
            "loaded": self.is_loaded,
            "device": self.device,
            "model": self.settings.TTS_MODEL_NAME,
            "sample_rate": self.sample_rate,
        }
