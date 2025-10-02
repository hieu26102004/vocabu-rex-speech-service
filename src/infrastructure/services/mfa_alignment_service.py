"""
Montreal Forced Aligner (MFA) implementation
Concrete implementation of forced alignment using MFA
"""

import asyncio
import json
import logging
import os
import shutil
import tempfile
import subprocess
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
import soundfile as sf
import librosa
import numpy as np

from src.domain.services.alignment_service import IAlignmentService, IAlignmentModelManager
from src.domain.entities.alignment_entities import (
    AlignmentResult,
    AlignmentStatistics,
    PhonemeAlignment,
    WordAlignment,
    SentenceAlignment,
    AlignmentQuality
)

logger = logging.getLogger(__name__)


class MFAAlignmentService(IAlignmentService):
    """
    Montreal Forced Aligner implementation
    """
    
    def __init__(self, 
                 mfa_executable_path: str = "mfa",
                 models_directory: str = None,
                 temp_directory: str = None):
        """
        Initialize MFA alignment service
        
        Args:
            mfa_executable_path: Path to MFA executable
            models_directory: Directory for MFA models
            temp_directory: Temporary directory for processing
        """
        self.mfa_executable = mfa_executable_path
        self.models_dir = Path(models_directory) if models_directory else Path.home() / ".mfa"
        self.temp_dir = Path(temp_directory) if temp_directory else Path(tempfile.gettempdir()) / "mfa_temp"
        
        # Create directories if they don't exist
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Default models for different languages
        self.default_models = {
            "english": "english_mfa",
            "en-us": "english_us_mfa", 
            "en-uk": "english_uk_mfa"
        }
    
    async def align_audio_with_text(
        self,
        audio_file_path: str,
        text: str,
        language: str = "english",
        model_name: Optional[str] = None
    ) -> AlignmentResult:
        """
        Perform forced alignment using MFA
        """
        start_time = datetime.now()
        
        try:
            # Validate inputs
            audio_path = Path(audio_file_path)
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
            
            if not await self.validate_audio_file(audio_file_path):
                raise ValueError(f"Invalid audio file: {audio_file_path}")
            
            # Preprocess audio
            processed_audio_path = await self.preprocess_audio(audio_file_path)
            
            # Get model name
            if not model_name:
                model_name = self.default_models.get(language, "english_mfa")
            
            # Create temporary workspace
            workspace_dir = self.temp_dir / f"alignment_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            workspace_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                # Prepare MFA input files
                audio_input_path = workspace_dir / "audio.wav"
                text_input_path = workspace_dir / "audio.txt"
                output_dir = workspace_dir / "output"
                
                # Copy audio file
                shutil.copy2(processed_audio_path, audio_input_path)
                
                # Write text file
                with open(text_input_path, 'w', encoding='utf-8') as f:
                    f.write(text.strip())
                
                # Run MFA alignment
                alignment_data = await self._run_mfa_alignment(
                    workspace_dir, output_dir, model_name, language
                )
                
                # Parse alignment results
                result = await self._parse_mfa_output(
                    alignment_data, audio_file_path, text, language, model_name
                )
                
                # Calculate processing time
                processing_time = (datetime.now() - start_time).total_seconds() * 1000
                result.processing_time_ms = processing_time
                
                return result
                
            finally:
                # Cleanup temporary files
                if workspace_dir.exists():
                    shutil.rmtree(workspace_dir, ignore_errors=True)
                
                # Remove processed audio if it's a temp file
                if processed_audio_path != audio_file_path and Path(processed_audio_path).exists():
                    Path(processed_audio_path).unlink(missing_ok=True)
        
        except Exception as e:
            logger.error(f"Alignment failed: {e}")
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return AlignmentResult(
                audio_file_path=audio_file_path,
                text=text,
                language=language,
                model_name=model_name or "unknown",
                sentence_alignment=SentenceAlignment(
                    text=text,
                    total_duration=0.0,
                    words=[],
                    quality=AlignmentQuality.POOR,
                    overall_confidence=0.0,
                    silence_segments=[]
                ),
                processing_time_ms=processing_time,
                timestamp=datetime.now(),
                metadata={"error": str(e)},
                success=False,
                error_message=str(e)
            )
    
    async def _run_mfa_alignment(
        self, 
        workspace_dir: Path, 
        output_dir: Path, 
        model_name: str,
        language: str
    ) -> Dict[str, Any]:
        """
        Execute MFA alignment command
        """
        try:
            # MFA alignment command
            cmd = [
                self.mfa_executable,
                "align",
                str(workspace_dir),
                model_name,
                model_name,  # Use same model for dictionary
                str(output_dir),
                "--clean",
                "--verbose"
            ]
            
            # Run MFA command
            logger.info(f"Running MFA command: {' '.join(cmd)}")
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=workspace_dir
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                error_msg = stderr.decode('utf-8') if stderr else "Unknown MFA error"
                raise RuntimeError(f"MFA alignment failed: {error_msg}")
            
            # Read TextGrid output
            textgrid_files = list(output_dir.glob("*.TextGrid"))
            if not textgrid_files:
                raise RuntimeError("No TextGrid output generated by MFA")
            
            textgrid_path = textgrid_files[0]
            return await self._parse_textgrid(textgrid_path)
            
        except Exception as e:
            logger.error(f"MFA execution failed: {e}")
            raise
    
    async def _parse_textgrid(self, textgrid_path: Path) -> Dict[str, Any]:
        """
        Parse MFA TextGrid output
        """
        try:
            import textgrid
            
            # Read TextGrid file
            tg = textgrid.TextGrid.fromFile(str(textgrid_path))
            
            alignment_data = {
                "words": [],
                "phones": [],
                "total_duration": 0.0
            }
            
            # Extract word tier
            word_tier = None
            phone_tier = None
            
            for tier in tg.tiers:
                if tier.name.lower() in ["words", "word"]:
                    word_tier = tier
                elif tier.name.lower() in ["phones", "phone"]:
                    phone_tier = tier
            
            if word_tier:
                for interval in word_tier:
                    if interval.mark and interval.mark.strip():
                        alignment_data["words"].append({
                            "word": interval.mark.strip(),
                            "start_time": interval.minTime,
                            "end_time": interval.maxTime
                        })
                        
                alignment_data["total_duration"] = max(
                    alignment_data["total_duration"],
                    word_tier[-1].maxTime if word_tier else 0.0
                )
            
            if phone_tier:
                for interval in phone_tier:
                    if interval.mark and interval.mark.strip():
                        alignment_data["phones"].append({
                            "phone": interval.mark.strip(),
                            "start_time": interval.minTime,
                            "end_time": interval.maxTime
                        })
            
            return alignment_data
            
        except ImportError:
            # Fallback: parse TextGrid manually
            return await self._parse_textgrid_manual(textgrid_path)
        except Exception as e:
            logger.error(f"TextGrid parsing failed: {e}")
            raise
    
    async def _parse_textgrid_manual(self, textgrid_path: Path) -> Dict[str, Any]:
        """
        Manual TextGrid parsing fallback
        """
        alignment_data = {
            "words": [],
            "phones": [],
            "total_duration": 0.0
        }
        
        try:
            with open(textgrid_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Simple regex-based parsing (basic implementation)
            import re
            
            # Find intervals
            interval_pattern = r'intervals \[(\d+)\]:\s*xmin = ([\d.]+)\s*xmax = ([\d.]+)\s*text = "([^"]*)"'
            matches = re.findall(interval_pattern, content, re.MULTILINE | re.DOTALL)
            
            for match in matches:
                _, start_time, end_time, text = match
                start_time = float(start_time)
                end_time = float(end_time)
                
                if text.strip():
                    # Determine if it's a word or phone based on content
                    if len(text.strip()) > 3 or ' ' in text.strip():
                        alignment_data["words"].append({
                            "word": text.strip(),
                            "start_time": start_time,
                            "end_time": end_time
                        })
                    else:
                        alignment_data["phones"].append({
                            "phone": text.strip(),
                            "start_time": start_time,
                            "end_time": end_time
                        })
                    
                    alignment_data["total_duration"] = max(
                        alignment_data["total_duration"], end_time
                    )
            
            return alignment_data
            
        except Exception as e:
            logger.error(f"Manual TextGrid parsing failed: {e}")
            # Return empty structure
            return alignment_data
    
    async def _parse_mfa_output(
        self,
        alignment_data: Dict[str, Any],
        audio_file_path: str,
        text: str,
        language: str,
        model_name: str
    ) -> AlignmentResult:
        """
        Convert MFA output to AlignmentResult
        """
        words = []
        current_phone_idx = 0
        phones = alignment_data.get("phones", [])
        
        # Process words and their phonemes
        for word_data in alignment_data.get("words", []):
            word_start = word_data["start_time"]
            word_end = word_data["end_time"]
            word_text = word_data["word"]
            
            # Find phonemes for this word
            word_phonemes = []
            while (current_phone_idx < len(phones) and 
                   phones[current_phone_idx]["start_time"] < word_end):
                
                phone_data = phones[current_phone_idx]
                
                # Only include phonemes that overlap with word timing
                if phone_data["end_time"] > word_start:
                    phoneme = PhonemeAlignment(
                        phoneme=phone_data["phone"],
                        start_time=phone_data["start_time"],
                        end_time=phone_data["end_time"],
                        confidence=0.8,  # Default confidence for MFA
                        duration=0.0  # Will be calculated in __post_init__
                    )
                    word_phonemes.append(phoneme)
                
                current_phone_idx += 1
            
            # Create word alignment
            word_alignment = WordAlignment(
                word=word_text,
                start_time=word_start,
                end_time=word_end,
                phonemes=word_phonemes,
                confidence=0.8,  # Default confidence
                duration=0.0  # Will be calculated in __post_init__
            )
            words.append(word_alignment)
        
        # Calculate overall confidence and quality
        overall_confidence = 0.8  # MFA generally provides good alignment
        quality = AlignmentQuality.GOOD
        
        if overall_confidence >= 0.9:
            quality = AlignmentQuality.EXCELLENT
        elif overall_confidence >= 0.7:
            quality = AlignmentQuality.GOOD
        elif overall_confidence >= 0.5:
            quality = AlignmentQuality.FAIR
        else:
            quality = AlignmentQuality.POOR
        
        # Create sentence alignment
        sentence_alignment = SentenceAlignment(
            text=text,
            total_duration=alignment_data.get("total_duration", 0.0),
            words=words,
            quality=quality,
            overall_confidence=overall_confidence,
            silence_segments=[]  # TODO: Extract silence segments
        )
        
        return AlignmentResult(
            audio_file_path=audio_file_path,
            text=text,
            language=language,
            model_name=model_name,
            sentence_alignment=sentence_alignment,
            processing_time_ms=0.0,  # Will be set by caller
            timestamp=datetime.now(),
            metadata={
                "mfa_version": "3.0.0",
                "word_count": len(words),
                "phoneme_count": sum(len(w.phonemes) for w in words)
            },
            success=True
        )
    
    async def batch_align(
        self,
        audio_text_pairs: List[tuple],
        language: str = "english",
        model_name: Optional[str] = None
    ) -> List[AlignmentResult]:
        """
        Batch alignment using MFA
        """
        results = []
        
        for audio_path, text in audio_text_pairs:
            try:
                result = await self.align_audio_with_text(
                    audio_path, text, language, model_name
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Batch alignment failed for {audio_path}: {e}")
                # Add failed result
                results.append(AlignmentResult(
                    audio_file_path=audio_path,
                    text=text,
                    language=language,
                    model_name=model_name or "unknown",
                    sentence_alignment=SentenceAlignment(
                        text=text, total_duration=0.0, words=[],
                        quality=AlignmentQuality.POOR, overall_confidence=0.0,
                        silence_segments=[]
                    ),
                    processing_time_ms=0.0,
                    timestamp=datetime.now(),
                    metadata={},
                    success=False,
                    error_message=str(e)
                ))
        
        return results
    
    async def align_with_phonemes(
        self,
        audio_file_path: str,
        phonemes: List[str],
        language: str = "english"
    ) -> AlignmentResult:
        """
        Align audio with pre-computed phoneme sequence
        """
        # Convert phonemes back to approximate text for MFA
        # This is a simplified approach - in practice, you might want
        # to use a phoneme-to-text converter
        
        # For now, create a placeholder text
        text = " ".join(phonemes)  # Simplified
        
        return await self.align_audio_with_text(
            audio_file_path, text, language
        )
    
    async def get_supported_languages(self) -> List[str]:
        """
        Get supported languages for MFA
        """
        return [
            "english", "en-us", "en-uk", "spanish", "french", 
            "german", "italian", "portuguese", "russian", "chinese"
        ]
    
    async def get_available_models(self, language: str) -> List[str]:
        """
        Get available MFA models for language
        """
        try:
            cmd = [self.mfa_executable, "model", "list"]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                output = stdout.decode('utf-8')
                # Parse model list (simplified)
                models = []
                for line in output.split('\n'):
                    if language.lower() in line.lower():
                        models.append(line.strip())
                return models
            else:
                logger.warning(f"Failed to list MFA models: {stderr.decode('utf-8')}")
                return [self.default_models.get(language, "english_mfa")]
                
        except Exception as e:
            logger.error(f"Error getting MFA models: {e}")
            return [self.default_models.get(language, "english_mfa")]
    
    async def validate_audio_file(self, audio_file_path: str) -> bool:
        """
        Validate audio file for MFA
        """
        try:
            audio_path = Path(audio_file_path)
            if not audio_path.exists():
                return False
            
            # Check file extension
            if audio_path.suffix.lower() not in ['.wav', '.flac', '.mp3']:
                return False
            
            # Load and validate audio
            data, sample_rate = sf.read(audio_file_path)
            
            # Check duration (should be > 0.1s, < 30min)
            duration = len(data) / sample_rate
            if duration < 0.1 or duration > 1800:
                return False
            
            # Check sample rate (should be reasonable)
            if sample_rate < 8000 or sample_rate > 48000:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Audio validation failed: {e}")
            return False
    
    async def preprocess_audio(
        self,
        audio_file_path: str,
        output_path: Optional[str] = None
    ) -> str:
        """
        Preprocess audio for MFA
        """
        try:
            # Load audio
            data, sample_rate = librosa.load(audio_file_path, sr=16000, mono=True)
            
            # Normalize audio
            data = librosa.util.normalize(data)
            
            # Remove leading/trailing silence
            data, _ = librosa.effects.trim(data, top_db=20)
            
            # Create output path
            if not output_path:
                output_path = str(self.temp_dir / f"preprocessed_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.wav")
            
            # Save preprocessed audio
            sf.write(output_path, data, 16000)
            
            return output_path
            
        except Exception as e:
            logger.error(f"Audio preprocessing failed: {e}")
            # Return original path if preprocessing fails
            return audio_file_path
    
    async def calculate_alignment_statistics(
        self,
        result: AlignmentResult
    ) -> AlignmentStatistics:
        """
        Calculate alignment statistics
        """
        return AlignmentStatistics.from_alignment_result(result)
    
    async def export_alignment_to_textgrid(
        self,
        result: AlignmentResult,
        output_path: str
    ) -> bool:
        """
        Export alignment to TextGrid format
        """
        try:
            import textgrid
            
            # Create TextGrid
            tg = textgrid.TextGrid(
                minTime=0.0,
                maxTime=result.sentence_alignment.total_duration
            )
            
            # Word tier
            word_tier = textgrid.IntervalTier(
                name="words",
                minTime=0.0,
                maxTime=result.sentence_alignment.total_duration
            )
            
            for word in result.sentence_alignment.words:
                interval = textgrid.Interval(
                    minTime=word.start_time,
                    maxTime=word.end_time,
                    mark=word.word
                )
                word_tier.append(interval)
            
            # Phone tier
            phone_tier = textgrid.IntervalTier(
                name="phones",
                minTime=0.0,
                maxTime=result.sentence_alignment.total_duration
            )
            
            for word in result.sentence_alignment.words:
                for phoneme in word.phonemes:
                    interval = textgrid.Interval(
                        minTime=phoneme.start_time,
                        maxTime=phoneme.end_time,
                        mark=phoneme.phoneme
                    )
                    phone_tier.append(interval)
            
            tg.append(word_tier)
            tg.append(phone_tier)
            
            # Write TextGrid file
            tg.write(output_path)
            return True
            
        except Exception as e:
            logger.error(f"TextGrid export failed: {e}")
            return False
    
    async def export_alignment_to_json(
        self,
        result: AlignmentResult,
        output_path: str
    ) -> bool:
        """
        Export alignment to JSON format
        """
        try:
            # Convert AlignmentResult to dict
            export_data = {
                "audio_file_path": result.audio_file_path,
                "text": result.text,
                "language": result.language,
                "model_name": result.model_name,
                "success": result.success,
                "processing_time_ms": result.processing_time_ms,
                "timestamp": result.timestamp.isoformat(),
                "sentence_alignment": {
                    "text": result.sentence_alignment.text,
                    "total_duration": result.sentence_alignment.total_duration,
                    "quality": result.sentence_alignment.quality.value,
                    "overall_confidence": result.sentence_alignment.overall_confidence,
                    "words": [
                        {
                            "word": word.word,
                            "start_time": word.start_time,
                            "end_time": word.end_time,
                            "duration": word.duration,
                            "confidence": word.confidence,
                            "phonemes": [
                                {
                                    "phoneme": p.phoneme,
                                    "start_time": p.start_time,
                                    "end_time": p.end_time,
                                    "duration": p.duration,
                                    "confidence": p.confidence
                                }
                                for p in word.phonemes
                            ]
                        }
                        for word in result.sentence_alignment.words
                    ]
                },
                "metadata": result.metadata
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            return True
            
        except Exception as e:
            logger.error(f"JSON export failed: {e}")
            return False


class MFAModelManager(IAlignmentModelManager):
    """
    MFA Model Manager implementation
    """
    
    def __init__(self, mfa_executable_path: str = "mfa"):
        self.mfa_executable = mfa_executable_path
    
    async def download_model(self, language: str, model_name: str) -> bool:
        """
        Download MFA model
        """
        try:
            cmd = [self.mfa_executable, "model", "download", "acoustic", model_name]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                logger.info(f"Successfully downloaded model: {model_name}")
                return True
            else:
                logger.error(f"Model download failed: {stderr.decode('utf-8')}")
                return False
                
        except Exception as e:
            logger.error(f"Model download error: {e}")
            return False
    
    async def list_installed_models(self) -> Dict[str, List[str]]:
        """
        List installed MFA models
        """
        try:
            cmd = [self.mfa_executable, "model", "list", "acoustic"]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                output = stdout.decode('utf-8')
                # Parse model list (simplified implementation)
                models = {}
                current_language = "english"
                
                for line in output.split('\n'):
                    line = line.strip()
                    if line and not line.startswith('-'):
                        if current_language not in models:
                            models[current_language] = []
                        models[current_language].append(line)
                
                return models
            else:
                logger.warning(f"Failed to list models: {stderr.decode('utf-8')}")
                return {}
                
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return {}
    
    async def get_model_info(self, language: str, model_name: str) -> Dict[str, Any]:
        """
        Get MFA model information
        """
        return {
            "name": model_name,
            "language": language,
            "type": "acoustic",
            "source": "mfa",
            "status": "available"
        }
    
    async def remove_model(self, language: str, model_name: str) -> bool:
        """
        Remove MFA model
        """
        try:
            # MFA doesn't have a direct remove command, but we can try
            logger.warning(f"Model removal not directly supported for: {model_name}")
            return False
            
        except Exception as e:
            logger.error(f"Model removal error: {e}")
            return False