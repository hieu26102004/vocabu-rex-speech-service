"""
Use cases for forced alignment functionality
Business logic orchestration for Step 2: Forced Alignment
"""

import asyncio
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
from pathlib import Path

from src.domain.services.alignment_service import IAlignmentService, IAlignmentModelManager
from src.domain.entities.alignment_entities import (
    AlignmentResult,
    AlignmentStatistics,
    PhonemeAlignment,
    WordAlignment,
    SentenceAlignment,
    AlignmentQuality
)
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
    AlignmentStatisticsDTO,
    SentenceAlignmentResponseDTO,
    WordAlignmentResponseDTO,
    PhonemeAlignmentResponseDTO,
    AlignmentConfigDTO,
    AlignmentProgressDTO
)
from src.infrastructure.utils.audio_processor import AudioProcessor, is_audio_file_valid

logger = logging.getLogger(__name__)


class AlignmentUseCases:
    """
    Use cases for forced alignment operations
    """
    
    def __init__(
        self, 
        alignment_service: IAlignmentService,
        model_manager: Optional[IAlignmentModelManager] = None,
        audio_processor: Optional[AudioProcessor] = None
    ):
        """
        Initialize alignment use cases
        
        Args:
            alignment_service: Alignment service implementation
            model_manager: Model manager for downloading/managing models
            audio_processor: Audio processing utilities
        """
        self.alignment_service = alignment_service
        self.model_manager = model_manager
        self.audio_processor = audio_processor or AudioProcessor()
        
        # Configuration
        self.config = AlignmentConfigDTO()
        
        # Task tracking for long-running operations
        self.active_tasks: Dict[str, AlignmentProgressDTO] = {}
    
    async def align_audio_with_text(
        self, 
        audio_file_path: str,
        request: AlignmentRequest
    ) -> AlignmentResponseDTO:
        """
        Perform forced alignment on audio file with text
        
        Args:
            audio_file_path: Path to audio file
            request: Alignment request parameters
            
        Returns:
            Alignment result with timing information
        """
        try:
            logger.info(f"Starting alignment for: {audio_file_path}")
            
            # Validate audio file first
            if not await self._validate_audio_file_internal(audio_file_path):
                raise ValueError(f"Invalid audio file: {audio_file_path}")
            
            # Preprocess audio if requested
            processed_audio_path = audio_file_path
            if request.preprocess_audio:
                processed_audio_path = self.audio_processor.preprocess_for_alignment(audio_file_path)
            
            # Perform alignment
            alignment_result = await self.alignment_service.align_audio_with_text(
                audio_file_path=processed_audio_path,
                text=request.text,
                language=request.language,
                model_name=request.model_name
            )
            
            # Calculate statistics
            statistics = await self.alignment_service.calculate_alignment_statistics(alignment_result)
            
            # Convert to response DTO
            response = self._convert_alignment_result_to_dto(alignment_result, statistics)
            
            logger.info(f"Alignment completed successfully for: {audio_file_path}")
            return response
            
        except Exception as e:
            logger.error(f"Alignment failed for {audio_file_path}: {e}")
            
            # Return error response
            return AlignmentResponseDTO(
                success=False,
                audio_file_path=audio_file_path,
                text=request.text,
                language=request.language,
                model_name=request.model_name or "unknown",
                sentence_alignment=SentenceAlignmentResponseDTO(
                    text=request.text,
                    total_duration=0.0,
                    words=[],
                    quality=AlignmentQuality.POOR,
                    overall_confidence=0.0,
                    speech_rate=0.0,
                    phoneme_rate=0.0,
                    silence_segments=[]
                ),
                statistics=AlignmentStatisticsDTO(
                    total_words=0,
                    total_phonemes=0,
                    total_duration=0.0,
                    average_word_duration=0.0,
                    average_phoneme_duration=0.0,
                    confidence_distribution={},
                    timing_accuracy=0.0,
                    speech_tempo="unknown"
                ),
                processing_time_ms=0.0,
                timestamp=datetime.now(),
                error_message=str(e)
            )
    
    async def batch_align(self, request: BatchAlignmentRequest) -> BatchAlignmentResponseDTO:
        """
        Perform batch alignment on multiple audio-text pairs
        
        Args:
            request: Batch alignment request
            
        Returns:
            Batch alignment results
        """
        start_time = datetime.now()
        
        try:
            logger.info(f"Starting batch alignment for {len(request.items)} items")
            
            results = []
            successful_items = 0
            failed_items = 0
            
            if request.parallel_processing and len(request.items) > 1:
                # Process in parallel
                tasks = []
                for item in request.items:
                    alignment_request = AlignmentRequest(
                        text=item['text'],
                        language=request.language,
                        model_name=request.model_name
                    )
                    task = self.align_audio_with_text(item['audio_file'], alignment_request)
                    tasks.append(task)
                
                # Wait for all tasks to complete
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Count successes and failures
                for result in results:
                    if isinstance(result, Exception):
                        failed_items += 1
                        # Convert exception to error response
                        # This would need proper error handling
                    elif result.success:
                        successful_items += 1
                    else:
                        failed_items += 1
            else:
                # Process sequentially
                for item in request.items:
                    try:
                        alignment_request = AlignmentRequest(
                            text=item['text'],
                            language=request.language,
                            model_name=request.model_name
                        )
                        result = await self.align_audio_with_text(item['audio_file'], alignment_request)
                        results.append(result)
                        
                        if result.success:
                            successful_items += 1
                        else:
                            failed_items += 1
                            
                    except Exception as e:
                        logger.error(f"Batch item failed: {e}")
                        failed_items += 1
                        # Add error result
                        # This would need proper error result creation
            
            # Calculate total processing time
            total_processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return BatchAlignmentResponseDTO(
                success=successful_items > 0,
                total_items=len(request.items),
                successful_items=successful_items,
                failed_items=failed_items,
                results=results,
                total_processing_time_ms=total_processing_time,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Batch alignment failed: {e}")
            
            total_processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return BatchAlignmentResponseDTO(
                success=False,
                total_items=len(request.items),
                successful_items=0,
                failed_items=len(request.items),
                results=[],
                total_processing_time_ms=total_processing_time,
                timestamp=datetime.now()
            )
    
    async def align_with_phonemes(
        self, 
        audio_file_path: str,
        request: PhonemeAlignmentRequest
    ) -> AlignmentResponseDTO:
        """
        Align audio with pre-computed phoneme sequence
        
        Args:
            audio_file_path: Path to audio file
            request: Phoneme alignment request
            
        Returns:
            Alignment result with phoneme timing
        """
        try:
            logger.info(f"Starting phoneme alignment for: {audio_file_path}")
            
            # Validate audio file
            if not await self._validate_audio_file_internal(audio_file_path):
                raise ValueError(f"Invalid audio file: {audio_file_path}")
            
            # Perform phoneme-based alignment
            alignment_result = await self.alignment_service.align_with_phonemes(
                audio_file_path=audio_file_path,
                phonemes=request.phonemes,
                language=request.language
            )
            
            # Calculate statistics
            statistics = await self.alignment_service.calculate_alignment_statistics(alignment_result)
            
            # Convert to response DTO
            response = self._convert_alignment_result_to_dto(alignment_result, statistics)
            
            logger.info(f"Phoneme alignment completed for: {audio_file_path}")
            return response
            
        except Exception as e:
            logger.error(f"Phoneme alignment failed for {audio_file_path}: {e}")
            raise
    
    async def validate_audio_file(
        self, 
        audio_file_path: str,
        request: AlignmentValidationRequest
    ) -> AudioValidationResponseDTO:
        """
        Validate audio file for alignment suitability
        
        Args:
            audio_file_path: Path to audio file
            request: Validation request parameters
            
        Returns:
            Audio validation results
        """
        try:
            # Basic file existence check
            audio_path = Path(audio_file_path)
            if not audio_path.exists():
                return AudioValidationResponseDTO(
                    valid=False,
                    file_path=audio_file_path,
                    error_message="File does not exist",
                    recommendations=["Check file path", "Ensure file exists"]
                )
            
            # Use audio processor for detailed validation
            validation_result = self.audio_processor.validate_audio_file(audio_file_path)
            
            if not validation_result.get("valid", False):
                return AudioValidationResponseDTO(
                    valid=False,
                    file_path=audio_file_path,
                    error_message=validation_result.get("error", "Unknown validation error"),
                    recommendations=self._get_validation_recommendations(validation_result)
                )
            
            # Extract features if requested
            features = None
            if request.extract_features:
                features = self.audio_processor.extract_audio_features(audio_file_path)
            
            # Build recommendations based on validation results
            recommendations = self._get_validation_recommendations(validation_result)
            
            return AudioValidationResponseDTO(
                valid=True,
                file_path=audio_file_path,
                format=validation_result.get("format"),
                duration=validation_result.get("duration"),
                sample_rate=validation_result.get("sample_rate"),
                channels=validation_result.get("channels"),
                file_size=validation_result.get("file_size"),
                quality_score=validation_result.get("quality_score"),
                features=features,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Audio validation failed for {audio_file_path}: {e}")
            return AudioValidationResponseDTO(
                valid=False,
                file_path=audio_file_path,
                error_message=str(e),
                recommendations=["Check file format", "Verify file integrity"]
            )
    
    async def get_supported_languages(self) -> SupportedLanguagesResponseDTO:
        """
        Get list of supported languages for alignment
        
        Returns:
            Supported languages list
        """
        try:
            languages = await self.alignment_service.get_supported_languages()
            
            return SupportedLanguagesResponseDTO(
                supported_languages=languages,
                total_languages=len(languages),
                default_language=self.config.default_language
            )
            
        except Exception as e:
            logger.error(f"Failed to get supported languages: {e}")
            # Return default list if service fails
            return SupportedLanguagesResponseDTO(
                supported_languages=["english"],
                total_languages=1,
                default_language="english"
            )
    
    async def get_available_models(self, language: str) -> AvailableModelsResponseDTO:
        """
        Get available alignment models for specific language
        
        Args:
            language: Language code
            
        Returns:
            Available models list
        """
        try:
            models = await self.alignment_service.get_available_models(language)
            
            return AvailableModelsResponseDTO(
                language=language,
                models=models,
                default_model=models[0] if models else None,
                total_models=len(models)
            )
            
        except Exception as e:
            logger.error(f"Failed to get available models for {language}: {e}")
            return AvailableModelsResponseDTO(
                language=language,
                models=[],
                total_models=0
            )
    
    async def export_alignment_result(
        self,
        alignment_result: AlignmentResponseDTO,
        output_path: str,
        format: str = "json"
    ) -> ExportResponseDTO:
        """
        Export alignment result to file
        
        Args:
            alignment_result: Alignment result to export
            output_path: Output file path
            format: Export format ("json", "textgrid")
            
        Returns:
            Export operation result
        """
        try:
            # Convert DTO back to domain entity for export
            # This would need a proper converter
            alignment_entity = self._convert_dto_to_alignment_result(alignment_result)
            
            success = False
            if format.lower() == "json":
                success = await self.alignment_service.export_alignment_to_json(
                    alignment_entity, output_path
                )
            elif format.lower() == "textgrid":
                success = await self.alignment_service.export_alignment_to_textgrid(
                    alignment_entity, output_path
                )
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
            if success:
                file_size = Path(output_path).stat().st_size if Path(output_path).exists() else None
                
                return ExportResponseDTO(
                    success=True,
                    export_format=format,
                    output_path=output_path,
                    file_size=file_size
                )
            else:
                return ExportResponseDTO(
                    success=False,
                    export_format=format,
                    output_path=output_path,
                    error_message="Export operation failed"
                )
                
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return ExportResponseDTO(
                success=False,
                export_format=format,
                output_path=output_path,
                error_message=str(e)
            )
    
    # Model management use cases (if model manager is available)
    
    async def download_model(self, language: str, model_name: str) -> ModelInfoResponseDTO:
        """
        Download alignment model for specific language
        
        Args:
            language: Language code
            model_name: Model name to download
            
        Returns:
            Model download result
        """
        if not self.model_manager:
            return ModelInfoResponseDTO(
                name=model_name,
                language=language,
                type="acoustic",
                source="unavailable",
                download_status="error",
                description="Model manager not available"
            )
        
        try:
            success = await self.model_manager.download_model(language, model_name)
            
            return ModelInfoResponseDTO(
                name=model_name,
                language=language,
                type="acoustic",
                source="mfa",
                download_status="installed" if success else "error"
            )
            
        except Exception as e:
            logger.error(f"Model download failed: {e}")
            return ModelInfoResponseDTO(
                name=model_name,
                language=language,
                type="acoustic",
                source="mfa",
                download_status="error",
                description=str(e)
            )
    
    async def list_installed_models(self) -> Dict[str, List[ModelInfoResponseDTO]]:
        """
        List all installed alignment models
        
        Returns:
            Dictionary mapping language -> list of model info
        """
        if not self.model_manager:
            return {}
        
        try:
            models_dict = await self.model_manager.list_installed_models()
            
            # Convert to DTO format
            result = {}
            for language, model_names in models_dict.items():
                result[language] = [
                    ModelInfoResponseDTO(
                        name=model_name,
                        language=language,
                        type="acoustic",
                        source="mfa",
                        download_status="installed"
                    )
                    for model_name in model_names
                ]
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to list installed models: {e}")
            return {}
    
    # Private helper methods
    
    async def _validate_audio_file_internal(self, audio_file_path: str) -> bool:
        """Internal audio validation"""
        try:
            return await self.alignment_service.validate_audio_file(audio_file_path)
        except Exception as e:
            logger.error(f"Internal audio validation failed: {e}")
            return False
    
    def _convert_alignment_result_to_dto(
        self, 
        result: AlignmentResult, 
        statistics: AlignmentStatistics
    ) -> AlignmentResponseDTO:
        """Convert domain entity to response DTO"""
        
        # Convert phoneme alignments
        def convert_phonemes(phonemes: List[PhonemeAlignment]) -> List[PhonemeAlignmentResponseDTO]:
            return [
                PhonemeAlignmentResponseDTO(
                    phoneme=p.phoneme,
                    start_time=p.start_time,
                    end_time=p.end_time,
                    duration=p.duration,
                    confidence=p.confidence
                )
                for p in phonemes
            ]
        
        # Convert word alignments
        words_dto = [
            WordAlignmentResponseDTO(
                word=word.word,
                start_time=word.start_time,
                end_time=word.end_time,
                duration=word.duration,
                confidence=word.confidence,
                phonemes=convert_phonemes(word.phonemes)
            )
            for word in result.sentence_alignment.words
        ]
        
        # Convert sentence alignment
        sentence_dto = SentenceAlignmentResponseDTO(
            text=result.sentence_alignment.text,
            total_duration=result.sentence_alignment.total_duration,
            words=words_dto,
            quality=result.sentence_alignment.quality,
            overall_confidence=result.sentence_alignment.overall_confidence,
            speech_rate=result.sentence_alignment.speech_rate,
            phoneme_rate=result.sentence_alignment.phoneme_rate,
            silence_segments=[
                {"start_time": seg[0], "end_time": seg[1]}
                for seg in result.sentence_alignment.silence_segments
            ]
        )
        
        # Convert statistics
        statistics_dto = AlignmentStatisticsDTO(
            total_words=statistics.total_words,
            total_phonemes=statistics.total_phonemes,
            total_duration=statistics.total_duration,
            average_word_duration=statistics.average_word_duration,
            average_phoneme_duration=statistics.average_phoneme_duration,
            confidence_distribution=statistics.confidence_distribution,
            timing_accuracy=statistics.timing_accuracy,
            speech_tempo=statistics.speech_tempo
        )
        
        return AlignmentResponseDTO(
            success=result.success,
            audio_file_path=result.audio_file_path,
            text=result.text,
            language=result.language,
            model_name=result.model_name,
            sentence_alignment=sentence_dto,
            statistics=statistics_dto,
            processing_time_ms=result.processing_time_ms,
            timestamp=result.timestamp,
            metadata=result.metadata,
            error_message=result.error_message
        )
    
    def _convert_dto_to_alignment_result(self, dto: AlignmentResponseDTO) -> AlignmentResult:
        """Convert response DTO back to domain entity (for export)"""
        # This would be implemented for export functionality
        # For now, return a placeholder
        pass
    
    def _get_validation_recommendations(self, validation_result: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on validation results"""
        recommendations = []
        
        if not validation_result.get("valid", False):
            recommendations.append("File format not suitable for alignment")
            return recommendations
        
        duration = validation_result.get("duration", 0)
        sample_rate = validation_result.get("sample_rate", 0)
        channels = validation_result.get("channels", 1)
        quality_score = validation_result.get("quality_score", 1.0)
        
        if duration < 1.0:
            recommendations.append("Audio is very short - may affect alignment quality")
        elif duration > 60.0:
            recommendations.append("Consider segmenting long audio for better processing")
        
        if sample_rate < 16000:
            recommendations.append("Low sample rate - consider resampling to 16kHz or higher")
        
        if channels > 1:
            recommendations.append("Stereo audio detected - mono audio works better for alignment")
        
        if quality_score < 0.7:
            recommendations.append("Audio quality may affect alignment accuracy")
            recommendations.append("Consider noise reduction preprocessing")
        
        if not recommendations:
            recommendations.append("Audio file is suitable for forced alignment")
        
        return recommendations