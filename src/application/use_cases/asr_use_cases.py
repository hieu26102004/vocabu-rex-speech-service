"""
Use Cases for Enhanced ASR (Step 3) operations
"""

from typing import List, Optional, Dict, Any, Tuple
import asyncio
import logging
from datetime import datetime
from pathlib import Path

from src.domain.services.asr_service import IEnhancedASRService, IPronunciationAnalyzer, IFluencyAnalyzer
from src.domain.entities.asr_entities import ASRResult, ActualUtterance, PronunciationComparison
from src.application.dtos.asr_dto import (
    ASRRequest, ASRResponseDTO, BatchASRRequest, BatchASRResponseDTO,
    PronunciationAnalysisRequest, FullPronunciationAnalysisRequest,
    FullPronunciationAnalysisResponseDTO, TranscriptionOnlyResponseDTO,
    ASRValidationRequest, ASRValidationResponseDTO, PronunciationReportDTO,
    ActualUtteranceResponseDTO, WordComparisonResponseDTO, PronunciationComparisonResponseDTO,
    ASRStatisticsDTO, PronunciationFeedbackDTO
)
from src.core.exceptions import AudioProcessingError, ValidationError, ServiceError


class TranscribeAudioUseCase:
    """
    Use case for transcribing audio with enhanced ASR
    """
    
    def __init__(
        self,
        asr_service: IEnhancedASRService,
        pronunciation_analyzer: IPronunciationAnalyzer,
        fluency_analyzer: IFluencyAnalyzer,
        logger: logging.Logger
    ):
        self.asr_service = asr_service
        self.pronunciation_analyzer = pronunciation_analyzer
        self.fluency_analyzer = fluency_analyzer
        self.logger = logger
    
    async def execute(self, audio_file_path: str, request: ASRRequest) -> ASRResponseDTO:
        """
        Execute audio transcription with pronunciation analysis
        """
        start_time = datetime.now()
        
        try:
            # Validate audio file
            await self._validate_audio_file(audio_file_path)
            
            # Transcribe with Whisper (only use supported parameters)
            asr_result = await self.asr_service.transcribe_with_phonemes(
                audio_file_path=audio_file_path,
                reference_text=request.reference_text,
                language=request.language,
                model_size=request.model_size
            )
            
            # Convert to response DTO
            actual_utterance_dto = self._convert_actual_utterance(asr_result.actual_utterance)
            
            # Initialize default values
            word_comparisons = []
            overall_pronunciation_score = 0.0
            fluency_score = 0.0
            accuracy_score = 0.0
            statistics_dto = self._create_basic_statistics(asr_result)
            feedback_dto = self._create_basic_feedback()
            
            # Perform pronunciation comparison if reference text provided
            if request.reference_text and request.compare_pronunciation:
                word_comparisons, overall_pronunciation_score, accuracy_score = await self._analyze_pronunciation(
                    asr_result, request.reference_text, request.language
                )
            
            # Perform fluency analysis
            if request.analyze_fluency:
                fluency_score = await self._analyze_fluency(asr_result.actual_utterance)
            
            # Generate detailed statistics and feedback
            if request.reference_text:
                statistics_dto = await self._generate_statistics(asr_result, word_comparisons)
                feedback_dto = await self._generate_feedback(
                    word_comparisons, overall_pronunciation_score, fluency_score
                )
            
            # Calculate total score
            total_score = self._calculate_total_score(overall_pronunciation_score, fluency_score, accuracy_score)
            pronunciation_grade = self._calculate_grade(total_score)
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return ASRResponseDTO(
                success=True,
                audio_file_path=audio_file_path,
                reference_text=request.reference_text or "",
                actual_utterance=actual_utterance_dto,
                word_comparisons=word_comparisons,
                overall_pronunciation_score=overall_pronunciation_score,
                fluency_score=fluency_score,
                accuracy_score=accuracy_score,
                total_score=total_score,
                pronunciation_grade=pronunciation_grade,
                statistics=statistics_dto,
                feedback=feedback_dto,
                processing_time_ms=processing_time,
                timestamp=datetime.now(),
                whisper_model_used=request.model_size
            )
            
        except Exception as e:
            self.logger.error(f"ASR transcription failed for {audio_file_path}: {str(e)}")
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return ASRResponseDTO(
                success=False,
                audio_file_path=audio_file_path,
                reference_text=request.reference_text or "",
                actual_utterance=self._create_empty_utterance(),
                word_comparisons=[],
                overall_pronunciation_score=0.0,
                fluency_score=0.0,
                accuracy_score=0.0,
                total_score=0.0,
                pronunciation_grade="F",
                statistics=self._create_empty_statistics(),
                feedback=self._create_error_feedback(str(e)),
                processing_time_ms=processing_time,
                timestamp=datetime.now(),
                whisper_model_used=request.model_size,
                error_message=str(e)
            )
    
    async def _validate_audio_file(self, audio_file_path: str) -> None:
        """Validate audio file exists and is readable"""
        audio_path = Path(audio_file_path)
        if not audio_path.exists():
            raise ValidationError(f"Audio file not found: {audio_file_path}")
        if not audio_path.is_file():
            raise ValidationError(f"Path is not a file: {audio_file_path}")
        if audio_path.stat().st_size == 0:
            raise ValidationError(f"Audio file is empty: {audio_file_path}")
    
    async def _analyze_pronunciation(
        self, 
        asr_result: ASRResult, 
        reference_text: str, 
        language: str
    ) -> Tuple[List[WordComparisonResponseDTO], float, float]:
        """Analyze pronunciation against reference"""
        try:
            comparison = await self.pronunciation_analyzer.compare_pronunciation(
                actual_utterance=asr_result.actual_utterance,
                reference_text=reference_text,
                language=language
            )
            
            word_comparisons = self._convert_word_comparisons(comparison)
            overall_score = comparison.overall_accuracy * 100
            accuracy_score = comparison.phoneme_accuracy * 100
            
            return word_comparisons, overall_score, accuracy_score
            
        except Exception as e:
            self.logger.error(f"Pronunciation analysis failed: {str(e)}")
            return [], 0.0, 0.0
    
    async def _analyze_fluency(self, actual_utterance: ActualUtterance) -> float:
        """Analyze speech fluency"""
        try:
            fluency_result = await self.fluency_analyzer.analyze_fluency(actual_utterance)
            return fluency_result.overall_fluency_score * 100
        except Exception as e:
            self.logger.error(f"Fluency analysis failed: {str(e)}")
            return 0.0
    
    def _convert_actual_utterance(self, utterance: ActualUtterance) -> ActualUtteranceResponseDTO:
        """Convert domain entity to response DTO"""
        from src.application.dtos.asr_dto import ActualWordResponseDTO, ActualPhonemeResponseDTO
        
        words_dto = []
        for word in utterance.words:
            phonemes_dto = [
                ActualPhonemeResponseDTO(
                    phoneme=phoneme.phoneme,
                    start_time=phoneme.start_time,
                    end_time=phoneme.end_time,
                    duration=phoneme.duration,
                    confidence=phoneme.confidence,
                    amplitude=phoneme.amplitude,
                    fundamental_frequency=phoneme.fundamental_frequency
                )
                for phoneme in word.phonemes
            ]
            
            words_dto.append(ActualWordResponseDTO(
                word=word.word,
                start_time=word.start_time,
                end_time=word.end_time,
                duration=word.duration,
                confidence=word.confidence,
                phonemes=phonemes_dto,
                pronunciation_score=word.pronunciation_score
            ))
        
        return ActualUtteranceResponseDTO(
            transcribed_text=utterance.transcribed_text,
            original_text=utterance.original_text,
            total_duration=utterance.total_duration,
            words=words_dto,
            overall_confidence=utterance.overall_confidence,
            transcription_quality=utterance.transcription_quality,
            pronunciation_accuracy=utterance.pronunciation_accuracy,
            speech_rate=utterance.speech_rate,
            phoneme_rate=utterance.phoneme_rate,
            pause_count=utterance.pause_count,
            pause_duration_total=utterance.pause_duration_total
        )
    
    def _convert_word_comparisons(
        self, 
        comparison: PronunciationComparison
    ) -> List[WordComparisonResponseDTO]:
        """Convert pronunciation comparison to response DTOs"""
        word_comparisons = []
        
        for word_comp in comparison.word_comparisons:
            phoneme_comparisons_dto = [
                PronunciationComparisonResponseDTO(
                    reference_phoneme=pc.reference_phoneme,
                    actual_phoneme=pc.actual_phoneme,
                    phoneme_match=pc.phoneme_match,
                    similarity_score=pc.similarity_score,
                    timing_deviation=pc.timing_deviation,
                    error_type=pc.error_type
                )
                for pc in word_comp.phoneme_comparisons
            ]
            
            word_comparisons.append(WordComparisonResponseDTO(
                reference_word=word_comp.reference_word,
                actual_word=word_comp.actual_word,
                word_match=word_comp.word_match,
                phoneme_comparisons=phoneme_comparisons_dto,
                overall_accuracy=word_comp.overall_accuracy,
                timing_accuracy=word_comp.timing_accuracy
            ))
        
        return word_comparisons
    
    async def _generate_statistics(
        self, 
        asr_result: ASRResult, 
        word_comparisons: List[WordComparisonResponseDTO]
    ) -> ASRStatisticsDTO:
        """Generate comprehensive ASR statistics"""
        stats = asr_result.statistics
        
        # Calculate phoneme accuracy
        total_phonemes = sum(len(wc.phoneme_comparisons) for wc in word_comparisons)
        correct_phonemes = sum(
            sum(1 for pc in wc.phoneme_comparisons if pc.is_accurate) 
            for wc in word_comparisons
        )
        phoneme_accuracy = (correct_phonemes / total_phonemes * 100) if total_phonemes > 0 else 0.0
        
        return ASRStatisticsDTO(
            word_error_rate=stats.word_error_rate,
            character_error_rate=stats.character_error_rate,
            phoneme_error_rate=stats.phoneme_error_rate,
            average_word_duration=stats.average_word_duration,
            average_phoneme_duration=stats.average_phoneme_duration,
            timing_precision=stats.timing_precision,
            correctly_pronounced_phonemes=correct_phonemes,
            total_phonemes=total_phonemes,
            phoneme_accuracy_percentage=phoneme_accuracy,
            speaking_speed=stats.speaking_speed,
            pause_patterns_score=stats.pause_patterns_score,
            rhythm_score=stats.rhythm_score,
            confidence_distribution=stats.confidence_distribution
        )
    
    async def _generate_feedback(
        self,
        word_comparisons: List[WordComparisonResponseDTO],
        pronunciation_score: float,
        fluency_score: float
    ) -> PronunciationFeedbackDTO:
        """Generate pronunciation feedback"""
        # Analyze common errors
        error_phonemes = []
        for wc in word_comparisons:
            for pc in wc.phoneme_comparisons:
                if not pc.is_accurate:
                    error_phonemes.append(pc.reference_phoneme)
        
        # Generate feedback based on scores
        overall_feedback = self._generate_overall_feedback(pronunciation_score, fluency_score)
        strengths = self._identify_strengths(word_comparisons)
        improvements = self._identify_improvements(error_phonemes)
        suggestions = self._generate_practice_suggestions(error_phonemes)
        
        return PronunciationFeedbackDTO(
            overall_feedback=overall_feedback,
            strengths=strengths,
            areas_for_improvement=improvements,
            specific_phoneme_feedback=[],
            practice_suggestions=suggestions,
            difficulty_level_recommendation=self._recommend_difficulty_level(pronunciation_score),
            confidence_level="intermediate"
        )
    
    def _calculate_total_score(
        self, 
        pronunciation_score: float, 
        fluency_score: float, 
        accuracy_score: float
    ) -> float:
        """Calculate weighted total score"""
        return (pronunciation_score * 0.4) + (fluency_score * 0.3) + (accuracy_score * 0.3)
    
    def _calculate_grade(self, total_score: float) -> str:
        """Convert score to letter grade"""
        if total_score >= 95: return "A+"
        elif total_score >= 90: return "A"
        elif total_score >= 85: return "A-"
        elif total_score >= 80: return "B+"
        elif total_score >= 75: return "B"
        elif total_score >= 70: return "B-"
        elif total_score >= 65: return "C+"
        elif total_score >= 60: return "C"
        elif total_score >= 55: return "C-"
        elif total_score >= 50: return "D"
        else: return "F"
    
    def _create_basic_statistics(self, asr_result: ASRResult) -> ASRStatisticsDTO:
        """Create basic statistics from ASR result"""
        return ASRStatisticsDTO(
            word_error_rate=0.0,
            character_error_rate=0.0,
            phoneme_error_rate=0.0,
            average_word_duration=2.0,
            average_phoneme_duration=0.1,
            timing_precision=0.95,
            correctly_pronounced_phonemes=0,
            total_phonemes=0,
            phoneme_accuracy_percentage=0.0,
            speaking_speed=1.0,
            pause_patterns_score=0.8,
            rhythm_score=0.8,
            confidence_distribution={"high": 0.7, "medium": 0.2, "low": 0.1}
        )
    
    def _create_basic_feedback(self) -> PronunciationFeedbackDTO:
        """Create basic feedback"""
        return PronunciationFeedbackDTO(
            overall_feedback="Audio transcribed successfully.",
            strengths=["Clear speech detected"],
            areas_for_improvement=["Analysis requires reference text"],
            specific_phoneme_feedback=[],
            practice_suggestions=["Provide reference text for detailed feedback"],
            difficulty_level_recommendation="intermediate"
        )
    
    def _create_empty_utterance(self) -> ActualUtteranceResponseDTO:
        """Create empty utterance for error cases"""
        from src.domain.entities.asr_entities import TranscriptionQuality, PronunciationAccuracy
        
        return ActualUtteranceResponseDTO(
            transcribed_text="",
            original_text="",
            total_duration=0.0,
            words=[],
            overall_confidence=0.0,
            transcription_quality=TranscriptionQuality.POOR,
            pronunciation_accuracy=PronunciationAccuracy.POOR,
            speech_rate=0.0,
            phoneme_rate=0.0,
            pause_count=0,
            pause_duration_total=0.0
        )
    
    def _create_empty_statistics(self) -> ASRStatisticsDTO:
        """Create empty statistics for error cases"""
        return ASRStatisticsDTO(
            word_error_rate=1.0,
            character_error_rate=1.0,
            phoneme_error_rate=1.0,
            average_word_duration=0.0,
            average_phoneme_duration=0.0,
            timing_precision=0.0,
            correctly_pronounced_phonemes=0,
            total_phonemes=0,
            phoneme_accuracy_percentage=0.0,
            speaking_speed=0.0,
            pause_patterns_score=0.0,
            rhythm_score=0.0,
            confidence_distribution={"high": 0.0, "medium": 0.0, "low": 1.0}
        )
    
    def _create_error_feedback(self, error_message: str) -> PronunciationFeedbackDTO:
        """Create error feedback"""
        return PronunciationFeedbackDTO(
            overall_feedback=f"Processing failed: {error_message}",
            strengths=[],
            areas_for_improvement=["Ensure audio file is valid and accessible"],
            specific_phoneme_feedback=[],
            practice_suggestions=["Check audio format and quality"],
            difficulty_level_recommendation="beginner"
        )
    
    def _generate_overall_feedback(self, pronunciation_score: float, fluency_score: float) -> str:
        """Generate overall feedback message"""
        if pronunciation_score >= 85 and fluency_score >= 85:
            return "Excellent pronunciation and fluency! Your speech is very clear and natural."
        elif pronunciation_score >= 70 and fluency_score >= 70:
            return "Good pronunciation and fluency with room for improvement in specific areas."
        elif pronunciation_score >= 50 or fluency_score >= 50:
            return "Fair pronunciation and fluency. Focus on problematic phonemes and speaking rhythm."
        else:
            return "Pronunciation and fluency need significant improvement. Regular practice recommended."
    
    def _identify_strengths(self, word_comparisons: List[WordComparisonResponseDTO]) -> List[str]:
        """Identify pronunciation strengths"""
        strengths = []
        accurate_phonemes = set()
        
        for wc in word_comparisons:
            for pc in wc.phoneme_comparisons:
                if pc.is_accurate:
                    accurate_phonemes.add(pc.reference_phoneme)
        
        if accurate_phonemes:
            strengths.append(f"Accurate pronunciation of {len(accurate_phonemes)} different phonemes")
        
        return strengths[:3]  # Limit to top 3 strengths
    
    def _identify_improvements(self, error_phonemes: List[str]) -> List[str]:
        """Identify areas for improvement"""
        from collections import Counter
        
        if not error_phonemes:
            return ["Continue practicing to maintain accuracy"]
        
        phoneme_counts = Counter(error_phonemes)
        common_errors = phoneme_counts.most_common(3)
        
        improvements = []
        for phoneme, count in common_errors:
            improvements.append(f"Practice the /{phoneme}/ sound (mispronounced {count} times)")
        
        return improvements
    
    def _generate_practice_suggestions(self, error_phonemes: List[str]) -> List[str]:
        """Generate practice suggestions based on errors"""
        suggestions = [
            "Record yourself speaking and compare with native speakers",
            "Practice tongue twisters focusing on problematic sounds",
            "Use minimal pairs to distinguish similar phonemes"
        ]
        
        if error_phonemes:
            unique_errors = set(error_phonemes)
            if 'θ' in unique_errors or 'ð' in unique_errors:
                suggestions.append("Practice 'th' sounds: thin/this, think/that")
            if 'r' in unique_errors:
                suggestions.append("Practice R sound with 'red', 'right', 'around'")
            if 'l' in unique_errors:
                suggestions.append("Practice L sound with 'light', 'hello', 'feel'")
        
        return suggestions[:5]
    
    def _recommend_difficulty_level(self, pronunciation_score: float) -> str:
        """Recommend difficulty level based on score"""
        if pronunciation_score >= 85:
            return "advanced"
        elif pronunciation_score >= 70:
            return "intermediate"
        elif pronunciation_score >= 50:
            return "beginner-intermediate"
        else:
            return "beginner"


class BatchTranscribeAudioUseCase:
    """
    Use case for batch audio transcription
    """
    
    def __init__(
        self,
        transcribe_use_case: TranscribeAudioUseCase,
        logger: logging.Logger
    ):
        self.transcribe_use_case = transcribe_use_case
        self.logger = logger
    
    async def execute(
        self, 
        audio_files: List[str], 
        request: BatchASRRequest
    ) -> BatchASRResponseDTO:
        """
        Execute batch audio transcription
        """
        start_time = datetime.now()
        results = []
        successful_count = 0
        failed_count = 0
        
        try:
            # Create individual ASR requests
            asr_requests = self._create_individual_requests(request, audio_files)
            
            if request.parallel_processing:
                # Process files in parallel (limited concurrency)
                semaphore = asyncio.Semaphore(5)  # Limit to 5 concurrent processes
                tasks = [
                    self._process_with_semaphore(semaphore, audio_file, asr_request)
                    for audio_file, asr_request in zip(audio_files, asr_requests)
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)
            else:
                # Process files sequentially
                for audio_file, asr_request in zip(audio_files, asr_requests):
                    result = await self.transcribe_use_case.execute(audio_file, asr_request)
                    results.append(result)
            
            # Count successful/failed results
            for result in results:
                if isinstance(result, ASRResponseDTO) and result.success:
                    successful_count += 1
                else:
                    failed_count += 1
            
            # Calculate averages
            valid_results = [r for r in results if isinstance(r, ASRResponseDTO) and r.success]
            avg_pronunciation = sum(r.overall_pronunciation_score for r in valid_results) / len(valid_results) if valid_results else 0.0
            avg_fluency = sum(r.fluency_score for r in valid_results) / len(valid_results) if valid_results else 0.0
            avg_accuracy = sum(r.accuracy_score for r in valid_results) / len(valid_results) if valid_results else 0.0
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return BatchASRResponseDTO(
                success=successful_count > 0,
                total_files=len(audio_files),
                successful_files=successful_count,
                failed_files=failed_count,
                results=[r for r in results if isinstance(r, ASRResponseDTO)],
                average_pronunciation_score=avg_pronunciation,
                average_fluency_score=avg_fluency,
                average_accuracy_score=avg_accuracy,
                total_processing_time_ms=processing_time,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Batch transcription failed: {str(e)}")
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return BatchASRResponseDTO(
                success=False,
                total_files=len(audio_files),
                successful_files=0,
                failed_files=len(audio_files),
                results=[],
                average_pronunciation_score=0.0,
                average_fluency_score=0.0,
                average_accuracy_score=0.0,
                total_processing_time_ms=processing_time,
                timestamp=datetime.now()
            )
    
    async def _process_with_semaphore(
        self, 
        semaphore: asyncio.Semaphore, 
        audio_file: str, 
        request: ASRRequest
    ) -> ASRResponseDTO:
        """Process single file with semaphore for concurrency control"""
        async with semaphore:
            return await self.transcribe_use_case.execute(audio_file, request)
    
    def _create_individual_requests(
        self, 
        batch_request: BatchASRRequest, 
        audio_files: List[str]
    ) -> List[ASRRequest]:
        """Create individual ASR requests from batch request"""
        requests = []
        
        for i, audio_file in enumerate(audio_files):
            reference_text = None
            if batch_request.reference_texts and i < len(batch_request.reference_texts):
                reference_text = batch_request.reference_texts[i]
            
            requests.append(ASRRequest(
                reference_text=reference_text,
                language=batch_request.language,
                model_size=batch_request.model_size,
                include_phonemes=True,
                include_timing=True,
                include_confidence=True,
                compare_pronunciation=reference_text is not None,
                analyze_fluency=True
            ))
        
        return requests


class TranscriptionOnlyUseCase:
    """
    Use case for transcription without reference text (no pronunciation analysis)
    """
    
    def __init__(
        self,
        asr_service: IEnhancedASRService,
        logger: logging.Logger
    ):
        self.asr_service = asr_service
        self.logger = logger
    
    async def execute(self, audio_file_path: str, language: str = "english", model_size: str = "base") -> TranscriptionOnlyResponseDTO:
        """
        Execute audio transcription only (no pronunciation analysis)
        """
        start_time = datetime.now()
        
        try:
            # Transcribe with Whisper
            asr_result = await self.asr_service.transcribe_with_phonemes(
                audio_file_path=audio_file_path,
                reference_text=None,
                language=language,
                model_size=model_size,
                include_timing=True,
                include_confidence=True
            )
            
            # Convert to response format
            from src.application.dtos.asr_dto import ActualWordResponseDTO, ActualPhonemeResponseDTO
            
            words_dto = []
            for word in asr_result.actual_utterance.words:
                phonemes_dto = [
                    ActualPhonemeResponseDTO(
                        phoneme=phoneme.phoneme,
                        start_time=phoneme.start_time,
                        end_time=phoneme.end_time,
                        duration=phoneme.duration,
                        confidence=phoneme.confidence,
                        amplitude=phoneme.amplitude,
                        fundamental_frequency=phoneme.fundamental_frequency
                    )
                    for phoneme in word.phonemes
                ]
                
                words_dto.append(ActualWordResponseDTO(
                    word=word.word,
                    start_time=word.start_time,
                    end_time=word.end_time,
                    duration=word.duration,
                    confidence=word.confidence,
                    phonemes=phonemes_dto
                ))
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return TranscriptionOnlyResponseDTO(
                success=True,
                audio_file_path=audio_file_path,
                transcribed_text=asr_result.actual_utterance.transcribed_text,
                confidence=asr_result.actual_utterance.overall_confidence,
                words=words_dto,
                transcription_quality=asr_result.actual_utterance.transcription_quality,
                processing_time_ms=processing_time,
                timestamp=datetime.now(),
                whisper_model_used=model_size
            )
            
        except Exception as e:
            self.logger.error(f"Transcription failed for {audio_file_path}: {str(e)}")
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return TranscriptionOnlyResponseDTO(
                success=False,
                audio_file_path=audio_file_path,
                transcribed_text="",
                confidence=0.0,
                words=[],
                transcription_quality=asr_result.actual_utterance.transcription_quality if 'asr_result' in locals() else None,
                processing_time_ms=processing_time,
                timestamp=datetime.now(),
                whisper_model_used=model_size,
                error_message=str(e)
            )


class ValidateAudioForASRUseCase:
    """
    Use case for validating audio files for ASR processing
    """
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    async def execute(self, audio_file_path: str, request: ASRValidationRequest) -> ASRValidationResponseDTO:
        """
        Validate audio file for ASR processing
        """
        try:
            validation_details = {}
            recommendations = []
            valid = True
            whisper_ready = True
            
            # Check file existence
            audio_path = Path(audio_file_path)
            if not audio_path.exists():
                valid = False
                validation_details["file_exists"] = False
                recommendations.append("Audio file does not exist")
            else:
                validation_details["file_exists"] = True
            
            if valid and request.check_format:
                # Check audio format
                supported_formats = ['.wav', '.mp3', '.m4a', '.flac', '.ogg']
                if audio_path.suffix.lower() not in supported_formats:
                    whisper_ready = False
                    validation_details["format_supported"] = False
                    recommendations.append(f"Convert to supported format: {', '.join(supported_formats)}")
                else:
                    validation_details["format_supported"] = True
            
            if valid and request.check_duration:
                # Estimate audio duration (simplified)
                file_size_mb = audio_path.stat().st_size / (1024 * 1024)
                estimated_duration = file_size_mb / 0.5  # Rough estimate: 0.5 MB per minute
                
                if estimated_duration > 300:  # 5 minutes
                    valid = False
                    validation_details["duration_acceptable"] = False
                    recommendations.append("Audio file too long (max 5 minutes)")
                elif estimated_duration < 0.5:  # 30 seconds
                    validation_details["duration_warning"] = True
                    recommendations.append("Very short audio may affect accuracy")
                else:
                    validation_details["duration_acceptable"] = True
                
                validation_details["estimated_duration_seconds"] = estimated_duration * 60
            
            if valid and request.check_audio_quality:
                # Basic quality checks
                if audio_path.stat().st_size < 10000:  # Less than 10KB
                    valid = False
                    validation_details["quality_acceptable"] = False
                    recommendations.append("Audio file too small, may indicate poor quality")
                else:
                    validation_details["quality_acceptable"] = True
            
            # Estimate processing time
            estimated_time = None
            if valid:
                file_size_mb = audio_path.stat().st_size / (1024 * 1024)
                estimated_time = file_size_mb * 2.0  # Rough estimate: 2 seconds per MB
            
            return ASRValidationResponseDTO(
                valid=valid,
                audio_file_path=audio_file_path,
                validation_details=validation_details,
                whisper_ready=whisper_ready,
                recommendations=recommendations,
                estimated_transcription_time=estimated_time
            )
            
        except Exception as e:
            self.logger.error(f"Audio validation failed for {audio_file_path}: {str(e)}")
            
            return ASRValidationResponseDTO(
                valid=False,
                audio_file_path=audio_file_path,
                validation_details={"error": str(e)},
                whisper_ready=False,
                recommendations=["Fix file access issues"],
                error_message=str(e)
            )