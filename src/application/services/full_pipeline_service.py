"""
Full Pipeline Integration Service - Combining Steps 1, 2, 3 for complete pronunciation assessment
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from pathlib import Path

from src.domain.services.asr_service import IEnhancedASRService, IPronunciationAnalyzer, IFluencyAnalyzer
from src.application.dtos.asr_dto import (
    FullPronunciationAnalysisRequest, FullPronunciationAnalysisResponseDTO,
    ASRRequest, ASRResponseDTO, PronunciationReportDTO
)
from src.application.use_cases.asr_use_cases import TranscribeAudioUseCase
from src.core.exceptions import (
    SpeechServiceException, ValidationError, ServiceError,
    PhonemizationError, AlignmentError, TranscriptionError
)


class FullPronunciationPipelineService:
    """
    Service that integrates all 4 steps of pronunciation assessment:
    Step 1: Phonemization (Text → Phonemes)
    Step 2: Forced Alignment (Reference Audio + Text → Timing)
    Step 3: Enhanced ASR (Actual Audio → Transcription + Phonemes + Timing)
    Step 4: Comprehensive Scoring (Compare all data → Final Assessment)
    """
    
    def __init__(
        self,
        transcribe_use_case: TranscribeAudioUseCase,
        logger: logging.Logger
    ):
        self.transcribe_use_case = transcribe_use_case
        self.logger = logger
        
        # Placeholder for Step 1 and Step 2 services (to be implemented)
        self.phonemization_service = None  # Step 1: Text → Phonemes
        self.alignment_service = None      # Step 2: Audio + Text → Timing
    
    async def analyze_full_pronunciation(
        self,
        audio_file_path: str,
        request: FullPronunciationAnalysisRequest
    ) -> FullPronunciationAnalysisResponseDTO:
        """
        Execute complete pronunciation analysis pipeline (Steps 1-4)
        """
        start_time = datetime.now()
        
        try:
            self.logger.info(f"Starting full pronunciation analysis for: {audio_file_path}")
            
            # Initialize results
            reference_phonemes = []
            phonemization_success = False
            reference_timing = None
            alignment_success = False
            asr_result = None
            asr_success = False
            
            # Step 1: Phonemization (if enabled)
            if request.include_phonemization:
                try:
                    reference_phonemes = await self._execute_phonemization(
                        request.text, request.language
                    )
                    phonemization_success = True
                    self.logger.info(f"Step 1 (Phonemization) completed: {len(reference_phonemes)} phonemes")
                except Exception as e:
                    self.logger.error(f"Step 1 (Phonemization) failed: {str(e)}")
                    phonemization_success = False
            
            # Step 2: Forced Alignment (if enabled and reference audio available)
            if request.include_alignment:
                try:
                    reference_timing = await self._execute_forced_alignment(
                        audio_file_path, request.text, request.language, 
                        request.alignment_model
                    )
                    alignment_success = True
                    self.logger.info("Step 2 (Forced Alignment) completed")
                except Exception as e:
                    self.logger.error(f"Step 2 (Forced Alignment) failed: {str(e)}")
                    alignment_success = False
            
            # Step 3: Enhanced ASR Analysis (always executed)
            if request.include_asr_analysis:
                try:
                    asr_request = ASRRequest(
                        reference_text=request.text,
                        language=request.language,
                        model_size=request.whisper_model_size,
                        include_phonemes=True,
                        include_timing=True,
                        include_confidence=True,
                        compare_pronunciation=True,
                        analyze_fluency=True
                    )
                    
                    asr_result = await self.transcribe_use_case.execute(audio_file_path, asr_request)
                    asr_success = asr_result.success
                    self.logger.info(f"Step 3 (Enhanced ASR) completed: success={asr_success}")
                except Exception as e:
                    self.logger.error(f"Step 3 (Enhanced ASR) failed: {str(e)}")
                    asr_success = False
            
            # Step 4: Comprehensive Analysis (combine all results)
            overall_assessment = await self._generate_comprehensive_assessment(
                reference_phonemes=reference_phonemes,
                reference_timing=reference_timing,
                asr_result=asr_result,
                request=request
            )
            
            # Generate learning recommendations
            learning_recommendations = await self._generate_learning_recommendations(
                asr_result, overall_assessment
            )
            
            # Generate next practice topics
            next_practice_topics = await self._generate_next_practice_topics(
                asr_result, request.native_language
            )
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return FullPronunciationAnalysisResponseDTO(
                success=asr_success,  # Overall success depends on at least ASR working
                audio_file_path=audio_file_path,
                reference_text=request.text,
                reference_phonemes=reference_phonemes,
                phonemization_success=phonemization_success,
                reference_timing=reference_timing,
                alignment_success=alignment_success,
                asr_result=asr_result or self._create_empty_asr_result(),
                asr_success=asr_success,
                overall_pronunciation_assessment=overall_assessment,
                learning_recommendations=learning_recommendations,
                next_practice_topics=next_practice_topics,
                total_processing_time_ms=processing_time,
                timestamp=datetime.now(),
                pipeline_version="1.0.0"
            )
            
        except Exception as e:
            self.logger.error(f"Full pronunciation analysis failed: {str(e)}")
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return FullPronunciationAnalysisResponseDTO(
                success=False,
                audio_file_path=audio_file_path,
                reference_text=request.text,
                reference_phonemes=[],
                phonemization_success=False,
                reference_timing=None,
                alignment_success=False,
                asr_result=self._create_empty_asr_result(),
                asr_success=False,
                overall_pronunciation_assessment={"error": str(e)},
                learning_recommendations=["Fix technical issues before assessment"],
                next_practice_topics=["Basic audio recording"],
                total_processing_time_ms=processing_time,
                timestamp=datetime.now(),
                pipeline_version="1.0.0"
            )
    
    async def _execute_phonemization(self, text: str, language: str) -> List[str]:
        """
        Execute Step 1: Convert text to phonemes
        TODO: Implement actual phonemization service
        """
        if self.phonemization_service is None:
            # Placeholder implementation - return mock phonemes
            words = text.lower().split()
            mock_phonemes = []
            
            # Simple mock phoneme mapping (English)
            phoneme_map = {
                'hello': ['h', 'ɛ', 'l', 'oʊ'],
                'world': ['w', 'ɜr', 'l', 'd'],
                'the': ['ð', 'ə'],
                'quick': ['k', 'w', 'ɪ', 'k'],
                'brown': ['b', 'r', 'aʊ', 'n'],
                'fox': ['f', 'ɑ', 'k', 's'],
                'jumps': ['dʒ', 'ʌ', 'm', 'p', 's'],
                'over': ['oʊ', 'v', 'ər'],
                'lazy': ['l', 'eɪ', 'z', 'i'],
                'dog': ['d', 'ɔ', 'g']
            }
            
            for word in words:
                if word in phoneme_map:
                    mock_phonemes.extend(phoneme_map[word])
                else:
                    # Generate phonemes based on letters (very simplified)
                    for char in word:
                        if char.isalpha():
                            mock_phonemes.append(char)
            
            self.logger.warning("Using mock phonemization - Step 1 service not implemented")
            return mock_phonemes
        
        try:
            # TODO: Call actual phonemization service
            phonemes = await self.phonemization_service.phonemize(text, language)
            return phonemes
        except Exception as e:
            raise PhonemizationError(
                message=f"Phonemization failed for text: {text}",
                text=text,
                language=language,
                cause=e
            )
    
    async def _execute_forced_alignment(
        self, 
        audio_file_path: str, 
        text: str, 
        language: str,
        alignment_model: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """
        Execute Step 2: Forced alignment between audio and text
        TODO: Implement actual forced alignment service
        """
        if self.alignment_service is None:
            # Placeholder implementation - return mock timing data
            words = text.lower().split()
            mock_timing = {
                "words": [],
                "phonemes": [],
                "total_duration": len(words) * 0.6,  # 600ms per word average
                "alignment_quality": 0.85
            }
            
            current_time = 0.0
            for i, word in enumerate(words):
                word_duration = 0.5 + (len(word) * 0.05)  # Duration based on word length
                mock_timing["words"].append({
                    "word": word,
                    "start_time": current_time,
                    "end_time": current_time + word_duration,
                    "confidence": 0.85 + (i % 3) * 0.05
                })
                current_time += word_duration + 0.1  # 100ms pause between words
            
            self.logger.warning("Using mock forced alignment - Step 2 service not implemented")
            return mock_timing
        
        try:
            # TODO: Call actual forced alignment service
            alignment_result = await self.alignment_service.align(
                audio_file_path, text, language, alignment_model
            )
            return alignment_result
        except Exception as e:
            raise AlignmentError(
                message=f"Forced alignment failed for audio: {audio_file_path}",
                audio_file_path=audio_file_path,
                text=text,
                cause=e
            )
    
    async def _generate_comprehensive_assessment(
        self,
        reference_phonemes: List[str],
        reference_timing: Optional[Dict[str, Any]],
        asr_result: Optional[ASRResponseDTO],
        request: FullPronunciationAnalysisRequest
    ) -> Dict[str, Any]:
        """
        Step 4: Generate comprehensive pronunciation assessment
        """
        assessment = {
            "pipeline_completeness": {
                "phonemization_completed": len(reference_phonemes) > 0,
                "alignment_completed": reference_timing is not None,
                "asr_completed": asr_result is not None and asr_result.success
            },
            "data_quality": {},
            "pronunciation_analysis": {},
            "recommendations": {}
        }
        
        # Analyze data quality
        if asr_result and asr_result.success:
            assessment["data_quality"] = {
                "audio_quality": "good" if asr_result.actual_utterance.overall_confidence > 0.8 else "fair",
                "transcription_quality": asr_result.actual_utterance.transcription_quality.value,
                "pronunciation_accuracy": asr_result.actual_utterance.pronunciation_accuracy.value
            }
        
        # Multi-step pronunciation analysis
        if asr_result and asr_result.success:
            # Use ASR results as primary analysis
            assessment["pronunciation_analysis"] = {
                "overall_score": asr_result.total_score,
                "pronunciation_score": asr_result.overall_pronunciation_score,
                "fluency_score": asr_result.fluency_score,
                "accuracy_score": asr_result.accuracy_score,
                "grade": asr_result.pronunciation_grade,
                "detailed_feedback": asr_result.feedback.overall_feedback
            }
            
            # Enhanced analysis if we have reference data
            if reference_phonemes:
                assessment["pronunciation_analysis"]["phoneme_level_analysis"] = {
                    "reference_phoneme_count": len(reference_phonemes),
                    "actual_phoneme_count": asr_result.actual_utterance.actual_phonemes_count,
                    "phoneme_match_rate": len(reference_phonemes) / max(asr_result.actual_utterance.actual_phonemes_count, 1)
                }
            
            if reference_timing:
                assessment["pronunciation_analysis"]["timing_analysis"] = {
                    "reference_duration": reference_timing.get("total_duration", 0),
                    "actual_duration": asr_result.actual_utterance.total_duration,
                    "timing_deviation": abs(reference_timing.get("total_duration", 0) - asr_result.actual_utterance.total_duration),
                    "speech_rate_analysis": "normal" if 0.8 <= (asr_result.actual_utterance.total_duration / max(reference_timing.get("total_duration", 1), 0.1)) <= 1.3 else "abnormal"
                }
        
        # Generate improvement recommendations
        assessment["recommendations"] = await self._generate_improvement_recommendations(
            asr_result, reference_phonemes, reference_timing
        )
        
        return assessment
    
    async def _generate_improvement_recommendations(
        self,
        asr_result: Optional[ASRResponseDTO],
        reference_phonemes: List[str],
        reference_timing: Optional[Dict[str, Any]]
    ) -> Dict[str, List[str]]:
        """Generate specific improvement recommendations based on multi-step analysis"""
        recommendations = {
            "immediate_focus": [],
            "practice_exercises": [],
            "long_term_goals": [],
            "technical_suggestions": []
        }
        
        if not asr_result or not asr_result.success:
            recommendations["technical_suggestions"] = [
                "Ensure clear audio recording",
                "Use quiet environment",
                "Speak at moderate pace",
                "Check microphone quality"
            ]
            return recommendations
        
        # Pronunciation-specific recommendations
        if asr_result.overall_pronunciation_score < 70:
            recommendations["immediate_focus"].extend([
                "Focus on accurate phoneme pronunciation",
                "Practice problem sounds identified in feedback",
                "Use phonetic transcription as reference"
            ])
        
        if asr_result.fluency_score < 70:
            recommendations["immediate_focus"].extend([
                "Work on speaking rhythm and pace",
                "Reduce excessive pauses",
                "Practice connected speech"
            ])
        
        # Exercise recommendations
        if asr_result.word_comparisons:
            problematic_phonemes = []
            for word_comp in asr_result.word_comparisons:
                for phoneme_comp in word_comp.phoneme_comparisons:
                    if not phoneme_comp.is_accurate:
                        problematic_phonemes.append(phoneme_comp.reference_phoneme)
            
            if problematic_phonemes:
                unique_problems = list(set(problematic_phonemes))
                recommendations["practice_exercises"] = [
                    f"Drill pronunciation of /{phoneme}/ sound" for phoneme in unique_problems[:3]
                ]
        
        # Long-term goals based on current level
        if asr_result.total_score >= 85:
            recommendations["long_term_goals"] = [
                "Maintain current pronunciation quality",
                "Focus on advanced prosodic features",
                "Work on natural intonation patterns"
            ]
        elif asr_result.total_score >= 70:
            recommendations["long_term_goals"] = [
                "Achieve consistent pronunciation accuracy",
                "Improve fluency and natural rhythm",
                "Master difficult phoneme contrasts"
            ]
        else:
            recommendations["long_term_goals"] = [
                "Build foundation in basic phoneme production",
                "Develop consistent speaking rhythm",
                "Improve overall intelligibility"
            ]
        
        return recommendations
    
    async def _generate_learning_recommendations(
        self,
        asr_result: Optional[ASRResponseDTO],
        overall_assessment: Dict[str, Any]
    ) -> List[str]:
        """Generate personalized learning recommendations"""
        recommendations = []
        
        if not asr_result or not asr_result.success:
            return [
                "Start with basic audio recording practice",
                "Ensure proper microphone setup",
                "Practice in quiet environment"
            ]
        
        # Based on overall score
        total_score = asr_result.total_score
        
        if total_score >= 90:
            recommendations.extend([
                "Excellent pronunciation! Focus on maintaining consistency",
                "Practice with more challenging vocabulary",
                "Work on advanced prosodic features like stress and intonation"
            ])
        elif total_score >= 75:
            recommendations.extend([
                "Good pronunciation foundation. Focus on specific weak areas",
                "Practice problematic phonemes with minimal pairs",
                "Work on fluency and natural speech rhythm"
            ])
        elif total_score >= 60:
            recommendations.extend([
                "Continue working on pronunciation fundamentals",
                "Use phonetic transcription as learning aid",
                "Practice with native speaker recordings for comparison"
            ])
        else:
            recommendations.extend([
                "Focus on basic phoneme production accuracy",
                "Start with slow, careful pronunciation practice",
                "Consider working with pronunciation instructor"
            ])
        
        # Add specific feedback from ASR result
        if asr_result.feedback.areas_for_improvement:
            recommendations.extend(asr_result.feedback.areas_for_improvement[:2])
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    async def _generate_next_practice_topics(
        self,
        asr_result: Optional[ASRResponseDTO],
        native_language: Optional[str]
    ) -> List[str]:
        """Generate next practice topics based on analysis"""
        topics = []
        
        if not asr_result or not asr_result.success:
            return ["Basic audio recording", "Clear speech production"]
        
        # Based on common error patterns
        if asr_result.word_comparisons:
            error_phonemes = []
            for word_comp in asr_result.word_comparisons:
                for phoneme_comp in word_comp.phoneme_comparisons:
                    if not phoneme_comp.is_accurate:
                        error_phonemes.append(phoneme_comp.reference_phoneme)
            
            # Generate topics for most common errors
            from collections import Counter
            common_errors = Counter(error_phonemes).most_common(3)
            
            for phoneme, count in common_errors:
                topics.append(f"Practice /{phoneme}/ sound production")
        
        # Fluency-based topics
        if asr_result.fluency_score < 70:
            topics.extend([
                "Connected speech practice",
                "Rhythm and stress patterns",
                "Pause and pacing control"
            ])
        
        # Language-specific recommendations
        if native_language:
            language_specific_topics = {
                'spanish': ['English /r/ vs Spanish /r/', 'Vowel reduction in unstressed syllables'],
                'chinese': ['English consonant clusters', 'Final consonant sounds'],
                'japanese': ['English /l/ vs /r/ distinction', 'English vowel length'],
                'german': ['English /θ/ and /ð/ sounds', 'Word-final devoicing'],
                'french': ['English /h/ sound', 'English vowel contrasts']
            }
            
            if native_language.lower() in language_specific_topics:
                topics.extend(language_specific_topics[native_language.lower()][:2])
        
        return topics[:5] if topics else ["General pronunciation practice", "Phonetic awareness"]
    
    def _create_empty_asr_result(self) -> ASRResponseDTO:
        """Create empty ASR result for error cases"""
        from src.application.dtos.asr_dto import (
            ActualUtteranceResponseDTO, ASRStatisticsDTO, PronunciationFeedbackDTO
        )
        from src.domain.entities.asr_entities import TranscriptionQuality, PronunciationAccuracy
        
        return ASRResponseDTO(
            success=False,
            audio_file_path="",
            reference_text="",
            actual_utterance=ActualUtteranceResponseDTO(
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
            ),
            word_comparisons=[],
            overall_pronunciation_score=0.0,
            fluency_score=0.0,
            accuracy_score=0.0,
            total_score=0.0,
            pronunciation_grade="F",
            statistics=ASRStatisticsDTO(
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
            ),
            feedback=PronunciationFeedbackDTO(
                overall_feedback="Analysis could not be completed",
                strengths=[],
                areas_for_improvement=["Ensure valid audio input"],
                specific_phoneme_feedback=[],
                practice_suggestions=["Check audio quality and format"],
                difficulty_level_recommendation="beginner"
            ),
            processing_time_ms=0.0,
            timestamp=datetime.now(),
            whisper_model_used="none"
        )


class PronunciationReportGeneratorService:
    """
    Service for generating comprehensive pronunciation assessment reports
    """
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    async def generate_comprehensive_report(
        self,
        session_results: List[FullPronunciationAnalysisResponseDTO],
        learner_id: Optional[str] = None
    ) -> PronunciationReportDTO:
        """
        Generate a comprehensive pronunciation assessment report
        """
        try:
            if not session_results:
                raise ValidationError("No session results provided for report generation")
            
            # Calculate overall metrics
            successful_sessions = [r for r in session_results if r.success and r.asr_success]
            
            if not successful_sessions:
                raise ValidationError("No successful sessions found for report generation")
            
            # Overall scores (average of all successful sessions)
            overall_pronunciation = sum(s.asr_result.overall_pronunciation_score for s in successful_sessions) / len(successful_sessions)
            overall_fluency = sum(s.asr_result.fluency_score for s in successful_sessions) / len(successful_sessions)
            overall_accuracy = sum(s.asr_result.accuracy_score for s in successful_sessions) / len(successful_sessions)
            
            # Analyze improvement trend
            if len(successful_sessions) >= 3:
                recent_scores = [s.asr_result.total_score for s in successful_sessions[-3:]]
                earlier_scores = [s.asr_result.total_score for s in successful_sessions[:3]]
                
                recent_avg = sum(recent_scores) / len(recent_scores)
                earlier_avg = sum(earlier_scores) / len(earlier_scores)
                
                if recent_avg > earlier_avg + 5:
                    trend = "improving"
                elif recent_avg < earlier_avg - 5:
                    trend = "declining"
                else:
                    trend = "stable"
            else:
                trend = "insufficient_data"
            
            # Phoneme accuracy breakdown
            phoneme_accuracy = await self._analyze_phoneme_accuracy(successful_sessions)
            
            # Common error patterns
            error_patterns = await self._analyze_error_patterns(successful_sessions)
            
            # Fluency metrics
            fluency_metrics = await self._analyze_fluency_metrics(successful_sessions)
            
            # Generate recommendations
            focus_areas = await self._identify_focus_areas(successful_sessions)
            practice_exercises = await self._recommend_practice_exercises(successful_sessions)
            
            # Session history
            session_history = [
                {
                    "date": s.timestamp.isoformat(),
                    "total_score": s.asr_result.total_score,
                    "pronunciation_score": s.asr_result.overall_pronunciation_score,
                    "fluency_score": s.asr_result.fluency_score,
                    "reference_text": s.reference_text
                }
                for s in successful_sessions
            ]
            
            return PronunciationReportDTO(
                learner_id=learner_id,
                assessment_date=datetime.now(),
                language=successful_sessions[0].asr_result.metadata.get("language", "english"),
                total_sessions=len(session_results),
                overall_pronunciation_score=overall_pronunciation,
                overall_fluency_score=overall_fluency,
                overall_accuracy_score=overall_accuracy,
                improvement_trend=trend,
                phoneme_accuracy_breakdown=phoneme_accuracy,
                common_error_patterns=error_patterns,
                fluency_metrics=fluency_metrics,
                focus_areas=focus_areas,
                practice_exercises=practice_exercises,
                estimated_improvement_time=self._estimate_improvement_time(overall_pronunciation),
                session_history=session_history,
                milestone_achievements=await self._identify_milestones(successful_sessions),
                next_milestones=await self._suggest_next_milestones(overall_pronunciation)
            )
            
        except Exception as e:
            self.logger.error(f"Report generation failed: {str(e)}")
            raise ServiceError(
                message="Failed to generate pronunciation report",
                service_name="ReportGenerator",
                operation="generate_report",
                cause=e
            )
    
    async def _analyze_phoneme_accuracy(
        self, 
        sessions: List[FullPronunciationAnalysisResponseDTO]
    ) -> Dict[str, float]:
        """Analyze accuracy by phoneme across all sessions"""
        phoneme_stats = {}
        
        for session in sessions:
            for word_comp in session.asr_result.word_comparisons:
                for phoneme_comp in word_comp.phoneme_comparisons:
                    phoneme = phoneme_comp.reference_phoneme
                    if phoneme not in phoneme_stats:
                        phoneme_stats[phoneme] = {"correct": 0, "total": 0}
                    
                    phoneme_stats[phoneme]["total"] += 1
                    if phoneme_comp.is_accurate:
                        phoneme_stats[phoneme]["correct"] += 1
        
        # Convert to percentages
        return {
            phoneme: (stats["correct"] / stats["total"] * 100) if stats["total"] > 0 else 0.0
            for phoneme, stats in phoneme_stats.items()
        }
    
    async def _analyze_error_patterns(
        self, 
        sessions: List[FullPronunciationAnalysisResponseDTO]
    ) -> List[Dict[str, Any]]:
        """Analyze common error patterns across sessions"""
        error_patterns = []
        substitution_errors = {}
        
        for session in sessions:
            for word_comp in session.asr_result.word_comparisons:
                for phoneme_comp in word_comp.phoneme_comparisons:
                    if not phoneme_comp.is_accurate and phoneme_comp.error_type == "substitution":
                        error_key = f"{phoneme_comp.reference_phoneme}→{phoneme_comp.actual_phoneme}"
                        substitution_errors[error_key] = substitution_errors.get(error_key, 0) + 1
        
        # Get most common substitution errors
        from collections import Counter
        common_substitutions = Counter(substitution_errors).most_common(5)
        
        for error, count in common_substitutions:
            ref_phoneme, actual_phoneme = error.split("→")
            error_patterns.append({
                "type": "substitution",
                "reference_phoneme": ref_phoneme,
                "actual_phoneme": actual_phoneme,
                "frequency": count,
                "description": f"Frequently substitutes /{ref_phoneme}/ with /{actual_phoneme}/"
            })
        
        return error_patterns
    
    async def _analyze_fluency_metrics(
        self, 
        sessions: List[FullPronunciationAnalysisResponseDTO]
    ) -> Dict[str, float]:
        """Analyze fluency metrics across sessions"""
        speech_rates = [s.asr_result.actual_utterance.speech_rate for s in sessions]
        pause_counts = [s.asr_result.actual_utterance.pause_count for s in sessions]
        
        return {
            "average_speech_rate": sum(speech_rates) / len(speech_rates),
            "average_pause_count": sum(pause_counts) / len(pause_counts),
            "speech_rate_consistency": 1.0 - (max(speech_rates) - min(speech_rates)) / max(speech_rates, 1),
            "fluency_trend": "improving" if sessions[-1].asr_result.fluency_score > sessions[0].asr_result.fluency_score else "stable"
        }
    
    async def _identify_focus_areas(
        self, 
        sessions: List[FullPronunciationAnalysisResponseDTO]
    ) -> List[str]:
        """Identify primary focus areas for improvement"""
        focus_areas = []
        
        # Average scores
        avg_pronunciation = sum(s.asr_result.overall_pronunciation_score for s in sessions) / len(sessions)
        avg_fluency = sum(s.asr_result.fluency_score for s in sessions) / len(sessions)
        avg_accuracy = sum(s.asr_result.accuracy_score for s in sessions) / len(sessions)
        
        if avg_pronunciation < 70:
            focus_areas.append("Phoneme-level pronunciation accuracy")
        if avg_fluency < 70:
            focus_areas.append("Speech fluency and rhythm")
        if avg_accuracy < 70:
            focus_areas.append("Overall intelligibility")
        
        return focus_areas[:3] if focus_areas else ["Continue current practice routine"]
    
    async def _recommend_practice_exercises(
        self, 
        sessions: List[FullPronunciationAnalysisResponseDTO]
    ) -> List[str]:
        """Recommend specific practice exercises"""
        exercises = []
        
        # Based on common errors
        all_errors = []
        for session in sessions:
            for word_comp in session.asr_result.word_comparisons:
                for phoneme_comp in word_comp.phoneme_comparisons:
                    if not phoneme_comp.is_accurate:
                        all_errors.append(phoneme_comp.reference_phoneme)
        
        if all_errors:
            from collections import Counter
            common_errors = Counter(all_errors).most_common(3)
            
            for phoneme, count in common_errors:
                exercises.append(f"Practice /{phoneme}/ sound with minimal pairs")
        
        exercises.extend([
            "Record and compare with native speakers",
            "Use shadowing technique with audio materials",
            "Practice connected speech patterns"
        ])
        
        return exercises[:5]
    
    def _estimate_improvement_time(self, current_score: float) -> str:
        """Estimate time needed for significant improvement"""
        if current_score >= 85:
            return "1-2 months for fine-tuning"
        elif current_score >= 70:
            return "2-4 months for noticeable improvement"
        elif current_score >= 50:
            return "4-6 months for significant improvement"
        else:
            return "6+ months for fundamental improvement"
    
    async def _identify_milestones(
        self, 
        sessions: List[FullPronunciationAnalysisResponseDTO]
    ) -> List[str]:
        """Identify achieved milestones"""
        milestones = []
        
        if sessions:
            latest_score = sessions[-1].asr_result.total_score
            
            if latest_score >= 90:
                milestones.append("Achieved excellent pronunciation level")
            elif latest_score >= 80:
                milestones.append("Achieved good pronunciation level")
            elif latest_score >= 70:
                milestones.append("Achieved satisfactory pronunciation level")
            elif latest_score >= 60:
                milestones.append("Achieved basic pronunciation competency")
        
        return milestones
    
    async def _suggest_next_milestones(self, current_score: float) -> List[str]:
        """Suggest next milestones to work toward"""
        if current_score >= 90:
            return ["Maintain consistency", "Master advanced prosodic features"]
        elif current_score >= 80:
            return ["Achieve 90+ score consistency", "Perfect problematic phonemes"]
        elif current_score >= 70:
            return ["Reach 80+ score level", "Improve fluency metrics"]
        elif current_score >= 60:
            return ["Reach 70+ score level", "Master basic phoneme production"]
        else:
            return ["Reach 60+ score level", "Establish pronunciation fundamentals"]