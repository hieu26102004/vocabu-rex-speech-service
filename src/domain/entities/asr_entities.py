"""
Domain entities for Enhanced ASR (Step 3)
Represents actual pronunciation capture and transcription results
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
from enum import Enum


class TranscriptionQuality(Enum):
    """Quality levels for ASR transcription"""
    EXCELLENT = "excellent"
    GOOD = "good" 
    FAIR = "fair"
    POOR = "poor"


class PronunciationAccuracy(Enum):
    """Pronunciation accuracy levels"""
    NATIVE = "native"
    FLUENT = "fluent"
    INTERMEDIATE = "intermediate"
    BEGINNER = "beginner"
    UNINTELLIGIBLE = "unintelligible"


@dataclass
class ActualPhoneme:
    """
    Represents an actual phoneme captured from speech
    """
    phoneme: str
    start_time: float
    end_time: float
    confidence: float  # ASR confidence 0.0-1.0
    duration: float
    amplitude: float  # Average amplitude during phoneme
    fundamental_frequency: Optional[float] = None  # F0 if available
    
    def __post_init__(self):
        self.duration = self.end_time - self.start_time


@dataclass
class ActualWord:
    """
    Represents an actual word captured from speech
    """
    word: str
    start_time: float
    end_time: float
    phonemes: List[ActualPhoneme]
    confidence: float
    duration: float
    pronunciation_score: Optional[float] = None  # Compared to reference
    
    def __post_init__(self):
        self.duration = self.end_time - self.start_time
    
    @property
    def phoneme_count(self) -> int:
        return len(self.phonemes)
    
    @property
    def average_phoneme_confidence(self) -> float:
        if not self.phonemes:
            return 0.0
        return sum(p.confidence for p in self.phonemes) / len(self.phonemes)


@dataclass
class ActualUtterance:
    """
    Represents complete actual utterance from speech
    """
    transcribed_text: str
    original_text: str  # Reference text
    total_duration: float
    words: List[ActualWord]
    overall_confidence: float
    transcription_quality: TranscriptionQuality
    pronunciation_accuracy: PronunciationAccuracy
    
    # Fluency metrics
    speech_rate: float  # words per minute
    phoneme_rate: float  # phonemes per second
    pause_count: int
    pause_duration_total: float
    
    @property
    def word_count(self) -> int:
        return len(self.words)
    
    @property
    def actual_phonemes_count(self) -> int:
        return sum(word.phoneme_count for word in self.words)
    
    @property
    def text_accuracy(self) -> float:
        """Word-level accuracy compared to reference text"""
        if not self.original_text:
            return 0.0
        # This would use WER (Word Error Rate) calculation
        return 1.0  # Placeholder


@dataclass
class PronunciationComparison:
    """
    Comparison between reference and actual pronunciation
    """
    reference_phoneme: str
    actual_phoneme: str
    phoneme_match: bool
    similarity_score: float  # 0.0-1.0
    timing_deviation: float  # seconds difference from reference
    error_type: Optional[str] = None  # "substitution", "insertion", "deletion"
    
    @property
    def is_accurate(self) -> bool:
        return self.phoneme_match and self.similarity_score >= 0.8


@dataclass
class WordComparison:
    """
    Word-level comparison between reference and actual
    """
    reference_word: str
    actual_word: str
    word_match: bool
    phoneme_comparisons: List[PronunciationComparison]
    overall_accuracy: float  # 0.0-1.0
    timing_accuracy: float  # How well timing matches reference
    
    @property
    def phoneme_accuracy_rate(self) -> float:
        if not self.phoneme_comparisons:
            return 0.0
        accurate_phonemes = sum(1 for pc in self.phoneme_comparisons if pc.is_accurate)
        return accurate_phonemes / len(self.phoneme_comparisons)


@dataclass
class ASRResult:
    """
    Complete ASR result with pronunciation analysis
    """
    audio_file_path: str
    reference_text: str
    actual_utterance: ActualUtterance
    word_comparisons: List[WordComparison]
    
    # Overall metrics
    overall_pronunciation_score: float  # 0.0-100.0
    fluency_score: float  # 0.0-100.0
    accuracy_score: float  # 0.0-100.0
    
    # Processing metadata
    processing_time_ms: float
    timestamp: datetime
    whisper_model_used: str
    success: bool = True
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    @property
    def total_score(self) -> float:
        """Combined score: pronunciation + fluency + accuracy"""
        return (self.overall_pronunciation_score + self.fluency_score + self.accuracy_score) / 3.0
    
    @property
    def pronunciation_grade(self) -> str:
        """Letter grade based on total score"""
        score = self.total_score
        if score >= 90:
            return "A+"
        elif score >= 85:
            return "A"
        elif score >= 80:
            return "B+"
        elif score >= 75:
            return "B"
        elif score >= 70:
            return "C+"
        elif score >= 65:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"


@dataclass
class EnhancedASRStatistics:
    """
    Statistical analysis of ASR and pronunciation results
    """
    # Transcription stats
    word_error_rate: float  # WER
    character_error_rate: float  # CER
    phoneme_error_rate: float  # PER
    
    # Timing stats
    average_word_duration: float
    average_phoneme_duration: float
    timing_precision: float
    
    # Pronunciation stats
    correctly_pronounced_phonemes: int
    total_phonemes: int
    phoneme_accuracy_percentage: float
    
    # Fluency stats
    speaking_speed: float  # compared to native speakers
    pause_patterns_score: float
    rhythm_score: float
    
    # Confidence distribution
    confidence_distribution: Dict[str, float]
    
    @classmethod
    def from_asr_result(cls, result: ASRResult) -> 'EnhancedASRStatistics':
        """Create statistics from ASR result"""
        utterance = result.actual_utterance
        
        # Calculate phoneme accuracy
        total_phonemes = 0
        correct_phonemes = 0
        
        for word_comp in result.word_comparisons:
            for phoneme_comp in word_comp.phoneme_comparisons:
                total_phonemes += 1
                if phoneme_comp.is_accurate:
                    correct_phonemes += 1
        
        phoneme_accuracy = (correct_phonemes / total_phonemes * 100) if total_phonemes > 0 else 0
        
        # Calculate timing stats
        word_durations = [word.duration for word in utterance.words]
        avg_word_duration = sum(word_durations) / len(word_durations) if word_durations else 0
        
        phoneme_durations = []
        for word in utterance.words:
            phoneme_durations.extend([p.duration for p in word.phonemes])
        avg_phoneme_duration = sum(phoneme_durations) / len(phoneme_durations) if phoneme_durations else 0
        
        # Confidence distribution
        all_confidences = []
        for word in utterance.words:
            for phoneme in word.phonemes:
                all_confidences.append(phoneme.confidence)
        
        confidence_dist = {}
        if all_confidences:
            excellent_count = sum(1 for c in all_confidences if c >= 0.9)
            good_count = sum(1 for c in all_confidences if 0.75 <= c < 0.9)
            fair_count = sum(1 for c in all_confidences if 0.5 <= c < 0.75)
            poor_count = sum(1 for c in all_confidences if c < 0.5)
            total = len(all_confidences)
            
            confidence_dist = {
                "excellent": (excellent_count / total) * 100,
                "good": (good_count / total) * 100,
                "fair": (fair_count / total) * 100,
                "poor": (poor_count / total) * 100
            }
        
        return cls(
            word_error_rate=1.0 - utterance.text_accuracy,  # Simplified
            character_error_rate=0.0,  # Would calculate properly
            phoneme_error_rate=(total_phonemes - correct_phonemes) / total_phonemes if total_phonemes > 0 else 0,
            average_word_duration=avg_word_duration,
            average_phoneme_duration=avg_phoneme_duration,
            timing_precision=0.85,  # Would calculate based on timing deviations
            correctly_pronounced_phonemes=correct_phonemes,
            total_phonemes=total_phonemes,
            phoneme_accuracy_percentage=phoneme_accuracy,
            speaking_speed=utterance.speech_rate / 150.0,  # Normalized to average speaking rate
            pause_patterns_score=85.0,  # Would analyze pause patterns
            rhythm_score=80.0,  # Would analyze speech rhythm
            confidence_distribution=confidence_dist
        )


@dataclass 
class PronunciationFeedback:
    """
    Detailed feedback for pronunciation improvement
    """
    overall_feedback: str
    strengths: List[str]
    areas_for_improvement: List[str]
    specific_phoneme_feedback: List[Dict[str, str]]  # [{"phoneme": "TH", "feedback": "..."}]
    practice_suggestions: List[str]
    difficulty_level_recommendation: str
    
    @classmethod
    def generate_from_result(cls, result: ASRResult) -> 'PronunciationFeedback':
        """Generate feedback from ASR result"""
        
        # Analyze common pronunciation errors
        problem_phonemes = []
        for word_comp in result.word_comparisons:
            for phoneme_comp in word_comp.phoneme_comparisons:
                if not phoneme_comp.is_accurate:
                    problem_phonemes.append({
                        "phoneme": phoneme_comp.reference_phoneme,
                        "actual": phoneme_comp.actual_phoneme,
                        "error_type": phoneme_comp.error_type
                    })
        
        # Generate feedback based on score
        score = result.total_score
        
        if score >= 85:
            overall = "Excellent pronunciation! Your speech is very clear and accurate."
            strengths = ["Clear articulation", "Good timing", "High confidence"]
        elif score >= 70:
            overall = "Good pronunciation with room for minor improvements."
            strengths = ["Generally clear speech", "Most phonemes correct"]
        elif score >= 55:
            overall = "Fair pronunciation. Focus on specific sound improvements."
            strengths = ["Basic communication clear"]
        else:
            overall = "Pronunciation needs significant improvement. Regular practice recommended."
            strengths = ["Effort acknowledged"]
        
        # Specific phoneme feedback
        phoneme_feedback = []
        for problem in problem_phonemes[:5]:  # Top 5 issues
            phoneme_feedback.append({
                "phoneme": problem["phoneme"],
                "feedback": f"Work on the {problem['phoneme']} sound - you pronounced it as {problem['actual']}"
            })
        
        # Practice suggestions
        suggestions = [
            "Practice with tongue twisters",
            "Record yourself speaking",
            "Listen to native speakers",
            "Use pronunciation apps",
            "Work with a language tutor"
        ]
        
        return cls(
            overall_feedback=overall,
            strengths=strengths,
            areas_for_improvement=[],  # Would analyze specific areas
            specific_phoneme_feedback=phoneme_feedback,
            practice_suggestions=suggestions,
            difficulty_level_recommendation="intermediate"
        )