"""
Domain entities for forced alignment
Represents alignment results, phoneme timing, and word boundaries
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class AlignmentQuality(Enum):
    """Quality levels for alignment results"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"


@dataclass
class PhonemeAlignment:
    """
    Represents alignment result for a single phoneme
    """
    phoneme: str
    start_time: float  # seconds
    end_time: float   # seconds
    confidence: float  # 0.0 to 1.0
    duration: float   # seconds (end_time - start_time)
    
    def __post_init__(self):
        self.duration = self.end_time - self.start_time


@dataclass
class WordAlignment:
    """
    Represents alignment result for a single word
    """
    word: str
    start_time: float
    end_time: float
    phonemes: List[PhonemeAlignment]
    confidence: float
    duration: float
    
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
class SentenceAlignment:
    """
    Represents alignment result for a complete sentence/utterance
    """
    text: str
    total_duration: float
    words: List[WordAlignment]
    quality: AlignmentQuality
    overall_confidence: float
    silence_segments: List[tuple]  # (start_time, end_time) pairs
    
    @property
    def word_count(self) -> int:
        return len(self.words)
    
    @property
    def total_phonemes(self) -> int:
        return sum(word.phoneme_count for word in self.words)
    
    @property
    def speech_rate(self) -> float:
        """Words per minute"""
        if self.total_duration <= 0:
            return 0.0
        return (self.word_count / self.total_duration) * 60
    
    @property
    def phoneme_rate(self) -> float:
        """Phonemes per second"""
        if self.total_duration <= 0:
            return 0.0
        return self.total_phonemes / self.total_duration


@dataclass
class AlignmentResult:
    """
    Complete forced alignment result
    """
    audio_file_path: str
    text: str
    language: str
    model_name: str
    sentence_alignment: SentenceAlignment
    processing_time_ms: float
    timestamp: datetime
    metadata: Dict[str, Any]
    success: bool = True
    error_message: Optional[str] = None
    
    @property
    def alignment_accuracy(self) -> float:
        """Overall alignment accuracy score"""
        return self.sentence_alignment.overall_confidence
    
    @property
    def timing_precision(self) -> float:
        """Measure of timing precision based on phoneme durations"""
        if not self.sentence_alignment.words:
            return 0.0
            
        phoneme_durations = []
        for word in self.sentence_alignment.words:
            phoneme_durations.extend([p.duration for p in word.phonemes])
        
        if not phoneme_durations:
            return 0.0
            
        # Calculate coefficient of variation (std/mean)
        import statistics
        mean_duration = statistics.mean(phoneme_durations)
        if mean_duration == 0:
            return 0.0
            
        std_duration = statistics.stdev(phoneme_durations) if len(phoneme_durations) > 1 else 0
        cv = std_duration / mean_duration
        
        # Convert to precision score (lower CV = higher precision)
        return max(0.0, 1.0 - min(cv, 1.0))


@dataclass
class AlignmentStatistics:
    """
    Statistical analysis of alignment results
    """
    total_words: int
    total_phonemes: int
    total_duration: float
    average_word_duration: float
    average_phoneme_duration: float
    confidence_distribution: Dict[str, float]  # quality_level -> percentage
    timing_accuracy: float
    speech_tempo: str  # "slow", "normal", "fast"
    
    @classmethod
    def from_alignment_result(cls, result: AlignmentResult) -> 'AlignmentStatistics':
        """Create statistics from alignment result"""
        sentence = result.sentence_alignment
        
        # Calculate averages
        avg_word_duration = sentence.total_duration / sentence.word_count if sentence.word_count > 0 else 0
        avg_phoneme_duration = sentence.total_duration / sentence.total_phonemes if sentence.total_phonemes > 0 else 0
        
        # Confidence distribution
        confidence_counts = {"excellent": 0, "good": 0, "fair": 0, "poor": 0}
        total_phonemes = 0
        
        for word in sentence.words:
            for phoneme in word.phonemes:
                total_phonemes += 1
                if phoneme.confidence >= 0.9:
                    confidence_counts["excellent"] += 1
                elif phoneme.confidence >= 0.75:
                    confidence_counts["good"] += 1
                elif phoneme.confidence >= 0.5:
                    confidence_counts["fair"] += 1
                else:
                    confidence_counts["poor"] += 1
        
        confidence_distribution = {}
        if total_phonemes > 0:
            for quality, count in confidence_counts.items():
                confidence_distribution[quality] = (count / total_phonemes) * 100
        
        # Speech tempo analysis
        speech_rate = sentence.speech_rate
        if speech_rate < 120:
            tempo = "slow"
        elif speech_rate > 180:
            tempo = "fast"
        else:
            tempo = "normal"
        
        return cls(
            total_words=sentence.word_count,
            total_phonemes=sentence.total_phonemes,
            total_duration=sentence.total_duration,
            average_word_duration=avg_word_duration,
            average_phoneme_duration=avg_phoneme_duration,
            confidence_distribution=confidence_distribution,
            timing_accuracy=result.timing_precision,
            speech_tempo=tempo
        )