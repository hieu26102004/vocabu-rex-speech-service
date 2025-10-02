"""
Domain entities for pronunciation assessment with forced alignment
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict, Any
from uuid import UUID


@dataclass
class PhonemeTarget:
    """Target phoneme from phonemization (Step 1)"""
    phoneme: str
    ipa: str
    word_index: int
    phoneme_index: int
    expected_duration: Optional[float] = None


@dataclass
class PhonemeAlignment:
    """Phoneme alignment result from Forced Alignment (Step 2)"""
    phoneme: str
    start_time: float
    end_time: float
    confidence: float
    word_index: int
    phoneme_index: int
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclass
class PhonemeActual:
    """Actual phoneme from ASR (Step 3)"""
    phoneme: str
    ipa: str
    confidence: float
    word_index: int
    phoneme_index: int


@dataclass
class PhonemeError:
    """Detailed phoneme error analysis (Step 4)"""
    phoneme_index: int
    word_index: int
    expected_phoneme: str
    actual_phoneme: Optional[str]
    error_type: str  # 'substitution', 'omission', 'insertion', 'distortion'
    severity: str    # 'minor', 'major', 'critical'
    start_time: float
    end_time: float
    confidence: float
    acoustic_score: Optional[float] = None
    formant_data: Optional[Dict[str, float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to API response format"""
        return {
            "phoneme": self.expected_phoneme,
            "expected": self.expected_phoneme,
            "actual": self.actual_phoneme,
            "error_type": self.error_type,
            "severity": self.severity,
            "timestamp_start": self.start_time,
            "timestamp_end": self.end_time,
            "confidence": self.confidence,
            "acoustic_score": self.acoustic_score,
            "formant_data": self.formant_data or {}
        }


@dataclass
class WordAnalysis:
    """Word-level pronunciation analysis"""
    word: str
    word_index: int
    start_time: float
    end_time: float
    target_phonemes: List[PhonemeTarget]
    actual_phonemes: List[PhonemeActual]
    alignment_results: List[PhonemeAlignment]
    errors: List[PhonemeError]
    
    # Scores
    word_score: float
    accuracy_percentage: float
    stress_accuracy: float
    rhythm_score: float
    
    # Error counts
    total_phoneme_errors: int
    substitution_errors: int
    omission_errors: int
    insertion_errors: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to API response format"""
        return {
            "word": self.word,
            "score": round(self.word_score, 2),
            "accuracy_percentage": round(self.accuracy_percentage, 2),
            "start_time": self.start_time,
            "end_time": self.end_time,
            "errors": [error.to_dict() for error in self.errors],
            "phoneme_breakdown": {
                "total": len(self.target_phonemes),
                "correct": len(self.target_phonemes) - self.total_phoneme_errors,
                "substitutions": self.substitution_errors,
                "omissions": self.omission_errors,
                "insertions": self.insertion_errors
            },
            "target_phonemes": [
                {"phoneme": p.phoneme, "ipa": p.ipa} 
                for p in self.target_phonemes
            ],
            "actual_phonemes": [
                {"phoneme": p.phoneme, "ipa": p.ipa, "confidence": p.confidence} 
                for p in self.actual_phonemes
            ]
        }


@dataclass
class PronunciationAssessment:
    """Complete pronunciation assessment result"""
    id: UUID
    session_id: UUID
    target_text: str
    spoken_text: Optional[str]
    
    # Step 1: Phonemization
    target_phonemes: List[List[PhonemeTarget]]  # Per word
    phonemization_engine: str
    
    # Step 2: Forced Alignment
    alignment_data: List[PhonemeAlignment]
    alignment_engine: str
    
    # Step 3: ASR Results
    asr_phonemes: List[List[PhonemeActual]]  # Per word
    asr_engine: str
    
    # Step 4: Analysis Results
    word_analyses: List[WordAnalysis]
    
    # Overall Scores
    overall_score: float
    accuracy_score: float
    fluency_score: float
    pronunciation_score: float
    
    # Processing info
    processing_duration_ms: int
    created_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to API response format"""
        return {
            "assessment_id": str(self.id),
            "target_text": self.target_text,
            "spoken_text": self.spoken_text,
            "overall_score": round(self.overall_score, 2),
            "scores": {
                "accuracy": round(self.accuracy_score, 2),
                "fluency": round(self.fluency_score, 2),
                "pronunciation": round(self.pronunciation_score, 2)
            },
            "word_level_analysis": [word.to_dict() for word in self.word_analyses],
            "summary": {
                "total_words": len(self.word_analyses),
                "total_errors": sum(w.total_phoneme_errors for w in self.word_analyses),
                "error_types": {
                    "substitutions": sum(w.substitution_errors for w in self.word_analyses),
                    "omissions": sum(w.omission_errors for w in self.word_analyses), 
                    "insertions": sum(w.insertion_errors for w in self.word_analyses)
                }
            },
            "processing_info": {
                "phonemization_engine": self.phonemization_engine,
                "alignment_engine": self.alignment_engine,
                "asr_engine": self.asr_engine,
                "duration_ms": self.processing_duration_ms
            },
            "timestamp": self.created_at.isoformat()
        }