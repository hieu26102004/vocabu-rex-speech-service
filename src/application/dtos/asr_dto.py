"""
DTOs for Enhanced ASR (Step 3) requests and responses
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from datetime import datetime
from enum import Enum

from src.domain.entities.asr_entities import TranscriptionQuality, PronunciationAccuracy


class ASRRequest(BaseModel):
    """
    Request DTO for enhanced ASR transcription
    """
    reference_text: Optional[str] = Field(None, max_length=5000, description="Reference text for comparison")
    language: str = Field(default="english", description="Language code")
    model_size: str = Field(default="base", description="Whisper model size")
    include_phonemes: bool = Field(default=True, description="Include phoneme-level analysis")
    include_timing: bool = Field(default=True, description="Include word/phoneme timing")
    include_confidence: bool = Field(default=True, description="Include confidence scores")
    compare_pronunciation: bool = Field(default=True, description="Compare with reference pronunciation")
    analyze_fluency: bool = Field(default=True, description="Analyze speech fluency")
    
    @validator('model_size')
    def validate_model_size(cls, v):
        """Validate Whisper model size"""
        valid_models = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]
        if v not in valid_models:
            raise ValueError(f"Invalid model size. Supported: {valid_models}")
        return v
    
    @validator('language')
    def validate_language(cls, v):
        """Validate language code"""
        supported_languages = [
            'english', 'en-us', 'en-uk', 'spanish', 'french', 
            'german', 'italian', 'portuguese', 'russian', 'chinese'
        ]
        if v.lower() not in supported_languages:
            raise ValueError(f"Unsupported language: {v}. Supported: {supported_languages}")
        return v.lower()


class BatchASRRequest(BaseModel):
    """
    Request DTO for batch ASR processing
    """
    audio_files: List[str] = Field(..., min_items=1, max_items=50, description="List of audio file identifiers")
    reference_texts: Optional[List[str]] = Field(None, description="List of reference texts")
    language: str = Field(default="english", description="Language for all files")
    model_size: str = Field(default="base", description="Whisper model size")
    parallel_processing: bool = Field(default=True, description="Process files in parallel")
    
    @validator('reference_texts')
    def validate_reference_texts(cls, v, values):
        """Validate reference texts match audio files count"""
        if v is not None and 'audio_files' in values:
            if len(v) != len(values['audio_files']):
                raise ValueError("Reference texts count must match audio files count")
        return v


class PronunciationAnalysisRequest(BaseModel):
    """
    Request DTO for pronunciation analysis with reference
    """
    reference_text: str = Field(..., min_length=1, description="Reference text")
    reference_phonemes: Optional[List[str]] = Field(None, description="Reference phonemes from Step 1")
    reference_alignment: Optional[Dict[str, Any]] = Field(None, description="Reference alignment from Step 2")
    language: str = Field(default="english", description="Language code")
    detailed_feedback: bool = Field(default=True, description="Include detailed feedback")
    native_language: Optional[str] = Field(None, description="Learner's native language for error prediction")


# Response DTOs

class ActualPhonemeResponseDTO(BaseModel):
    """
    Response DTO for actual phoneme data
    """
    phoneme: str
    start_time: float
    end_time: float
    duration: float
    confidence: float
    amplitude: float
    fundamental_frequency: Optional[float] = None


class ActualWordResponseDTO(BaseModel):
    """
    Response DTO for actual word data
    """
    word: str
    start_time: float
    end_time: float
    duration: float
    confidence: float
    phonemes: List[ActualPhonemeResponseDTO]
    pronunciation_score: Optional[float] = None
    
    @property
    def phoneme_count(self) -> int:
        return len(self.phonemes)


class ActualUtteranceResponseDTO(BaseModel):
    """
    Response DTO for actual utterance
    """
    transcribed_text: str
    original_text: str
    total_duration: float
    words: List[ActualWordResponseDTO]
    overall_confidence: float
    transcription_quality: TranscriptionQuality
    pronunciation_accuracy: PronunciationAccuracy
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


class PronunciationComparisonResponseDTO(BaseModel):
    """
    Response DTO for phoneme comparison
    """
    reference_phoneme: str
    actual_phoneme: str
    phoneme_match: bool
    similarity_score: float
    timing_deviation: float
    error_type: Optional[str] = None
    
    @property
    def is_accurate(self) -> bool:
        return self.phoneme_match and self.similarity_score >= 0.8


class WordComparisonResponseDTO(BaseModel):
    """
    Response DTO for word comparison
    """
    reference_word: str
    actual_word: str
    word_match: bool
    phoneme_comparisons: List[PronunciationComparisonResponseDTO]
    overall_accuracy: float
    timing_accuracy: float
    
    @property
    def phoneme_accuracy_rate(self) -> float:
        if not self.phoneme_comparisons:
            return 0.0
        accurate_phonemes = sum(1 for pc in self.phoneme_comparisons if pc.is_accurate)
        return accurate_phonemes / len(self.phoneme_comparisons)


class ASRStatisticsDTO(BaseModel):
    """
    Response DTO for ASR statistics
    """
    word_error_rate: float
    character_error_rate: float
    phoneme_error_rate: float
    average_word_duration: float
    average_phoneme_duration: float
    timing_precision: float
    correctly_pronounced_phonemes: int
    total_phonemes: int
    phoneme_accuracy_percentage: float
    speaking_speed: float  # normalized to average
    pause_patterns_score: float
    rhythm_score: float
    confidence_distribution: Dict[str, float]


class PronunciationFeedbackDTO(BaseModel):
    """
    Response DTO for pronunciation feedback
    """
    overall_feedback: str
    strengths: List[str]
    areas_for_improvement: List[str]
    specific_phoneme_feedback: List[Dict[str, str]]
    practice_suggestions: List[str]
    difficulty_level_recommendation: str
    confidence_level: str = "intermediate"


class ASRResponseDTO(BaseModel):
    """
    Main response DTO for ASR results
    """
    success: bool
    audio_file_path: str
    reference_text: str
    actual_utterance: ActualUtteranceResponseDTO
    word_comparisons: List[WordComparisonResponseDTO]
    overall_pronunciation_score: float  # 0-100
    fluency_score: float  # 0-100
    accuracy_score: float  # 0-100
    total_score: float  # Combined score
    pronunciation_grade: str  # Letter grade (A+, A, B+, etc.)
    statistics: ASRStatisticsDTO
    feedback: PronunciationFeedbackDTO
    processing_time_ms: float
    timestamp: datetime
    whisper_model_used: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    error_message: Optional[str] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class BatchASRResponseDTO(BaseModel):
    """
    Response DTO for batch ASR processing
    """
    success: bool
    total_files: int
    successful_files: int
    failed_files: int
    results: List[ASRResponseDTO]
    average_pronunciation_score: float
    average_fluency_score: float
    average_accuracy_score: float
    total_processing_time_ms: float
    timestamp: datetime
    
    @property
    def success_rate(self) -> float:
        if self.total_files == 0:
            return 0.0
        return (self.successful_files / self.total_files) * 100


class TranscriptionOnlyResponseDTO(BaseModel):
    """
    Response DTO for transcription-only (no reference text)
    """
    success: bool
    audio_file_path: str
    transcribed_text: str
    confidence: float
    words: List[ActualWordResponseDTO]
    transcription_quality: TranscriptionQuality
    processing_time_ms: float
    timestamp: datetime
    whisper_model_used: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    error_message: Optional[str] = None


class ModelInfoResponseDTO(BaseModel):
    """
    Response DTO for Whisper model information
    """
    name: str
    type: str = "whisper"
    parameters: Optional[str] = None
    size: Optional[str] = None
    speed: Optional[str] = None
    download_status: str = "available"
    accuracy_description: Optional[str] = None


class SupportedLanguagesASRResponseDTO(BaseModel):
    """
    Response DTO for supported ASR languages
    """
    supported_languages: List[str]
    total_languages: int
    default_language: str = "english"
    whisper_languages: List[str]  # Languages specifically supported by Whisper


# Integration DTOs (for combining Steps 1, 2, 3)

class FullPronunciationAnalysisRequest(BaseModel):
    """
    Request DTO for complete pronunciation analysis (Steps 1+2+3)
    """
    text: str = Field(..., min_length=1, max_length=5000, description="Text to analyze")
    language: str = Field(default="english", description="Language code")
    whisper_model_size: str = Field(default="base", description="Whisper model size")
    alignment_model: Optional[str] = Field(None, description="Forced alignment model")
    include_phonemization: bool = Field(default=True, description="Include Step 1: Phonemization")
    include_alignment: bool = Field(default=True, description="Include Step 2: Forced Alignment")
    include_asr_analysis: bool = Field(default=True, description="Include Step 3: ASR Analysis")
    detailed_feedback: bool = Field(default=True, description="Include detailed feedback")
    native_language: Optional[str] = Field(None, description="Learner's native language")


class FullPronunciationAnalysisResponseDTO(BaseModel):
    """
    Response DTO for complete pronunciation analysis
    """
    success: bool
    audio_file_path: str
    reference_text: str
    
    # Step 1: Phonemization results
    reference_phonemes: List[str]
    phonemization_success: bool
    
    # Step 2: Forced Alignment results
    reference_timing: Optional[Dict[str, Any]] = None
    alignment_success: bool
    
    # Step 3: ASR Analysis results
    asr_result: ASRResponseDTO
    asr_success: bool
    
    # Combined analysis
    overall_pronunciation_assessment: Dict[str, Any]
    learning_recommendations: List[str]
    next_practice_topics: List[str]
    
    # Processing info
    total_processing_time_ms: float
    timestamp: datetime
    pipeline_version: str = "1.0.0"


class PronunciationReportDTO(BaseModel):
    """
    Comprehensive pronunciation assessment report
    """
    learner_id: Optional[str] = None
    assessment_date: datetime
    language: str
    total_sessions: int
    
    # Overall scores
    overall_pronunciation_score: float
    overall_fluency_score: float
    overall_accuracy_score: float
    improvement_trend: str  # "improving", "stable", "declining"
    
    # Detailed analysis
    phoneme_accuracy_breakdown: Dict[str, float]  # phoneme -> accuracy percentage
    common_error_patterns: List[Dict[str, Any]]
    fluency_metrics: Dict[str, float]
    
    # Recommendations
    focus_areas: List[str]
    practice_exercises: List[str]
    estimated_improvement_time: str
    
    # Progress tracking
    session_history: List[Dict[str, Any]]
    milestone_achievements: List[str]
    next_milestones: List[str]


# Validation DTOs

class ASRValidationRequest(BaseModel):
    """
    Request DTO for ASR validation
    """
    check_audio_quality: bool = Field(default=True, description="Validate audio quality")
    check_duration: bool = Field(default=True, description="Validate audio duration")
    check_format: bool = Field(default=True, description="Validate audio format")
    whisper_compatibility: bool = Field(default=True, description="Check Whisper compatibility")


class ASRValidationResponseDTO(BaseModel):
    """
    Response DTO for ASR validation
    """
    valid: bool
    audio_file_path: str
    validation_details: Dict[str, Any]
    whisper_ready: bool
    recommendations: List[str]
    estimated_transcription_time: Optional[float] = None
    error_message: Optional[str] = None


# Configuration DTOs

class ASRConfigDTO(BaseModel):
    """
    DTO for ASR configuration
    """
    default_model_size: str = "base"
    default_language: str = "english"
    max_audio_duration: float = 300.0  # 5 minutes
    min_audio_duration: float = 0.5    # 500ms
    confidence_threshold: float = 0.7
    enable_phoneme_analysis: bool = True
    enable_fluency_analysis: bool = True
    batch_size_limit: int = 50
    
    @validator('confidence_threshold')
    def validate_confidence_threshold(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Confidence threshold must be between 0 and 1")
        return v


# Error DTOs

class ASRErrorDTO(BaseModel):
    """
    DTO for ASR errors
    """
    error_type: str
    error_message: str
    error_code: Optional[str] = None
    audio_file_path: Optional[str] = None
    processing_stage: str  # "preprocessing", "transcription", "analysis"
    suggestions: List[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }