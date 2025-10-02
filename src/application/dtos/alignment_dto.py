"""
DTOs for forced alignment requests and responses
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from datetime import datetime
from enum import Enum

from src.domain.entities.alignment_entities import AlignmentQuality


class AlignmentRequest(BaseModel):
    """
    Request DTO for audio-text alignment
    """
    text: str = Field(..., min_length=1, max_length=5000, description="Text to align with audio")
    language: str = Field(default="english", description="Language code (e.g., 'english', 'en-us')")
    model_name: Optional[str] = Field(None, description="Specific alignment model to use")
    include_phonemes: bool = Field(default=True, description="Include phoneme-level alignment")
    include_confidence: bool = Field(default=True, description="Include confidence scores")
    preprocess_audio: bool = Field(default=True, description="Apply audio preprocessing")
    
    @validator('text')
    def validate_text(cls, v):
        """Validate alignment text"""
        if not v or not v.strip():
            raise ValueError("Text cannot be empty")
        
        # Remove excessive whitespace
        v = ' '.join(v.split())
        
        # Check for invalid characters
        invalid_chars = set('<>{}[]|\\^~`')
        if any(char in v for char in invalid_chars):
            raise ValueError(f"Text contains invalid characters: {invalid_chars}")
        
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


class BatchAlignmentRequest(BaseModel):
    """
    Request DTO for batch alignment
    """
    items: List[Dict[str, str]] = Field(..., min_items=1, max_items=100, 
                                       description="List of {text, audio_file} pairs")
    language: str = Field(default="english", description="Language for all items")
    model_name: Optional[str] = Field(None, description="Model to use for all items")
    parallel_processing: bool = Field(default=True, description="Process items in parallel")
    
    @validator('items')
    def validate_items(cls, v):
        """Validate batch items"""
        for i, item in enumerate(v):
            if 'text' not in item or 'audio_file' not in item:
                raise ValueError(f"Item {i} missing required fields: text, audio_file")
            
            if not item['text'].strip():
                raise ValueError(f"Item {i} has empty text")
        
        return v


class PhonemeAlignmentRequest(BaseModel):
    """
    Request DTO for phoneme-based alignment
    """
    phonemes: List[str] = Field(..., min_items=1, description="List of phonemes (ARPAbet format)")
    language: str = Field(default="english", description="Language code")
    
    @validator('phonemes')
    def validate_phonemes(cls, v):
        """Validate phoneme list"""
        if not v:
            raise ValueError("Phonemes list cannot be empty")
        
        # Basic validation for ARPAbet format
        for phoneme in v:
            if not phoneme.strip():
                raise ValueError("Empty phoneme found")
            
            # ARPAbet phonemes are typically uppercase letters/digits
            if not phoneme.replace('0', '').replace('1', '').replace('2', '').isalpha():
                if phoneme not in ['SIL', 'SP']:  # Special silence markers
                    raise ValueError(f"Invalid phoneme format: {phoneme}")
        
        return v


class AlignmentValidationRequest(BaseModel):
    """
    Request DTO for audio file validation
    """
    check_format: bool = Field(default=True, description="Validate audio format")
    check_duration: bool = Field(default=True, description="Validate audio duration")
    check_quality: bool = Field(default=True, description="Assess audio quality")
    extract_features: bool = Field(default=False, description="Extract audio features")


# Response DTOs

class PhonemeAlignmentResponseDTO(BaseModel):
    """
    Response DTO for phoneme alignment data
    """
    phoneme: str
    start_time: float
    end_time: float
    duration: float
    confidence: float


class WordAlignmentResponseDTO(BaseModel):
    """
    Response DTO for word alignment data
    """
    word: str
    start_time: float
    end_time: float
    duration: float
    confidence: float
    phonemes: List[PhonemeAlignmentResponseDTO]
    
    @property
    def phoneme_count(self) -> int:
        return len(self.phonemes)


class SentenceAlignmentResponseDTO(BaseModel):
    """
    Response DTO for sentence alignment data
    """
    text: str
    total_duration: float
    words: List[WordAlignmentResponseDTO]
    quality: AlignmentQuality
    overall_confidence: float
    speech_rate: float  # words per minute
    phoneme_rate: float  # phonemes per second
    silence_segments: List[Dict[str, float]]  # [{"start_time": x, "end_time": y}, ...]
    
    @property
    def word_count(self) -> int:
        return len(self.words)
    
    @property
    def total_phonemes(self) -> int:
        return sum(word.phoneme_count for word in self.words)


class AlignmentStatisticsDTO(BaseModel):
    """
    Response DTO for alignment statistics
    """
    total_words: int
    total_phonemes: int
    total_duration: float
    average_word_duration: float
    average_phoneme_duration: float
    confidence_distribution: Dict[str, float]  # quality_level -> percentage
    timing_accuracy: float
    speech_tempo: str  # "slow", "normal", "fast"


class AlignmentResponseDTO(BaseModel):
    """
    Main response DTO for alignment results
    """
    success: bool
    audio_file_path: str
    text: str
    language: str
    model_name: str
    sentence_alignment: SentenceAlignmentResponseDTO
    statistics: AlignmentStatisticsDTO
    processing_time_ms: float
    timestamp: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)
    error_message: Optional[str] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class BatchAlignmentResponseDTO(BaseModel):
    """
    Response DTO for batch alignment
    """
    success: bool
    total_items: int
    successful_items: int
    failed_items: int
    results: List[AlignmentResponseDTO]
    total_processing_time_ms: float
    timestamp: datetime
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage"""
        if self.total_items == 0:
            return 0.0
        return (self.successful_items / self.total_items) * 100


class AudioValidationResponseDTO(BaseModel):
    """
    Response DTO for audio validation
    """
    valid: bool
    file_path: str
    format: Optional[str] = None
    duration: Optional[float] = None
    sample_rate: Optional[int] = None
    channels: Optional[int] = None
    file_size: Optional[int] = None
    quality_score: Optional[float] = None
    features: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    recommendations: List[str] = Field(default_factory=list)


class SupportedLanguagesResponseDTO(BaseModel):
    """
    Response DTO for supported languages
    """
    supported_languages: List[str]
    total_languages: int
    default_language: str = "english"


class AvailableModelsResponseDTO(BaseModel):
    """
    Response DTO for available models
    """
    language: str
    models: List[str]
    default_model: Optional[str] = None
    total_models: int


class ModelInfoResponseDTO(BaseModel):
    """
    Response DTO for model information
    """
    name: str
    language: str
    type: str
    source: str
    size: Optional[str] = None
    accuracy: Optional[float] = None
    description: Optional[str] = None
    download_status: str = "unknown"  # "installed", "available", "downloading", "error"


class ExportResponseDTO(BaseModel):
    """
    Response DTO for alignment export
    """
    success: bool
    export_format: str  # "textgrid", "json", "csv"
    output_path: str
    file_size: Optional[int] = None
    error_message: Optional[str] = None


# Utility DTOs

class AlignmentProgressDTO(BaseModel):
    """
    DTO for tracking alignment progress (for long-running tasks)
    """
    task_id: str
    status: str  # "pending", "processing", "completed", "failed"
    progress_percentage: float = Field(ge=0, le=100)
    current_step: str
    estimated_time_remaining: Optional[float] = None  # seconds
    result: Optional[AlignmentResponseDTO] = None
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class AlignmentConfigDTO(BaseModel):
    """
    DTO for alignment configuration
    """
    default_language: str = "english"
    default_sample_rate: int = 16000
    max_audio_duration: float = 300.0  # 5 minutes
    min_audio_duration: float = 0.1    # 100ms
    supported_formats: List[str] = Field(default_factory=lambda: ["wav", "flac", "mp3"])
    preprocessing_enabled: bool = True
    batch_size_limit: int = 100
    parallel_processing: bool = True
    confidence_threshold: float = 0.5
    
    @validator('confidence_threshold')
    def validate_confidence_threshold(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Confidence threshold must be between 0 and 1")
        return v


# Error DTOs

class AlignmentErrorDTO(BaseModel):
    """
    DTO for alignment errors
    """
    error_type: str
    error_message: str
    error_code: Optional[str] = None
    details: Dict[str, Any] = Field(default_factory=dict)
    suggestions: List[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }