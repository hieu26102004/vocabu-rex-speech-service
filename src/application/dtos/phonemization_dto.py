"""
DTOs for phonemization requests and responses
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime


class PhonemizationRequest(BaseModel):
    """Request DTO for text phonemization"""
    
    text: str = Field(
        ..., 
        min_length=1, 
        max_length=1000,
        description="Text to be converted to phonemes"
    )
    language: str = Field(
        default="en-us",
        description="Language code for phonemization"
    )
    backend: Optional[str] = Field(
        default=None,
        description="Phonemization backend to use (espeak, g2p, festival)"
    )
    include_stress: bool = Field(
        default=True,
        description="Include stress markers in output"
    )
    include_syllables: bool = Field(
        default=True,
        description="Include syllable count in output"
    )
    
    @validator('text')
    def validate_text(cls, v):
        if not v.strip():
            raise ValueError('Text cannot be empty')
        return v.strip()
    
    @validator('language')
    def validate_language(cls, v):
        supported_langs = ["en-us", "en-gb", "en-au", "en", "es", "fr", "de", "it", "pt"]
        if v not in supported_langs:
            raise ValueError(f'Language {v} not supported. Supported: {supported_langs}')
        return v

    class Config:
        schema_extra = {
            "example": {
                "text": "Hello world",
                "language": "en-us",
                "backend": "espeak",
                "include_stress": True,
                "include_syllables": True
            }
        }


class WordPhonemizationRequest(BaseModel):
    """Request DTO for word list phonemization"""
    
    words: List[str] = Field(
        ...,
        min_items=1,
        max_items=100,
        description="List of words to phonemize"
    )
    language: str = Field(
        default="en-us",
        description="Language code for phonemization"
    )
    backend: Optional[str] = Field(
        default=None,
        description="Phonemization backend to use"
    )
    
    @validator('words')
    def validate_words(cls, v):
        cleaned_words = []
        for word in v:
            if word and word.strip():
                cleaned_words.append(word.strip())
        if not cleaned_words:
            raise ValueError('At least one valid word required')
        return cleaned_words

    class Config:
        schema_extra = {
            "example": {
                "words": ["hello", "world", "pronunciation"],
                "language": "en-us",
                "backend": "espeak"
            }
        }


class PhonemeResultDTO(BaseModel):
    """DTO for individual word phonemization result"""
    
    word: str = Field(..., description="Original word")
    phonemes: List[str] = Field(..., description="List of phoneme symbols")
    ipa: str = Field(..., description="IPA representation")
    stress_pattern: Optional[str] = Field(None, description="Stress pattern markers")
    syllable_count: Optional[int] = Field(None, description="Number of syllables")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    
    class Config:
        schema_extra = {
            "example": {
                "word": "hello",
                "phonemes": ["h", "ə", "ˈl", "oʊ"],
                "ipa": "/həˈloʊ/",
                "stress_pattern": "ˈ",
                "syllable_count": 2,
                "confidence": 0.95
            }
        }


class PhonemizationResponseDTO(BaseModel):
    """Response DTO for phonemization result"""
    
    success: bool = Field(True, description="Success status")
    original_text: str = Field(..., description="Original input text")
    language: str = Field(..., description="Language used for phonemization")
    backend: str = Field(..., description="Backend engine used")
    words: List[PhonemeResultDTO] = Field(..., description="Phonemization results per word")
    
    # Summary statistics
    total_words: int = Field(..., description="Total number of words processed")
    total_phonemes: int = Field(..., description="Total number of phonemes")
    processing_time_ms: Optional[int] = Field(None, description="Processing time in milliseconds")
    
    # Metadata
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "original_text": "Hello world",
                "language": "en-us",
                "backend": "espeak",
                "words": [
                    {
                        "word": "hello",
                        "phonemes": ["h", "ə", "ˈl", "oʊ"],
                        "ipa": "/həˈloʊ/",
                        "stress_pattern": "ˈ",
                        "syllable_count": 2,
                        "confidence": 0.95
                    },
                    {
                        "word": "world", 
                        "phonemes": ["w", "ɜː", "l", "d"],
                        "ipa": "/wɜːld/",
                        "syllable_count": 1,
                        "confidence": 0.98
                    }
                ],
                "total_words": 2,
                "total_phonemes": 8,
                "processing_time_ms": 45,
                "timestamp": "2025-10-02T10:30:00Z"
            }
        }


class IpaConversionRequest(BaseModel):
    """Request DTO for IPA representation"""
    
    text: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Text to convert to IPA"
    )
    language: str = Field(
        default="en-us",
        description="Language code"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "text": "Hello world",
                "language": "en-us"
            }
        }


class IpaConversionResponseDTO(BaseModel):
    """Response DTO for IPA conversion"""
    
    success: bool = Field(True, description="Success status")
    original_text: str = Field(..., description="Original input text")
    ipa_representation: str = Field(..., description="IPA representation")
    language: str = Field(..., description="Language used")
    processing_time_ms: Optional[int] = Field(None, description="Processing time")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "original_text": "Hello world",
                "ipa_representation": "/həˈloʊ wɜːld/",
                "language": "en-us",
                "processing_time_ms": 25,
                "timestamp": "2025-10-02T10:30:00Z"
            }
        }


class SupportedLanguagesResponseDTO(BaseModel):
    """Response DTO for supported languages"""
    
    success: bool = Field(True, description="Success status")
    supported_languages: List[str] = Field(..., description="List of supported language codes")
    default_language: str = Field(..., description="Default language code")
    total_languages: int = Field(..., description="Total number of supported languages")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "supported_languages": ["en-us", "en-gb", "en-au", "es", "fr", "de"],
                "default_language": "en-us", 
                "total_languages": 6
            }
        }


class ValidationErrorDTO(BaseModel):
    """DTO for validation errors"""
    
    success: bool = Field(False, description="Success status")
    error: str = Field(..., description="Error message")
    field: Optional[str] = Field(None, description="Field that caused the error")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "success": False,
                "error": "Text cannot be empty",
                "field": "text",
                "details": {"provided_value": ""},
                "timestamp": "2025-10-02T10:30:00Z"
            }
        }