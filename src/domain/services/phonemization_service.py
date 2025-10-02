"""
Domain service interface for phonemization (Step 1)
Abstract interface for converting text to phonemes
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class PhonemeResult:
    """Result of phonemization for a single word"""
    word: str
    phonemes: List[str]  # List of phoneme symbols
    ipa: str             # IPA representation
    stress_pattern: Optional[str] = None  # Stress markers
    syllable_count: Optional[int] = None
    confidence: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "word": self.word,
            "phonemes": self.phonemes,
            "ipa": self.ipa,
            "stress_pattern": self.stress_pattern,
            "syllable_count": self.syllable_count,
            "confidence": self.confidence
        }


@dataclass 
class PhonemizationResult:
    """Complete phonemization result for input text"""
    original_text: str
    words: List[PhonemeResult]
    language: str
    backend: str
    processing_time_ms: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_text": self.original_text,
            "words": [word.to_dict() for word in self.words],
            "language": self.language, 
            "backend": self.backend,
            "processing_time_ms": self.processing_time_ms,
            "total_words": len(self.words),
            "total_phonemes": sum(len(word.phonemes) for word in self.words)
        }


class IPhonemizationService(ABC):
    """
    Abstract interface for phonemization services
    Converts text to phonemic representation (Step 1 of pronunciation assessment)
    """
    
    @abstractmethod
    async def phonemize_text(
        self, 
        text: str, 
        language: str = "en-us"
    ) -> PhonemizationResult:
        """
        Convert text to phonemes
        
        Args:
            text: Input text to phonemize
            language: Language code (e.g., "en-us", "en-gb")
            
        Returns:
            PhonemizationResult with phonemes for each word
        """
        pass
    
    @abstractmethod
    async def phonemize_words(
        self,
        words: List[str],
        language: str = "en-us"
    ) -> List[PhonemeResult]:
        """
        Convert list of words to phonemes
        
        Args:
            words: List of words to phonemize
            language: Language code
            
        Returns:
            List of PhonemeResult objects
        """
        pass
    
    @abstractmethod
    async def get_ipa_representation(
        self,
        text: str,
        language: str = "en-us"
    ) -> str:
        """
        Get IPA representation of text
        
        Args:
            text: Input text
            language: Language code
            
        Returns:
            IPA representation string
        """
        pass
    
    @abstractmethod
    def get_supported_languages(self) -> List[str]:
        """
        Get list of supported language codes
        
        Returns:
            List of supported language codes
        """
        pass
    
    @abstractmethod
    def validate_text(self, text: str) -> bool:
        """
        Validate if text can be phonemized
        
        Args:
            text: Text to validate
            
        Returns:
            True if text is valid for phonemization
        """
        pass