"""
Domain service interface for forced alignment
Abstracts the forced alignment functionality
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from pathlib import Path

from src.domain.entities.alignment_entities import (
    AlignmentResult,
    AlignmentStatistics,
    AlignmentQuality
)


class IAlignmentService(ABC):
    """
    Abstract interface for forced alignment services
    Supports Montreal Forced Aligner and potentially other alignment backends
    """
    
    @abstractmethod
    async def align_audio_with_text(
        self,
        audio_file_path: str,
        text: str,
        language: str = "english",
        model_name: Optional[str] = None
    ) -> AlignmentResult:
        """
        Perform forced alignment on audio file with given text
        
        Args:
            audio_file_path: Path to audio file (WAV format preferred)
            text: Text to align with audio
            language: Language code (e.g., "english", "en-us")
            model_name: Specific alignment model to use
            
        Returns:
            AlignmentResult with phoneme and word timing information
        """
        pass
    
    @abstractmethod
    async def batch_align(
        self,
        audio_text_pairs: List[tuple],  # [(audio_path, text), ...]
        language: str = "english",
        model_name: Optional[str] = None
    ) -> List[AlignmentResult]:
        """
        Perform batch forced alignment on multiple audio-text pairs
        
        Args:
            audio_text_pairs: List of (audio_file_path, text) tuples
            language: Language code
            model_name: Specific alignment model to use
            
        Returns:
            List of AlignmentResult objects
        """
        pass
    
    @abstractmethod
    async def align_with_phonemes(
        self,
        audio_file_path: str,
        phonemes: List[str],
        language: str = "english"
    ) -> AlignmentResult:
        """
        Align audio with pre-computed phoneme sequence
        
        Args:
            audio_file_path: Path to audio file
            phonemes: List of phonemes (ARPAbet format)
            language: Language code
            
        Returns:
            AlignmentResult with phoneme timing
        """
        pass
    
    @abstractmethod
    async def get_supported_languages(self) -> List[str]:
        """
        Get list of supported languages for alignment
        
        Returns:
            List of supported language codes
        """
        pass
    
    @abstractmethod
    async def get_available_models(self, language: str) -> List[str]:
        """
        Get available alignment models for a specific language
        
        Args:
            language: Language code
            
        Returns:
            List of available model names
        """
        pass
    
    @abstractmethod
    async def validate_audio_file(self, audio_file_path: str) -> bool:
        """
        Validate if audio file is suitable for forced alignment
        
        Args:
            audio_file_path: Path to audio file
            
        Returns:
            True if audio file is valid, False otherwise
        """
        pass
    
    @abstractmethod
    async def preprocess_audio(
        self,
        audio_file_path: str,
        output_path: Optional[str] = None
    ) -> str:
        """
        Preprocess audio file for optimal alignment results
        (normalize volume, convert format, remove noise, etc.)
        
        Args:
            audio_file_path: Input audio file path
            output_path: Output path (if None, creates temp file)
            
        Returns:
            Path to preprocessed audio file
        """
        pass
    
    @abstractmethod
    async def calculate_alignment_statistics(
        self,
        result: AlignmentResult
    ) -> AlignmentStatistics:
        """
        Calculate detailed statistics from alignment result
        
        Args:
            result: AlignmentResult to analyze
            
        Returns:
            AlignmentStatistics with detailed metrics
        """
        pass
    
    @abstractmethod
    async def export_alignment_to_textgrid(
        self,
        result: AlignmentResult,
        output_path: str
    ) -> bool:
        """
        Export alignment result to Praat TextGrid format
        
        Args:
            result: AlignmentResult to export
            output_path: Path for TextGrid file
            
        Returns:
            True if export successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def export_alignment_to_json(
        self,
        result: AlignmentResult,
        output_path: str
    ) -> bool:
        """
        Export alignment result to JSON format
        
        Args:
            result: AlignmentResult to export
            output_path: Path for JSON file
            
        Returns:
            True if export successful, False otherwise
        """
        pass


class IAlignmentModelManager(ABC):
    """
    Abstract interface for managing alignment models
    """
    
    @abstractmethod
    async def download_model(self, language: str, model_name: str) -> bool:
        """
        Download alignment model for specific language
        
        Args:
            language: Language code
            model_name: Model name to download
            
        Returns:
            True if download successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def list_installed_models(self) -> Dict[str, List[str]]:
        """
        List all installed alignment models
        
        Returns:
            Dictionary mapping language -> list of model names
        """
        pass
    
    @abstractmethod
    async def get_model_info(self, language: str, model_name: str) -> Dict[str, Any]:
        """
        Get information about specific alignment model
        
        Args:
            language: Language code
            model_name: Model name
            
        Returns:
            Dictionary with model information (size, accuracy, etc.)
        """
        pass
    
    @abstractmethod
    async def remove_model(self, language: str, model_name: str) -> bool:
        """
        Remove installed alignment model
        
        Args:
            language: Language code
            model_name: Model name to remove
            
        Returns:
            True if removal successful, False otherwise
        """
        pass