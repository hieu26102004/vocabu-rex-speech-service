"""
Domain service interface for Enhanced ASR (Step 3)
Abstracts enhanced speech recognition with phoneme-level analysis
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path

from src.domain.entities.asr_entities import (
    ASRResult,
    ActualUtterance,
    PronunciationComparison,
    WordComparison,
    EnhancedASRStatistics,
    PronunciationFeedback
)
from src.domain.entities.alignment_entities import AlignmentResult


class IEnhancedASRService(ABC):
    """
    Abstract interface for enhanced ASR services
    Supports Whisper-based transcription with phoneme-level analysis
    """
    
    @abstractmethod
    async def transcribe_with_phonemes(
        self,
        audio_file_path: str,
        reference_text: Optional[str] = None,
        language: str = "english",
        model_size: str = "base"
    ) -> ASRResult:
        """
        Transcribe audio with phoneme-level analysis
        
        Args:
            audio_file_path: Path to audio file
            reference_text: Expected text for comparison
            language: Language code
            model_size: Whisper model size (tiny, base, small, medium, large)
            
        Returns:
            ASRResult with actual pronunciation and comparison
        """
        pass
    
    @abstractmethod
    async def transcribe_and_align(
        self,
        audio_file_path: str,
        reference_text: str,
        reference_alignment: Optional[AlignmentResult] = None,
        language: str = "english"
    ) -> ASRResult:
        """
        Transcribe audio and compare with reference alignment
        
        Args:
            audio_file_path: Path to audio file
            reference_text: Reference text
            reference_alignment: Reference alignment from Step 2
            language: Language code
            
        Returns:
            ASRResult with pronunciation comparison
        """
        pass
    
    @abstractmethod
    async def batch_transcribe(
        self,
        audio_files: List[str],
        reference_texts: Optional[List[str]] = None,
        language: str = "english"
    ) -> List[ASRResult]:
        """
        Batch transcription with pronunciation analysis
        
        Args:
            audio_files: List of audio file paths
            reference_texts: List of reference texts (optional)
            language: Language code
            
        Returns:
            List of ASRResult objects
        """
        pass
    
    @abstractmethod
    async def compare_pronunciation(
        self,
        actual_utterance: ActualUtterance,
        reference_text: str,
        reference_phonemes: Optional[List[str]] = None
    ) -> List[WordComparison]:
        """
        Compare actual pronunciation with reference
        
        Args:
            actual_utterance: Captured actual pronunciation
            reference_text: Reference text
            reference_phonemes: Reference phoneme sequence
            
        Returns:
            List of word-level comparisons
        """
        pass
    
    @abstractmethod
    async def calculate_pronunciation_score(
        self,
        word_comparisons: List[WordComparison],
        fluency_metrics: Dict[str, float]
    ) -> Tuple[float, float, float]:
        """
        Calculate pronunciation, fluency, and accuracy scores
        
        Args:
            word_comparisons: Word-level pronunciation comparisons
            fluency_metrics: Fluency analysis metrics
            
        Returns:
            Tuple of (pronunciation_score, fluency_score, accuracy_score)
        """
        pass
    
    @abstractmethod
    async def analyze_fluency(
        self,
        actual_utterance: ActualUtterance,
        reference_duration: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Analyze speech fluency metrics
        
        Args:
            actual_utterance: Captured speech
            reference_duration: Expected duration for comparison
            
        Returns:
            Dictionary with fluency metrics
        """
        pass
    
    @abstractmethod
    async def get_supported_languages(self) -> List[str]:
        """
        Get languages supported by ASR service
        
        Returns:
            List of supported language codes
        """
        pass
    
    @abstractmethod
    async def get_available_models(self) -> List[str]:
        """
        Get available Whisper model sizes
        
        Returns:
            List of model names (tiny, base, small, medium, large)
        """
        pass
    
    @abstractmethod
    async def validate_audio_for_transcription(self, audio_file_path: str) -> bool:
        """
        Validate if audio file is suitable for transcription
        
        Args:
            audio_file_path: Path to audio file
            
        Returns:
            True if audio is suitable, False otherwise
        """
        pass


class IPronunciationAnalyzer(ABC):
    """
    Abstract interface for pronunciation analysis
    """
    
    @abstractmethod
    async def analyze_phoneme_accuracy(
        self,
        actual_phonemes: List[str],
        reference_phonemes: List[str]
    ) -> List[PronunciationComparison]:
        """
        Analyze phoneme-level pronunciation accuracy
        
        Args:
            actual_phonemes: Phonemes from speech recognition
            reference_phonemes: Expected phonemes
            
        Returns:
            List of phoneme comparisons
        """
        pass
    
    @abstractmethod
    async def calculate_similarity_score(
        self,
        actual_phoneme: str,
        reference_phoneme: str
    ) -> float:
        """
        Calculate similarity score between two phonemes
        
        Args:
            actual_phoneme: Recognized phoneme
            reference_phoneme: Expected phoneme
            
        Returns:
            Similarity score 0.0-1.0
        """
        pass
    
    @abstractmethod
    async def identify_error_type(
        self,
        actual_phonemes: List[str],
        reference_phonemes: List[str],
        index: int
    ) -> str:
        """
        Identify type of pronunciation error
        
        Args:
            actual_phonemes: Recognized phoneme sequence
            reference_phonemes: Expected phoneme sequence
            index: Index of error
            
        Returns:
            Error type: "substitution", "insertion", "deletion"
        """
        pass
    
    @abstractmethod
    async def generate_feedback(
        self,
        asr_result: ASRResult
    ) -> PronunciationFeedback:
        """
        Generate detailed pronunciation feedback
        
        Args:
            asr_result: Complete ASR analysis result
            
        Returns:
            Structured pronunciation feedback
        """
        pass


class IFluencyAnalyzer(ABC):
    """
    Abstract interface for speech fluency analysis
    """
    
    @abstractmethod
    async def analyze_speech_rate(
        self,
        utterance: ActualUtterance,
        reference_rate: Optional[float] = None
    ) -> float:
        """
        Analyze speaking rate (words per minute)
        
        Args:
            utterance: Actual speech utterance
            reference_rate: Expected rate for comparison
            
        Returns:
            Speech rate score 0.0-100.0
        """
        pass
    
    @abstractmethod
    async def analyze_pause_patterns(
        self,
        utterance: ActualUtterance
    ) -> Dict[str, float]:
        """
        Analyze pause patterns in speech
        
        Args:
            utterance: Actual speech utterance
            
        Returns:
            Dictionary with pause analysis metrics
        """
        pass
    
    @abstractmethod
    async def analyze_rhythm(
        self,
        utterance: ActualUtterance
    ) -> float:
        """
        Analyze speech rhythm and timing
        
        Args:
            utterance: Actual speech utterance
            
        Returns:
            Rhythm score 0.0-100.0
        """
        pass
    
    @abstractmethod
    async def calculate_fluency_score(
        self,
        speech_rate_score: float,
        pause_score: float,
        rhythm_score: float
    ) -> float:
        """
        Calculate overall fluency score
        
        Args:
            speech_rate_score: Speaking rate score
            pause_score: Pause pattern score
            rhythm_score: Speech rhythm score
            
        Returns:
            Overall fluency score 0.0-100.0
        """
        pass


class IWhisperModelManager(ABC):
    """
    Abstract interface for managing Whisper models
    """
    
    @abstractmethod
    async def download_model(self, model_size: str) -> bool:
        """
        Download Whisper model
        
        Args:
            model_size: Model size (tiny, base, small, medium, large)
            
        Returns:
            True if download successful
        """
        pass
    
    @abstractmethod
    async def list_downloaded_models(self) -> List[str]:
        """
        List downloaded Whisper models
        
        Returns:
            List of downloaded model names
        """
        pass
    
    @abstractmethod
    async def get_model_info(self, model_size: str) -> Dict[str, Any]:
        """
        Get information about Whisper model
        
        Args:
            model_size: Model size
            
        Returns:
            Dictionary with model information
        """
        pass
    
    @abstractmethod
    async def remove_model(self, model_size: str) -> bool:
        """
        Remove downloaded Whisper model
        
        Args:
            model_size: Model size to remove
            
        Returns:
            True if removal successful
        """
        pass


class IPronunciationIntegrator(ABC):
    """
    Abstract interface for integrating Steps 1, 2, and 3
    """
    
    @abstractmethod
    async def full_pronunciation_analysis(
        self,
        audio_file_path: str,
        reference_text: str,
        language: str = "english"
    ) -> ASRResult:
        """
        Complete pronunciation analysis using all 3 steps
        
        Step 1: Text → Phonemes
        Step 2: Audio + Phonemes → Reference Timing  
        Step 3: Audio → Actual Pronunciation + Comparison
        
        Args:
            audio_file_path: Path to audio file
            reference_text: Reference text to compare against
            language: Language code
            
        Returns:
            Complete ASR result with pronunciation analysis
        """
        pass
    
    @abstractmethod
    async def segment_and_analyze(
        self,
        audio_file_path: str,
        reference_text: str,
        segment_duration: float = 5.0
    ) -> List[ASRResult]:
        """
        Segment long audio and analyze each segment
        
        Args:
            audio_file_path: Path to long audio file
            reference_text: Reference text
            segment_duration: Duration of each segment in seconds
            
        Returns:
            List of ASRResult for each segment
        """
        pass
    
    @abstractmethod
    async def generate_pronunciation_report(
        self,
        asr_results: List[ASRResult]
    ) -> Dict[str, Any]:
        """
        Generate comprehensive pronunciation assessment report
        
        Args:
            asr_results: List of ASR analysis results
            
        Returns:
            Comprehensive pronunciation report
        """
        pass