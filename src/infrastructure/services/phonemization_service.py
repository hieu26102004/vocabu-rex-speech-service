"""
Concrete implementation of phonemization service
Uses phonemizer, g2p-en, and espeak backends
"""

import time
import re
import logging
from typing import List, Dict, Any, Optional
import asyncio
from functools import lru_cache

try:
    from phonemizer import phonemize
    from phonemizer.backend import EspeakBackend, FestivalBackend
    from phonemizer.separator import Separator
except ImportError as e:
    logging.error(f"Phonemizer not installed: {e}")
    phonemize = None
    EspeakBackend = None

try:
    from g2p_en import G2p
except ImportError as e:
    logging.error(f"G2P-EN not installed: {e}")
    G2p = None

from src.domain.services.phonemization_service import (
    IPhonemizationService, 
    PhonemizationResult, 
    PhonemeResult
)
from src.shared.exceptions import (
    SpeechServiceException,
    AudioProcessingError
)
from src.shared.config import get_settings


class PhonemizationService(IPhonemizationService):
    """
    Concrete implementation using multiple phonemization backends
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.logger = logging.getLogger(__name__)
        self._g2p_model = None
        self._espeak_backend = None
        self._initialize_backends()
    
    def _initialize_backends(self):
        """Initialize phonemization backends"""
        try:
            # Initialize G2P model for English
            if G2p:
                self._g2p_model = G2p()
                self.logger.info("G2P-EN model initialized")
            
            # Initialize eSpeak backend
            if EspeakBackend:
                self._espeak_backend = EspeakBackend(
                    language=self.settings.PHONEMIZER_LANGUAGE,
                    preserve_punctuation=True,
                    with_stress=True
                )
                self.logger.info("eSpeak backend initialized")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize phonemization backends: {e}")
            raise AudioProcessingError(
                f"Failed to initialize phonemization: {e}",
                {"backend": self.settings.PHONEMIZER_BACKEND}
            )
    
    async def phonemize_text(
        self, 
        text: str, 
        language: str = "en-us"
    ) -> PhonemizationResult:
        """
        Convert text to phonemes using configured backend
        """
        start_time = time.time()
        
        try:
            # Validate input
            if not self.validate_text(text):
                raise AudioProcessingError(
                    f"Invalid text for phonemization: {text}",
                    {"text": text, "language": language}
                )
            
            # Preprocess text
            words = self._preprocess_text(text)
            
            # Phonemize each word
            word_results = await self.phonemize_words(words, language)
            
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            return PhonemizationResult(
                original_text=text,
                words=word_results,
                language=language,
                backend=self.settings.PHONEMIZER_BACKEND,
                processing_time_ms=processing_time_ms
            )
            
        except Exception as e:
            self.logger.error(f"Phonemization failed for text '{text}': {e}")
            raise AudioProcessingError(
                f"Phonemization failed: {e}",
                {"text": text, "language": language}
            )
    
    async def phonemize_words(
        self,
        words: List[str],
        language: str = "en-us"
    ) -> List[PhonemeResult]:
        """
        Convert list of words to phonemes
        """
        results = []
        
        for word in words:
            try:
                # Choose backend based on configuration
                if self.settings.PHONEMIZER_BACKEND == "g2p" and self._g2p_model:
                    result = await self._phonemize_with_g2p(word)
                elif self.settings.PHONEMIZER_BACKEND == "espeak" and self._espeak_backend:
                    result = await self._phonemize_with_espeak(word, language)
                else:
                    # Fallback to basic phonemizer
                    result = await self._phonemize_with_phonemizer(word, language)
                
                results.append(result)
                
            except Exception as e:
                self.logger.warning(f"Failed to phonemize word '{word}': {e}")
                # Create fallback result
                results.append(PhonemeResult(
                    word=word,
                    phonemes=[word],  # Fallback to original word
                    ipa=f"/{word}/",
                    confidence=0.0
                ))
        
        return results
    
    async def _phonemize_with_g2p(self, word: str) -> PhonemeResult:
        """Phonemize using G2P-EN model"""
        try:
            phonemes = self._g2p_model(word)
            
            # Convert to IPA-like format
            ipa = "/" + " ".join(phonemes) + "/"
            
            # Count syllables (approximation)
            syllable_count = len([p for p in phonemes if self._is_vowel_phoneme(p)])
            
            return PhonemeResult(
                word=word,
                phonemes=phonemes,
                ipa=ipa,
                syllable_count=syllable_count,
                confidence=0.9
            )
            
        except Exception as e:
            self.logger.error(f"G2P phonemization failed for '{word}': {e}")
            raise AudioProcessingError(f"G2P phonemization failed: {e}")
    
    async def _phonemize_with_espeak(self, word: str, language: str) -> PhonemeResult:
        """Phonemize using eSpeak backend"""
        try:
            # Use phonemizer with eSpeak
            separator = Separator(phone=' ', word='|')
            
            phonemes_str = phonemize(
                word,
                language=language,
                backend='espeak',
                separator=separator,
                strip=True,
                preserve_punctuation=False,
                with_stress=True
            )
            
            # Parse phonemes and stress
            phonemes = phonemes_str.split()
            
            # Extract stress pattern
            stress_pattern = self._extract_stress_pattern(phonemes_str)
            
            # Count syllables
            syllable_count = len([p for p in phonemes if self._is_vowel_phoneme(p)])
            
            return PhonemeResult(
                word=word,
                phonemes=phonemes,
                ipa=f"/{phonemes_str}/",
                stress_pattern=stress_pattern,
                syllable_count=syllable_count,
                confidence=0.95
            )
            
        except Exception as e:
            self.logger.error(f"eSpeak phonemization failed for '{word}': {e}")
            raise AudioProcessingError(f"eSpeak phonemization failed: {e}")
    
    async def _phonemize_with_phonemizer(self, word: str, language: str) -> PhonemeResult:
        """Fallback phonemization using basic phonemizer"""
        try:
            separator = Separator(phone=' ', word='|')
            
            phonemes_str = phonemize(
                word,
                language=language,
                backend=self.settings.PHONEMIZER_BACKEND,
                separator=separator,
                strip=True
            )
            
            phonemes = phonemes_str.split()
            
            return PhonemeResult(
                word=word,
                phonemes=phonemes,
                ipa=f"/{phonemes_str}/",
                confidence=0.8
            )
            
        except Exception as e:
            self.logger.error(f"Basic phonemization failed for '{word}': {e}")
            raise AudioProcessingError(f"Basic phonemization failed: {e}")
    
    async def get_ipa_representation(
        self,
        text: str,
        language: str = "en-us"
    ) -> str:
        """Get IPA representation of text"""
        try:
            result = await self.phonemize_text(text, language)
            ipa_parts = [word.ipa.strip('/') for word in result.words]
            return f"/{' '.join(ipa_parts)}/"
            
        except Exception as e:
            self.logger.error(f"IPA representation failed for '{text}': {e}")
            raise AudioProcessingError(f"IPA representation failed: {e}")
    
    def get_supported_languages(self) -> List[str]:
        """Get supported language codes"""
        return [
            "en-us",    # American English
            "en-gb",    # British English  
            "en-au",    # Australian English
            "en",       # Generic English
            "es",       # Spanish
            "fr",       # French
            "de",       # German
            "it",       # Italian
            "pt",       # Portuguese
        ]
    
    def validate_text(self, text: str) -> bool:
        """Validate text for phonemization"""
        if not text or not text.strip():
            return False
        
        # Check for supported characters
        if not re.match(r'^[a-zA-Z\s\'-.,!?]+$', text.strip()):
            return False
        
        # Check length
        if len(text.strip()) > 1000:  # Reasonable limit
            return False
        
        return True
    
    def _preprocess_text(self, text: str) -> List[str]:
        """Preprocess text and extract words"""
        # Clean text
        text = re.sub(r'[^\w\s\'-]', '', text)
        
        # Split into words
        words = text.split()
        
        # Filter valid words
        words = [word.strip("'\"") for word in words if word.strip()]
        
        return words
    
    def _is_vowel_phoneme(self, phoneme: str) -> bool:
        """Check if phoneme is a vowel (for syllable counting)"""
        vowel_patterns = ['a', 'e', 'i', 'o', 'u', 'æ', 'ɛ', 'ɪ', 'ɔ', 'ʊ', 'ə', 'ʌ']
        return any(v in phoneme.lower() for v in vowel_patterns)
    
    def _extract_stress_pattern(self, phonemes_str: str) -> Optional[str]:
        """Extract stress pattern from phonemized string"""
        try:
            # Look for stress markers (primary ˈ, secondary ˌ)
            stress_marks = re.findall(r'[ˈˌ]', phonemes_str)
            if stress_marks:
                return ''.join(stress_marks)
            return None
        except:
            return None
    
    @lru_cache(maxsize=1000)
    def _cached_phonemize(self, word: str, language: str) -> str:
        """Cached phonemization for frequently used words"""
        try:
            return phonemize(
                word,
                language=language,
                backend=self.settings.PHONEMIZER_BACKEND,
                strip=True
            )
        except:
            return word