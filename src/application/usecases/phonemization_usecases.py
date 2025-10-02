"""
Use cases for phonemization operations (Step 1 of pronunciation assessment)
Business logic layer for text-to-phoneme conversion
"""

import logging
from typing import List, Optional
from datetime import datetime

from src.domain.services.phonemization_service import (
    IPhonemizationService,
    PhonemizationResult,
    PhonemeResult
)
from src.application.dtos.phonemization_dto import (
    PhonemizationRequest,
    WordPhonemizationRequest,
    IpaConversionRequest,
    PhonemizationResponseDTO,
    PhonemeResultDTO,
    IpaConversionResponseDTO,
    SupportedLanguagesResponseDTO
)
from src.shared.exceptions import (
    ValidationError,
    AudioProcessingError
)
from src.shared.config import get_settings


class PhonemizationUseCases:
    """
    Use cases for phonemization operations
    Orchestrates business logic for Step 1 of pronunciation assessment
    """
    
    def __init__(self, phonemization_service: IPhonemizationService):
        self.phonemization_service = phonemization_service
        self.settings = get_settings()
        self.logger = logging.getLogger(__name__)
    
    async def phonemize_text(self, request: PhonemizationRequest) -> PhonemizationResponseDTO:
        """
        Phonemize input text and return detailed results
        
        Args:
            request: Phonemization request with text and options
            
        Returns:
            Detailed phonemization response
        """
        try:
            # Validate request
            self._validate_phonemization_request(request)
            
            # Determine backend
            backend = request.backend or self.settings.PHONEMIZER_BACKEND
            
            # Perform phonemization
            result = await self.phonemization_service.phonemize_text(
                text=request.text,
                language=request.language
            )
            
            # Convert to response DTO
            response = self._convert_to_response_dto(result, request)
            
            self.logger.info(
                f"Phonemization completed: {len(result.words)} words, "
                f"{sum(len(w.phonemes) for w in result.words)} phonemes"
            )
            
            return response
            
        except Exception as e:
            self.logger.error(f"Phonemization use case failed: {e}")
            if isinstance(e, (ValidationError, AudioProcessingError)):
                raise e
            raise AudioProcessingError(f"Phonemization failed: {e}")
    
    async def phonemize_words(self, request: WordPhonemizationRequest) -> List[PhonemeResultDTO]:
        """
        Phonemize list of words
        
        Args:
            request: Word list phonemization request
            
        Returns:
            List of phoneme results for each word
        """
        try:
            # Validate request
            self._validate_word_request(request)
            
            # Perform phonemization
            results = await self.phonemization_service.phonemize_words(
                words=request.words,
                language=request.language
            )
            
            # Convert to DTOs
            dto_results = [
                self._convert_phoneme_result_to_dto(result) 
                for result in results
            ]
            
            self.logger.info(f"Word phonemization completed: {len(dto_results)} words")
            
            return dto_results
            
        except Exception as e:
            self.logger.error(f"Word phonemization use case failed: {e}")
            if isinstance(e, (ValidationError, AudioProcessingError)):
                raise e
            raise AudioProcessingError(f"Word phonemization failed: {e}")
    
    async def get_ipa_representation(self, request: IpaConversionRequest) -> IpaConversionResponseDTO:
        """
        Get IPA representation of text
        
        Args:
            request: IPA conversion request
            
        Returns:
            IPA representation response
        """
        try:
            # Validate request  
            self._validate_ipa_request(request)
            
            start_time = datetime.utcnow()
            
            # Get IPA representation
            ipa = await self.phonemization_service.get_ipa_representation(
                text=request.text,
                language=request.language
            )
            
            end_time = datetime.utcnow()
            processing_time_ms = int((end_time - start_time).total_seconds() * 1000)
            
            response = IpaConversionResponseDTO(
                success=True,
                original_text=request.text,
                ipa_representation=ipa,
                language=request.language,
                processing_time_ms=processing_time_ms,
                timestamp=end_time
            )
            
            self.logger.info(f"IPA conversion completed for text: '{request.text}'")
            
            return response
            
        except Exception as e:
            self.logger.error(f"IPA conversion use case failed: {e}")
            if isinstance(e, (ValidationError, AudioProcessingError)):
                raise e
            raise AudioProcessingError(f"IPA conversion failed: {e}")
    
    async def get_supported_languages(self) -> SupportedLanguagesResponseDTO:
        """
        Get list of supported languages for phonemization
        
        Returns:
            Supported languages response
        """
        try:
            languages = self.phonemization_service.get_supported_languages()
            
            response = SupportedLanguagesResponseDTO(
                success=True,
                supported_languages=languages,
                default_language=self.settings.PHONEMIZER_LANGUAGE,
                total_languages=len(languages)
            )
            
            self.logger.info(f"Supported languages retrieved: {len(languages)} languages")
            
            return response
            
        except Exception as e:
            self.logger.error(f"Get supported languages failed: {e}")
            raise AudioProcessingError(f"Failed to get supported languages: {e}")
    
    async def validate_text_for_phonemization(self, text: str, language: str = "en-us") -> bool:
        """
        Validate if text can be phonemized
        
        Args:
            text: Text to validate
            language: Language code
            
        Returns:
            True if text is valid for phonemization
        """
        try:
            return self.phonemization_service.validate_text(text)
        except Exception as e:
            self.logger.error(f"Text validation failed: {e}")
            return False
    
    def _validate_phonemization_request(self, request: PhonemizationRequest):
        """Validate phonemization request"""
        if not request.text or not request.text.strip():
            raise ValidationError(
                "Text cannot be empty",
                {"field": "text", "value": request.text}
            )
        
        if not self.phonemization_service.validate_text(request.text):
            raise ValidationError(
                "Text contains invalid characters or is too long",
                {"field": "text", "value": request.text}
            )
        
        supported_languages = self.phonemization_service.get_supported_languages()
        if request.language not in supported_languages:
            raise ValidationError(
                f"Language '{request.language}' not supported",
                {"field": "language", "supported": supported_languages}
            )
    
    def _validate_word_request(self, request: WordPhonemizationRequest):
        """Validate word phonemization request"""
        if not request.words:
            raise ValidationError(
                "Words list cannot be empty",
                {"field": "words"}
            )
        
        for i, word in enumerate(request.words):
            if not self.phonemization_service.validate_text(word):
                raise ValidationError(
                    f"Invalid word at index {i}: '{word}'",
                    {"field": "words", "index": i, "value": word}
                )
    
    def _validate_ipa_request(self, request: IpaConversionRequest):
        """Validate IPA conversion request"""
        if not request.text or not request.text.strip():
            raise ValidationError(
                "Text cannot be empty",
                {"field": "text", "value": request.text}
            )
        
        if not self.phonemization_service.validate_text(request.text):
            raise ValidationError(
                "Text contains invalid characters",
                {"field": "text", "value": request.text}
            )
    
    def _convert_to_response_dto(
        self, 
        result: PhonemizationResult, 
        request: PhonemizationRequest
    ) -> PhonemizationResponseDTO:
        """Convert domain result to response DTO"""
        word_dtos = [
            self._convert_phoneme_result_to_dto(word_result) 
            for word_result in result.words
        ]
        
        return PhonemizationResponseDTO(
            success=True,
            original_text=result.original_text,
            language=result.language,
            backend=result.backend,
            words=word_dtos,
            total_words=len(word_dtos),
            total_phonemes=sum(len(w.phonemes) for w in word_dtos),
            processing_time_ms=result.processing_time_ms,
            timestamp=datetime.utcnow()
        )
    
    def _convert_phoneme_result_to_dto(self, result: PhonemeResult) -> PhonemeResultDTO:
        """Convert domain phoneme result to DTO"""
        return PhonemeResultDTO(
            word=result.word,
            phonemes=result.phonemes,
            ipa=result.ipa,
            stress_pattern=result.stress_pattern,
            syllable_count=result.syllable_count,
            confidence=result.confidence
        )