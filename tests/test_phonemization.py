"""
Unit tests for phonemization service (Step 1)
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch

from src.domain.services.phonemization_service import (
    IPhonemizationService,
    PhonemizationResult,
    PhonemeResult
)
from src.infrastructure.services.phonemization_service import PhonemizationService
from src.application.usecases.phonemization_usecases import PhonemizationUseCases
from src.application.dtos.phonemization_dto import (
    PhonemizationRequest,
    WordPhonemizationRequest,
    IpaConversionRequest
)
from src.shared.exceptions import ValidationError, AudioProcessingError


class TestPhonemizationService:
    """Test cases for phonemization service implementation"""
    
    @pytest.fixture
    def mock_phonemization_service(self):
        """Create mock phonemization service"""
        service = Mock(spec=IPhonemizationService)
        return service
    
    @pytest.fixture
    def phonemization_use_cases(self, mock_phonemization_service):
        """Create phonemization use cases with mocked service"""
        return PhonemizationUseCases(mock_phonemization_service)
    
    @pytest.mark.asyncio
    async def test_phonemize_text_success(self, phonemization_use_cases, mock_phonemization_service):
        """Test successful text phonemization"""
        # Arrange
        request = PhonemizationRequest(
            text="hello world",
            language="en-us"
        )
        
        expected_result = PhonemizationResult(
            original_text="hello world",
            words=[
                PhonemeResult(
                    word="hello",
                    phonemes=["h", "ə", "ˈl", "oʊ"],
                    ipa="/həˈloʊ/",
                    confidence=0.95
                ),
                PhonemeResult(
                    word="world",
                    phonemes=["w", "ɜː", "l", "d"],
                    ipa="/wɜːld/",
                    confidence=0.98
                )
            ],
            language="en-us",
            backend="espeak",
            processing_time_ms=50
        )
        
        mock_phonemization_service.phonemize_text = AsyncMock(return_value=expected_result)
        mock_phonemization_service.validate_text = Mock(return_value=True)
        mock_phonemization_service.get_supported_languages = Mock(
            return_value=["en-us", "en-gb", "es"]
        )
        
        # Act
        result = await phonemization_use_cases.phonemize_text(request)
        
        # Assert
        assert result.success is True
        assert result.original_text == "hello world"
        assert len(result.words) == 2
        assert result.words[0].word == "hello"
        assert result.words[1].word == "world"
        assert result.total_words == 2
        assert result.total_phonemes == 8
    
    @pytest.mark.asyncio
    async def test_phonemize_text_validation_error(self, phonemization_use_cases, mock_phonemization_service):
        """Test phonemization with invalid text"""
        # Arrange
        request = PhonemizationRequest(
            text="",  # Empty text
            language="en-us"
        )
        
        # Act & Assert
        with pytest.raises(ValidationError) as exc_info:
            await phonemization_use_cases.phonemize_text(request)
        
        assert "Text cannot be empty" in str(exc_info.value)
    
    @pytest.mark.asyncio 
    async def test_phonemize_words_success(self, phonemization_use_cases, mock_phonemization_service):
        """Test successful word list phonemization"""
        # Arrange
        request = WordPhonemizationRequest(
            words=["hello", "world"],
            language="en-us"
        )
        
        expected_results = [
            PhonemeResult(
                word="hello",
                phonemes=["h", "ə", "ˈl", "oʊ"],
                ipa="/həˈloʊ/",
                confidence=0.95
            ),
            PhonemeResult(
                word="world",
                phonemes=["w", "ɜː", "l", "d"],
                ipa="/wɜːld/",
                confidence=0.98
            )
        ]
        
        mock_phonemization_service.phonemize_words = AsyncMock(return_value=expected_results)
        mock_phonemization_service.validate_text = Mock(return_value=True)
        
        # Act
        result = await phonemization_use_cases.phonemize_words(request)
        
        # Assert
        assert len(result) == 2
        assert result[0].word == "hello"
        assert result[1].word == "world"
        assert all(r.confidence > 0.9 for r in result)
    
    @pytest.mark.asyncio
    async def test_get_ipa_representation(self, phonemization_use_cases, mock_phonemization_service):
        """Test IPA representation conversion"""
        # Arrange
        request = IpaConversionRequest(
            text="hello world",
            language="en-us"
        )
        
        expected_ipa = "/həˈloʊ wɜːld/"
        
        mock_phonemization_service.get_ipa_representation = AsyncMock(return_value=expected_ipa)
        mock_phonemization_service.validate_text = Mock(return_value=True)
        
        # Act
        result = await phonemization_use_cases.get_ipa_representation(request)
        
        # Assert
        assert result.success is True
        assert result.ipa_representation == expected_ipa
        assert result.original_text == "hello world"
        assert result.language == "en-us"
    
    @pytest.mark.asyncio
    async def test_get_supported_languages(self, phonemization_use_cases, mock_phonemization_service):
        """Test getting supported languages"""
        # Arrange
        expected_languages = ["en-us", "en-gb", "en-au", "es", "fr", "de"]
        mock_phonemization_service.get_supported_languages = Mock(return_value=expected_languages)
        
        # Act
        result = await phonemization_use_cases.get_supported_languages()
        
        # Assert
        assert result.success is True
        assert len(result.supported_languages) == 6
        assert "en-us" in result.supported_languages
        assert result.total_languages == 6


class TestPhonemizationServiceImplementation:
    """Test cases for concrete phonemization service implementation"""
    
    @pytest.fixture
    def service(self):
        """Create phonemization service instance"""
        with patch('src.infrastructure.services.phonemization_service.G2p'), \
             patch('src.infrastructure.services.phonemization_service.EspeakBackend'):
            return PhonemizationService()
    
    def test_validate_text_valid(self, service):
        """Test text validation with valid input"""
        assert service.validate_text("hello world") is True
        assert service.validate_text("Hello, world!") is True
        assert service.validate_text("test-word") is True
    
    def test_validate_text_invalid(self, service):
        """Test text validation with invalid input"""
        assert service.validate_text("") is False
        assert service.validate_text("   ") is False  
        assert service.validate_text("hello@world") is False
        assert service.validate_text("text with 123 numbers") is False
        assert service.validate_text("x" * 1001) is False  # Too long
    
    def test_preprocess_text(self, service):
        """Test text preprocessing"""
        words = service._preprocess_text("Hello, world! How are you?")
        assert words == ["Hello", "world", "How", "are", "you"]
    
    def test_is_vowel_phoneme(self, service):
        """Test vowel phoneme detection"""
        assert service._is_vowel_phoneme("a") is True
        assert service._is_vowel_phoneme("ə") is True
        assert service._is_vowel_phoneme("ɛ") is True
        assert service._is_vowel_phoneme("b") is False
        assert service._is_vowel_phoneme("k") is False
    
    def test_extract_stress_pattern(self, service):
        """Test stress pattern extraction"""
        assert service._extract_stress_pattern("həˈloʊ") == "ˈ"
        assert service._extract_stress_pattern("ˌhɛloʊˈwɜːld") == "ˌˈ"
        assert service._extract_stress_pattern("hello") is None


class TestPhonemizationIntegration:
    """Integration tests for phonemization pipeline"""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_end_to_end_phonemization(self):
        """Test complete phonemization pipeline"""
        # This would require actual phonemization libraries to be installed
        # For now, we'll skip this test
        pytest.skip("Requires phonemization libraries to be installed")
    
    @pytest.mark.integration
    def test_phonemization_performance(self):
        """Test phonemization performance with large text"""
        # Performance test - skip for now
        pytest.skip("Performance test - implement when needed")


# Fixtures for test data
@pytest.fixture
def sample_phoneme_results():
    """Sample phoneme results for testing"""
    return [
        PhonemeResult(
            word="hello",
            phonemes=["h", "ə", "ˈl", "oʊ"],
            ipa="/həˈloʊ/",
            stress_pattern="ˈ",
            syllable_count=2,
            confidence=0.95
        ),
        PhonemeResult(
            word="world",
            phonemes=["w", "ɜː", "l", "d"],
            ipa="/wɜːld/",
            syllable_count=1,
            confidence=0.98
        )
    ]


@pytest.fixture
def sample_phonemization_result(sample_phoneme_results):
    """Sample phonemization result for testing"""
    return PhonemizationResult(
        original_text="hello world",
        words=sample_phoneme_results,
        language="en-us",
        backend="espeak",
        processing_time_ms=45
    )