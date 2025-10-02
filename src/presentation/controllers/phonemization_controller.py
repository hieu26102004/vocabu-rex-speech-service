"""
FastAPI controller for phonemization endpoints (Step 1)
Handles HTTP requests for text-to-phoneme conversion
"""

import logging
from typing import List
from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.responses import JSONResponse

from src.application.usecases.phonemization_usecases import PhonemizationUseCases
from src.application.dtos.phonemization_dto import (
    PhonemizationRequest,
    WordPhonemizationRequest,
    IpaConversionRequest,
    PhonemizationResponseDTO,
    PhonemeResultDTO,
    IpaConversionResponseDTO,
    SupportedLanguagesResponseDTO,
    ValidationErrorDTO
)
from src.infrastructure.services.phonemization_service import PhonemizationService
from src.shared.exceptions import (
    ValidationError,
    AudioProcessingError,
    SpeechServiceException
)


# Create router for phonemization endpoints
phonemization_router = APIRouter(
    prefix="/phonemization",
    tags=["Phonemization (Step 1)"]
)

# Logger
logger = logging.getLogger(__name__)


def get_phonemization_use_cases() -> PhonemizationUseCases:
    """
    Dependency injection for phonemization use cases
    """
    phonemization_service = PhonemizationService()
    return PhonemizationUseCases(phonemization_service)


@phonemization_router.post(
    "/text",
    response_model=PhonemizationResponseDTO,
    status_code=status.HTTP_200_OK,
    summary="Phonemize Text",
    description="Convert text to phonemes (Step 1 of pronunciation assessment)"
)
async def phonemize_text(
    request: PhonemizationRequest,
    use_cases: PhonemizationUseCases = Depends(get_phonemization_use_cases)
) -> PhonemizationResponseDTO:
    """
    **Step 1 of Pronunciation Assessment: Phonemization**
    
    Convert input text to phonemic representation using various backends:
    - **eSpeak**: Fast, accurate, supports stress markers
    - **G2P**: Grapheme-to-Phoneme for English
    - **Festival**: Alternative TTS-based phonemization
    
    **Example Usage:**
    ```json
    {
        "text": "Hello world",
        "language": "en-us", 
        "backend": "espeak",
        "include_stress": true,
        "include_syllables": true
    }
    ```
    
    **Returns:**
    - Phoneme breakdown for each word
    - IPA representation
    - Syllable counts and stress patterns
    - Processing timing information
    """
    try:
        result = await use_cases.phonemize_text(request)
        
        logger.info(
            f"Text phonemization successful: '{request.text}' -> "
            f"{result.total_phonemes} phonemes in {result.processing_time_ms}ms"
        )
        
        return result
        
    except ValidationError as e:
        logger.warning(f"Validation error in phonemization: {e.message}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "success": False,
                "error": e.message,
                "details": e.details
            }
        )
    except AudioProcessingError as e:
        logger.error(f"Audio processing error in phonemization: {e.message}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "success": False,
                "error": e.message,
                "details": e.details
            }
        )
    except Exception as e:
        logger.error(f"Unexpected error in phonemization: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "success": False,
                "error": "Internal server error during phonemization",
                "details": {"error_type": type(e).__name__}
            }
        )


@phonemization_router.post(
    "/words",
    response_model=List[PhonemeResultDTO],
    status_code=status.HTTP_200_OK,
    summary="Phonemize Word List",
    description="Convert list of words to phonemes"
)
async def phonemize_words(
    request: WordPhonemizationRequest,
    use_cases: PhonemizationUseCases = Depends(get_phonemization_use_cases)
) -> List[PhonemeResultDTO]:
    """
    **Batch Phonemization for Word Lists**
    
    Efficiently process multiple words in a single request.
    Useful for preprocessing vocabulary lists or batch processing.
    
    **Example Usage:**
    ```json
    {
        "words": ["hello", "world", "pronunciation", "assessment"],
        "language": "en-us",
        "backend": "espeak"
    }
    ```
    
    **Returns:**
    - Individual phoneme results for each word
    - Confidence scores per word
    - IPA representations
    """
    try:
        results = await use_cases.phonemize_words(request)
        
        logger.info(f"Word list phonemization successful: {len(results)} words processed")
        
        return results
        
    except ValidationError as e:
        logger.warning(f"Validation error in word phonemization: {e.message}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "success": False,
                "error": e.message,
                "details": e.details
            }
        )
    except Exception as e:
        logger.error(f"Error in word phonemization: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "success": False,
                "error": "Internal server error during word phonemization"
            }
        )


@phonemization_router.post(
    "/ipa",
    response_model=IpaConversionResponseDTO,
    status_code=status.HTTP_200_OK,
    summary="Get IPA Representation",
    description="Convert text to International Phonetic Alphabet (IPA) format"
)
async def get_ipa_representation(
    request: IpaConversionRequest,
    use_cases: PhonemizationUseCases = Depends(get_phonemization_use_cases)
) -> IpaConversionResponseDTO:
    """
    **IPA (International Phonetic Alphabet) Conversion**
    
    Convert text to standardized IPA representation for linguistic analysis.
    
    **Example Usage:**
    ```json
    {
        "text": "Hello world",
        "language": "en-us"
    }
    ```
    
    **Returns:**
    - Complete IPA transcription: `/həˈloʊ wɜːld/`
    - Processing timing
    - Language information
    """
    try:
        result = await use_cases.get_ipa_representation(request)
        
        logger.info(f"IPA conversion successful: '{request.text}' -> '{result.ipa_representation}'")
        
        return result
        
    except ValidationError as e:
        logger.warning(f"Validation error in IPA conversion: {e.message}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "success": False,
                "error": e.message,
                "details": e.details
            }
        )
    except Exception as e:
        logger.error(f"Error in IPA conversion: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "success": False,
                "error": "Internal server error during IPA conversion"
            }
        )


@phonemization_router.get(
    "/languages",
    response_model=SupportedLanguagesResponseDTO,
    status_code=status.HTTP_200_OK,
    summary="Get Supported Languages",
    description="List all supported languages for phonemization"
)
async def get_supported_languages(
    use_cases: PhonemizationUseCases = Depends(get_phonemization_use_cases)
) -> SupportedLanguagesResponseDTO:
    """
    **Supported Languages for Phonemization**
    
    Get complete list of language codes supported by the phonemization engines.
    
    **Returns:**
    - List of supported language codes (e.g., "en-us", "en-gb", "es", "fr")
    - Default language setting
    - Total count of supported languages
    """
    try:
        result = await use_cases.get_supported_languages()
        
        logger.info(f"Supported languages retrieved: {result.total_languages} languages")
        
        return result
        
    except Exception as e:
        logger.error(f"Error getting supported languages: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "success": False,
                "error": "Failed to retrieve supported languages"
            }
        )


@phonemization_router.get(
    "/validate/{text}",
    status_code=status.HTTP_200_OK,
    summary="Validate Text",
    description="Check if text can be phonemized"
)
async def validate_text(
    text: str,
    language: str = "en-us",
    use_cases: PhonemizationUseCases = Depends(get_phonemization_use_cases)
) -> JSONResponse:
    """
    **Text Validation for Phonemization**
    
    Check if input text is valid for phonemization:
    - Contains supported characters
    - Within length limits
    - Language compatibility
    
    **Parameters:**
    - **text**: Text to validate
    - **language**: Language code (optional, default: en-us)
    
    **Returns:**
    - Validation status (true/false)
    - Reason for rejection (if invalid)
    """
    try:
        is_valid = await use_cases.validate_text_for_phonemization(text, language)
        
        return JSONResponse(
            content={
                "success": True,
                "valid": is_valid,
                "text": text,
                "language": language,
                "message": "Text is valid for phonemization" if is_valid else "Text is not valid for phonemization"
            },
            status_code=status.HTTP_200_OK
        )
        
    except Exception as e:
        logger.error(f"Error validating text: {e}")
        return JSONResponse(
            content={
                "success": False,
                "valid": False,
                "error": "Validation error",
                "details": str(e)
            },
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@phonemization_router.get(
    "/health",
    status_code=status.HTTP_200_OK,
    summary="Phonemization Health Check",
    description="Check health of phonemization service"
)
async def health_check(
    use_cases: PhonemizationUseCases = Depends(get_phonemization_use_cases)
) -> JSONResponse:
    """
    **Health Check for Phonemization Service**
    
    Verify that phonemization engines are working properly.
    
    **Returns:**
    - Service status
    - Available backends
    - Sample phonemization test
    """
    try:
        # Test phonemization with simple text
        test_result = await use_cases.validate_text_for_phonemization("hello")
        
        # Get supported languages
        languages = await use_cases.get_supported_languages()
        
        return JSONResponse(
            content={
                "success": True,
                "status": "healthy",
                "service": "phonemization",
                "test_passed": test_result,
                "supported_languages_count": languages.total_languages,
                "message": "Phonemization service is operational"
            },
            status_code=status.HTTP_200_OK
        )
        
    except Exception as e:
        logger.error(f"Phonemization health check failed: {e}")
        return JSONResponse(
            content={
                "success": False,
                "status": "unhealthy",
                "service": "phonemization",
                "error": str(e),
                "message": "Phonemization service is not operational"
            },
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE
        )