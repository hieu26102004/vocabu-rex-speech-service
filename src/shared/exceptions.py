"""
Custom exceptions for the speech service
"""

from datetime import datetime
from typing import Dict, Any, Optional


class SpeechServiceException(Exception):
    """Base exception for speech service"""
    
    def __init__(
        self, 
        message: str, 
        error_code: str = "SPEECH_ERROR",
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details or {}
        self.timestamp = datetime.utcnow().isoformat()
        super().__init__(self.message)


class AudioProcessingError(SpeechServiceException):
    """Audio processing related errors"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="AUDIO_PROCESSING_ERROR",
            status_code=400,
            details=details
        )


class SpeechRecognitionError(SpeechServiceException):
    """Speech recognition related errors"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="SPEECH_RECOGNITION_ERROR",
            status_code=422,
            details=details
        )


class PronunciationAssessmentError(SpeechServiceException):
    """Pronunciation assessment related errors"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="PRONUNCIATION_ASSESSMENT_ERROR",
            status_code=422,
            details=details
        )


class FileUploadError(SpeechServiceException):
    """File upload related errors"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="FILE_UPLOAD_ERROR",
            status_code=400,
            details=details
        )


class ValidationError(SpeechServiceException):
    """Validation related errors"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            status_code=422,
            details=details
        )


class AuthenticationError(SpeechServiceException):
    """Authentication related errors"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="AUTHENTICATION_ERROR",
            status_code=401,
            details=details
        )


class AuthorizationError(SpeechServiceException):
    """Authorization related errors"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="AUTHORIZATION_ERROR",
            status_code=403,
            details=details
        )


class ExternalServiceError(SpeechServiceException):
    """External service related errors"""
    
    def __init__(self, message: str, service_name: str = "", details: Optional[Dict[str, Any]] = None):
        details = details or {}
        if service_name:
            details["service"] = service_name
            
        super().__init__(
            message=message,
            error_code="EXTERNAL_SERVICE_ERROR",
            status_code=502,
            details=details
        )


class RateLimitExceededError(SpeechServiceException):
    """Rate limit exceeded errors"""
    
    def __init__(self, message: str = "Rate limit exceeded", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="RATE_LIMIT_EXCEEDED",
            status_code=429,
            details=details
        )