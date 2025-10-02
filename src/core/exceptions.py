"""
Core exception classes for the speech service
"""

from typing import Optional, Dict, Any
from enum import Enum


class ErrorCode(Enum):
    """Error codes for different types of exceptions"""
    # Audio processing errors
    AUDIO_FILE_NOT_FOUND = "AUDIO_001"
    AUDIO_FORMAT_UNSUPPORTED = "AUDIO_002"
    AUDIO_CORRUPTED = "AUDIO_003"
    AUDIO_TOO_LONG = "AUDIO_004"
    AUDIO_TOO_SHORT = "AUDIO_005"
    AUDIO_QUALITY_POOR = "AUDIO_006"
    
    # Validation errors
    INVALID_INPUT = "VALIDATION_001"
    MISSING_REQUIRED_FIELD = "VALIDATION_002"
    INVALID_FORMAT = "VALIDATION_003"
    OUT_OF_RANGE = "VALIDATION_004"
    
    # Service errors
    MODEL_LOADING_FAILED = "SERVICE_001"
    TRANSCRIPTION_FAILED = "SERVICE_002"
    PHONEMIZATION_FAILED = "SERVICE_003"
    ALIGNMENT_FAILED = "SERVICE_004"
    ANALYSIS_FAILED = "SERVICE_005"
    
    # Configuration errors
    CONFIG_MISSING = "CONFIG_001"
    CONFIG_INVALID = "CONFIG_002"
    
    # External service errors
    EXTERNAL_API_ERROR = "EXTERNAL_001"
    NETWORK_ERROR = "EXTERNAL_002"
    TIMEOUT_ERROR = "EXTERNAL_003"


class SpeechServiceException(Exception):
    """
    Base exception class for all speech service errors
    """
    
    def __init__(
        self,
        message: str,
        error_code: Optional[ErrorCode] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.cause = cause
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API responses"""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code.value if self.error_code else None,
            "details": self.details,
            "cause": str(self.cause) if self.cause else None
        }


class AudioProcessingError(SpeechServiceException):
    """
    Exception raised when audio processing fails
    """
    
    def __init__(
        self,
        message: str,
        audio_file_path: Optional[str] = None,
        error_code: Optional[ErrorCode] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        self.audio_file_path = audio_file_path
        details = details or {}
        if audio_file_path:
            details["audio_file_path"] = audio_file_path
        
        super().__init__(message, error_code, details, cause)


class ValidationError(SpeechServiceException):
    """
    Exception raised when input validation fails
    """
    
    def __init__(
        self,
        message: str,
        field_name: Optional[str] = None,
        field_value: Optional[Any] = None,
        error_code: Optional[ErrorCode] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        self.field_name = field_name
        self.field_value = field_value
        
        details = details or {}
        if field_name:
            details["field_name"] = field_name
        if field_value is not None:
            details["field_value"] = str(field_value)
        
        super().__init__(message, error_code, details, cause)


class ServiceError(SpeechServiceException):
    """
    Exception raised when a service operation fails
    """
    
    def __init__(
        self,
        message: str,
        service_name: Optional[str] = None,
        operation: Optional[str] = None,
        error_code: Optional[ErrorCode] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        self.service_name = service_name
        self.operation = operation
        
        details = details or {}
        if service_name:
            details["service_name"] = service_name
        if operation:
            details["operation"] = operation
        
        super().__init__(message, error_code, details, cause)


class ModelLoadingError(ServiceError):
    """
    Exception raised when model loading fails
    """
    
    def __init__(
        self,
        message: str,
        model_name: Optional[str] = None,
        model_path: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        self.model_name = model_name
        self.model_path = model_path
        
        details = details or {}
        if model_name:
            details["model_name"] = model_name
        if model_path:
            details["model_path"] = model_path
        
        super().__init__(
            message=message,
            service_name="ModelLoader",
            operation="load_model",
            error_code=ErrorCode.MODEL_LOADING_FAILED,
            details=details,
            cause=cause
        )


class TranscriptionError(ServiceError):
    """
    Exception raised when audio transcription fails
    """
    
    def __init__(
        self,
        message: str,
        audio_file_path: Optional[str] = None,
        model_name: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        self.audio_file_path = audio_file_path
        self.model_name = model_name
        
        details = details or {}
        if audio_file_path:
            details["audio_file_path"] = audio_file_path
        if model_name:
            details["model_name"] = model_name
        
        super().__init__(
            message=message,
            service_name="ASRService",
            operation="transcribe",
            error_code=ErrorCode.TRANSCRIPTION_FAILED,
            details=details,
            cause=cause
        )


class PhonemizationError(ServiceError):
    """
    Exception raised when phonemization fails
    """
    
    def __init__(
        self,
        message: str,
        text: Optional[str] = None,
        language: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        self.text = text
        self.language = language
        
        details = details or {}
        if text:
            details["text"] = text[:100] + "..." if len(text) > 100 else text
        if language:
            details["language"] = language
        
        super().__init__(
            message=message,
            service_name="PhonemizationService",
            operation="phonemize",
            error_code=ErrorCode.PHONEMIZATION_FAILED,
            details=details,
            cause=cause
        )


class AlignmentError(ServiceError):
    """
    Exception raised when forced alignment fails
    """
    
    def __init__(
        self,
        message: str,
        audio_file_path: Optional[str] = None,
        text: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        self.audio_file_path = audio_file_path
        self.text = text
        
        details = details or {}
        if audio_file_path:
            details["audio_file_path"] = audio_file_path
        if text:
            details["text"] = text[:100] + "..." if len(text) > 100 else text
        
        super().__init__(
            message=message,
            service_name="AlignmentService",
            operation="align",
            error_code=ErrorCode.ALIGNMENT_FAILED,
            details=details,
            cause=cause
        )


class PronunciationAnalysisError(ServiceError):
    """
    Exception raised when pronunciation analysis fails
    """
    
    def __init__(
        self,
        message: str,
        reference_text: Optional[str] = None,
        actual_text: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        self.reference_text = reference_text
        self.actual_text = actual_text
        
        details = details or {}
        if reference_text:
            details["reference_text"] = reference_text[:50] + "..." if len(reference_text) > 50 else reference_text
        if actual_text:
            details["actual_text"] = actual_text[:50] + "..." if len(actual_text) > 50 else actual_text
        
        super().__init__(
            message=message,
            service_name="PronunciationAnalyzer",
            operation="analyze",
            error_code=ErrorCode.ANALYSIS_FAILED,
            details=details,
            cause=cause
        )


class ConfigurationError(SpeechServiceException):
    """
    Exception raised when configuration is missing or invalid
    """
    
    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        config_value: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        self.config_key = config_key
        self.config_value = config_value
        
        details = details or {}
        if config_key:
            details["config_key"] = config_key
        if config_value:
            details["config_value"] = config_value
        
        error_code = ErrorCode.CONFIG_MISSING if config_value is None else ErrorCode.CONFIG_INVALID
        
        super().__init__(message, error_code, details, cause)


class ExternalServiceError(SpeechServiceException):
    """
    Exception raised when external service calls fail
    """
    
    def __init__(
        self,
        message: str,
        service_name: Optional[str] = None,
        endpoint: Optional[str] = None,
        status_code: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        self.service_name = service_name
        self.endpoint = endpoint
        self.status_code = status_code
        
        details = details or {}
        if service_name:
            details["service_name"] = service_name
        if endpoint:
            details["endpoint"] = endpoint
        if status_code:
            details["status_code"] = status_code
        
        super().__init__(message, ErrorCode.EXTERNAL_API_ERROR, details, cause)


class TimeoutError(SpeechServiceException):
    """
    Exception raised when operations timeout
    """
    
    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        self.operation = operation
        self.timeout_seconds = timeout_seconds
        
        details = details or {}
        if operation:
            details["operation"] = operation
        if timeout_seconds:
            details["timeout_seconds"] = timeout_seconds
        
        super().__init__(message, ErrorCode.TIMEOUT_ERROR, details, cause)


# Convenience functions for common exceptions

def audio_file_not_found_error(file_path: str) -> AudioProcessingError:
    """Create an audio file not found error"""
    return AudioProcessingError(
        message=f"Audio file not found: {file_path}",
        audio_file_path=file_path,
        error_code=ErrorCode.AUDIO_FILE_NOT_FOUND
    )


def audio_format_unsupported_error(file_path: str, format_detected: str) -> AudioProcessingError:
    """Create an unsupported audio format error"""
    return AudioProcessingError(
        message=f"Unsupported audio format '{format_detected}' for file: {file_path}",
        audio_file_path=file_path,
        error_code=ErrorCode.AUDIO_FORMAT_UNSUPPORTED,
        details={"format_detected": format_detected}
    )


def validation_required_field_error(field_name: str) -> ValidationError:
    """Create a required field validation error"""
    return ValidationError(
        message=f"Required field '{field_name}' is missing",
        field_name=field_name,
        error_code=ErrorCode.MISSING_REQUIRED_FIELD
    )


def validation_invalid_value_error(field_name: str, value: Any, expected: str) -> ValidationError:
    """Create an invalid value validation error"""
    return ValidationError(
        message=f"Invalid value for field '{field_name}': {value}. Expected: {expected}",
        field_name=field_name,
        field_value=value,
        error_code=ErrorCode.INVALID_FORMAT,
        details={"expected": expected}
    )


def model_loading_failed_error(model_name: str, cause: Exception) -> ModelLoadingError:
    """Create a model loading failed error"""
    return ModelLoadingError(
        message=f"Failed to load model '{model_name}': {str(cause)}",
        model_name=model_name,
        cause=cause
    )


def transcription_failed_error(audio_path: str, model_name: str, cause: Exception) -> TranscriptionError:
    """Create a transcription failed error"""
    return TranscriptionError(
        message=f"Transcription failed for audio '{audio_path}' using model '{model_name}': {str(cause)}",
        audio_file_path=audio_path,
        model_name=model_name,
        cause=cause
    )


def timeout_error(operation: str, timeout_seconds: float) -> TimeoutError:
    """Create a timeout error"""
    return TimeoutError(
        message=f"Operation '{operation}' timed out after {timeout_seconds} seconds",
        operation=operation,
        timeout_seconds=timeout_seconds
    )