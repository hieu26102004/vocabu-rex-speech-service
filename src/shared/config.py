"""
Shared configuration management
"""

from functools import lru_cache
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings"""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    # Database Configuration
    DATABASE_URL: str = Field(
        default="postgresql+asyncpg://vocabu_user:vocabu_password@localhost:5435/vocabu_speech_db",
        description="Database connection URL"
    )
    DATABASE_ECHO: bool = Field(default=False, description="Enable SQLAlchemy echo")
    
    # Redis Configuration
    REDIS_URL: str = Field(
        default="redis://localhost:6382", 
        description="Redis connection URL"
    )
    REDIS_DECODE_RESPONSES: bool = Field(
        default=True, 
        description="Decode Redis responses"
    )
    
    # FastAPI Configuration
    ENVIRONMENT: str = Field(default="development", description="Environment")
    DEBUG: bool = Field(default=True, description="Debug mode")
    APP_NAME: str = Field(default="VocabuRex Speech Service", description="App name")
    APP_VERSION: str = Field(default="1.0.0", description="App version")
    PORT: int = Field(default=3005, description="Server port")
    HOST: str = Field(default="0.0.0.0", description="Server host")
    
    # CORS Configuration
    ALLOWED_ORIGINS: str = Field(
        default="http://localhost:3000,http://localhost:8080",
        description="Allowed CORS origins"
    )
    ALLOWED_METHODS: str = Field(
        default="GET,POST,PUT,DELETE,OPTIONS",
        description="Allowed CORS methods"
    )
    ALLOWED_HEADERS: str = Field(
        default="*",
        description="Allowed CORS headers"
    )
    
    # JWT Configuration
    JWT_SECRET_KEY: str = Field(
        default="your-super-secret-jwt-key-here-change-in-production",
        description="JWT secret key"
    )
    JWT_ALGORITHM: str = Field(default="HS256", description="JWT algorithm")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(
        default=30, 
        description="Access token expiration in minutes"
    )
    REFRESH_TOKEN_EXPIRE_DAYS: int = Field(
        default=7, 
        description="Refresh token expiration in days"
    )
    
    # File Upload Configuration
    MAX_FILE_SIZE_MB: int = Field(
        default=50, 
        description="Maximum file size in MB"
    )
    UPLOAD_DIR: str = Field(
        default="./uploads", 
        description="Upload directory"
    )
    ALLOWED_AUDIO_FORMATS: str = Field(
        default="wav,mp3,m4a,flac,ogg",
        description="Allowed audio formats"
    )
    
    # Speech Recognition Configuration
    # Local Whisper Configuration
    WHISPER_MODEL: str = Field(
        default="base",
        description="Local Whisper model name (tiny, base, small, medium, large, large-v2, large-v3)"
    )
    WHISPER_MODEL_PATH: str = Field(
        default="./models/whisper",
        description="Local Whisper model cache directory"
    )
    WHISPER_DEVICE: str = Field(
        default="cpu",
        description="Whisper computation device (cpu, cuda, auto)"
    )
    WHISPER_COMPUTE_TYPE: str = Field(
        default="int8",
        description="Whisper compute type (int8, int16, float16, float32)"
    )
    
    # Google Cloud Speech (Optional)
    GOOGLE_CLOUD_CREDENTIALS_PATH: str = Field(
        default="./credentials/google-speech-credentials.json",
        description="Google Cloud credentials path"
    )
    GOOGLE_CLOUD_PROJECT_ID: str = Field(
        default="",
        description="Google Cloud project ID"
    )
    GOOGLE_SPEECH_ENABLED: bool = Field(
        default=False,
        description="Enable Google Speech service"
    )
    
    # Azure Speech (Optional)  
    AZURE_SPEECH_KEY: str = Field(
        default="",
        description="Azure Speech service key"
    )
    AZURE_SPEECH_REGION: str = Field(
        default="",
        description="Azure Speech service region"
    )
    AZURE_SPEECH_ENABLED: bool = Field(
        default=False,
        description="Enable Azure Speech service"
    )
    
    # Default Speech Engine
    DEFAULT_SPEECH_ENGINE: str = Field(
        default="whisper",
        description="Default speech recognition engine (whisper, google, azure)"
    )
    
    # Audio Processing
    SAMPLE_RATE: int = Field(default=16000, description="Audio sample rate")
    CHUNK_SIZE: int = Field(default=1024, description="Audio chunk size")
    AUDIO_FORMAT: str = Field(default="wav", description="Default audio format")
    NOISE_REDUCTION_ENABLED: bool = Field(
        default=True, 
        description="Enable noise reduction"
    )
    VOICE_ACTIVITY_DETECTION: bool = Field(
        default=True, 
        description="Enable voice activity detection"
    )
    
    # Pronunciation Assessment & Forced Alignment
    PRONUNCIATION_THRESHOLD: float = Field(
        default=0.7, 
        description="Pronunciation accuracy threshold"
    )
    SIMILARITY_THRESHOLD: float = Field(
        default=0.8, 
        description="Voice similarity threshold"
    )
    CONFIDENCE_THRESHOLD: float = Field(
        default=0.6, 
        description="Recognition confidence threshold"
    )
    
    # Phonemization Configuration
    PHONEMIZER_BACKEND: str = Field(
        default="espeak", 
        description="Phonemizer backend (espeak, festival, segments)"
    )
    PHONEMIZER_LANGUAGE: str = Field(
        default="en-us", 
        description="Language for phonemization"
    )
    
    # Forced Alignment Configuration
    FORCED_ALIGNMENT_ENGINE: str = Field(
        default="mfa", 
        description="Forced alignment engine (mfa, gentle, wav2vec2)"
    )
    MFA_MODEL_PATH: str = Field(
        default="./models/mfa", 
        description="Montreal Forced Aligner model path"
    )
    MFA_ACOUSTIC_MODEL: str = Field(
        default="english_us_arpa", 
        description="MFA acoustic model name"
    )
    MFA_DICTIONARY: str = Field(
        default="english_us_arpa", 
        description="MFA pronunciation dictionary"
    )
    
    # Error Detection Thresholds
    PHONEME_SUBSTITUTION_THRESHOLD: float = Field(
        default=0.7, 
        description="Threshold for phoneme substitution detection"
    )
    PHONEME_OMISSION_THRESHOLD: float = Field(
        default=0.5, 
        description="Threshold for phoneme omission detection"
    )
    WORD_ACCURACY_THRESHOLD: float = Field(
        default=0.8, 
        description="Threshold for word-level accuracy"
    )
    
    # Logging Configuration
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    LOG_FORMAT: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format"
    )
    LOG_FILE: str = Field(
        default="speech_service.log", 
        description="Log file name"
    )
    LOG_MAX_BYTES: int = Field(
        default=10485760, 
        description="Log file max size in bytes"
    )
    LOG_BACKUP_COUNT: int = Field(
        default=5, 
        description="Log file backup count"
    )
    
    # Service Mode Configuration
    SERVICE_MODE: str = Field(
        default="both",
        description="Service mode: 'stt' (only STT/Whisper), 'tts' (only TTS/VibeVoice), 'both' (all features)"
    )
    
    # TTS Configuration (VibeVoice)
    TTS_MODEL_NAME: str = Field(
        default="microsoft/VibeVoice-Realtime-0.5B",
        description="VibeVoice model name from HuggingFace"
    )
    TTS_MODEL_PATH: str = Field(
        default="./models/vibevoice",
        description="Local cache directory for TTS model"
    )
    TTS_DEVICE: str = Field(
        default="auto",
        description="TTS computation device (cpu, cuda, auto)"
    )
    TTS_SAMPLE_RATE: int = Field(
        default=24000,
        description="TTS output audio sample rate"
    )
    TTS_MAX_TEXT_LENGTH: int = Field(
        default=500,
        description="Maximum text length for TTS"
    )
    TTS_VOICE_STYLE: str = Field(
        default="friendly",
        description="Voice style/personality"
    )
    
    # Health Check
    HEALTH_CHECK_TIMEOUT: int = Field(
        default=30, 
        description="Health check timeout in seconds"
    )
    
    # Rate Limiting
    RATE_LIMIT_REQUESTS: int = Field(
        default=100, 
        description="Rate limit requests per window"
    )
    RATE_LIMIT_WINDOW_SECONDS: int = Field(
        default=60, 
        description="Rate limit window in seconds"
    )
    
    # Circuit Breaker
    CIRCUIT_BREAKER_FAILURE_THRESHOLD: int = Field(
        default=5, 
        description="Circuit breaker failure threshold"
    )
    CIRCUIT_BREAKER_TIMEOUT_SECONDS: int = Field(
        default=60, 
        description="Circuit breaker timeout in seconds"
    )
    CIRCUIT_BREAKER_RECOVERY_TIMEOUT: int = Field(
        default=300, 
        description="Circuit breaker recovery timeout in seconds"
    )
    
    # Monitoring
    METRICS_ENABLED: bool = Field(
        default=True, 
        description="Enable metrics collection"
    )
    TRACING_ENABLED: bool = Field(
        default=False, 
        description="Enable distributed tracing"
    )
    
    @property
    def allowed_audio_formats_list(self) -> List[str]:
        """Get allowed audio formats as a list"""
        return [fmt.strip() for fmt in self.ALLOWED_AUDIO_FORMATS.split(",")]
    
    @property
    def max_file_size_bytes(self) -> int:
        """Get max file size in bytes"""
        return self.MAX_FILE_SIZE_MB * 1024 * 1024


@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings"""
    return Settings()