"""
Audio processing utilities for forced alignment
Handles audio format conversion, validation, and preprocessing
"""

import os
import logging
import tempfile
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from enum import Enum

import librosa
import soundfile as sf
import numpy as np
from scipy import signal

logger = logging.getLogger(__name__)


class AudioFormat(Enum):
    """Supported audio formats"""
    WAV = "wav"
    FLAC = "flac"
    MP3 = "mp3"
    M4A = "m4a"
    OGG = "ogg"


class AudioProcessor:
    """
    Audio processing utilities for speech alignment
    """
    
    def __init__(self, target_sample_rate: int = 16000):
        """
        Initialize audio processor
        
        Args:
            target_sample_rate: Target sample rate for processing
        """
        self.target_sample_rate = target_sample_rate
        self.temp_dir = Path(tempfile.gettempdir()) / "vocabu_rex_audio"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
    
    def validate_audio_file(self, file_path: str) -> Dict[str, Any]:
        """
        Validate audio file and return properties
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Dictionary with validation results and audio properties
        """
        try:
            audio_path = Path(file_path)
            
            if not audio_path.exists():
                return {
                    "valid": False,
                    "error": "File does not exist",
                    "file_path": file_path
                }
            
            # Check file extension
            extension = audio_path.suffix.lower()
            supported_formats = [f".{fmt.value}" for fmt in AudioFormat]
            
            if extension not in supported_formats:
                return {
                    "valid": False,
                    "error": f"Unsupported format: {extension}",
                    "supported_formats": supported_formats
                }
            
            # Load audio properties
            try:
                data, sample_rate = sf.read(file_path)
                duration = len(data) / sample_rate
                
                # Check if mono or stereo
                channels = 1 if data.ndim == 1 else data.shape[1]
                
                # File size
                file_size = audio_path.stat().st_size
                
                return {
                    "valid": True,
                    "file_path": file_path,
                    "format": extension[1:],  # Remove dot
                    "duration": duration,
                    "sample_rate": sample_rate,
                    "channels": channels,
                    "file_size": file_size,
                    "samples": len(data) if data.ndim == 1 else data.shape[0],
                    "bit_depth": "16-bit",  # Default assumption
                    "is_mono": channels == 1,
                    "quality_score": self._calculate_quality_score(
                        duration, sample_rate, channels, file_size
                    )
                }
                
            except Exception as e:
                return {
                    "valid": False,
                    "error": f"Could not read audio file: {str(e)}",
                    "file_path": file_path
                }
        
        except Exception as e:
            logger.error(f"Audio validation failed: {e}")
            return {
                "valid": False,
                "error": f"Validation error: {str(e)}",
                "file_path": file_path
            }
    
    def _calculate_quality_score(
        self, 
        duration: float, 
        sample_rate: int, 
        channels: int, 
        file_size: int
    ) -> float:
        """
        Calculate audio quality score for alignment suitability
        
        Returns:
            Quality score from 0.0 to 1.0
        """
        score = 1.0
        
        # Duration scoring
        if duration < 0.5:
            score -= 0.3  # Too short
        elif duration > 300:
            score -= 0.2  # Very long
        
        # Sample rate scoring
        if sample_rate < 8000:
            score -= 0.4  # Too low
        elif sample_rate < 16000:
            score -= 0.2  # Low but acceptable
        elif sample_rate > 48000:
            score -= 0.1  # Unnecessarily high
        
        # Channel scoring
        if channels > 2:
            score -= 0.1  # Multi-channel not ideal
        
        # Bitrate estimation and scoring
        estimated_bitrate = (file_size * 8) / duration if duration > 0 else 0
        if estimated_bitrate < 64000:
            score -= 0.3  # Too compressed
        elif estimated_bitrate > 320000:
            score -= 0.1  # Unnecessarily high
        
        return max(0.0, min(1.0, score))
    
    def convert_to_target_format(
        self, 
        input_path: str, 
        output_path: Optional[str] = None,
        target_format: AudioFormat = AudioFormat.WAV
    ) -> str:
        """
        Convert audio to target format for alignment
        
        Args:
            input_path: Input audio file path
            output_path: Output path (if None, creates temp file)
            target_format: Target audio format
            
        Returns:
            Path to converted audio file
        """
        try:
            if not output_path:
                timestamp = str(int(librosa.get_duration(path=input_path) * 1000))
                output_path = str(self.temp_dir / f"converted_{timestamp}.{target_format.value}")
            
            # Load audio
            data, sample_rate = librosa.load(
                input_path, 
                sr=self.target_sample_rate, 
                mono=True
            )
            
            # Save in target format
            sf.write(output_path, data, self.target_sample_rate, format=target_format.value.upper())
            
            logger.info(f"Converted {input_path} to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Audio conversion failed: {e}")
            raise
    
    def preprocess_for_alignment(
        self, 
        input_path: str, 
        output_path: Optional[str] = None
    ) -> str:
        """
        Preprocess audio for optimal forced alignment results
        
        Args:
            input_path: Input audio file path
            output_path: Output path (if None, creates temp file)
            
        Returns:
            Path to preprocessed audio file
        """
        try:
            if not output_path:
                timestamp = str(int(librosa.get_duration(path=input_path) * 1000))
                output_path = str(self.temp_dir / f"preprocessed_{timestamp}.wav")
            
            # Load audio
            data, sample_rate = librosa.load(
                input_path, 
                sr=self.target_sample_rate, 
                mono=True
            )
            
            # Apply preprocessing steps
            data = self._normalize_volume(data)
            data = self._remove_silence(data, sample_rate)
            data = self._apply_noise_reduction(data, sample_rate)
            data = self._apply_preemphasis(data)
            
            # Ensure minimum duration
            min_duration = 0.1  # 100ms minimum
            min_samples = int(min_duration * sample_rate)
            if len(data) < min_samples:
                # Pad with silence if too short
                padding = min_samples - len(data)
                data = np.pad(data, (0, padding), mode='constant')
            
            # Save preprocessed audio
            sf.write(output_path, data, self.target_sample_rate)
            
            logger.info(f"Preprocessed {input_path} to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Audio preprocessing failed: {e}")
            raise
    
    def _normalize_volume(self, data: np.ndarray) -> np.ndarray:
        """Normalize audio volume"""
        try:
            # RMS normalization
            rms = np.sqrt(np.mean(data**2))
            if rms > 0:
                target_rms = 0.1  # Target RMS level
                data = data * (target_rms / rms)
            
            # Peak normalization to prevent clipping
            peak = np.max(np.abs(data))
            if peak > 0.95:
                data = data * (0.95 / peak)
            
            return data
            
        except Exception as e:
            logger.warning(f"Volume normalization failed: {e}")
            return data
    
    def _remove_silence(self, data: np.ndarray, sample_rate: int) -> np.ndarray:
        """Remove leading and trailing silence"""
        try:
            # Use librosa's trim function
            data_trimmed, _ = librosa.effects.trim(
                data, 
                top_db=20,  # Threshold in dB
                frame_length=2048,
                hop_length=512
            )
            
            return data_trimmed if len(data_trimmed) > 0 else data
            
        except Exception as e:
            logger.warning(f"Silence removal failed: {e}")
            return data
    
    def _apply_noise_reduction(self, data: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply basic noise reduction"""
        try:
            # Simple high-pass filter to remove low-frequency noise
            nyquist = sample_rate / 2
            cutoff = 80  # Hz
            normalized_cutoff = cutoff / nyquist
            
            if normalized_cutoff < 1.0:
                b, a = signal.butter(5, normalized_cutoff, btype='high')
                data_filtered = signal.filtfilt(b, a, data)
                return data_filtered
            else:
                return data
                
        except Exception as e:
            logger.warning(f"Noise reduction failed: {e}")
            return data
    
    def _apply_preemphasis(self, data: np.ndarray, alpha: float = 0.97) -> np.ndarray:
        """Apply preemphasis filter"""
        try:
            # Preemphasis: y[n] = x[n] - alpha * x[n-1]
            return np.append(data[0], data[1:] - alpha * data[:-1])
            
        except Exception as e:
            logger.warning(f"Preemphasis failed: {e}")
            return data
    
    def extract_audio_features(self, file_path: str) -> Dict[str, Any]:
        """
        Extract audio features for analysis
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Dictionary with extracted features
        """
        try:
            # Load audio
            data, sample_rate = librosa.load(file_path, sr=None)
            
            # Basic features
            duration = len(data) / sample_rate
            rms_energy = np.sqrt(np.mean(data**2))
            zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(data))
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=data, sr=sample_rate)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=data, sr=sample_rate)
            
            # MFCC features
            mfccs = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=13)
            
            # Tempo and beat
            tempo, beat_frames = librosa.beat.beat_track(y=data, sr=sample_rate)
            
            return {
                "duration": duration,
                "sample_rate": sample_rate,
                "rms_energy": float(rms_energy),
                "zero_crossing_rate": float(np.mean(zero_crossing_rate)),
                "spectral_centroid_mean": float(np.mean(spectral_centroids)),
                "spectral_centroid_std": float(np.std(spectral_centroids)),
                "spectral_rolloff_mean": float(np.mean(spectral_rolloff)),
                "mfcc_mean": mfccs.mean(axis=1).tolist(),
                "mfcc_std": mfccs.std(axis=1).tolist(),
                "tempo": float(tempo),
                "beat_count": len(beat_frames),
                "signal_to_noise_ratio": self._estimate_snr(data)
            }
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return {}
    
    def _estimate_snr(self, data: np.ndarray) -> float:
        """
        Estimate Signal-to-Noise Ratio
        
        Args:
            data: Audio signal data
            
        Returns:
            Estimated SNR in dB
        """
        try:
            # Simple SNR estimation
            # Assume noise is in the quieter portions of the signal
            
            # Calculate energy in sliding windows
            window_size = 1024
            hop_length = 512
            
            # Frame the signal
            frames = librosa.util.frame(data, frame_length=window_size, hop_length=hop_length)
            
            # Calculate RMS for each frame
            rms_per_frame = np.sqrt(np.mean(frames**2, axis=0))
            
            # Estimate signal and noise levels
            # Signal: higher percentile of RMS values
            # Noise: lower percentile of RMS values
            signal_level = np.percentile(rms_per_frame, 90)
            noise_level = np.percentile(rms_per_frame, 10)
            
            if noise_level > 0:
                snr = 20 * np.log10(signal_level / noise_level)
                return float(snr)
            else:
                return float('inf')  # Perfect signal
                
        except Exception as e:
            logger.warning(f"SNR estimation failed: {e}")
            return 0.0
    
    def segment_audio(
        self, 
        file_path: str, 
        segment_duration: float = 10.0,
        overlap: float = 1.0
    ) -> list:
        """
        Segment long audio file into smaller chunks for processing
        
        Args:
            file_path: Path to audio file
            segment_duration: Duration of each segment in seconds
            overlap: Overlap between segments in seconds
            
        Returns:
            List of segment file paths
        """
        try:
            # Load audio
            data, sample_rate = librosa.load(file_path, sr=self.target_sample_rate)
            total_duration = len(data) / sample_rate
            
            if total_duration <= segment_duration:
                # No need to segment
                return [file_path]
            
            segments = []
            segment_samples = int(segment_duration * sample_rate)
            overlap_samples = int(overlap * sample_rate)
            step_samples = segment_samples - overlap_samples
            
            start_idx = 0
            segment_idx = 0
            
            while start_idx < len(data):
                end_idx = min(start_idx + segment_samples, len(data))
                segment_data = data[start_idx:end_idx]
                
                # Save segment
                segment_path = str(self.temp_dir / f"segment_{segment_idx:03d}.wav")
                sf.write(segment_path, segment_data, sample_rate)
                segments.append(segment_path)
                
                segment_idx += 1
                start_idx += step_samples
                
                # Break if we've covered the whole file
                if end_idx >= len(data):
                    break
            
            logger.info(f"Segmented {file_path} into {len(segments)} parts")
            return segments
            
        except Exception as e:
            logger.error(f"Audio segmentation failed: {e}")
            return [file_path]  # Return original if segmentation fails
    
    def cleanup_temp_files(self):
        """Remove temporary files created during processing"""
        try:
            if self.temp_dir.exists():
                for temp_file in self.temp_dir.glob("*"):
                    if temp_file.is_file():
                        temp_file.unlink(missing_ok=True)
                logger.info("Temporary audio files cleaned up")
                
        except Exception as e:
            logger.warning(f"Temp file cleanup failed: {e}")


# Utility functions
def get_audio_duration(file_path: str) -> float:
    """Get audio file duration in seconds"""
    try:
        return librosa.get_duration(path=file_path)
    except Exception as e:
        logger.error(f"Could not get duration for {file_path}: {e}")
        return 0.0


def convert_audio_format(
    input_path: str, 
    output_path: str, 
    target_format: str = "wav",
    sample_rate: int = 16000
) -> bool:
    """
    Convert audio file to target format
    
    Args:
        input_path: Input file path
        output_path: Output file path  
        target_format: Target format (wav, flac, etc.)
        sample_rate: Target sample rate
        
    Returns:
        True if conversion successful, False otherwise
    """
    try:
        data, sr = librosa.load(input_path, sr=sample_rate, mono=True)
        sf.write(output_path, data, sample_rate, format=target_format.upper())
        return True
    except Exception as e:
        logger.error(f"Audio format conversion failed: {e}")
        return False


def is_audio_file_valid(file_path: str) -> bool:
    """Quick validation check for audio files"""
    try:
        processor = AudioProcessor()
        validation_result = processor.validate_audio_file(file_path)
        return validation_result.get("valid", False)
    except Exception:
        return False