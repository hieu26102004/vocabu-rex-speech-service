"""
Fluency Analyzer Implementation  
Analyzes speech fluency, rhythm, and timing patterns
"""

import logging
from typing import Dict, List
import numpy as np
from scipy import stats

from src.domain.services.asr_service import IFluencyAnalyzer
from src.domain.entities.asr_entities import ActualUtterance

logger = logging.getLogger(__name__)


class FluencyAnalyzerService(IFluencyAnalyzer):
    """
    Implementation of speech fluency analysis service
    """
    
    def __init__(self):
        # Reference values for fluency analysis
        self.reference_speech_rates = {
            "native": (140, 180),      # words per minute
            "fluent": (120, 160),
            "intermediate": (100, 140),
            "beginner": (80, 120)
        }
        
        self.reference_pause_patterns = {
            "acceptable_pause_rate": 0.15,  # 15% of speech time
            "max_pause_duration": 2.0,      # 2 seconds
            "min_pause_duration": 0.2       # 200ms
        }
    
    async def analyze_speech_rate(
        self,
        utterance: ActualUtterance,
        reference_rate: float = None
    ) -> float:
        """
        Analyze speaking rate (words per minute)
        """
        try:
            speech_rate = utterance.speech_rate
            
            # Determine target rate category
            if reference_rate:
                target_min, target_max = reference_rate * 0.85, reference_rate * 1.15
            else:
                # Use native speaker range as default
                target_min, target_max = self.reference_speech_rates["native"]
            
            # Calculate score based on how close to target range
            if target_min <= speech_rate <= target_max:
                score = 100.0
            elif speech_rate < target_min:
                # Too slow
                ratio = speech_rate / target_min
                score = max(50.0, ratio * 100.0)
            else:
                # Too fast
                ratio = target_max / speech_rate
                score = max(50.0, ratio * 100.0)
            
            return min(100.0, score)
            
        except Exception as e:
            logger.error(f"Speech rate analysis failed: {e}")
            return 50.0
    
    async def analyze_pause_patterns(
        self,
        utterance: ActualUtterance
    ) -> Dict[str, float]:
        """
        Analyze pause patterns in speech
        """
        try:
            total_duration = utterance.total_duration
            pause_count = utterance.pause_count
            pause_duration = utterance.pause_duration_total
            
            if total_duration <= 0:
                return {"pause_score": 0.0, "pause_rate": 0.0}
            
            # Calculate pause rate (percentage of total time)
            pause_rate = pause_duration / total_duration
            
            # Calculate pause frequency (pauses per minute)
            pause_frequency = (pause_count / total_duration) * 60
            
            # Score pause appropriateness
            acceptable_rate = self.reference_pause_patterns["acceptable_pause_rate"]
            
            if pause_rate <= acceptable_rate:
                rate_score = 100.0
            else:
                # Penalty for too many/long pauses
                excess_ratio = pause_rate / acceptable_rate
                rate_score = max(30.0, 100.0 / excess_ratio)
            
            # Score pause frequency
            if 2 <= pause_frequency <= 8:  # 2-8 pauses per minute is reasonable
                frequency_score = 100.0
            elif pause_frequency < 2:
                # Too few pauses (might indicate rushed speech)
                frequency_score = 85.0
            else:
                # Too many pauses
                frequency_score = max(40.0, 100.0 - (pause_frequency - 8) * 10)
            
            # Combined pause score
            pause_score = (rate_score + frequency_score) / 2.0
            
            return {
                "pause_score": pause_score,
                "pause_rate": pause_rate * 100,  # as percentage
                "pause_frequency": pause_frequency,
                "pause_count": pause_count,
                "total_pause_duration": pause_duration
            }
            
        except Exception as e:
            logger.error(f"Pause pattern analysis failed: {e}")
            return {"pause_score": 50.0, "pause_rate": 0.0}
    
    async def analyze_rhythm(
        self,
        utterance: ActualUtterance
    ) -> float:
        """
        Analyze speech rhythm and timing consistency
        """
        try:
            words = utterance.words
            
            if len(words) < 3:
                return 80.0  # Default for very short utterances
            
            # Calculate word duration variability
            word_durations = [word.duration for word in words]
            
            # Remove outliers (very short function words)
            filtered_durations = [d for d in word_durations if d > 0.1]
            
            if len(filtered_durations) < 2:
                return 70.0
            
            # Calculate coefficient of variation (CV)
            mean_duration = np.mean(filtered_durations)
            std_duration = np.std(filtered_durations)
            
            if mean_duration <= 0:
                return 50.0
            
            cv = std_duration / mean_duration
            
            # Good rhythm has moderate variability (CV ~ 0.3-0.6)
            if 0.3 <= cv <= 0.6:
                rhythm_score = 100.0
            elif cv < 0.3:
                # Too monotonous
                rhythm_score = 70.0 + (cv / 0.3) * 30.0
            else:
                # Too variable/choppy
                rhythm_score = max(50.0, 100.0 - (cv - 0.6) * 100.0)
            
            # Analyze timing regularity
            regularity_score = self._analyze_timing_regularity(word_durations)
            
            # Combined rhythm score
            final_score = (rhythm_score + regularity_score) / 2.0
            
            return min(100.0, max(0.0, final_score))
            
        except Exception as e:
            logger.error(f"Rhythm analysis failed: {e}")
            return 60.0
    
    async def calculate_fluency_score(
        self,
        speech_rate_score: float,
        pause_score: float,
        rhythm_score: float
    ) -> float:
        """
        Calculate overall fluency score
        """
        try:
            # Weighted combination
            weights = {
                "speech_rate": 0.4,
                "pause_patterns": 0.3,
                "rhythm": 0.3
            }
            
            fluency_score = (
                speech_rate_score * weights["speech_rate"] +
                pause_score * weights["pause_patterns"] +
                rhythm_score * weights["rhythm"]
            )
            
            return min(100.0, max(0.0, fluency_score))
            
        except Exception as e:
            logger.error(f"Fluency score calculation failed: {e}")
            return 50.0
    
    def _analyze_timing_regularity(self, word_durations: List[float]) -> float:
        """
        Analyze timing regularity using statistical measures
        """
        try:
            if len(word_durations) < 3:
                return 75.0
            
            # Calculate inter-word interval variation
            intervals = []
            for i in range(len(word_durations) - 1):
                intervals.append(abs(word_durations[i+1] - word_durations[i]))
            
            if not intervals:
                return 75.0
            
            # Measure consistency (lower variation = more regular)
            mean_interval = np.mean(intervals)
            std_interval = np.std(intervals)
            
            if mean_interval <= 0:
                return 60.0
            
            # Regularity coefficient (lower is better)
            regularity_coeff = std_interval / mean_interval
            
            # Convert to score (0-100)
            if regularity_coeff <= 0.5:
                score = 100.0
            elif regularity_coeff <= 1.0:
                score = 100.0 - (regularity_coeff - 0.5) * 60.0
            else:
                score = max(30.0, 70.0 - regularity_coeff * 20.0)
            
            return score
            
        except Exception as e:
            logger.warning(f"Timing regularity analysis failed: {e}")
            return 60.0
    
    def _detect_hesitations_and_fillers(
        self,
        utterance: ActualUtterance
    ) -> Dict[str, Any]:
        """
        Detect hesitations and filler words
        """
        try:
            filler_words = {"um", "uh", "er", "ah", "like", "you know"}
            
            total_words = len(utterance.words)
            filler_count = 0
            
            for word in utterance.words:
                if word.word.lower() in filler_words:
                    filler_count += 1
            
            filler_rate = (filler_count / total_words) * 100 if total_words > 0 else 0
            
            # Score based on filler usage
            if filler_rate <= 2.0:  # Less than 2% fillers
                filler_score = 100.0
            elif filler_rate <= 5.0:
                filler_score = 85.0
            elif filler_rate <= 10.0:
                filler_score = 70.0
            else:
                filler_score = max(40.0, 100.0 - filler_rate * 5.0)
            
            return {
                "filler_count": filler_count,
                "filler_rate": filler_rate,
                "filler_score": filler_score
            }
            
        except Exception as e:
            logger.warning(f"Filler detection failed: {e}")
            return {"filler_score": 80.0}