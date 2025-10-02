"""
Pronunciation Analyzer Implementation
Analyzes phoneme-level pronunciation accuracy and provides feedback
"""

import logging
from typing import List, Dict, Any
from datetime import datetime

from src.domain.services.asr_service import IPronunciationAnalyzer
from src.domain.entities.asr_entities import (
    ASRResult,
    PronunciationComparison,
    PronunciationFeedback
)

logger = logging.getLogger(__name__)


class PronunciationAnalyzerService(IPronunciationAnalyzer):
    """
    Implementation of pronunciation analysis service
    """
    
    def __init__(self):
        # Phoneme similarity mappings
        self.phoneme_similarities = {
            # Vowel similarities
            "AH0": ["AH1", "AH2", "UH0"],
            "IH0": ["IH1", "IH2", "I"],
            "EH0": ["EH1", "EH2", "E"],
            # Consonant similarities  
            "TH": ["F", "S", "T"],
            "DH": ["V", "Z", "D"],
            "R": ["W", "L"],
            "L": ["R", "W"]
        }
        
        # Common pronunciation errors by native language
        self.common_errors = {
            "spanish": {
                "TH": "T",
                "DH": "D", 
                "SH": "S",
                "ZH": "S"
            },
            "chinese": {
                "TH": "S",
                "DH": "Z",
                "R": "L",
                "L": "R"
            },
            "german": {
                "TH": "S",
                "DH": "Z",
                "W": "V"
            }
        }
    
    async def analyze_phoneme_accuracy(
        self,
        actual_phonemes: List[str],
        reference_phonemes: List[str]
    ) -> List[PronunciationComparison]:
        """
        Analyze phoneme-level pronunciation accuracy
        """
        try:
            comparisons = []
            
            # Use dynamic programming for alignment
            aligned_pairs = self._align_phoneme_sequences(
                actual_phonemes, reference_phonemes
            )
            
            for actual_phoneme, reference_phoneme, alignment_type in aligned_pairs:
                similarity_score = await self.calculate_similarity_score(
                    actual_phoneme, reference_phoneme
                )
                
                phoneme_match = actual_phoneme == reference_phoneme
                error_type = None if phoneme_match else alignment_type
                
                comparison = PronunciationComparison(
                    reference_phoneme=reference_phoneme or "<MISSING>",
                    actual_phoneme=actual_phoneme or "<EXTRA>",
                    phoneme_match=phoneme_match,
                    similarity_score=similarity_score,
                    timing_deviation=0.0,  # Would calculate from timing data
                    error_type=error_type
                )
                
                comparisons.append(comparison)
            
            return comparisons
            
        except Exception as e:
            logger.error(f"Phoneme accuracy analysis failed: {e}")
            return []
    
    async def calculate_similarity_score(
        self,
        actual_phoneme: str,
        reference_phoneme: str
    ) -> float:
        """
        Calculate similarity score between two phonemes
        """
        try:
            if not actual_phoneme or not reference_phoneme:
                return 0.0
            
            if actual_phoneme == reference_phoneme:
                return 1.0
            
            # Check phoneme similarity mappings
            if reference_phoneme in self.phoneme_similarities:
                similar_phonemes = self.phoneme_similarities[reference_phoneme]
                if actual_phoneme in similar_phonemes:
                    return 0.8  # High similarity
            
            # Check reverse mapping
            if actual_phoneme in self.phoneme_similarities:
                similar_phonemes = self.phoneme_similarities[actual_phoneme]
                if reference_phoneme in similar_phonemes:
                    return 0.8
            
            # Phonetic feature-based similarity
            feature_similarity = self._calculate_phonetic_feature_similarity(
                actual_phoneme, reference_phoneme
            )
            
            return feature_similarity
            
        except Exception as e:
            logger.warning(f"Similarity calculation failed: {e}")
            return 0.3  # Default low similarity
    
    async def identify_error_type(
        self,
        actual_phonemes: List[str],
        reference_phonemes: List[str],
        index: int
    ) -> str:
        """
        Identify type of pronunciation error
        """
        try:
            if index >= len(actual_phonemes):
                return "deletion"  # Missing phoneme
            elif index >= len(reference_phonemes):
                return "insertion"  # Extra phoneme
            elif actual_phonemes[index] != reference_phonemes[index]:
                return "substitution"  # Wrong phoneme
            else:
                return "match"  # Correct phoneme
                
        except Exception as e:
            logger.warning(f"Error type identification failed: {e}")
            return "unknown"
    
    async def generate_feedback(
        self,
        asr_result: ASRResult
    ) -> PronunciationFeedback:
        """
        Generate detailed pronunciation feedback
        """
        try:
            return PronunciationFeedback.generate_from_result(asr_result)
            
        except Exception as e:
            logger.error(f"Feedback generation failed: {e}")
            return PronunciationFeedback(
                overall_feedback="Unable to generate feedback",
                strengths=[],
                areas_for_improvement=["Review pronunciation basics"],
                specific_phoneme_feedback=[],
                practice_suggestions=["Practice with audio resources"],
                difficulty_level_recommendation="beginner"
            )
    
    def _align_phoneme_sequences(
        self,
        actual: List[str],
        reference: List[str]
    ) -> List[tuple]:
        """
        Align phoneme sequences using simple dynamic programming
        """
        try:
            # Simple alignment algorithm
            aligned_pairs = []
            
            i, j = 0, 0
            while i < len(actual) or j < len(reference):
                if i >= len(actual):
                    # Deletion (missing phoneme)
                    aligned_pairs.append((None, reference[j], "deletion"))
                    j += 1
                elif j >= len(reference):
                    # Insertion (extra phoneme)
                    aligned_pairs.append((actual[i], None, "insertion"))
                    i += 1
                elif actual[i] == reference[j]:
                    # Match
                    aligned_pairs.append((actual[i], reference[j], "match"))
                    i += 1
                    j += 1
                else:
                    # Substitution (for now, advance both)
                    aligned_pairs.append((actual[i], reference[j], "substitution"))
                    i += 1
                    j += 1
            
            return aligned_pairs
            
        except Exception as e:
            logger.warning(f"Phoneme alignment failed: {e}")
            # Fallback: simple pairing
            max_len = max(len(actual), len(reference))
            aligned_pairs = []
            
            for idx in range(max_len):
                act_phoneme = actual[idx] if idx < len(actual) else None
                ref_phoneme = reference[idx] if idx < len(reference) else None
                
                if act_phoneme and ref_phoneme:
                    error_type = "match" if act_phoneme == ref_phoneme else "substitution"
                elif act_phoneme:
                    error_type = "insertion"
                else:
                    error_type = "deletion"
                
                aligned_pairs.append((act_phoneme, ref_phoneme, error_type))
            
            return aligned_pairs
    
    def _calculate_phonetic_feature_similarity(
        self,
        phoneme1: str,
        phoneme2: str
    ) -> float:
        """
        Calculate similarity based on phonetic features
        """
        try:
            # Simplified phonetic feature analysis
            # In practice, would use detailed phonetic feature matrices
            
            # Vowel vs consonant
            vowels = {"A", "E", "I", "O", "U", "AH", "EH", "IH", "OW", "UH"}
            
            p1_vowel = any(v in phoneme1 for v in vowels)
            p2_vowel = any(v in phoneme2 for v in vowels)
            
            if p1_vowel != p2_vowel:
                return 0.1  # Very different (vowel vs consonant)
            
            # Same category (both vowels or both consonants)
            if p1_vowel and p2_vowel:
                # Vowel similarity (simplified)
                return 0.6
            else:
                # Consonant similarity (simplified)
                return 0.5
                
        except Exception as e:
            logger.warning(f"Phonetic feature similarity failed: {e}")
            return 0.3