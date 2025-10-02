"""
Demo script for Step 2: Forced Alignment
Test the complete phonemization + forced alignment pipeline
"""

import asyncio
import json
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.infrastructure.services.mfa_alignment_service import MFAAlignmentService, MFAModelManager
from src.infrastructure.services.phonemization_service import PhonemizationService
from src.infrastructure.utils.audio_processor import AudioProcessor
from src.application.usecases.alignment_usecases import AlignmentUseCases
from src.application.usecases.phonemization_usecases import PhonemizationUseCases
from src.application.dtos.alignment_dto import (
    AlignmentRequest,
    AlignmentValidationRequest,
    PhonemeAlignmentRequest
)
from src.application.dtos.phonemization_dto import PhonemizationRequest


async def demo_step2_forced_alignment():
    """
    Demo Step 2: Forced Alignment pipeline
    """
    print("=" * 70)
    print("🎯 VocabuRex Speech Service - Step 2: Forced Alignment Demo")
    print("=" * 70)
    
    try:
        print("\n1. Initializing Services...")
        
        # Initialize all services
        phonemization_service = PhonemizationService()
        alignment_service = MFAAlignmentService()
        model_manager = MFAModelManager()
        audio_processor = AudioProcessor()
        
        # Initialize use cases
        phonemization_use_cases = PhonemizationUseCases(phonemization_service)
        alignment_use_cases = AlignmentUseCases(
            alignment_service, model_manager, audio_processor
        )
        
        print("✅ All services initialized successfully")
        
        print("\n2. Testing Audio Processing Utilities...")
        
        # Demo audio validation (with dummy file path)
        test_audio_path = "demo_audio.wav"  # Would be real file in practice
        validation_request = AlignmentValidationRequest(
            check_format=True,
            check_duration=True,
            check_quality=True,
            extract_features=False
        )
        
        print(f"📋 Audio validation for: {test_audio_path}")
        print("   - Format validation: enabled")
        print("   - Duration validation: enabled") 
        print("   - Quality assessment: enabled")
        
        print("\n3. Testing Supported Languages and Models...")
        
        # Get supported languages
        languages_response = await alignment_use_cases.get_supported_languages()
        print(f"✅ {languages_response.total_languages} languages supported:")
        for lang in languages_response.supported_languages[:5]:  # Show first 5
            print(f"   • {lang}")
        
        # Get available models for English
        models_response = await alignment_use_cases.get_available_models("english")
        print(f"✅ {models_response.total_models} models available for English:")
        for model in models_response.models[:3]:  # Show first 3
            print(f"   • {model}")
        
        print("\n4. Demo: Complete Phonemization + Alignment Pipeline...")
        
        # Step 1: Phonemize text first
        test_text = "The quick brown fox jumps over the lazy dog"
        print(f"📝 Input text: '{test_text}'")
        
        phonemization_request = PhonemizationRequest(
            text=test_text,
            language="en-us",
            include_stress=True
        )
        
        print("\n   4a. Step 1: Text → Phonemes")
        phonemization_result = await phonemization_use_cases.phonemize_text(phonemization_request)
        
        if phonemization_result.success:
            print(f"   ✅ Phonemization completed in {phonemization_result.processing_time_ms:.1f}ms")
            
            # Extract phonemes for alignment
            all_phonemes = []
            for word_result in phonemization_result.words:
                all_phonemes.extend(word_result.phonemes)
            
            print(f"   📊 Generated {len(all_phonemes)} phonemes from {phonemization_result.total_words} words")
            print(f"   🔤 Phoneme sequence: {' '.join(all_phonemes[:10])}..." if len(all_phonemes) > 10 else f"   🔤 Phoneme sequence: {' '.join(all_phonemes)}")
            
            print("\n   4b. Step 2: Audio + Phonemes → Timing Alignment")
            
            # Demo phoneme-based alignment
            phoneme_alignment_request = PhonemeAlignmentRequest(
                phonemes=all_phonemes,
                language="english"
            )
            
            # This would work with a real audio file
            print(f"   🎵 Would align audio with {len(all_phonemes)} phonemes")
            print("   📍 Expected output:")
            print("      • Precise start/end times for each phoneme")
            print("      • Word boundaries with timing")
            print("      • Confidence scores for alignment quality")
            print("      • Speech rate and phoneme rate analysis")
            
            # Demo text-based alignment request
            print("\n   4c. Alternative: Direct Text → Audio Alignment")
            alignment_request = AlignmentRequest(
                text=test_text,
                language="english",
                include_phonemes=True,
                include_confidence=True,
                preprocess_audio=True
            )
            
            print(f"   📋 Alignment request configured:")
            print(f"      • Text: '{alignment_request.text}'")
            print(f"      • Language: {alignment_request.language}")
            print(f"      • Include phonemes: {alignment_request.include_phonemes}")
            print(f"      • Include confidence: {alignment_request.include_confidence}")
            print(f"      • Preprocess audio: {alignment_request.preprocess_audio}")
            
        else:
            print(f"   ❌ Phonemization failed: {phonemization_result.error_message}")
        
        print("\n5. Demo: Audio Processing Features...")
        
        # Demo audio processor capabilities
        print("   🎧 Audio Processing Pipeline:")
        print("      • Format validation (WAV, MP3, FLAC support)")
        print("      • Sample rate conversion (target: 16kHz)")
        print("      • Mono conversion for optimal alignment")
        print("      • Volume normalization")
        print("      • Silence trimming")
        print("      • Noise reduction (high-pass filter)")
        print("      • Preemphasis filtering")
        print("      • Quality assessment scoring")
        
        print("\n6. Demo: Expected Alignment Output Structure...")
        
        print("   📊 Alignment Result Structure:")
        print("   {")
        print("     'success': true,")
        print("     'sentence_alignment': {")
        print("       'text': 'The quick brown fox...',")
        print("       'total_duration': 2.45,")
        print("       'words': [")
        print("         {")
        print("           'word': 'The',")
        print("           'start_time': 0.0,")
        print("           'end_time': 0.12,")
        print("           'confidence': 0.95,")
        print("           'phonemes': [")
        print("             {")
        print("               'phoneme': 'DH',")
        print("               'start_time': 0.0,")
        print("               'end_time': 0.03,")
        print("               'confidence': 0.92")
        print("             },")
        print("             {")
        print("               'phoneme': 'AH0',")
        print("               'start_time': 0.03,")
        print("               'end_time': 0.12,")
        print("               'confidence': 0.88")
        print("             }")
        print("           ]")
        print("         }")
        print("       ],")
        print("       'quality': 'excellent',")
        print("       'overall_confidence': 0.91,")
        print("       'speech_rate': 145.2,  // words per minute")
        print("       'phoneme_rate': 12.8   // phonemes per second")
        print("     },")
        print("     'statistics': {")
        print("       'total_words': 9,")
        print("       'total_phonemes': 32,")
        print("       'timing_accuracy': 0.94,")
        print("       'speech_tempo': 'normal'")
        print("     }")
        print("   }")
        
        print("\n7. Demo: API Endpoints Available...")
        
        print("   🌐 Step 2 API Endpoints:")
        print("      • POST /api/v1/alignment/align")
        print("        - Upload audio + text → get timing alignment")
        print("      • POST /api/v1/alignment/batch-align")
        print("        - Process multiple audio-text pairs")
        print("      • POST /api/v1/alignment/align-phonemes")
        print("        - Upload audio + phoneme sequence → get timing")
        print("      • POST /api/v1/alignment/validate-audio")
        print("        - Validate audio file for alignment suitability")
        print("      • GET  /api/v1/alignment/languages")
        print("        - List supported languages")
        print("      • GET  /api/v1/alignment/models/{language}")
        print("        - List available models for language")
        print("      • POST /api/v1/alignment/export/{format}")
        print("        - Export results to TextGrid/JSON")
        
        print("\n🎉 Step 2: Forced Alignment Demo Completed!")
        
        print("\n💡 Integration with Step 1:")
        print("   ✅ Step 1 provides phonemes → Step 2 adds precise timing")
        print("   ✅ Combined pipeline: Text → Phonemes → Audio Alignment")
        print("   ✅ Foundation ready for Step 3: Enhanced ASR")
        
    except ImportError as e:
        print(f"\n❌ Import Error: {e}")
        print("📝 Note: This demo requires MFA and audio libraries:")
        print("   pip install montreal-forced-aligner librosa soundfile")
        print("   # Also install MFA: conda install -c conda-forge montreal-forced-aligner")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


async def demo_step1_step2_integration():
    """
    Demo integration between Step 1 (Phonemization) and Step 2 (Alignment)
    """
    print("\n" + "=" * 70)
    print("🔗 Step 1 + Step 2 Integration Demo")
    print("=" * 70)
    
    try:
        print("\n🎯 Integration Workflow:")
        print("   1. User provides: Audio file + Text")
        print("   2. Step 1: Text → Phonemes (ARPAbet)")
        print("   3. Step 2: Audio + Phonemes → Precise Timing")
        print("   4. Result: Complete phoneme-level alignment with timestamps")
        
        # Demo data
        sample_text = "Hello world"
        print(f"\n📝 Sample text: '{sample_text}'")
        
        # Step 1 simulation
        print("\n   Step 1 Output (Phonemization):")
        step1_output = {
            "words": [
                {
                    "word": "Hello",
                    "phonemes": ["HH", "AH0", "L", "OW1"],
                    "ipa": "həˈloʊ"
                },
                {
                    "word": "world", 
                    "phonemes": ["W", "ER1", "L", "D"],
                    "ipa": "wɝld"
                }
            ],
            "total_phonemes": 8
        }
        
        for word in step1_output["words"]:
            print(f"      '{word['word']}' → {word['phonemes']} → {word['ipa']}")
        
        # Step 2 simulation
        print("\n   Step 2 Output (Forced Alignment):")
        print("      Phoneme timing alignment:")
        
        simulated_timing = [
            ("HH", 0.00, 0.05), ("AH0", 0.05, 0.12), ("L", 0.12, 0.18), ("OW1", 0.18, 0.35),
            ("W", 0.45, 0.52), ("ER1", 0.52, 0.68), ("L", 0.68, 0.75), ("D", 0.75, 0.85)
        ]
        
        for phoneme, start, end in simulated_timing:
            duration = end - start
            print(f"         {phoneme:4} | {start:.2f}s - {end:.2f}s | {duration*1000:.0f}ms")
        
        print(f"\n   📊 Complete Results:")
        print(f"      • Total duration: 0.85s")
        print(f"      • Speech rate: ~84 words/minute")
        print(f"      • Phoneme rate: ~9.4 phonemes/second")
        print(f"      • Average phoneme duration: ~106ms")
        
        print("\n🎯 Ready for Step 3: Enhanced Whisper ASR!")
        print("   • Step 1+2 provide reference timing")
        print("   • Step 3 will compare actual pronunciation vs reference")
        print("   • Step 4 will score pronunciation accuracy")
        
    except Exception as e:
        print(f"❌ Integration demo failed: {e}")


if __name__ == "__main__":
    print("🚀 Starting Step 2: Forced Alignment Demo...")
    
    # Run demos
    asyncio.run(demo_step2_forced_alignment())
    asyncio.run(demo_step1_step2_integration())
    
    print("\n🏁 All demos completed!")
    print("\n💻 Next steps:")
    print("   1. Install MFA: conda install -c conda-forge montreal-forced-aligner")
    print("   2. Install audio libraries: pip install librosa soundfile")  
    print("   3. Start server: python src/main.py")
    print("   4. Test endpoints: http://localhost:3005/docs")
    print("   5. Proceed to Step 3: Enhanced Whisper ASR implementation")