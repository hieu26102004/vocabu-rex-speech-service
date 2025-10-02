"""
Demo script for Step 1: Phonemization
Test the phonemization pipeline with sample text
"""

import asyncio
import json
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.infrastructure.services.phonemization_service import PhonemizationService
from src.application.usecases.phonemization_usecases import PhonemizationUseCases
from src.application.dtos.phonemization_dto import (
    PhonemizationRequest,
    WordPhonemizationRequest,
    IpaConversionRequest
)


async def demo_text_phonemization():
    """Demo text phonemization"""
    print("=" * 60)
    print("🎤 VocabuRex Speech Service - Step 1: Phonemization Demo")
    print("=" * 60)
    
    try:
        # Initialize service
        print("\n1. Initializing Phonemization Service...")
        phonemization_service = PhonemizationService()
        use_cases = PhonemizationUseCases(phonemization_service)
        print("✅ Service initialized successfully")
        
        # Test 1: Basic text phonemization
        print("\n2. Testing Text Phonemization...")
        request = PhonemizationRequest(
            text="Hello world, how are you?",
            language="en-us",
            include_stress=True,
            include_syllables=True
        )
        
        print(f"Input text: '{request.text}'")
        result = await use_cases.phonemize_text(request)
        
        print(f"✅ Phonemization completed in {result.processing_time_ms}ms")
        print(f"   - Total words: {result.total_words}")
        print(f"   - Total phonemes: {result.total_phonemes}")
        print(f"   - Backend: {result.backend}")
        
        print("\n📊 Word-by-word breakdown:")
        for word_result in result.words:
            print(f"   • '{word_result.word}' -> {word_result.phonemes}")
            print(f"     IPA: {word_result.ipa}")
            if word_result.stress_pattern:
                print(f"     Stress: {word_result.stress_pattern}")
            if word_result.syllable_count:
                print(f"     Syllables: {word_result.syllable_count}")
            print(f"     Confidence: {word_result.confidence:.2f}")
            print()
        
        # Test 2: Word list phonemization
        print("\n3. Testing Word List Phonemization...")
        word_request = WordPhonemizationRequest(
            words=["pronunciation", "assessment", "phonemes", "linguistics"],
            language="en-us"
        )
        
        word_results = await use_cases.phonemize_words(word_request)
        print(f"✅ {len(word_results)} words phonemized")
        
        for word in word_results:
            print(f"   • '{word.word}' -> {' '.join(word.phonemes)} ({word.ipa})")
        
        # Test 3: IPA representation
        print("\n4. Testing IPA Conversion...")
        ipa_request = IpaConversionRequest(
            text="The quick brown fox jumps over the lazy dog",
            language="en-us"
        )
        
        ipa_result = await use_cases.get_ipa_representation(ipa_request)
        print(f"Text: '{ipa_result.original_text}'")
        print(f"IPA:  {ipa_result.ipa_representation}")
        print(f"✅ IPA conversion completed in {ipa_result.processing_time_ms}ms")
        
        # Test 4: Supported languages
        print("\n5. Checking Supported Languages...")
        languages = await use_cases.get_supported_languages()
        print(f"✅ {languages.total_languages} languages supported:")
        for lang in languages.supported_languages:
            print(f"   • {lang}")
        
        # Test 5: Text validation
        print("\n6. Testing Text Validation...")
        test_texts = [
            "valid english text",
            "invalid@text#with$symbols",
            "",
            "x" * 1001  # Too long
        ]
        
        for text in test_texts:
            is_valid = await use_cases.validate_text_for_phonemization(text)
            status = "✅ Valid" if is_valid else "❌ Invalid"
            preview = text[:50] + "..." if len(text) > 50 else text
            print(f"   {status}: '{preview}'")
        
        print("\n🎉 All phonemization tests completed successfully!")
        
    except ImportError as e:
        print(f"\n❌ Import Error: {e}")
        print("📝 Note: This demo requires phonemization libraries to be installed:")
        print("   pip install phonemizer g2p-en")
        print("   # Also install espeak: apt-get install espeak espeak-data")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


async def demo_json_output():
    """Demo with JSON output for API testing"""
    print("\n" + "=" * 60)
    print("📋 JSON API Response Demo")
    print("=" * 60)
    
    try:
        phonemization_service = PhonemizationService()
        use_cases = PhonemizationUseCases(phonemization_service)
        
        request = PhonemizationRequest(
            text="Hello world",
            language="en-us"
        )
        
        result = await use_cases.phonemize_text(request)
        
        # Convert to dict for JSON serialization
        json_response = {
            "success": result.success,
            "original_text": result.original_text,
            "language": result.language,
            "backend": result.backend,
            "words": [
                {
                    "word": w.word,
                    "phonemes": w.phonemes,
                    "ipa": w.ipa,
                    "stress_pattern": w.stress_pattern,
                    "syllable_count": w.syllable_count,
                    "confidence": w.confidence
                }
                for w in result.words
            ],
            "total_words": result.total_words,
            "total_phonemes": result.total_phonemes,
            "processing_time_ms": result.processing_time_ms,
            "timestamp": result.timestamp.isoformat()
        }
        
        print("📤 Sample API Response:")
        print(json.dumps(json_response, indent=2))
        
    except Exception as e:
        print(f"❌ JSON demo failed: {e}")


if __name__ == "__main__":
    print("🎯 Starting Phonemization Demo...")
    
    # Run the demo
    asyncio.run(demo_text_phonemization())
    asyncio.run(demo_json_output())
    
    print("\n🔚 Demo completed!")
    print("\n💡 Next steps:")
    print("   1. Install phonemization dependencies")
    print("   2. Start the FastAPI server: python src/main.py")
    print("   3. Test API endpoints at: http://localhost:3005/docs")
    print("   4. Proceed to Step 2: Forced Alignment implementation")