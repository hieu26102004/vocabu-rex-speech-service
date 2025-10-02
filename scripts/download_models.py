#!/usr/bin/env python3
"""
Script to download and setup local Whisper models
"""

import os
import sys
import argparse
from pathlib import Path
import whisper
from faster_whisper import WhisperModel

# Available Whisper model sizes
AVAILABLE_MODELS = [
    "tiny",      # ~39 MB, English-only
    "tiny.en",   # ~39 MB, English-only
    "base",      # ~74 MB, multilingual  
    "base.en",   # ~74 MB, English-only
    "small",     # ~244 MB, multilingual
    "small.en",  # ~244 MB, English-only
    "medium",    # ~769 MB, multilingual
    "medium.en", # ~769 MB, English-only
    "large",     # ~1550 MB, multilingual
    "large-v2",  # ~1550 MB, multilingual
    "large-v3",  # ~1550 MB, multilingual
]

def setup_model_directory():
    """Create models directory if it doesn't exist"""
    models_dir = Path("./models/whisper")
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir

def download_openai_whisper_model(model_name: str):
    """Download OpenAI Whisper model"""
    print(f"📥 Downloading OpenAI Whisper model: {model_name}")
    try:
        model = whisper.load_model(model_name)
        print(f"✅ OpenAI Whisper model '{model_name}' downloaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Failed to download OpenAI Whisper model '{model_name}': {e}")
        return False

def download_faster_whisper_model(model_name: str, models_dir: Path):
    """Download Faster-Whisper model"""
    print(f"📥 Downloading Faster-Whisper model: {model_name}")
    try:
        # This will download and cache the model
        model = WhisperModel(
            model_name, 
            device="cpu", 
            compute_type="int8",
            download_root=str(models_dir)
        )
        # Test the model by getting info
        info = model.get_model_info()
        print(f"✅ Faster-Whisper model '{model_name}' downloaded successfully!")
        print(f"   Model info: {info}")
        return True
    except Exception as e:
        print(f"❌ Failed to download Faster-Whisper model '{model_name}': {e}")
        return False

def list_available_models():
    """List all available Whisper models"""
    print("📋 Available Whisper models:")
    print("\n🔸 Recommended for development:")
    print("   • tiny    - Fastest, least accurate (~39 MB)")
    print("   • base    - Good balance of speed/accuracy (~74 MB)")
    print("   • small   - Better accuracy, slower (~244 MB)")
    
    print("\n🔸 Production models:")
    print("   • medium  - High accuracy (~769 MB)")
    print("   • large   - Best accuracy, slowest (~1550 MB)")
    print("   • large-v2 - Improved large model (~1550 MB)")
    print("   • large-v3 - Latest large model (~1550 MB)")
    
    print("\n🔸 English-only variants (faster for English):")
    for model in AVAILABLE_MODELS:
        if model.endswith('.en'):
            print(f"   • {model}")

def main():
    parser = argparse.ArgumentParser(description="Download Whisper models for local usage")
    parser.add_argument(
        "models", 
        nargs="*", 
        help="Model names to download (e.g., base small medium)"
    )
    parser.add_argument(
        "--list", 
        action="store_true", 
        help="List available models"
    )
    parser.add_argument(
        "--engine",
        choices=["openai", "faster", "both"],
        default="both",
        help="Which Whisper engine to download for (default: both)"
    )
    parser.add_argument(
        "--recommended",
        action="store_true",
        help="Download recommended models (tiny, base, small)"
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_available_models()
        return
    
    # Setup models directory
    models_dir = setup_model_directory()
    print(f"📁 Models directory: {models_dir.absolute()}")
    
    # Determine which models to download
    models_to_download = args.models
    if args.recommended:
        models_to_download = ["tiny", "base", "small"]
    elif not models_to_download:
        models_to_download = ["base"]  # Default to base model
    
    # Validate model names
    invalid_models = [m for m in models_to_download if m not in AVAILABLE_MODELS]
    if invalid_models:
        print(f"❌ Invalid model names: {invalid_models}")
        print("Use --list to see available models")
        sys.exit(1)
    
    print(f"🎯 Downloading models: {models_to_download}")
    print(f"🔧 Using engine(s): {args.engine}")
    
    success_count = 0
    total_count = len(models_to_download)
    
    for model_name in models_to_download:
        print(f"\n{'='*50}")
        print(f"Processing model: {model_name}")
        print('='*50)
        
        if args.engine in ["openai", "both"]:
            if download_openai_whisper_model(model_name):
                success_count += 1
        
        if args.engine in ["faster", "both"]:
            if download_faster_whisper_model(model_name, models_dir):
                success_count += 1
    
    print(f"\n{'='*50}")
    print(f"📊 Download Summary")
    print('='*50)
    
    if args.engine == "both":
        total_count *= 2  # Both engines
    
    print(f"✅ Successful downloads: {success_count}/{total_count}")
    
    if success_count == total_count:
        print("🎉 All models downloaded successfully!")
    else:
        print("⚠️  Some downloads failed. Check the error messages above.")
        
    print(f"\n💡 To use the downloaded models, set in your .env file:")
    print(f"   WHISPER_MODEL={models_to_download[0]}")
    print(f"   WHISPER_MODEL_PATH=./models/whisper")

if __name__ == "__main__":
    main()