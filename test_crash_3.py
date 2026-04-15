import torch
import librosa
import asyncio

async def test_conflict():
    print("Loading inside async...")
    from faster_whisper import WhisperModel
    model = WhisperModel('small', device='cuda', compute_type='float16')
    print("Loaded inside async!")

if __name__ == "__main__":
    asyncio.run(test_conflict())
