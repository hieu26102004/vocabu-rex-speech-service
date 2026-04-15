import asyncio
from faster_whisper import WhisperModel
async def main():
    print("Loading...")
    model = WhisperModel('small', device='cuda', compute_type='float16')
    print('Loaded!')

if __name__ == '__main__':
    asyncio.run(main())
