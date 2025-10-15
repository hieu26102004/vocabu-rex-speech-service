# VocabuRex Speech Service with GPU Support
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_VISIBLE_DEVICES=0

# Install system dependencies including GPU support
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libffi-dev \
    libssl-dev \
    libasound2-dev \
    portaudio19-dev \
    ffmpeg \
    libsndfile1 \
    curl \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python -m pip install --upgrade pip

# Set work directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt ./

# Install PyTorch with CUDA support first (will use host GPU) - fix torchaudio compatibility
RUN pip install torch==2.3.1+cu121 torchvision==0.18.1+cu121 torchaudio==2.3.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
RUN pip install -r requirements.txt

# Download Whisper models (tiny, base, small, medium, large)
RUN python scripts/download_whisper_models.py

# Copy source code
COPY . .

# Create necessary directories
RUN mkdir -p uploads warmup models logs

# Create non-root user
RUN useradd --create-home --shell /bin/bash --uid 1000 speech_user \
    && chown -R speech_user:speech_user /app

# Switch to non-root user
USER speech_user

# Expose port
EXPOSE 3005

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:3005/speech/health || exit 1

# Run the application
CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "3005"]