#!/bin/bash

# VocabuRex Speech Service Docker Startup Script

echo "🚀 Starting VocabuRex Speech Service with Docker..."

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p uploads models logs warmup

# Check if GPU is available
if command -v nvidia-smi &> /dev/null; then
    echo "🎮 NVIDIA GPU detected - using GPU-accelerated containers"
    GPU_COMPOSE="--profile gpu"
else
    echo "💻 No GPU detected - using CPU-only containers"
    GPU_COMPOSE=""
fi

# Build and start containers
echo "🔨 Building Docker containers..."
docker compose build

echo "🚀 Starting containers..."
docker compose up -d $GPU_COMPOSE

# Wait for service to be ready
echo "⏳ Waiting for service to be ready..."
sleep 10

# Check health
echo "🔍 Checking service health..."
curl -f http://localhost:3005/speech/health || {
    echo "❌ Service health check failed"
    echo "📋 Container logs:"
    docker compose logs vocabu-rex-speech-service --tail=50
    exit 1
}

echo "✅ VocabuRex Speech Service is running!"
echo "📡 Service available at: http://localhost:3005"
echo "📖 API Documentation: http://localhost:3005/docs"
echo "🏥 Health check: http://localhost:3005/speech/health"
echo ""
echo "🎯 Internal API Endpoints:"
echo "  POST /speech/transcribe"
echo "  POST /speech/transcribe-only"
echo "  GET  /speech/health"
echo ""
echo "🔧 Management commands:"
echo "  docker compose logs vocabu-rex-speech-service  # View logs"
echo "  docker compose stop                             # Stop service"
echo "  docker compose restart vocabu-rex-speech-service # Restart"
echo "  docker compose down                             # Stop and remove"