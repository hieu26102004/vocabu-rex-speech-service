@echo off
REM VocabuRex Speech Service Docker Startup Script for Windows

echo 🚀 Starting VocabuRex Speech Service with Docker...

REM Create necessary directories
echo 📁 Creating directories...
if not exist uploads mkdir uploads
if not exist models mkdir models
if not exist logs mkdir logs
if not exist warmup mkdir warmup

REM Check if GPU is available
nvidia-smi >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo 🎮 NVIDIA GPU detected - using GPU-accelerated containers
    set GPU_COMPOSE=--profile gpu
) else (
    echo 💻 No GPU detected - using CPU-only containers
    set GPU_COMPOSE=
)

REM Build and start containers
echo 🔨 Building Docker containers...
docker compose build

if %ERRORLEVEL% NEQ 0 (
    echo ❌ Docker build failed
    pause
    exit /b 1
)

echo 🚀 Starting containers...
docker compose up -d %GPU_COMPOSE%

if %ERRORLEVEL% NEQ 0 (
    echo ❌ Failed to start containers
    pause
    exit /b 1
)

REM Wait for service to be ready
echo ⏳ Waiting for service to be ready...
timeout /t 15 /nobreak >nul

REM Check health
echo 🔍 Checking service health...
curl -f http://localhost:3005/speech/health
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Service health check failed
    echo 📋 Container logs:
    docker compose logs vocabu-rex-speech-service --tail=50
    pause
    exit /b 1
)

echo.
echo ✅ VocabuRex Speech Service is running!
echo 📡 Service available at: http://localhost:3005
echo 📖 API Documentation: http://localhost:3005/docs
echo 🏥 Health check: http://localhost:3005/speech/health
echo.
echo 🎯 Internal API Endpoints:
echo   POST /speech/transcribe
echo   POST /speech/transcribe-only
echo   GET  /speech/health
echo.
echo 🔧 Management commands:
echo   docker compose logs vocabu-rex-speech-service  ^& View logs
echo   docker compose stop                             ^& Stop service
echo   docker compose restart vocabu-rex-speech-service ^& Restart
echo   docker compose down                             ^& Stop and remove
echo.
pause