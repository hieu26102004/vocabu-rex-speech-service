# 🎤 VocabuRex Speech Service

> **Speech Processing Service cho VocabuRex** - Xử lý Speech-to-Text, đánh giá phát âm chi tiết với Forced Alignment

## 📋 Tổng quan

Speech Service là một microservice trong hệ sinh thái VocabuRex, chuyên xử lý các tác vụ liên quan đến giọng nói với **quy trình đánh giá phát âm chi tiết 4 bước**:

### 🎯 **Quy trình Đánh giá Phát âm (Pronunciation Assessment)**

| Bước | Quá trình | Công cụ | Mục đích |
|------|-----------|---------|----------|
| 1 | **Phonemization** | Phonemizer, G2P | Chuyển văn bản chuẩn ("hello") → âm vị IPA (/hɛloʊ/) |
| 2 | **Forced Alignment** | Montreal Forced Aligner (MFA) | Căn chỉnh cưỡng bức - xác định thời gian chính xác từng phoneme |
| 3 | **ASR Transcription** | Local Whisper | Chuyển giọng nói người dùng thành văn bản thực tế |
| 4 | **Error Analysis** | Custom Logic | So sánh & phát hiện lỗi chi tiết (substitution, omission, insertion) |

### 📊 **Kết quả chi tiết**
```json
{
  "word": "hello",
  "score": 75,
  "errors": [
    {
      "phoneme": "l", 
      "expected": "l",
      "actual": "r", 
      "error_type": "substitution",
      "severity": "major",
      "timestamp_start": 0.3,
      "timestamp_end": 0.4
    }
  ]
}
```

### 🔧 **Các tính năng khác**
- **Speech-to-Text (STT)**: Chuyển đổi giọng nói thành văn bản
- **Voice Chatbot**: Hội thoại bằng giọng nói với AI
- **Audio Processing**: Xử lý và phân tích file âm thanh
- **Voice Similarity**: So sánh độ tương tự giọng nói
- **Voice Chatbot**: Hội thoại bằng giọng nói với AI
- **Audio Processing**: Xử lý và phân tích file âm thanh
- **Voice Similarity**: So sánh độ tương tự giọng nói

## 🏗️ Kiến trúc

### Clean Architecture
```
src/
├── presentation/       # Controllers, Routers, Middleware
├── application/        # Use cases, DTOs, Business Logic  
├── domain/            # Entities, Repositories, Services
├── infrastructure/    # External services, Database, Redis
└── shared/           # Config, Exceptions, Utilities
```

### Tech Stack
- **Framework**: FastAPI + Python 3.11
- **Database**: PostgreSQL 15
- **Cache**: Redis 7
- **Container**: Docker + Docker Compose
- **Speech Engine**: Local Whisper (OpenAI Whisper + Faster-Whisper)
- **ML Framework**: PyTorch
- **Optional**: Google Cloud Speech, Azure Speech

## 🚀 Cài đặt và Chạy

### Prerequisites
```bash
# Cần cài đặt
- Docker & Docker Compose
- Python 3.11+ (nếu chạy local)
- Poetry (nếu chạy local)
```

### 1. Clone và Setup
```bash
# Clone repository
git clone <repo-url>
cd vocabu-rex-speech-service

# Copy environment file
cp .env.example .env

# Chỉnh sửa .env với thông tin của bạn
```

### 2. Chạy với Docker (Khuyến nghị)
```bash
# Chạy tất cả services
docker-compose up -d

# Chỉ chạy database và redis
docker-compose up -d speech-db redis-speech

# Xem logs
docker-compose logs -f speech-service

# Chạy với tools (pgAdmin, Redis Commander)
docker-compose --profile tools up -d
```

### 3. Chạy Local Development
```bash
# Cài đặt dependencies
poetry install

# Download Whisper models
python scripts/download_models.py --recommended
# hoặc chỉ model cơ bản
python scripts/download_models.py base

# Chạy database
docker-compose up -d speech-db redis-speech  

# Activate virtual environment
poetry shell

# Chạy development server
python src/main.py
# hoặc
uvicorn src.main:app --host 0.0.0.0 --port 3005 --reload
```

## 📡 API Endpoints

### Health Check
```http
GET /health
```

### Speech Recognition
```http
POST /api/v1/speech/recognize
Content-Type: multipart/form-data

{
  "audio_file": "<audio-file>",
  "language": "en-US",
  "engine": "whisper"
}
```

### Pronunciation Assessment
```http
POST /api/v1/speech/pronunciation/assess
Content-Type: multipart/form-data

{
  "audio_file": "<audio-file>",
  "target_text": "Hello world",
  "language": "en-US"
}
```

### Voice Conversation
```http
POST /api/v1/speech/conversation/start
WebSocket: /api/v1/speech/conversation/ws/{session_id}
```

## 🗄️ Database Schema

### Các bảng chính:
- `audio_sessions` - Theo dõi phiên xử lý âm thanh
- `speech_recognitions` - Kết quả nhận dạng giọng nói  
- `pronunciation_assessments` - Đánh giá phát âm
- `voice_profiles` - Profile giọng nói người dùng
- `conversation_sessions` - Phiên hội thoại
- `audio_files` - Metadata file âm thanh

## 🔧 Configuration

### Environment Variables
```bash
# Database
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5435/vocabu_speech_db

# Redis  
REDIS_URL=redis://localhost:6382

# Speech Services
OPENAI_API_KEY=your-openai-key
GOOGLE_CLOUD_CREDENTIALS_PATH=./credentials/google-speech.json
AZURE_SPEECH_KEY=your-azure-key
AZURE_SPEECH_REGION=your-region

# File Upload
MAX_FILE_SIZE_MB=50
ALLOWED_AUDIO_FORMATS=wav,mp3,m4a,flac,ogg
```

## � Whisper Models

### Model Sizes & Performance
| Model | Size | Speed | Accuracy | Use Case |
|-------|------|-------|----------|----------|
| `tiny` | ~39 MB | Fastest | Basic | Development, Testing |
| `base` | ~74 MB | Fast | Good | Development, Light Production |
| `small` | ~244 MB | Medium | Better | Production |
| `medium` | ~769 MB | Slow | High | High Accuracy Needed |
| `large-v3` | ~1550 MB | Slowest | Best | Maximum Accuracy |

### Downloading Models
```bash
# Download recommended models (tiny, base, small)
python scripts/download_models.py --recommended

# Download specific models
python scripts/download_models.py base medium large-v3

# List available models
python scripts/download_models.py --list

# Download for specific engine
python scripts/download_models.py --engine faster base
```

### GPU Support
```bash
# Để sử dụng GPU (nếu có CUDA):
WHISPER_DEVICE=cuda
WHISPER_COMPUTE_TYPE=float16

# Tự động detect device:
WHISPER_DEVICE=auto
```

## �🧪 Testing

```bash
# Chạy tests
poetry run pytest

# Test với coverage
poetry run pytest --cov=src

# Test specific file
poetry run pytest tests/test_speech_recognition.py
```

## 📊 Monitoring

### Health Endpoints
- **Service Health**: `GET /health`
- **Database Health**: `GET /health/db` 
- **Redis Health**: `GET /health/redis`

### Management Tools
- **pgAdmin**: http://localhost:5051 (admin@vocaburex.com/admin123)
- **Redis Commander**: http://localhost:8082

## 🔄 Development Workflow

### 1. Code Style
```bash
# Format code
poetry run black src/ tests/
poetry run isort src/ tests/

# Lint
poetry run flake8 src/ tests/
poetry run mypy src/
```

### 2. Database Migration
```bash
# Tạo migration mới
alembic revision --autogenerate -m "Description"

# Apply migration
alembic upgrade head

# Rollback
alembic downgrade -1
```

### 3. Docker Development
```bash
# Build image
docker-compose build speech-service

# Reset database
docker-compose down -v
docker-compose up -d speech-db
```

## 📚 API Documentation

- **Swagger UI**: http://localhost:3005/docs
- **ReDoc**: http://localhost:3005/redoc

## 🤝 Integration với Services khác

### VocabuRex Gateway
- Port: `3005`
- Health check: `/health`
- Rate limiting: 100 requests/minute

### Auth Service Integration
```python
# Headers cần thiết
Authorization: Bearer <jwt-token>
X-User-ID: <user-uuid>
```

## 🚧 Roadmap

- [ ] Implement WebSocket real-time speech recognition
- [ ] Add voice cloning capabilities  
- [ ] Integrate with AI service for conversation
- [ ] Add speech synthesis (Text-to-Speech)
- [ ] Performance optimization for large audio files
- [ ] Multi-language support expansion

## 📝 License

Private - VocabuRex Team Only