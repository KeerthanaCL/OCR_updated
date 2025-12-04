# Document Extraction API - OCR Project

An intelligent document extraction API with OCR (Optical Character Recognition) and AI-powered parsing capabilities. This project leverages multiple OCR engines and AI providers to extract and process text from documents and images.

## 🎯 Features

- **Multiple OCR Engines**: Tesseract, TrOCR (Transformer-based OCR), EasyOCR
- **AI-Powered Extraction**: Integration with OpenAI, Google Gemini, Claude, Groq, and Ollama
- **Document Processing**: Support for PDFs and images
- **Intelligent Parsing**: Document segmentation and structured data extraction
- **Async Operations**: Built on FastAPI with async/await support
- **Database Support**: SQLAlchemy with SQLite (extensible to other databases)
- **Task Queue**: Celery integration with Redis for background processing
- **Handwritten & Printed Text**: Support for both handwritten and printed text recognition

## 🛠️ Tech Stack

### Backend
- **Framework**: FastAPI
- **Server**: Uvicorn
- **OCR**: Tesseract, TrOCR, EasyOCR
- **AI Models**: Transformers, PyTorch
- **Computer Vision**: OpenCV, torchvision
- **PDF Processing**: PDF2Image, PyMuPDF
- **Database**: SQLAlchemy with SQLite
- **Task Queue**: Celery with Redis
- **Async**: aiofiles

### Frontend
- (Coming soon)

## 📋 Prerequisites

- Python 3.8 or higher
- Tesseract OCR installed on your system
- Redis (for Celery task queue)
- Virtual environment manager (venv or conda)

### System-specific Installation

#### Windows
```bash
# Install Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki
# Download the installer and run it
```

#### macOS
```bash
brew install tesseract
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt-get install tesseract-ocr
```

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/KeerthanaCL/OCR_updated.git
cd OCR_updated
```

### 2. Create Virtual Environment
```bash
# Using venv
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 4. Environment Configuration

Create a `.env` file in the `backend` directory with the following configuration:

```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Google Gemini Configuration
GEMINI_API_KEY=your_gemini_api_key_here

# Application Settings
DEBUG=True
APP_NAME="Document Extraction API"
APP_VERSION="1.0.0"

# Tesseract Configuration (Windows path example)
# TESSERACT_PATH=C:\Program Files\Tesseract-OCR\tesseract.exe
# For Linux/macOS:
# TESSERACT_PATH=/usr/bin/tesseract

# Database
DATABASE_URL=sqlite:///./documents.db

# Redis (for Celery)
REDIS_URL=redis://localhost:6379/0

# AI Provider Selection
# Options: openai, gemini, claude, groq, ollama
AI_PROVIDER=gemini

# Optional: Data Science API Integration
DATA_SCIENCE_API_URL=
DATA_SCIENCE_API_KEY=
```

### 5. Install Additional System Dependencies

#### Tesseract Language Data (Optional)
```bash
# Install additional languages for Tesseract
# On Linux:
sudo apt-get install tesseract-ocr-[language-code]

# Example for Spanish:
sudo apt-get install tesseract-ocr-spa
```

### 6. Redis Setup (for Celery)

You can run Redis using Docker:
```bash
docker run -d -p 6379:6379 redis:latest
```

Or install locally:
- **Windows**: Use Windows Subsystem for Linux (WSL) or download from https://github.com/microsoftarchive/redis/releases
- **macOS**: `brew install redis`
- **Linux**: `sudo apt-get install redis-server`

## 📁 Project Structure

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application entry point
│   ├── config.py            # Configuration settings
│   ├── database.py          # Database setup and utilities
│   ├── models.py            # SQLAlchemy models
│   ├── api/
│   │   ├── upload.py        # File upload endpoints
│   │   ├── extract.py       # OCR extraction endpoints
│   │   └── segment_extraction.py  # Document segmentation
│   ├── agents/              # AI agents for processing
│   ├── services/            # Business logic services
│   └── utils/               # Utility functions
├── uploads/                 # Uploaded files directory
├── requirements.txt         # Python dependencies
└── documents.db            # SQLite database (auto-created)

frontend/
└── (Coming soon)
```

## 🎮 Running the Application

### Start the FastAPI Server

```bash
cd backend

# With auto-reload (development)
python -m uvicorn app.main:app --reload

# Production mode
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The API will be available at: `http://localhost:8000`

### API Documentation

Once the server is running, you can access:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

### Health Check

```bash
curl http://localhost:8000/health
```

## 📡 API Endpoints

### Upload Documents
- **POST** `/upload` - Upload a document for processing

### Extract Text
- **POST** `/extract` - Extract text from uploaded document
- **GET** `/extract/{document_id}` - Retrieve extraction results

### Segment Extraction
- **POST** `/segment` - Extract specific segments from documents

### Root & Health
- **GET** `/` - Root endpoint with API info
- **GET** `/health` - Health check endpoint

## 🔄 Running Celery Workers (Optional)

For background task processing:

```bash
# In a separate terminal
celery -A app.celery worker --loglevel=info
```

## 🧪 Testing

```bash
# Run tests (if available)
pytest

# With coverage
pytest --cov=app
```

## 🔧 Configuration Details

### OCR Settings
- **Tesseract Confidence Threshold**: 60
- **Tesseract Language**: English (eng) - modify in config
- **TrOCR Model**: 
  - Printed text: `microsoft/trocr-base-printed`
  - Handwritten: `microsoft/trocr-base-handwritten`

### AI Provider Settings
Configure in `config.py`:
- **OpenAI**: GPT-4O with structured output
- **Google Gemini**: Gemini 2.0 Flash
- **Default**: Gemini (can be changed via `AI_PROVIDER` env var)

### File Upload Limits
- **Max File Size**: 10 MB (configurable)
- **Upload Directory**: `./uploads`

## 🐛 Troubleshooting

### Tesseract Not Found
- Verify Tesseract installation
- Update `TESSERACT_PATH` in `.env` file
- Ensure Tesseract is in your system PATH

### Redis Connection Error
- Verify Redis is running
- Check `REDIS_URL` configuration
- Ensure port 6379 is accessible

### CORS Issues
- API allows all origins by default (`allow_origins=["*"]`)
- Modify in `main.py` if needed for production

### Out of Memory with Large Models
- Reduce batch size
- Use smaller models (TrOCR instead of EasyOCR)
- Enable GPU acceleration if available

## 📚 Dependencies Overview

| Package | Purpose |
|---------|---------|
| fastapi | Web framework |
| uvicorn | ASGI server |
| pytesseract | Tesseract wrapper |
| transformers | Hugging Face models |
| torch | Deep learning framework |
| easyocr | Easy OCR library |
| sqlalchemy | ORM |
| celery | Task queue |
| redis | Caching/message broker |
| python-dotenv | Environment variables |

## 🌟 Future Enhancements

- [ ] Frontend application
- [ ] Advanced document classification
- [ ] Multi-language support improvements
- [ ] Batch processing API
- [ ] Document versioning
- [ ] Audit logging
- [ ] Rate limiting
- [ ] API authentication

## 📝 Environment Variables Summary

```
OPENAI_API_KEY          - OpenAI API key
GEMINI_API_KEY          - Google Gemini API key
DEBUG                   - Enable debug mode
TESSERACT_PATH          - Path to Tesseract executable
DATABASE_URL            - Database connection URL
REDIS_URL               - Redis connection URL
AI_PROVIDER             - AI provider to use (openai/gemini/claude/groq/ollama)
```

## 🤝 Contributing

1. Create a new branch: `git checkout -b feature/your-feature`
2. Commit changes: `git commit -am 'Add your feature'`
3. Push to branch: `git push origin feature/your-feature`
4. Submit a pull request

## 📄 License

This project is part of CirrusLabs AI Detection initiative.

## 📞 Support

For issues and questions, please create an issue on the repository or contact the development team.
.
---
