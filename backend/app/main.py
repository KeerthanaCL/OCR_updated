from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
import logging
from app.config import get_settings
from app.api.upload import router as upload_router
from app.api.extract import router as extract_router
from app.api.analysis import router as analysis_router
from app.api.orchestration import router as orchestration_router
from app.api.monitoring import router as monitoring_router
from app.api.text_upload import router as text_upload_router
from app.api.appeal import router as appeal_router
from app.api.paddleocr_test import router as paddleocr_test_router
from app.api.confidence import router as confidence_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
 
logger = logging.getLogger(__name__)
settings = get_settings()
 
class LargeFileMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Allow large uploads for specific routes
        if request.url.path.startswith("/api/v1/upload"):
            # Check Content-Length header
            content_length = request.headers.get("content-length")
            if content_length:
                content_length = int(content_length)
               
                # ADD DEBUG LOGGING
                logger.info(f"📦 Upload request: {content_length / (1024*1024):.2f}MB")
                logger.info(f"📊 Max allowed: {settings.max_file_size / (1024*1024):.2f}MB")
               
                if content_length > settings.max_file_size:
                    logger.warning(f"❌ File rejected: {content_length / (1024*1024):.2f}MB > {settings.max_file_size / (1024*1024):.2f}MB")
                    return JSONResponse(
                        status_code=413,
                        content={
                            "detail": f"File too large. Max size: {settings.max_file_size // (1024*1024)}MB"
                        }
                    )
       
        response = await call_next(request)
        return response
 
# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Intelligent document extraction API with OCR and parsing capabilities"
)
 
app.add_middleware(LargeFileMiddleware)  # File size check first
 
# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
 
# ADD THIS STARTUP EVENT HANDLER
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Starting OCR AI Detection API...")
    logger.info(f"Config max_file_size (raw): {settings.max_file_size}")
    logger.info(f"Config max_file_size (MB): {settings.max_file_size / (1024*1024):.2f}MB")
    logger.info(f"Config upload_dir: {settings.upload_dir}")
   
    logger.info(f"Max Upload Size: {settings.max_file_size // (1024*1024)}MB")
    logger.info(f"OCR Strategy: Tesseract → AWS Textract Fallback")
    logger.info(f"Textract Fallback Threshold: {settings.textract_fallback_threshold}%")
    logger.info("API Ready")
 
# Include routers
app.include_router(upload_router)
app.include_router(extract_router)
app.include_router(analysis_router)
app.include_router(orchestration_router)
app.include_router(monitoring_router)
app.include_router(paddleocr_test_router)
app.include_router(text_upload_router)
app.include_router(appeal_router)
app.include_router(confidence_router)
 
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "max_upload_size_mb": settings.max_file_size // (1024*1024),
        "ocr_strategy": {
            "primary": "Tesseract (FREE, fast)",
            "fallback": "AWS Textract (PAID, accurate)",
            "threshold": f"{settings.textract_fallback_threshold}%"
        },
        "api_approaches": {
            "approach_1": "Individual endpoints (/ai-detection, /horizon/*)",
            "approach_2": "Orchestrated processing (/process, /status)"
        }
    }
 
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}
 
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        timeout_keep_alive=300,
        limit_concurrency=10
    )
 