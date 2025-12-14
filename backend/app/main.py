from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
from app.config import get_settings
from app.api import upload, extract, monitoring
from app.api import analysis, orchestration

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)
settings = get_settings()

# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Intelligent document extraction API with OCR and parsing capabilities"
)

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
    logger.info("API Ready")

# Include routers
app.include_router(upload.router)
app.include_router(extract.router)
app.include_router(analysis.router)
app.include_router(orchestration.router)
app.include_router(monitoring.router)  

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "approaches": {
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
        reload=settings.debug
    )