from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from sqlalchemy.orm import Session
from app.models import ExtractionRequest, ExtractionResult, JobStatus, OCRMethod
from app.agents.orchestrator import AgentOrchestrator
from app.services.storage import StorageService
from app.database import get_db, Extraction
import uuid
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["extraction"])

# Global orchestrator instance
orchestrator = AgentOrchestrator()

@router.post("/extract", response_model=ExtractionResult)
async def extract_text(
    request: ExtractionRequest,
    db: Session = Depends(get_db)
):
    """
    Extract text from uploaded document.
    
    Uses Tesseract OCR first, falls back to TrOCR if confidence is low.
    """
    try:
        # Get file path
        storage = StorageService()
        file_path = storage.get_file_path(request.document_id)
        
        # Create job
        job_id = orchestrator.create_job(request.document_id)
        
        # Execute extraction
        results = orchestrator.execute_full_pipeline(
            job_id=job_id,
            file_path=file_path,
            use_preprocessing=request.use_preprocessing,
            force_trocr=request.force_trocr,
            apply_parsing=False  # Only extraction
        )
        
        extraction_data = results['extraction']

        # FIX: Handle both string and OCRMethod enum
        method_used = extraction_data['method_used']
        if isinstance(method_used, str):
            # Convert string to enum
            method_used_str = method_used
            try:
                method_used_enum = OCRMethod(method_used)
            except ValueError:
                # If not a valid enum value, use TESSERACT as default
                method_used_enum = OCRMethod.TESSERACT
        else:
            method_used_enum = method_used
            method_used_str = method_used.value
        
        # Save to database
        extraction_id = str(uuid.uuid4())
        db_extraction = Extraction(
            id=extraction_id,
            document_id=request.document_id,
            text=extraction_data['text'],
            confidence=extraction_data['confidence'],
            method_used=method_used_str,
            pages=extraction_data['pages'],
            processing_time=extraction_data['processing_time'],
            extraction_metadata=extraction_data['metadata']
        )
        db.add(db_extraction)
        db.commit()
        
        logger.info(f"Extraction completed: {extraction_id}")
        
        return ExtractionResult(
            document_id=request.document_id,
            text=extraction_data['text'],
            confidence=extraction_data['confidence'],
            method_used=method_used_enum,
            pages=extraction_data['pages'],
            processing_time=extraction_data['processing_time'],
            metadata=extraction_data['metadata']
        )
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Document not found")
    except Exception as e:
        logger.error(f"Extraction failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/extract/status/{job_id}", response_model=JobStatus)
async def get_extraction_status(job_id: str):
    """Get status of an extraction job"""
    status = orchestrator.get_job_status(job_id)
    
    if not status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return status