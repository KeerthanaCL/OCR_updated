"""
Orchestration Endpoints - Approach 2
Single endpoint with background processing and polling
"""
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
import logging

from app.database import get_db, Document, Extraction
from app.models import ProcessRequest, ProcessResponse, JobStatusResponse
from app.agents.orchestrator_agent import OrchestratorAgent
from datetime import datetime

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["orchestration"])

# Initialize orchestrator
orchestrator = OrchestratorAgent()


@router.post("/process", response_model=ProcessResponse)
async def start_orchestrated_processing(
    request: ProcessRequest,
    db: Session = Depends(get_db)
):
    """
    Start complete document processing in background
    
    Supports TWO modes:
    1. Complete processing (document_id): Extraction + All analyses
    2. Analysis only (extraction_id): Just AI detection + Horizon segments
    
    Frontend Usage:
    ```
    // Option 1: Complete processing (extraction + analyses)
    const { job_id } = await fetch('/api/v1/process', {
        method: 'POST',
        body: JSON.stringify({ document_id: 'abc123' })
    }).then(r => r.json());
    
    // Option 2: Analysis only (text already extracted)
    const { job_id } = await fetch('/api/v1/process', {
        method: 'POST',
        body: JSON.stringify({ extraction_id: 'xyz789' })
    }).then(r => r.json());
    
    // Poll for results
    const interval = setInterval(async () => {
        const status = await fetch(`/api/v1/status/${job_id}`).then(r => r.json());
        if (status.status === 'complete') {
            clearInterval(interval);
            // Use results: status.ai_detection, status.references, etc.
        }
    }, 1000);
    ```
    """
    try:
        if request.document_id:
            # Mode 1: Complete processing (extraction + analyses)
            from app.database import Document
            
            document = db.query(Document).filter(Document.id == request.document_id).first()
            if not document:
                raise HTTPException(status_code=404, detail="Document not found")
            
            logger.info(f"Starting COMPLETE processing for document: {request.document_id}")
            
            job_id = await orchestrator.process_complete(
                document_id=request.document_id,
                db=db
            )
            
            return ProcessResponse(
                job_id=job_id,
                status="processing",
                message="Complete processing started (extraction + analyses). Poll /status/{job_id} for results.",
                started_at=datetime.utcnow().isoformat()
            )
            
        elif request.extraction_id:
            # Mode 2: Analysis only (extraction already done)
            from app.database import Extraction
            
            extraction = db.query(Extraction).filter(Extraction.id == request.extraction_id).first()
            if not extraction:
                raise HTTPException(status_code=404, detail="Extraction not found")
            
            logger.info(f"Starting ANALYSIS processing for extraction: {request.extraction_id}")
            
            job_id = await orchestrator.process_full_analysis(
                extraction_id=request.extraction_id,
                db=db
            )
            
            return ProcessResponse(
                job_id=job_id,
                status="processing",
                message="Analysis started (AI detection + Horizon). Poll /status/{job_id} for results.",
                started_at=datetime.utcnow().isoformat()
            )
            
        else:
            raise HTTPException(
                status_code=400,
                detail="Either document_id or extraction_id must be provided"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start processing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{job_id}", response_model=JobStatusResponse)
async def get_job_status(
    job_id: str,
    db: Session = Depends(get_db)
):
    """
    Get current job status and progressive results
    
    Returns results as they become available.
    Poll this endpoint every 1-2 seconds until status = 'complete'
    
    Frontend Usage (Approach 2):
    ```
    const status = await fetch(`/api/v1/status/${job_id}`).then(r => r.json());
    
    // Update UI with available results
    if (status.ai_detection) {
        displayAIDetection(status.ai_detection);
    }
    if (status.references) {
        displayReferences(status.references);
    }
    // ... etc
    
    // Check if complete
    if (status.status === 'complete') {
        // All done!
    }
    ```
    """
    try:
        result = orchestrator.get_job_status(job_id, db)
        
        if "error" in result and result["error"] == "Job not found":
            raise HTTPException(status_code=404, detail="Job not found")
        
        return JobStatusResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get job status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))