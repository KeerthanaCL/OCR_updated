"""
Text Analysis Endpoint
Process pre-extracted text directly (no OCR needed)
"""

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
import logging
import uuid
from datetime import datetime

from app.utils import cancellation_manager
from app.database import get_db, Extraction
from app.models import ProcessRequest, ProcessResponse
from app.agents.orchestrator_agent import OrchestratorAgent

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["text-analysis"])

# Initialize orchestrator
orchestrator = OrchestratorAgent()


class TextAnalysisRequest(BaseModel):
    """Request for analyzing pre-extracted text"""
    text: str = Field(..., min_length=50, description="Text to analyze (minimum 50 characters)")
    document_name: str = Field(default="Pasted Text", description="Optional name for this text")
    skip_extraction: bool = Field(default=True, description="Always true for this endpoint")


class TextAnalysisResponse(BaseModel):
    """Response from text analysis"""
    success: bool
    extraction_id: str
    job_id: str
    message: str
    text_length: int
    word_count: int


@router.post("/analyze-text", response_model=TextAnalysisResponse)
async def analyze_text(
    request: TextAnalysisRequest,
    db: Session = Depends(get_db)
):
        # STEP 3: Block future requests if cancelled
    if not cancellation_manager.accept_requests:
        raise HTTPException(
            status_code=503,
            detail="Processing has been cancelled by user"
        )

    """
    Analyze pre-extracted text directly (no OCR needed).
    
    **Use this endpoint when:**
    - User pastes text directly
    - Text is already extracted from document
    - You want to skip OCR and go straight to analysis
    
    **Workflow:**
    1. Save text as "extraction" record (no OCR performed)
    2. Start background analysis job (AI detection + Horizon segments)
    3. Poll /status/{job_id} for results
    
    **Example:**
    ```
    response = await fetch('/api/v1/analyze-text', {
        method: 'POST',
        body: JSON.stringify({
            text: "Your insurance appeal text here...",
            document_name: "My Appeal Letter"
        })
    })
    
    # Get job_id from response
    job_id = response.job_id
    
    # Poll for results
    status = await fetch(`/api/v1/status/${job_id}`)
    ```
    """
    
    try:
        logger.info(f"Text analysis request: {len(request.text)} characters")
        
        # Validate text length
        if len(request.text) < 50:
            raise HTTPException(
                status_code=400,
                detail="Text too short. Minimum 50 characters required."
            )
        
        # Create extraction record (no OCR, just store the text)
        extraction_id = str(uuid.uuid4())
        word_count = len(request.text.split())
        
        db_extraction = Extraction(
            id=extraction_id,
            document_id="direct-text-input",  # Special marker
            text=request.text,
            confidence=100.0,  # Perfect confidence (no OCR errors)
            method_used="direct_input",
            pages=1,
            processing_time=0.0,
            extraction_metadata={
                "source": "direct_text_input",
                "document_name": request.document_name,
                "character_count": len(request.text),
                "word_count": word_count,
                "input_timestamp": datetime.utcnow().isoformat()
            }
        )
        
        db.add(db_extraction)
        db.commit()
        db.refresh(db_extraction)
        
        logger.info(f"Created extraction record: {extraction_id}")
        
        # Start background analysis (AI detection + Horizon segments)
        job_id = await orchestrator.process_full_analysis(
            extraction_id=extraction_id,
            db=db
        )
        
        logger.info(f"Started analysis job: {job_id}")
        
        return TextAnalysisResponse(
            success=True,
            extraction_id=extraction_id,
            job_id=job_id,
            message=f"Text analysis started. Poll /status/{job_id} for results.",
            text_length=len(request.text),
            word_count=word_count
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Text analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze-text-complete", response_model=TextAnalysisResponse)
async def analyze_text_synchronous(
    request: TextAnalysisRequest,
    db: Session = Depends(get_db)
):
    """
    Synchronous version: Wait for all analyses to complete before returning.
    
    **Warning:** This can take 30-60 seconds. Use async version above for better UX.
    
    **Use this when:**
    - You need immediate results
    - User is willing to wait
    - Simple testing/debugging
    """
    
    try:
        # Same as above, but wait for completion
        logger.info(f"Synchronous text analysis: {len(request.text)} characters")
        
        # Create extraction
        extraction_id = str(uuid.uuid4())
        word_count = len(request.text.split())
        
        db_extraction = Extraction(
            id=extraction_id,
            document_id="direct-text-input-sync",
            text=request.text,
            confidence=100.0,
            method_used="direct_input_sync",
            pages=1,
            processing_time=0.0,
            extraction_metadata={
                "source": "direct_text_input_synchronous",
                "document_name": request.document_name,
                "character_count": len(request.text),
                "word_count": word_count
            }
        )
        
        db.add(db_extraction)
        db.commit()
        
        # Start and WAIT for analysis
        job_id = await orchestrator.process_full_analysis(
            extraction_id=extraction_id,
            db=db
        )
        
        # Poll until complete (with timeout)
        import asyncio
        max_wait = 120  # 2 minutes
        poll_interval = 2  # 2 seconds
        elapsed = 0
        
        while elapsed < max_wait:
            status = orchestrator.get_job_status(job_id, db)
            
            if status['status'] == 'complete':
                logger.info(f"Analysis complete after {elapsed}s")
                return TextAnalysisResponse(
                    success=True,
                    extraction_id=extraction_id,
                    job_id=job_id,
                    message="Analysis complete. Check /status/{job_id} for full results.",
                    text_length=len(request.text),
                    word_count=word_count
                )
            
            if status['status'] == 'failed':
                raise HTTPException(
                    status_code=500,
                    detail=f"Analysis failed: {status.get('error')}"
                )
            
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval
        
        # Timeout
        raise HTTPException(
            status_code=408,
            detail=f"Analysis timeout after {max_wait}s. Check /status/{job_id} later."
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Synchronous text analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))