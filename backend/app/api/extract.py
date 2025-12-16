from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from app.models import (
    ExtractionRequest, 
    DocumentStatus,
    OCRMethod
)
from app.agents.orchestrator_agent import OrchestratorAgent
from app.services.storage import StorageService
from app.database import get_db, Document, Extraction
import numpy as np
import uuid
import logging

logger = logging.getLogger(__name__)

def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

router = APIRouter(prefix="/api/v1", tags=["extraction"])

# Global orchestrator instance
orchestrator = OrchestratorAgent()

@router.post("/extract")
async def extract_document(
    request: ExtractionRequest,
    db: Session = Depends(get_db)
):
    """
    Extract text from uploaded document (Page 1)
    
    This endpoint only performs OCR extraction.
    For analysis, use either:
    - Approach 1: Individual endpoints (/ai-detection, /horizon/*)
    - Approach 2: Orchestrated processing (/process, /status)
    """
    try:
        # Step 1: Get document from database
        document = db.query(Document).filter(Document.id == request.document_id).first()
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Update document status
        document.status = DocumentStatus.EXTRACTING.value
        db.commit()
        
        # Step 2: Get file path
        storage = StorageService()
        file_path = storage.get_file_path(request.document_id)
        
        logger.info(f"Processing document: {document.filename} (ID: {request.document_id})")
        
        # Step 3: Execute complete processing pipeline via orchestrator
        result = await orchestrator.extract_only(
            document_id=request.document_id,
            db=db
        )
        
        if not result.get('success'):
            document.status = DocumentStatus.FAILED.value
            db.commit()
            raise HTTPException(
                status_code=500, 
                detail=result.get('error', 'Processing failed')
            )
        
        extraction_id = result['extraction_id']

        # Step 3: Update document status
        document.status = DocumentStatus.COMPLETED.value
        db.commit()

        logger.info(f"Document extraction completed: {extraction_id}")

        # Step 4: Return extraction result
        extraction = db.query(Extraction).filter(Extraction.id == extraction_id).first()

        if not extraction:
            raise HTTPException(
                status_code=500,
                detail=f"Extraction record not found in database for ID: {extraction_id}"
            )
        
        return {
            'success': True,
            'document_id': request.document_id,
            'extraction_id': extraction_id,
            'text': extraction.text or '',
            'confidence': float(extraction.confidence or 0.0),
            'method_used': extraction.method_used or 'unknown',
            'pages': extraction.pages or 0,
            'processing_time': float(extraction.processing_time or 0.0),
            'metadata': extraction.extraction_metadata or {}
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Extraction failed: {e}", exc_info=True)
        if 'document' in locals():
            document.status = DocumentStatus.FAILED.value
            db.commit()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/extract/status/{document_id}")
async def get_extraction_status(document_id: str, db: Session = Depends(get_db)):
    """Get extraction status for a document."""
    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    extraction = db.query(Extraction).filter(
        Extraction.document_id == document_id
    ).order_by(Extraction.created_at.desc()).first()

    response = {
        'document_id': document_id,
        'status': document.status,
        'filename': document.filename,
        'created_at': document.created_at.isoformat(),
        'updated_at': document.updated_at.isoformat()
    }

    if extraction:
        response['extraction'] = {
            'extraction_id': extraction.id,
            'confidence': extraction.confidence,
            'method_used': extraction.method_used,
            'pages': extraction.pages,
            'processing_time': extraction.processing_time,
            'text_length': len(extraction.text) if extraction.text else 0
        }

    return response