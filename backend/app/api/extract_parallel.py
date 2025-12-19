"""
Parallel Multi-Method Extraction Endpoint

Runs Tesseract, EasyOCR, and TrOCR in parallel, then selects the best result.
Maximizes accuracy at the cost of processing time (3x slower but most accurate).
"""

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from app.models import ExtractionRequest, DocumentStatus
from app.agents.extraction_agent import LangGraphExtractionAgent
from app.database import get_db, Document, Extraction
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["extraction-parallel"])

# Global extraction agent
extraction_agent = LangGraphExtractionAgent()

@router.post("/extract/parallel")
async def extract_document_parallel_best(
    request: ExtractionRequest,
    db: Session = Depends(get_db)
):
    """
    Extract text using ALL OCR methods in parallel (Tesseract, EasyOCR, TrOCR).
    
    **Strategy:** Run all 3 methods simultaneously for each page, then select
    the result with highest confidence.
    
    **Use this when:**
    - Maximum accuracy is required
    - Processing time is not critical
    - You want to compare all OCR methods
    
    **Processing time:** ~3x slower than regular extraction (all methods run in parallel)
    
    **Returns:** Best result + comparison data for all methods
    """
    try:
        # Get document
        document = db.query(Document).filter(Document.id == request.document_id).first()
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")

        # Update status
        document.status = DocumentStatus.EXTRACTING.value
        db.commit()

        logger.info(f"Starting PARALLEL extraction: {document.filename} (ID: {request.document_id})")

        # Run parallel extraction
        result = await extraction_agent.execute_parallel_best(
            document_id=request.document_id,
            db=db
        )

        if not result.get('success'):
            document.status = DocumentStatus.FAILED.value
            db.commit()
            raise HTTPException(
                status_code=500,
                detail=result.get('error', 'Parallel extraction failed')
            )

        extraction_id = result.get('extraction_id')
        
        if not extraction_id:
            document.status = DocumentStatus.FAILED.value
            db.commit()
            raise HTTPException(
                status_code=500,
                detail="Extraction failed: No extraction ID returned"
            )

        # Update document status
        document.status = DocumentStatus.COMPLETED.value
        db.commit()

        logger.info(f"Parallel extraction completed: {extraction_id}, best: {result['method_used']}")

        # Get extraction from database
        extraction = db.query(Extraction).filter(Extraction.id == extraction_id).first()
        
        if not extraction:
            raise HTTPException(
                status_code=500,
                detail="Extraction record not found in database"
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
            'metadata': extraction.extraction_metadata or {},
            'all_results': result.get('all_results', [])  # Comparison data
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Parallel extraction failed: {str(e)}", exc_info=True)
        if 'document' in locals():
            document.status = DocumentStatus.FAILED.value
            db.commit()
        raise HTTPException(status_code=500, detail=str(e))