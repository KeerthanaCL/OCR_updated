from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from app.models import (
    SegmentExtractionRequest,
    ReferencesExtractionResponse,
    MedicalContextResponse,
    LegalContextResponse
)
from app.services.openai_extraction_service import OpenAIExtractionService
from app.services.appeals_service import AppealsExtractionService
from app.database import get_db, Extraction
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/segments", tags=["segment-extraction"])

# Global services
openai_service = OpenAIExtractionService()
appeals_service = AppealsExtractionService()


async def get_text_for_extraction(
    request: SegmentExtractionRequest,
    db: Session
) -> str:
    """Helper to get text from document_id or direct text."""
    if request.text:
        text = request.text
    elif request.document_id:
        extraction = db.query(Extraction).filter(
            Extraction.document_id == request.document_id
        ).order_by(Extraction.created_at.desc()).first()
        
        if not extraction:
            raise HTTPException(
                status_code=404,
                detail=f"No extraction found for document {request.document_id}"
            )
        text = extraction.text
    else:
        raise HTTPException(
            status_code=400,
            detail="Either document_id or text must be provided"
        )
    
    # Extract appeals section if requested
    if request.extract_appeals_first:
        appeals_text, found = appeals_service.extract_appeals_section(text)
        if found and appeals_text:
            return appeals_text
    
    return text


@router.post("/references", response_model=ReferencesExtractionResponse)
async def extract_references(
    request: SegmentExtractionRequest,
    db: Session = Depends(get_db)
):
    """
    Extract all references (case numbers, contact info, providers) using OpenAI.
    
    Workflow:
    1. Get text from document or directly
    2. Optionally extract appeals section first
    3. Send entire text to OpenAI for reference extraction
    4. Return structured references
    """
    try:
        text = await get_text_for_extraction(request, db)
        
        logger.info(f"Extracting references from {len(text)} characters")
        result = await openai_service.extract_references(text)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Reference extraction failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Reference extraction failed: {str(e)}"
        )


@router.post("/medical", response_model=MedicalContextResponse)
async def extract_medical_context(
    request: SegmentExtractionRequest,
    db: Session = Depends(get_db)
):
    """
    Extract medical context (conditions, medications, history) using OpenAI.
    
    Workflow:
    1. Get text from document or directly
    2. Optionally extract appeals section first
    3. Send entire text to OpenAI for medical extraction
    4. Return structured medical information
    """
    try:
        text = await get_text_for_extraction(request, db)
        
        logger.info(f"Extracting medical context from {len(text)} characters")
        result = await openai_service.extract_medical_context(text)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Medical extraction failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Medical extraction failed: {str(e)}"
        )


@router.post("/legal", response_model=LegalContextResponse)
async def extract_legal_context(
    request: SegmentExtractionRequest,
    db: Session = Depends(get_db)
):
    """
    Extract legal context (claims, rights, statutes, deadlines) using OpenAI.
    
    Workflow:
    1. Get text from document or directly
    2. Optionally extract appeals section first
    3. Send entire text to OpenAI for legal extraction
    4. Return structured legal information
    """
    try:
        text = await get_text_for_extraction(request, db)
        
        logger.info(f"Extracting legal context from {len(text)} characters")
        result = await openai_service.extract_legal_context(text)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Legal extraction failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Legal extraction failed: {str(e)}"
        )