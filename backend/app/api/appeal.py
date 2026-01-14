from fastapi import APIRouter, HTTPException, Depends, Body
from sqlalchemy.orm import Session
from pydantic import BaseModel
import re
import logging
from app.models import ExtractionRequest, DocumentStatus  # From your extract.py
from app.database import get_db, Document, Extraction

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["extraction"])

class AppealRequest(BaseModel):
    extraction_id: str

class AppealResponse(BaseModel):
    success: bool
    extraction_id: str
    appeal_text: str
    full_text_length: int
    appeal_length: int

def extract_appeal_section(full_text: str) -> str:
    """Extract appeal section using patterns from insurance denial letters."""
    if not full_text:
        return ""
    
    lines = full_text.split('\n')
    appeal_start = None
    appeal_end = len(lines)
    
    # Keywords from your sample appeals + insurance docs
    appeal_keywords = [
        r"re:\s*appeal", r"appeal\s+(?:against|of|for|denial)",
        r"dear\s+(?:claims|review|committee)", r"policy\s+number", r"claim\s+number"
    ]
    
    for i, line in enumerate(lines):
        line_lower = line.lower().strip()
        if any(re.search(kw, line_lower, re.IGNORECASE) for kw in appeal_keywords):
            appeal_start = i
            break
    
    if appeal_start is not None:
        closing_keywords = [r"sincerely", r"thank you", r"regards"]
        for j in range(appeal_start + 5, len(lines)):
            if any(re.search(kw, lines[j].lower(), re.IGNORECASE) for kw in closing_keywords):
                appeal_end = j + 1
                break
    
    appeal_lines = lines[appeal_start:appeal_end] if appeal_start else []
    return '\n'.join(appeal_lines).strip()

def sanitize_sensitive_info(text: str) -> str:
    """Copy from extract.py"""
    if not text:
        return text
    dob_pattern1 = r'\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b'
    ssn_pattern = r'\b\d{3}-\d{2}-\d{4}\b'
    text = re.sub(dob_pattern1, '<D.O.B>', text, flags=re.IGNORECASE)
    text = re.sub(ssn_pattern, '<S.S.N>', text)
    return text

@router.post("/appeal")
async def appeal_extraction(
    request: AppealRequest = Body(...),
    db: Session = Depends(get_db)
):
    try:
        # Exact query match from your extract.py
        extraction = db.query(Extraction).filter(Extraction.id == request.extraction_id).first()
        if not extraction:
            raise HTTPException(status_code=404, detail="Extraction not found")
        
        full_text = extraction.text or ''
        sanitized_full = sanitize_sensitive_info(full_text)
        
        appeal_text = extract_appeal_section(sanitized_full)
        
        if not appeal_text:
            raise HTTPException(status_code=400, detail="No appeal section detected in document")
        
        logger.info(f"Appeal extracted: {len(appeal_text)} chars from {len(sanitized_full)} total")
        
        return AppealResponse(
            success=True,
            extraction_id=request.extraction_id,
            appeal_text=appeal_text,
            full_text_length=len(sanitized_full),
            appeal_length=len(appeal_text)
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Appeal extraction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
