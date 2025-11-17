from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from app.models import ParsingRequest, ParsingResult
from app.agents.insurance_parsing_agent import InsuranceClaimParsingAgent
from app.database import get_db, Extraction, Parsing
import uuid
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["parsing"])

@router.post("/parse", response_model=ParsingResult)
async def parse_text(
    request: ParsingRequest,
    db: Session = Depends(get_db)
):
    """
    Parse extracted text into structured fields.
    
    Extracts: emails, phones, dates, amounts, URLs, and custom patterns.
    """
    try:
        # Get extraction from database
        extraction = db.query(Extraction).filter(
            Extraction.id == request.extraction_id
        ).first()
        
        if not extraction:
            raise HTTPException(status_code=404, detail="Extraction not found")
        
        # Execute parsing
        parser = InsuranceClaimParsingAgent()
        result = parser.execute(
            text=extraction.text,
            parsing_rules=request.parsing_rules
        )
        
        # Save to database
        parsing_id = str(uuid.uuid4())
        db_parsing = Parsing(
            id=parsing_id,
            document_id=request.document_id,
            extraction_id=request.extraction_id,
            fields=[field.dict() for field in result['fields']],
            parsing_method=result['parsing_method'],
            processing_time=result['processing_time']
        )
        db.add(db_parsing)
        db.commit()
        
        logger.info(f"Parsing completed: {parsing_id}")
        
        return ParsingResult(
            document_id=request.document_id,
            extraction_id=request.extraction_id,
            fields=result['fields'],
            parsing_method=result['parsing_method'],
            processing_time=result['processing_time']
        )
        
    except Exception as e:
        logger.error(f"Parsing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))