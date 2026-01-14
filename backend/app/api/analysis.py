"""
Analysis Endpoints - Approach 1
Individual endpoints for progressive loading
"""
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
import logging
import uuid
from datetime import datetime

from app.database import get_db, Extraction, AIDetectionResult, HorizonResult
from app.models import (
    AIDetectionRequest, AIDetectionResponse,
    HorizonRequest, HorizonSegmentResponse
)
from app.agents.horizon_agent import HorizonAgent
from app.services.ai_detection_service import get_ai_detection_service
from app.utils import cancellation_manager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["analysis"])

# Initialize services
horizon_agent = HorizonAgent()
ai_detection_service = get_ai_detection_service()


@router.post("/ai-detection", response_model=AIDetectionResponse)
async def detect_ai_content(
    request: AIDetectionRequest,
    db: Session = Depends(get_db)
):
        # STEP 3: Block future requests if cancelled
    if not cancellation_manager.accept_requests:
        raise HTTPException(
            status_code=503,
            detail="Processing has been cancelled by user"
        )

    """
    Detect AI-generated content using external API
    
    Frontend Usage (Approach 1):
    ```
    const result = await fetch('/api/v1/ai-detection', {
        method: 'POST',
        body: JSON.stringify({ extraction_id: 'abc123' })
    });
    ```
    """
    try:
        logger.info(f"AI Detection request for extraction: {request.extraction_id}")
        
        # Get extracted text from database
        extraction = db.query(Extraction).filter(
            Extraction.id == request.extraction_id
        ).first()
        
        if not extraction:
            raise HTTPException(status_code=404, detail="Extraction not found")
        
        # Check if already processed (caching)
        cached_result = db.query(AIDetectionResult).filter(
            AIDetectionResult.extraction_id == request.extraction_id
        ).first()
        
        if cached_result:
            logger.info(f"Returning cached AI detection result")
            return AIDetectionResponse(
                success=True,
                extraction_id=request.extraction_id,
                is_ai_generated=cached_result.is_ai_generated,
                confidence=cached_result.confidence,
                detection_method=cached_result.detection_method,
                flags=cached_result.flags or [],
                summary="Cached result",
                processing_time=0.0
            )
        
        # Call external AI detection service
        result = await ai_detection_service.detect(extraction.text)
        
        # Save result to database
        if result['success']:
            ai_result = AIDetectionResult(
                id=str(uuid.uuid4()),
                extraction_id=request.extraction_id,
                is_ai_generated=result['is_ai_generated'],
                confidence=result['confidence'],
                detection_method=result['detection_method'],
                flags=result['flags'],
                raw_response=result.get('raw_response', {})
            )
            db.add(ai_result)
            db.commit()
            
            logger.info(f"AI Detection complete: {result['is_ai_generated']}")
        
        return AIDetectionResponse(
            success=result['success'],
            extraction_id=request.extraction_id,
            is_ai_generated=result['is_ai_generated'],
            confidence=result['confidence'],
            detection_method=result['detection_method'],
            flags=result['flags'],
            summary=result['summary'],
            processing_time=result['processing_time'],
            error=result.get('error')
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"AI Detection failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/horizon/references", response_model=HorizonSegmentResponse)
async def extract_references(
    request: HorizonRequest,
    db: Session = Depends(get_db)
):
    """
    Extract and validate references (research papers, citations)
    
    Frontend Usage (Approach 1):
    ```
    const result = await fetch('/api/v1/horizon/references', {
        method: 'POST',
        body: JSON.stringify({ extraction_id: 'abc123' })
    });
    ```
    """
    try:
        logger.info(f"References extraction request for: {request.extraction_id}")
        
        # Get extracted text
        extraction = db.query(Extraction).filter(
            Extraction.id == request.extraction_id
        ).first()
        
        if not extraction:
            raise HTTPException(status_code=404, detail="Extraction not found")
        
        # Check cache
        cached_result = db.query(HorizonResult).filter(
            HorizonResult.extraction_id == request.extraction_id,
            HorizonResult.segment_type == "references"
        ).first()
        
        if cached_result:
            logger.info("Returning cached references result")
            return HorizonSegmentResponse(
                success=True,
                extraction_id=request.extraction_id,
                segment_type="references",
                data=cached_result.data,
                validation=cached_result.validation,
                processing_time=0.0
            )
        
        # Extract references using Horizon Agent
        result = await horizon_agent.extract_references(extraction.text)
        
        # Save to database
        if result['success']:
            horizon_result = HorizonResult(
                id=str(uuid.uuid4()),
                extraction_id=request.extraction_id,
                segment_type="references",
                data=result['data'],
                validation=result['validation'],
                confidence=result['validation']['confidence'],
                processing_time=result['processing_time']
            )
            db.add(horizon_result)
            db.commit()
            
            logger.info(f"References extracted: {result['data'].get('total_count', 0)} found")
        
        return HorizonSegmentResponse(
            success=result['success'],
            extraction_id=request.extraction_id,
            segment_type="references",
            data=result['data'],
            validation=result['validation'],
            processing_time=result['processing_time'],
            error=result.get('error')
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"References extraction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/horizon/medical", response_model=HorizonSegmentResponse)
async def extract_medical(
    request: HorizonRequest,
    db: Session = Depends(get_db)
):
    """
    Extract and validate medical context (conditions, medications, history)
    
    Frontend Usage (Approach 1):
    ```
    const result = await fetch('/api/v1/horizon/medical', {
        method: 'POST',
        body: JSON.stringify({ extraction_id: 'abc123' })
    });
    ```
    """
    try:
        logger.info(f"Medical extraction request for: {request.extraction_id}")
        
        # Get extracted text
        extraction = db.query(Extraction).filter(
            Extraction.id == request.extraction_id
        ).first()
        
        if not extraction:
            raise HTTPException(status_code=404, detail="Extraction not found")
        
        # Check cache
        cached_result = db.query(HorizonResult).filter(
            HorizonResult.extraction_id == request.extraction_id,
            HorizonResult.segment_type == "medical"
        ).first()
        
        if cached_result:
            logger.info("Returning cached medical result")
            return HorizonSegmentResponse(
                success=True,
                extraction_id=request.extraction_id,
                segment_type="medical",
                data=cached_result.data,
                validation=cached_result.validation,
                processing_time=0.0
            )
        
        # Extract medical context using Horizon Agent
        result = await horizon_agent.extract_medical(extraction.text)
        
        # Save to database
        if result['success']:
            horizon_result = HorizonResult(
                id=str(uuid.uuid4()),
                extraction_id=request.extraction_id,
                segment_type="medical",
                data=result['data'],
                validation=result['validation'],
                confidence=result['validation']['confidence'],
                processing_time=result['processing_time']
            )
            db.add(horizon_result)
            db.commit()
            
            logger.info(f"Medical extracted: {len(result['data'].get('conditions', []))} conditions")
        
        return HorizonSegmentResponse(
            success=result['success'],
            extraction_id=request.extraction_id,
            segment_type="medical",
            data=result['data'],
            validation=result['validation'],
            processing_time=result['processing_time'],
            error=result.get('error')
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Medical extraction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/horizon/legal", response_model=HorizonSegmentResponse)
async def extract_legal(
    request: HorizonRequest,
    db: Session = Depends(get_db)
):
    """
    Extract and validate legal context (claims, statutes, deadlines)
    
    Frontend Usage (Approach 1):
    ```
    const result = await fetch('/api/v1/horizon/legal', {
        method: 'POST',
        body: JSON.stringify({ extraction_id: 'abc123' })
    });
    ```
    """
    try:
        logger.info(f"Legal extraction request for: {request.extraction_id}")
        
        # Get extracted text
        extraction = db.query(Extraction).filter(
            Extraction.id == request.extraction_id
        ).first()
        
        if not extraction:
            raise HTTPException(status_code=404, detail="Extraction not found")
        
        # Check cache
        cached_result = db.query(HorizonResult).filter(
            HorizonResult.extraction_id == request.extraction_id,
            HorizonResult.segment_type == "legal"
        ).first()
        
        if cached_result:
            logger.info("Returning cached legal result")
            return HorizonSegmentResponse(
                success=True,
                extraction_id=request.extraction_id,
                segment_type="legal",
                data=cached_result.data,
                validation=cached_result.validation,
                processing_time=0.0
            )
        
        # Extract legal context using Horizon Agent
        result = await horizon_agent.extract_legal(extraction.text)
        
        # Save to database
        if result['success']:
            horizon_result = HorizonResult(
                id=str(uuid.uuid4()),
                extraction_id=request.extraction_id,
                segment_type="legal",
                data=result['data'],
                validation=result['validation'],
                confidence=result['validation']['confidence'],
                processing_time=result['processing_time']
            )
            db.add(horizon_result)
            db.commit()
            
            logger.info(f"Legal extracted: {len(result['data'].get('legal_claims', []))} claims")
        
        return HorizonSegmentResponse(
            success=result['success'],
            extraction_id=request.extraction_id,
            segment_type="legal",
            data=result['data'],
            validation=result['validation'],
            processing_time=result['processing_time'],
            error=result.get('error')
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Legal extraction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))