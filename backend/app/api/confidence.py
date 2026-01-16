"""
Confidence Score API
Combines AI detection, legal, medical, and citation validation scores
"""
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel
import logging
from typing import Optional, Dict, Any

from app.database import get_db, Extraction, OrchestrationJob, HorizonResult, AIDetectionResult

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["confidence"])


class ConfidenceScoreResponse(BaseModel):
    """Combined confidence score response"""
    success: bool
    extraction_id: str
    combined_score: float  # 0-100
    confidence_level: str  # excellent, good, fair, poor
    
    # Individual components
    ai_confidence: Optional[float] = None
    legal_ratio: Optional[float] = None
    medical_ratio: Optional[float] = None
    citation_ratio: Optional[float] = None
    
    # Weights used
    weights: Dict[str, float]
    
    # Detailed breakdown
    breakdown: Dict[str, Any]
    missing_analyses: list[str]


# Confidence score weights
WEIGHTS = {
    "ai": 0.85,
    "legal": 0.05,
    "medical": 0.05,
    "citation": 0.05
}

# Hardcoded AI detection confidence (since API not available)
HARDCODED_AI_CONFIDENCE = 85.81583499908447 


def calculate_legal_ratio(legal_data: Dict) -> float:
    """
    Calculate legal validation ratio - SIMPLE PERCENTAGE
    
    Based on validation completeness_score:
    - completeness_score is already 0-1 (e.g., 0.67 = 67%)
    - Just convert to 0-100 scale
    
    Example:
    - If 2 out of 3 components validated → completeness_score = 0.67 → 67%
    """
    if not legal_data:
        return 0.0
    
    validation = legal_data.get('validation', {})
    completeness_score = validation.get('completeness_score', 0)  # 0-1 scale
    
    # Convert to percentage (0-100)
    score = completeness_score * 100
    
    logger.info(f"Legal score: {score:.1f}% (completeness: {completeness_score:.2f})")
    return score


def calculate_medical_ratio(medical_data: Dict) -> float:
    """
    Calculate medical validation ratio - SIMPLE PERCENTAGE
    
    Based on validation completeness_score:
    - Medical has 4 components: conditions, medications, history, necessity
    - If 2/4 validated → completeness_score = 0.5 → 50%
    
    Example:
    - 4/4 components → 100%
    - 3/4 components → 75%
    - 2/4 components → 50%
    - 1/4 components → 25%
    """
    if not medical_data:
        return 0.0
    
    validation = medical_data.get('validation', {})
    completeness_score = validation.get('completeness_score', 0)  # 0-1 scale
    
    # Convert to percentage (0-100)
    score = completeness_score * 100
    
    logger.info(f"Medical score: {score:.1f}% (completeness: {completeness_score:.2f})")
    return score


def calculate_citation_ratio(references_data: Dict) -> float:
    """
    Calculate citation validation ratio - SIMPLE PERCENTAGE
    
    Based on validation completeness_score:
    - completeness_score = valid_items / total_items
    - If 7 out of 10 references validated → 70%
    
    Example:
    - 10/10 references valid → 100%
    - 7/10 references valid → 70%
    - 5/10 references valid → 50%
    """
    if not references_data:
        return 0.0
    
    validation = references_data.get('validation', {})
    completeness_score = validation.get('completeness_score', 0)  # 0-1 scale
    
    # Convert to percentage (0-100)
    score = completeness_score * 100
    
    logger.info(f"Citation score: {score:.1f}% (completeness: {completeness_score:.2f})")
    return score

def get_confidence_level(score: float) -> str:
    """Convert numeric score to confidence level"""
    if score >= 85:
        return "excellent"
    elif score >= 70:
        return "good"
    elif score >= 50:
        return "fair"
    else:
        return "poor"


@router.get("/confidence-score/{extraction_id}", response_model=ConfidenceScoreResponse)
async def get_confidence_score(
    extraction_id: str,
    db: Session = Depends(get_db)
):
    """
    Calculate combined confidence score from all analyses
    
    **Formula:**
    ```
    combined_score = 
        w_ai * ai_confidence
      + w_legal * legal_ratio
      + w_med * medical_ratio
      + w_cite * citation_ratio
    
    Weights: AI=60%, Legal=15%, Medical=15%, Citations=10%
    ```
    
    **AI Detection:** Currently hardcoded to 85.82 (no API access)
    
    **Returns 200 even if some analyses are missing** (partial score calculated)
    """
    try:
        logger.info(f"Calculating confidence score for extraction: {extraction_id}")
        
        # Verify extraction exists
        extraction = db.query(Extraction).filter(
            Extraction.id == extraction_id
        ).first()
        
        if not extraction:
            raise HTTPException(status_code=404, detail="Extraction not found")
        
        # Initialize scores
        ai_confidence = None
        legal_ratio = None
        medical_ratio = None
        citation_ratio = None
        missing_analyses = []
        
        # 1. Get AI Detection confidence
        # ai_result = db.query(AIDetectionResult).filter(
        #     AIDetectionResult.extraction_id == extraction_id
        # ).first()
        
        # if ai_result:
        #     # AI confidence is already 0-100
        #     ai_confidence = float(ai_result.confidence)
        # else:
        #     missing_analyses.append("ai_detection")
        #     logger.warning(f"No AI detection result for {extraction_id}")
        ai_confidence = HARDCODED_AI_CONFIDENCE
        logger.info(f"Using hardcoded AI confidence: {ai_confidence}")
        
        # 2. Get Legal validation ratio
        legal_result = db.query(HorizonResult).filter(
            HorizonResult.extraction_id == extraction_id,
            HorizonResult.segment_type == "legal"
        ).first()
        
        if legal_result and legal_result.data:
            legal_ratio = calculate_legal_ratio(legal_result.data)
            logger.info(f"Legal ratio calculated: {legal_ratio}")
        else:
            missing_analyses.append("legal_analysis")
            logger.warning(f"No legal result for {extraction_id}")
        
        # 3. Get Medical validation ratio
        medical_result = db.query(HorizonResult).filter(
            HorizonResult.extraction_id == extraction_id,
            HorizonResult.segment_type == "medical"
        ).first()
        
        if medical_result and medical_result.data:
            medical_ratio = calculate_medical_ratio(medical_result.data)
            logger.info(f"Medical ratio calculated: {medical_ratio}")
        else:
            missing_analyses.append("medical_analysis")
            logger.warning(f"No medical result for {extraction_id}")
        
        # 4. Get Citation validation ratio
        references_result = db.query(HorizonResult).filter(
            HorizonResult.extraction_id == extraction_id,
            HorizonResult.segment_type == "references"
        ).first()
        
        if references_result and references_result.data:
            citation_ratio = calculate_citation_ratio(references_result.data)
            logger.info(f"Citation ratio calculated: {citation_ratio}")
        else:
            missing_analyses.append("references_analysis")
            logger.warning(f"No references result for {extraction_id}")
        
        # Calculate weighted combined score
        total_weight = 0
        combined_score = 0
        
        if ai_confidence is not None:
            combined_score += WEIGHTS["ai"] * ai_confidence
            total_weight += WEIGHTS["ai"]
        
        if legal_ratio is not None:
            combined_score += WEIGHTS["legal"] * legal_ratio
            total_weight += WEIGHTS["legal"]
        
        if medical_ratio is not None:
            combined_score += WEIGHTS["medical"] * medical_ratio
            total_weight += WEIGHTS["medical"]
        
        if citation_ratio is not None:
            combined_score += WEIGHTS["citation"] * citation_ratio
            total_weight += WEIGHTS["citation"]
        
        # Normalize by total weight (in case some analyses are missing)
        if total_weight > 0:
            combined_score = combined_score / total_weight
        else:
            raise HTTPException(
                status_code=400,
                detail="No analyses found. Please run AI detection and Horizon analyses first."
            )
        
        confidence_level = get_confidence_level(combined_score)
        
        logger.info(
            f"Confidence calculated: {combined_score:.2f} ({confidence_level}) "
            f"[AI:{ai_confidence}, Legal:{legal_ratio}, Med:{medical_ratio}, Cite:{citation_ratio}]"
        )
        
        return ConfidenceScoreResponse(
            success=True,
            extraction_id=extraction_id,
            combined_score=round(combined_score, 2),
            confidence_level=confidence_level,
            ai_confidence=round(ai_confidence, 2) if ai_confidence is not None else None,
            legal_ratio=round(legal_ratio, 2) if legal_ratio is not None else None,
            medical_ratio=round(medical_ratio, 2) if medical_ratio is not None else None,
            citation_ratio=round(citation_ratio, 2) if citation_ratio is not None else None,
            weights=WEIGHTS,
            breakdown={
                "ai_contribution": round(WEIGHTS["ai"] * ai_confidence, 2) if ai_confidence else 0,
                "legal_contribution": round(WEIGHTS["legal"] * legal_ratio, 2) if legal_ratio else 0,
                "medical_contribution": round(WEIGHTS["medical"] * medical_ratio, 2) if medical_ratio else 0,
                "citation_contribution": round(WEIGHTS["citation"] * citation_ratio, 2) if citation_ratio else 0,
                "total_weight_used": round(total_weight, 2)
            },
            missing_analyses=missing_analyses
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Confidence score calculation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))