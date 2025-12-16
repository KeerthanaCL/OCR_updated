"""
Horizon Agent
Handles segmentation and validation of references, medical, and legal content
"""
import logging
import time
from typing import Dict, Any
from app.services.openai_extraction_service import OpenAIExtractionService

logger = logging.getLogger(__name__)

class HorizonAgent:
    """
    Agent for extracting and validating document segments.
    Handles: References, Medical Context, Legal Context
    """
    
    def __init__(self):
        self.openai_service = OpenAIExtractionService()
        logger.info("Horizon Agent initialized")
    
    async def extract_references(self, text: str) -> Dict[str, Any]:
        """
        Extract and validate references (research papers, citations)
        
        Args:
            text: Extracted document text
            
        Returns:
            Dict with references data and validation
        """
        start_time = time.time()
        
        try:
            logger.info("Extracting references with validation...")
            
            # Call OpenAI with combined extraction + validation prompt
            result = await self.openai_service.extract_references(text)
            
            # Add validation metrics
            validation = self._validate_references(result)
            
            processing_time = time.time() - start_time
            
            return {
                "success": True,
                "data": {
                    "research_papers": [r.dict() for r in result.research_papers], 
                    "online_resources": [r.dict() for r in result.online_resources],
                    "summary": result.summary,
                    "total_count": len(result.research_papers) + len(result.online_resources)
                },
                "validation": validation,
                "processing_time": processing_time
            }
            
        except Exception as e:
            logger.error(f"References extraction failed: {e}", exc_info=True)
            return self._error_response("references", str(e), time.time() - start_time)
    
    async def extract_medical(self, text: str) -> Dict[str, Any]:
        """
        Extract and validate medical context
        
        Args:
            text: Extracted document text
            
        Returns:
            Dict with medical data and validation
        """
        start_time = time.time()
        
        try:
            logger.info("Extracting medical context with validation...")
            
            result = await self.openai_service.extract_medical_context(text)
            
            validation = self._validate_medical(result)
            
            processing_time = time.time() - start_time
            
            return {
                "success": True,
                "data": {
                    "patient_name": result.patient_name,
                    "conditions": [c.dict() for c in result.conditions],
                    "medications": [m.dict() for m in result.medications],
                    "medical_history": result.medical_history,
                    "medical_necessity_argument": result.medical_necessity_argument,
                    "providers_mentioned": result.providers_mentioned
                },
                "validation": validation,
                "processing_time": processing_time
            }
            
        except Exception as e:
            logger.error(f"Medical extraction failed: {e}", exc_info=True)
            return self._error_response("medical", str(e), time.time() - start_time)
    
    async def extract_legal(self, text: str) -> Dict[str, Any]:
        """
        Extract and validate legal context
        
        Args:
            text: Extracted document text
            
        Returns:
            Dict with legal data and validation
        """
        start_time = time.time()
        
        try:
            logger.info("Extracting legal context with validation...")
            
            result = await self.openai_service.extract_legal_context(text)
            
            validation = self._validate_legal(result)
            
            processing_time = time.time() - start_time
            
            return {
                "success": True,
                "data": {
                    "appeal_type": result.appeal_type,
                    "legal_claims": [c.dict() for c in result.legal_claims],
                    "legal_summary": result.legal_summary,
                    "statutes_cited": result.statutes_cited,
                    "deadlines_mentioned": result.deadlines_mentioned
                },
                "validation": validation,
                "processing_time": processing_time
            }
            
        except Exception as e:
            logger.error(f"Legal extraction failed: {e}", exc_info=True)
            return self._error_response("legal", str(e), time.time() - start_time)
    
    def _validate_references(self, result) -> Dict:
        """Validate references extraction quality"""
        total_refs = len(result.research_papers) + len(result.online_resources)
        
        is_complete = total_refs > 0
        confidence = result.confidence
        quality_issues = []
        
        if total_refs == 0:
            quality_issues.append("No references found")
        if confidence < 0.7:
            quality_issues.append("Low confidence extraction")
        
        completeness_score = min(1.0, total_refs / 5.0)  # Expect ~5 references
        
        return {
            "is_complete": is_complete,
            "confidence": confidence,
            "quality_issues": quality_issues,
            "completeness_score": completeness_score,
            "missing_fields": [] if is_complete else ["references"]
        }
    
    def _validate_medical(self, result) -> Dict:
        """Validate medical extraction quality"""
        has_conditions = len(result.conditions) > 0
        has_medications = len(result.medications) > 0
        has_history = bool(result.medical_history)
        
        is_complete = has_conditions and has_medications
        confidence = result.confidence
        quality_issues = []
        missing_fields = []
        
        if not has_conditions:
            quality_issues.append("No medical conditions found")
            missing_fields.append("conditions")
        if not has_medications:
            quality_issues.append("No medications found")
            missing_fields.append("medications")
        if not has_history:
            quality_issues.append("Medical history missing")
            missing_fields.append("history")
        
        completeness_score = sum([has_conditions, has_medications, has_history]) / 3.0
        
        return {
            "is_complete": is_complete,
            "confidence": confidence,
            "quality_issues": quality_issues,
            "completeness_score": completeness_score,
            "missing_fields": missing_fields
        }
    
    def _validate_legal(self, result) -> Dict:
        """Validate legal extraction quality"""
        has_claims = len(result.legal_claims) > 0
        has_summary = bool(result.legal_summary)
        has_statutes = len(result.statutes_cited) > 0
        
        is_complete = has_claims and has_summary
        confidence = result.confidence
        quality_issues = []
        missing_fields = []
        
        if not has_claims:
            quality_issues.append("No legal claims found")
            missing_fields.append("claims")
        if not has_summary:
            quality_issues.append("Legal summary missing")
            missing_fields.append("summary")
        if not has_statutes:
            quality_issues.append("No statutes cited")
        
        completeness_score = sum([has_claims, has_summary, has_statutes]) / 3.0
        
        return {
            "is_complete": is_complete,
            "confidence": confidence,
            "quality_issues": quality_issues,
            "completeness_score": completeness_score,
            "missing_fields": missing_fields
        }
    
    def _error_response(self, segment_type: str, error_msg: str, processing_time: float) -> Dict:
        """Return error response"""
        return {
            "success": False,
            "data": {},
            "validation": {
                "is_complete": False,
                "confidence": 0.0,
                "quality_issues": [f"Extraction failed: {error_msg}"],
                "completeness_score": 0.0,
                "missing_fields": [segment_type]
            },
            "processing_time": processing_time,
            "error": error_msg
        }