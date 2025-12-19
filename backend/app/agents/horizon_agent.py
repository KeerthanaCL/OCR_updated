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
            # Determine overall validation status
            is_validated = validation['is_complete'] and validation['confidence'] >= 0.7
            validation_status = self._get_validation_status(validation)
            
            processing_time = time.time() - start_time
            
            return {
                "success": True,
                "is_validated": is_validated,  # Clear boolean flag
                "validation_status": validation_status,  # "validated", "partial", "failed"
                "data": {
                    "research_papers": [r.dict() for r in result.research_papers], 
                    "online_resources": [r.dict() for r in result.online_resources],
                    "summary": result.summary,
                    "total_count": len(result.research_papers) + len(result.online_resources)
                },
                "validation": validation,
                "validation_summary": self._get_validation_summary(validation),
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
            # Determine validation status
            is_validated = validation['is_complete'] and validation['confidence'] >= 0.7
            validation_status = self._get_validation_status(validation)
            
            processing_time = time.time() - start_time
            
            return {
                "success": True,
                "is_validated": is_validated,  # Clear boolean flag
                "validation_status": validation_status,  # "validated", "partial", "failed"
                "data": {
                    "patient_name": result.patient_name,
                    "conditions": [c.dict() for c in result.conditions],
                    "medications": [m.dict() for m in result.medications],
                    "medical_history": result.medical_history,
                    "medical_necessity_argument": result.medical_necessity_argument,
                    "providers_mentioned": result.providers_mentioned
                },
                "validation": validation,
                "validation_summary": self._get_validation_summary(validation),
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
            # Determine validation status
            is_validated = validation['is_complete'] and validation['confidence'] >= 0.7
            validation_status = self._get_validation_status(validation)
            
            processing_time = time.time() - start_time
            
            return {
                "success": True,
                "is_validated": is_validated,  # Clear boolean flag
                "validation_status": validation_status,  # "validated", "partial", "failed"
                "data": {
                    "appeal_type": result.appeal_type,
                    "legal_claims": [c.dict() for c in result.legal_claims],
                    "legal_summary": result.legal_summary,
                    "statutes_cited": result.statutes_cited,
                    "deadlines_mentioned": result.deadlines_mentioned
                },
                "validation": validation,
                "validation_summary": self._get_validation_summary(validation),
                "processing_time": processing_time
            }
            
        except Exception as e:
            logger.error(f"Legal extraction failed: {e}", exc_info=True)
            return self._error_response("legal", str(e), time.time() - start_time)
        
    def _get_validation_status(self, validation: Dict) -> str:
        """
        Determine validation status: 'validated', 'partial', or 'failed'
        
        Rules:
        - validated: is_complete=True, confidence>=0.7, completeness_score>=0.8
        - partial: is_complete=True but lower scores, or some quality issues
        - failed: is_complete=False or critical issues
        """
        is_complete = validation.get('is_complete', False)
        confidence = validation.get('confidence', 0.0)
        completeness_score = validation.get('completeness_score', 0.0)
        quality_issues = validation.get('quality_issues', [])
        
        # Failed validation
        if not is_complete:
            return "failed"
        
        if confidence < 0.5 or completeness_score < 0.5:
            return "failed"
        
        # Full validation
        if is_complete and confidence >= 0.7 and completeness_score >= 0.8 and len(quality_issues) == 0:
            return "validated"
        
        # Partial validation (acceptable but with warnings)
        return "partial"


    def _get_validation_summary(self, validation: Dict) -> str:
        """
        Generate human-readable validation summary
        """
        status = self._get_validation_status(validation)
        completeness = validation.get('completeness_score', 0.0) * 100
        confidence = validation.get('confidence', 0.0)
        quality_issues = validation.get('quality_issues', [])
        
        if status == "validated":
            return f"Validation passed: {completeness:.0f}% complete, {confidence:.0f}% confidence"
        
        elif status == "partial":
            issues_summary = f", {len(quality_issues)} minor issues" if quality_issues else ""
            return f"Partial validation: {completeness:.0f}% complete, {confidence:.0f}% confidence{issues_summary}"
        
        else:  # failed
            issues_summary = f": {', '.join(quality_issues[:2])}" if quality_issues else ""
            return f"Validation failed: {completeness:.0f}% complete{issues_summary}"
    
    def _validate_references(self, result) -> Dict:
        """Enhanced validation with clear pass/fail logic"""
    
        research_papers = result.research_papers
        online_resources = result.online_resources
        
        # Detailed checks
        quality_issues = []
        missing_fields = []
        
        # Check research papers
        valid_papers = 0
        for paper in research_papers:
            content = paper.content.lower()
            # Check for DOI, PMID, or journal patterns
            if any(keyword in content for keyword in ['doi:', 'pmid:', 'journal', 'vol.', 'issue', 'pubmed']):
                valid_papers += 1
            else:
                quality_issues.append(f"Paper missing identifiers: {paper.content[:50]}")
        
        # Check URLs
        valid_urls = 0
        for resource in online_resources:
            content = resource.content
            if content.startswith(('http://', 'https://', 'www.')):
                valid_urls += 1
            else:
                quality_issues.append(f"Invalid URL format: {content}")
        
        # Calculate completeness
        total_refs = len(research_papers) + len(online_resources)
        valid_refs = valid_papers + valid_urls
        
        completeness_score = valid_refs / total_refs if total_refs > 0 else 0.0
        is_complete = completeness_score >= 0.8  # 80% threshold
        
        if total_refs == 0:
            quality_issues.append("No references found")
            missing_fields.append("references")
        
        if valid_papers < len(research_papers):
            quality_issues.append(f"Only {valid_papers}/{len(research_papers)} papers have valid identifiers")
        
        if valid_urls < len(online_resources):
            quality_issues.append(f"Only {valid_urls}/{len(online_resources)} URLs are valid")
        
        # Determine if complete (at least 80% of items are valid)
        is_complete = completeness_score >= 0.8 and total_refs > 0
        
        return {
            "is_complete": is_complete,
            "confidence": result.confidence,
            "quality_issues": quality_issues,
            "completeness_score": completeness_score,
            "missing_fields": missing_fields,
            "valid_items": valid_refs,
            "total_items": total_refs,
            "validation_criteria": {
                "min_completeness_required": 0.8,
                "min_confidence_required": 0.7,
                "achieved_completeness": completeness_score,
                "achieved_confidence": result.confidence
            }
        }
    
    def _validate_medical(self, result) -> Dict:
        """Enhanced medical validation"""
    
        has_conditions = len(result.conditions) > 0
        has_medications = len(result.medications) > 0
        has_history = bool(result.medical_history and len(result.medical_history) > 20)
        has_necessity = bool(result.medical_necessity_argument and len(result.medical_necessity_argument) > 20)
        
        quality_issues = []
        missing_fields = []
        
        # Detailed checks
        if not has_conditions:
            quality_issues.append("No medical conditions found")
            missing_fields.append("conditions")
        else:
            # Check condition quality
            conditions_with_details = sum(1 for c in result.conditions if c.severity or c.treatment)
            if conditions_with_details < len(result.conditions) / 2:
                quality_issues.append(f"Only {conditions_with_details}/{len(result.conditions)} conditions have details")
        
        if not has_medications:
            quality_issues.append("No medications found")
            missing_fields.append("medications")
        else:
            # Check medication quality
            meds_with_details = sum(1 for m in result.medications if m.dosage or m.frequency)
            if meds_with_details < len(result.medications) / 2:
                quality_issues.append(f"Only {meds_with_details}/{len(result.medications)} medications have dosage/frequency")
        
        if not has_history:
            quality_issues.append("Medical history too brief or missing")
            missing_fields.append("history")
        
        if not has_necessity:
            quality_issues.append("Medical necessity argument missing or too brief")
            missing_fields.append("necessity_argument")
        
        # Completeness scoring
        completeness_factors = [has_conditions, has_medications, has_history, has_necessity]
        completeness_score = sum(completeness_factors) / len(completeness_factors)
        
        is_complete = has_conditions and has_medications  # Minimum requirement
        
        return {
            "is_complete": is_complete,
            "confidence": result.confidence,
            "quality_issues": quality_issues,
            "completeness_score": completeness_score,
            "missing_fields": missing_fields,
            "stats": {
                "conditions_count": len(result.conditions),
                "medications_count": len(result.medications),
                "providers_count": len(result.providers_mentioned),
                "has_history": has_history,
                "has_necessity_argument": has_necessity
            },
            "validation_criteria": {
                "min_completeness_required": 0.5,  # 50% (at least conditions + medications)
                "min_confidence_required": 0.7,
                "achieved_completeness": completeness_score,
                "achieved_confidence": result.confidence
            }
        }
    
    def _validate_legal(self, result) -> Dict:
        """Enhanced legal validation"""
    
        has_claims = len(result.legal_claims) > 0
        has_summary = bool(result.legal_summary and len(result.legal_summary) > 20)
        has_statutes = len(result.statutes_cited) > 0
        has_deadlines = len(result.deadlines_mentioned) > 0
        
        quality_issues = []
        missing_fields = []
        
        if not has_claims:
            quality_issues.append("No legal claims found")
            missing_fields.append("claims")
        else:
            # Check claim quality
            claims_with_statutes = sum(1 for c in result.legal_claims if c.relevant_statute)
            if claims_with_statutes == 0 and has_statutes:
                quality_issues.append("Claims don't reference the cited statutes")
        
        if not has_summary:
            quality_issues.append("Legal summary too brief or missing")
            missing_fields.append("summary")
        
        if not has_statutes:
            quality_issues.append("No statutes or regulations cited")
        
        # if not has_deadlines:
        #     quality_issues.append("No deadlines mentioned (may be normal)")
        
        # Completeness scoring
        completeness_factors = [has_claims, has_summary, has_statutes]
        completeness_score = sum(completeness_factors) / len(completeness_factors)
        
        is_complete = has_claims and has_summary
        
        return {
            "is_complete": is_complete,
            "confidence": result.confidence,
            "quality_issues": quality_issues,
            "completeness_score": completeness_score,
            "missing_fields": missing_fields,
            "stats": {
                "claims_count": len(result.legal_claims),
                "statutes_count": len(result.statutes_cited),
                "deadlines_count": len(result.deadlines_mentioned),
                "has_summary": has_summary
            },
            "validation_criteria": {
                "min_completeness_required": 0.67,  # 67% (at least 2 of 3 fields)
                "min_confidence_required": 0.7,
                "achieved_completeness": completeness_score,
                "achieved_confidence": result.confidence
            }
        }

    def _error_response(self, segment_type: str, error_msg: str, processing_time: float) -> Dict:
        """Return error response"""
        return {
            "success": False,
            "is_validated": False,
            "validation_status": "failed",
            "data": {},
            "validation": {
                "is_complete": False,
                "confidence": 0.0,
                "quality_issues": [f"Extraction failed: {error_msg}"],
                "completeness_score": 0.0,
                "missing_fields": [segment_type]
            },
            "validation_summary": f"Extraction failed: {error_msg}",
            "processing_time": processing_time,
            "error": error_msg
        }