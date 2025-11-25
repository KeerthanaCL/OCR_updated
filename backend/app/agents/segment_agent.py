import logging
from typing import Dict, Any
import time

from app.agents.base_agent import BaseAgent
from app.services.appeals_service import AppealsExtractionService
from app.services.openai_extraction_service import OpenAIExtractionService

logger = logging.getLogger(__name__)

class SegmentAgent(BaseAgent):
    """
    Intelligent agent for document segmentation and validation.
    
    Agent Capabilities:
    - Autonomous quality assessment
    - Retry logic with improved prompts
    - Learning from segmentation patterns
    - Confidence-based validation
    
    Goals:
    - Extract complete, accurate segments
    - Validate segment quality
    - Minimize extraction errors
    """
    
    def __init__(self):
        super().__init__("SegmentationAgent")

        self.appeals_service = AppealsExtractionService()
        self.openai_service = OpenAIExtractionService()  # If using AI for extraction

        # Agent Goals
        self.goals = {
            'min_reference_count': 3,
            'min_medical_conditions': 1,
            'require_validation': True,
            'max_retry_attempts': 2
        }
        
        # Agent State
        self.state['performance_metrics'] = {
            'references': {'success_rate': 0.0, 'avg_count': 0.0},
            'medical': {'success_rate': 0.0, 'avg_conditions': 0.0},
            'legal': {'success_rate': 0.0, 'avg_claims': 0.0}
        }
        
        logger.info(f"{self.agent_name} initialized")

    def assess_text_quality(self, text: str) -> Dict[str, Any]:
        """Agent assesses text quality before segmentation"""
        assessment = {
            'length': len(text),
            'has_content': len(text.strip()) > 100,
            'word_count': len(text.split()),
            'avg_word_length': sum(len(word) for word in text.split()) / max(len(text.split()), 1),
            'quality': 'good' if len(text) > 500 else 'poor'
        }
        
        logger.info(f"Text Assessment: {assessment['word_count']} words, quality={assessment['quality']}")
        return assessment
    
    async def extract_references(self, text: str, retry_count: int = 0) -> Dict[str, Any]:
        """
        Extract and validate reference segments from text.
        
        Args:
            text: Extracted document text
            
        Returns:
            Dictionary containing reference information
        """
        start_time = time.time()
        try:
            # Assess text quality
            quality = self.assess_text_quality(text)
            
            if not quality['has_content']:
                logger.warning("Insufficient text for reference extraction")
                return {
                    'success': False,
                    'error': 'Insufficient text content',
                    'references': []
                }
            
            logger.info(f"Extracting references (attempt {retry_count + 1})...")
            
            # Execute extraction
            result = await self.openai_service.extract_references(text)
            
            # Validate result
            validation = self._validate_references(result)
            
            if not validation['is_valid'] and retry_count < self.goals['max_retry_attempts']:
                logger.warning(f"Validation failed: {validation['reason']}, retrying...")
                return await self.extract_references(text, retry_count + 1)
            
            # Record execution
            success = validation['is_valid']
            self.record_execution('extract_references', {
                'reference_count': len(result.references) if hasattr(result, 'references') else 0,
                'validation': validation,
                'processing_time': time.time() - start_time
            }, success)
            
            # Learn from execution
            self.learn_from_execution('references', {
                'count': len(result.references) if hasattr(result, 'references') else 0,
                'success': success
            })
            
            return result.dict()
            
        except Exception as e:
            logger.error(f"Reference extraction failed: {str(e)}")
            self.record_execution('extract_references', {'error': str(e)}, False)
            return {
                'success': False,
                'error': str(e)
            }
    
    async def extract_medical_segment(self, text: str, retry_count: int = 0) -> Dict[str, Any]:
        """
        Extract and validate medical information from text.
        
        Args:
            text: Extracted document text
            
        Returns:
            Dictionary containing medical segment information
        """
        start_time = time.time()
        
        try:
            quality = self.assess_text_quality(text)
            
            if not quality['has_content']:
                return {
                    'success': False,
                    'error': 'Insufficient text content'
                }
            
            logger.info(f"Extracting medical context (attempt {retry_count + 1})...")
            
            result = await self.openai_service.extract_medical_context(text)
            
            # Validate
            validation = self._validate_medical(result)
            
            if not validation['is_valid'] and retry_count < self.goals['max_retry_attempts']:
                logger.warning(f"Medical validation failed, retrying...")
                return await self.extract_medical_segment(text, retry_count + 1)
            
            success = validation['is_valid']
            self.record_execution('extract_medical', {
                'conditions_count': len(result.conditions) if hasattr(result, 'conditions') else 0,
                'medications_count': len(result.medications) if hasattr(result, 'medications') else 0,
                'processing_time': time.time() - start_time
            }, success)
            
            return result.dict()
            
        except Exception as e:
            logger.error(f"Medical extraction failed: {str(e)}")
            self.record_execution('extract_medical', {'error': str(e)}, False)
            return {
                'success': False,
                'error': str(e)
            }
        
    async def extract_legal_segment(self, text: str, retry_count: int = 0) -> Dict[str, Any]:
        """
        Extract and validate legal information from text.
        
        Args:
            text: Extracted document text
            
        Returns:
            Dictionary containing legal segment information
        """
        start_time = time.time()
        
        try:
            quality = self.assess_text_quality(text)
            
            if not quality['has_content']:
                return {
                    'success': False,
                    'error': 'Insufficient text content'
                }
            
            logger.info(f"Extracting legal context (attempt {retry_count + 1})...")
            
            result = await self.openai_service.extract_legal_context(text)
            
            # Validate
            validation = self._validate_legal(result)
            
            if not validation['is_valid'] and retry_count < self.goals['max_retry_attempts']:
                logger.warning(f"Legal validation failed, retrying...")
                return await self.extract_legal_segment(text, retry_count + 1)
            
            success = validation['is_valid']
            self.record_execution('extract_legal', {
                'claims_count': len(result.claims) if hasattr(result, 'claims') else 0,
                'processing_time': time.time() - start_time
            }, success)
            
            return result.dict()
            
        except Exception as e:
            logger.error(f"Legal extraction failed: {str(e)}")
            self.record_execution('extract_legal', {'error': str(e)}, False)
            return {
                'success': False,
                'error': str(e)
            }
        
    def _validate_references(self, result: Any) -> Dict[str, Any]:
        """Validate reference extraction results"""
        if not hasattr(result, 'references'):
            return {'is_valid': False, 'reason': 'No references field'}
        
        ref_count = len(result.references)
        
        if ref_count < self.goals['min_reference_count']:
            return {
                'is_valid': False,
                'reason': f'Too few references: {ref_count} < {self.goals["min_reference_count"]}'
            }
        
        return {'is_valid': True, 'reference_count': ref_count}
    
    def _validate_medical(self, result: Any) -> Dict[str, Any]:
        """Validate medical extraction results"""
        if not hasattr(result, 'conditions'):
            return {'is_valid': False, 'reason': 'No conditions field'}
        
        condition_count = len(result.conditions)
        
        if condition_count < self.goals['min_medical_conditions']:
            return {
                'is_valid': False,
                'reason': f'No medical conditions found'
            }
        
        return {'is_valid': True, 'condition_count': condition_count}
    
    def _validate_legal(self, result: Any) -> Dict[str, Any]:
        """Validate legal extraction results"""
        if not hasattr(result, 'claims'):
            return {'is_valid': False, 'reason': 'No claims field'}
        
        return {'is_valid': True, 'claim_count': len(result.claims)}
        
    def extract_appeals_section(self, text: str) -> Dict[str, Any]:
        """
        Extract appeals section from full document text.
        Useful for pre-processing before segment extraction.
        """
        try:
            appeals_text, found = self.appeals_service.extract_appeals_section(text)
            return {
                'success': found,
                'appeals_text': appeals_text,
                'found': found
            }
        except Exception as e:
            logger.error(f"Appeals section extraction failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'found': False
            }
    
    async def extract_all_segments(self, text: str, extract_appeals_first: bool = False) -> Dict[str, Any]:
        """
        Extract all segments (references, medical, legal) in one call.
        
        Args:
            text: Extracted document text
            extract_appeals_first: Whether to extract appeals section before segmentation
        """
        logger.info(f"{self.agent_name} executing full segmentation")

        # Optionally extract appeals section first
        working_text = text
        if extract_appeals_first:
            appeals_result = self.extract_appeals_section(text)
            if appeals_result['found']:
                working_text = appeals_result['appeals_text']
                logger.info(f"Using appeals section: {len(working_text)} chars")
        
        # Execute all extractions
        references = await self.extract_references(working_text)
        medical = await self.extract_medical_segment(working_text)
        legal = await self.extract_legal_segment(working_text)
        
        # Aggregate success
        all_success = all([
            references.get('success', True),
            medical.get('success', True),
            legal.get('success', True)
        ])
        
        logger.info(f"Segmentation complete: success={all_success}")
        
        return {
            'references': references,
            'medical': medical,
            'legal': legal,
            'overall_success': all_success
        }