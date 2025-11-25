import logging
from typing import Dict, Any
from app.services.appeals_service import AppealsExtractionService
from app.services.openai_extraction_service import OpenAIExtractionService

logger = logging.getLogger(__name__)

class SegmentAgent:
    """
    Agent responsible for segmenting extracted text into:
    - References
    - Medical information
    - Legal information
    """
    
    def __init__(self):
        self.appeals_service = AppealsExtractionService()
        self.openai_service = OpenAIExtractionService()  # If using AI for extraction
    
    async def extract_references(self, text: str) -> Dict[str, Any]:
        """
        Extract and validate reference segments from text.
        
        Args:
            text: Extracted document text
            
        Returns:
            Dictionary containing reference information
        """
        try:
            logger.info("Extracting references...")
            result = await self.openai_service.extract_references(text)
            return result.dict()
        except Exception as e:
            logger.error(f"Reference extraction failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def extract_medical_segment(self, text: str) -> Dict[str, Any]:
        """
        Extract and validate medical information from text.
        
        Args:
            text: Extracted document text
            
        Returns:
            Dictionary containing medical segment information
        """
        try:
            logger.info("Extracting medical segments...")
            # Use your existing service logic
            result = await self.openai_service.extract_medical_context(text)
            return result.dict()
        except Exception as e:
            logger.error(f"Medical extraction failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def extract_legal_segment(self, text: str) -> Dict[str, Any]:
        """
        Extract and validate legal information from text.
        
        Args:
            text: Extracted document text
            
        Returns:
            Dictionary containing legal segment information
        """
        try:
            logger.info("Extracting legal segments...")
            # Use your existing service logic
            result = await self.openai_service.extract_legal_context(text)
            return result.dict()
        except Exception as e:
            logger.error(f"Legal extraction failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
        
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
        # Optionally extract appeals section first
        if extract_appeals_first:
            appeals_result = self.extract_appeals_section(text)
            if appeals_result['found']:
                text = appeals_result['appeals_text']
                logger.info(f"Using extracted appeals section: {len(text)} chars")
        
        return {
            'references': await self.extract_references(text),
            'medical': await self.extract_medical_segment(text),
            'legal': await self.extract_legal_segment(text)
        }