import re
import logging
from typing import Tuple, Optional
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

class AppealsExtractionService:
    """
    Service for detecting and extracting appeals sections from documents.
    Segmentation is now handled by OpenAI in openai_extraction_service.py
    """
    
    def __init__(self):
        # Common appeals section headers
        self.appeals_keywords = [
            r"appeal",
            r"reconsideration",
            r"grievance",
            r"dispute",
            r"review request",
            r"denial review",
            r"adverse decision"
        ]
    
    def extract_appeals_section(self, full_text: str) -> Tuple[Optional[str], bool]:
        """
        Extract the appeals section from full document text.
        
        Args:
            full_text: Complete document text
            
        Returns:
            Tuple of (appeals_text, found)
            - appeals_text: The extracted appeals section or None
            - found: Boolean indicating if appeals section was detected
        """
        try:
            text_lower = full_text.lower()
            
            # Look for appeals section headers
            for keyword in self.appeals_keywords:
                pattern = rf"(?:^|\n)(.{{0,50}}{keyword}.{{0,50}})\n"
                matches = re.finditer(pattern, text_lower, re.IGNORECASE | re.MULTILINE)
                
                for match in matches:
                    start_pos = match.start()
                    appeals_text = full_text[start_pos:]
                    
                    # Try to find where appeals section ends
                    section_end_patterns = [
                        r"\n(?:attachments?|exhibits?|appendix|signature)\s*\n",
                        r"\n\s*---+\s*\n",
                        r"\n{3,}"  # Multiple blank lines
                    ]
                    
                    for end_pattern in section_end_patterns:
                        end_match = re.search(end_pattern, appeals_text, re.IGNORECASE)
                        if end_match:
                            appeals_text = appeals_text[:end_match.start()]
                            break
                    
                    # Validate minimum length
                    min_length = getattr(settings, 'appeals_min_length', 100)
                    if len(appeals_text.strip()) >= min_length:
                        logger.info(f"Appeals section found: {len(appeals_text)} chars")
                        return appeals_text.strip(), True
            
            # Check if entire document is an appeal
            appeal_score = sum(1 for keyword in self.appeals_keywords if keyword in text_lower)
            
            min_length = getattr(settings, 'appeals_min_length', 100)
            if appeal_score >= 2 and len(full_text) >= min_length:
                logger.info("Entire document appears to be an appeal")
                return full_text, True
            
            logger.warning("No appeals section found in document")
            return None, False
            
        except Exception as e:
            logger.error(f"Error extracting appeals section: {e}", exc_info=True)
            return None, False