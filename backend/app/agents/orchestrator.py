import logging
from typing import Dict, Any
from app.agents.extraction_agent import ExtractionAgent
from app.agents.segment_agent import SegmentAgent

logger = logging.getLogger(__name__)

class OrchestratorAgent:
    """
    Main orchestrator that coordinates the document processing pipeline:
    1. ExtractionAgent: OCR and text extraction
    2. SegmentAgent: Segment text into references, medical, and legal
    """
    
    def __init__(self):
        self.extraction_agent = ExtractionAgent()
        self.segment_agent = SegmentAgent()
    
    async def process_document(
        self, 
        file_path: str,
        use_preprocessing: bool = True,
        correct_orientation: bool = True,
        force_trocr: bool = False,
        use_easyocr: bool = False,
        extract_appeals_first: bool = True
    ) -> Dict[str, Any]:
        """
        Complete document processing pipeline.
        
        Args:
            file_path: Path to the document file
            use_preprocessing: Whether to preprocess images
            correct_orientation: Auto-correct document orientation
            force_trocr: Force TrOCR usage
            use_easyocr: Force EasyOCR usage
            extract_appeals_first: Extract appeals section before segmentation
            
        Returns:
            Complete extraction and segmentation results
        """
        try:
            # Step 1: Extract text using OCR
            logger.info(f"Starting document processing for: {file_path}")
            extraction_result = self.extraction_agent.execute(
                file_path=file_path,
                use_preprocessing=use_preprocessing,
                correct_orientation=correct_orientation,
                force_trocr=force_trocr,
                use_easyocr=use_easyocr
            )
            
            extracted_text = extraction_result.get('text', '')
            
            if not extracted_text:
                logger.warning("No text extracted from document")
                return {
                    'success': False,
                    'error': 'No text could be extracted from the document',
                    'extraction': extraction_result
                }
            
            logger.info(f"Text extracted: {len(extracted_text)} characters")
            
            # Step 2: Segment the extracted text
            logger.info("Starting text segmentation...")
            segmentation_result = await self.segment_agent.extract_all_segments(
                text=extracted_text,
                extract_appeals_first=extract_appeals_first
            )
            
            # Step 3: Combine results
            logger.info("Document processing complete")
            return {
                'success': True,
                'extraction': extraction_result,
                'segmentation': segmentation_result,
                'metadata': {
                    'file_path': file_path,
                    'ocr_method': extraction_result.get('method_used'),
                    'confidence': extraction_result.get('confidence'),
                    'processing_time': extraction_result.get('processing_time'),
                    'text_length': len(extracted_text)
                }
            }
            
        except Exception as e:
            logger.error(f"Orchestration failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }