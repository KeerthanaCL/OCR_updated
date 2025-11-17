import easyocr
import cv2
from typing import Dict, Tuple, List
import logging
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

class EasyOCRService:
    """
    EasyOCR service for fallback when Tesseract confidence is low.
    Often performs better on complex layouts and handwritten text.
    """
    
    def __init__(self):
        self.reader = None
        self.confidence_threshold = settings.tesseract_confidence_threshold
        logger.info("EasyOCRService initialized (lazy loading)")
    
    def _load_reader(self):
        """Lazy load EasyOCR reader (heavy operation)"""
        if self.reader is None:
            logger.info("Loading EasyOCR reader...")
            self.reader = easyocr.Reader(['en'], gpu=False)
            logger.info("EasyOCR reader loaded successfully")
    
    def extract_text(self, image_path: str) -> Tuple[str, float, Dict]:
        """
        Extract text using EasyOCR.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Tuple of (text, confidence, metadata)
        """
        try:
            self._load_reader()
            
            logger.info(f"Starting EasyOCR for: {image_path}")
            
            # Extract text
            results = self.reader.readtext(image_path)
            
            # Combine all text and calculate average confidence
            extracted_texts = []
            confidences = []
            
            for (bbox, text, conf) in results:
                extracted_texts.append(text)
                confidences.append(conf * 100)  # Convert to percentage
            
            full_text = ' '.join(extracted_texts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            logger.info(
                f"EasyOCR completed: confidence={avg_confidence:.2f}%, "
                f"text_blocks={len(results)}"
            )
            
            metadata = {
                'engine': 'easyocr',
                'text_length': len(full_text),
                'word_count': len(full_text.split()),
                'average_confidence': round(avg_confidence, 2),
                'detected_text_blocks': len(results),
                'low_confidence_words': sum(1 for c in confidences if c < self.confidence_threshold),
                'page_count': 1
            }
            
            return full_text.strip(), avg_confidence, metadata
            
        except Exception as e:
            logger.error(f"EasyOCR failed: {str(e)}", exc_info=True)
            raise
    
    def extract_from_multiple_images(
        self,
        image_paths: List[str]
    ) -> Tuple[str, float, Dict]:
        """Extract text from multiple images"""
        self._load_reader()
        
        all_texts = []
        all_confidences = []
        total_blocks = 0
        
        logger.info(f"Processing {len(image_paths)} pages with EasyOCR")
        
        for page_num, image_path in enumerate(image_paths, start=1):
            try:
                text, confidence, metadata = self.extract_text(image_path)
                
                all_texts.append(f"\n{'='*60}\nPage {page_num}\n{'='*60}\n{text}")
                all_confidences.append(confidence)
                total_blocks += metadata['detected_text_blocks']
                
                logger.info(f"Page {page_num}: confidence={confidence:.2f}%")
                
            except Exception as e:
                logger.error(f"Failed to process page {page_num}: {str(e)}")
                all_texts.append(f"\n{'='*60}\nPage {page_num} (ERROR)\n{'='*60}\n")
                all_confidences.append(0.0)
        
        combined_text = "\n".join(all_texts)
        overall_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0
        
        metadata = {
            'engine': 'easyocr',
            'page_count': len(image_paths),
            'overall_confidence': round(overall_confidence, 2),
            'total_text_blocks': total_blocks,
            'successful_pages': sum(1 for c in all_confidences if c > 0)
        }
        
        return combined_text, overall_confidence, metadata