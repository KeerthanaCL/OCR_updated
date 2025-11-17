import time
import logging
from typing import Dict, Any, List
from pathlib import Path
from app.services.tesseract_service import TesseractService
from app.services.trocr_service import TrOCRService
from app.services.easyocr_service import EasyOCRService
from app.services.pdf_converter import PDFConverter
from app.services.orientation_detector import OrientationDetector
from app.models import OCRMethod

logger = logging.getLogger(__name__)

class ExtractionAgent:
    """
    Intelligent agent for document text extraction.
    Uses Tesseract first, then TrOCR if confidence is low.
    """
    
    def __init__(self):
        self.tesseract = TesseractService()
        self.easyocr = None
        self.trocr = None  # Lazy loading
        self.pdf_converter = PDFConverter(method="pymupdf")
        self.orientation_detector = OrientationDetector()

    def _load_easyocr(self):
        """Lazy load EasyOCR"""
        if self.easyocr is None:
            logger.info("Loading EasyOCR...")
            self.easyocr = EasyOCRService()
    
    def _load_trocr(self):
        """Lazy load TrOCR model (heavy operation)"""
        if self.trocr is None:
            logger.info("Loading TrOCR model...")
            self.trocr = TrOCRService()

    def _is_pdf(self, file_path: str) -> bool:
        """Check if file is a PDF"""
        return Path(file_path).suffix.lower() == '.pdf'
    
    def _correct_image_orientation(self, image_path: str) -> Dict[str, Any]:
        """
        Detect and correct image orientation before OCR.
        Critical for landscape insurance forms.
        """
        try:
            corrected_image, orientation_info = self.orientation_detector.detect_and_correct_orientation(image_path)
            
            # Save corrected image if rotation was applied
            if orientation_info['was_corrected']:
                corrected_path = str(Path(image_path).with_suffix('')) + '_corrected.png'
                corrected_image.save(corrected_path)
                logger.info(f"Saved corrected image to {corrected_path}")
                return {'path': corrected_path, 'info': orientation_info}
            
            return {'path': image_path, 'info': orientation_info}
            
        except Exception as e:
            logger.warning(f"Orientation correction failed: {str(e)}, using original")
            return {'path': image_path, 'info': {'was_corrected': False, 'error': str(e)}}
    
    def execute(
        self, 
        file_path: str,
        use_preprocessing: bool = True,
        force_trocr: bool = False,
        use_easyocr: bool = False,
        correct_orientation: bool = True
    ) -> Dict[str, Any]:
        """
        Execute extraction with automatic PDF handling and orientation correction.
        
        Workflow for PDF:
        1. Convert PDF pages to images
        2. Process each image with OCR
        3. Combine results
        4. Clean up temporary images
        
        Workflow for Images:
        1. Direct OCR processing
        
        Args:
            file_path: Path to document (PDF or image)
            use_preprocessing: Whether to preprocess images
            force_trocr: Force TrOCR usage
            correct_orientation: Auto-correct landscape/rotated documents
            
        Returns:
            Dictionary with extraction results
        """
        start_time = time.time()
        
        try:
            # Check if PDF
            if self._is_pdf(file_path):
                return self._process_pdf(
                    file_path, use_preprocessing, force_trocr, 
                    use_easyocr, correct_orientation, start_time
                )
            else:
                return self._process_image(
                    file_path, use_preprocessing, force_trocr,
                    use_easyocr, correct_orientation, start_time
                )
            
        except Exception as e:
            logger.error(f"Extraction agent failed: {str(e)}")
            raise

    def _process_pdf(
        self,
        pdf_path: str,
        use_preprocessing: bool,
        force_trocr: bool,
        use_easyocr: bool,
        correct_orientation: bool,
        start_time: float
    ) -> Dict[str, Any]:
        """Process PDF document"""
        
        # Convert PDF to images
        output_dir = Path(pdf_path).parent / "temp_images"
        image_paths = self.pdf_converter.convert(pdf_path, str(output_dir))
        
        logger.info(f"PDF converted to {len(image_paths)} images")

        # Correct orientation for each page
        if correct_orientation:
            corrected_paths = []
            orientation_results = []
            
            for img_path in image_paths:
                result = self._correct_image_orientation(img_path)
                corrected_paths.append(result['path'])
                orientation_results.append(result['info'])
            
            image_paths = corrected_paths
        
        try:
            if force_trocr:
                result = self._extract_with_trocr_multi(
                    image_paths, start_time
                )
            elif use_easyocr:
                return self._extract_with_easyocr_multi(image_paths, start_time)
            
            # Try Tesseract first
            text, confidence, metadata = self.tesseract.extract_from_multiple_images(
                image_paths, preprocess=use_preprocessing
            )
            
            # Check if fallback needed
            if confidence < self.tesseract.confidence_threshold:
                logger.warning(
                    f"Tesseract confidence ({confidence:.2f}%) below threshold, "
                    f"trying EasyOCR"
                )
                try:
                    return self._extract_with_easyocr_multi(image_paths, start_time)
                except Exception as e:
                    logger.warning(f"EasyOCR failed: {str(e)}, trying TrOCR")
                    return self._extract_with_trocr_multi(image_paths, start_time)
            
            # Return Tesseract result
            processing_time = time.time() - start_time
            return {
                'text': text,
                'confidence': confidence,
                'method_used': 'tesseract',
                'pages': metadata['page_count'],
                'processing_time': processing_time,
                'metadata': metadata
            }
            
        finally:
            self._cleanup_temp_images(output_dir)

    def _process_image(
        self,
        image_path: str,
        use_preprocessing: bool,
        force_trocr: bool,
        use_easyocr: bool,
        correct_orientation: bool,
        start_time: float
    ) -> Dict[str, Any]:
        """Process single image with orientation correction"""

        # Correct orientation if needed
        if correct_orientation:
            orientation_result = self._correct_image_orientation(image_path)
            image_path = orientation_result['path']
            orientation_info = orientation_result['info']
        else:
            orientation_info = None
        
        if force_trocr:
            result = self._extract_with_trocr_single(image_path, start_time)
        elif use_easyocr:
            return self._extract_with_easyocr_single(image_path, start_time)
        
        # Try Tesseract
        text, confidence, metadata = self.tesseract.extract_text_with_confidence(
            image_path, preprocess=use_preprocessing
        )
        
        # Check fallback
        if confidence < self.tesseract.confidence_threshold:
            logger.warning(f"Low confidence ({confidence:.2f}%), trying EasyOCR")
            try:
                return self._extract_with_easyocr_single(image_path, start_time)
            except Exception as e:
                logger.warning(f"EasyOCR failed, trying TrOCR")
                return self._extract_with_trocr_single(image_path, start_time)
        
        processing_time = time.time() - start_time
        return {
            'text': text,
            'confidence': confidence,
            'method_used': 'tesseract',
            'pages': 1,
            'processing_time': processing_time,
            'metadata': metadata
        }
    
    def _extract_with_trocr_multi(
        self,
        image_paths: List[str],
        start_time: float
    ) -> Dict[str, Any]:
        """Extract using TrOCR for multiple images"""
        self._load_trocr()
        
        text, metadata = self.trocr.extract_from_multiple_images(image_paths)
        processing_time = time.time() - start_time
        
        return {
            'text': text,
            'confidence': 95.0,
            'method_used': 'trocr',
            'pages': metadata['page_count'],
            'processing_time': processing_time,
            'metadata': metadata
        }
    
    def _extract_with_easyocr_single(
        self, image_path: str, start_time: float
    ) -> Dict[str, Any]:
        """Extract using EasyOCR for single image"""
        self._load_easyocr()
        
        text, confidence, metadata = self.easyocr.extract_text(image_path)
        processing_time = time.time() - start_time
        
        return {
            'text': text,
            'confidence': confidence,
            'method_used': 'easyocr',
            'pages': 1,
            'processing_time': processing_time,
            'metadata': metadata
        }
    
    def _extract_with_easyocr_multi(
        self, image_paths: List[str], start_time: float
    ) -> Dict[str, Any]:
        """Extract using EasyOCR for multiple images"""
        self._load_easyocr()
        
        text, confidence, metadata = self.easyocr.extract_from_multiple_images(image_paths)
        processing_time = time.time() - start_time
        
        return {
            'text': text,
            'confidence': confidence,
            'method_used': 'easyocr',
            'pages': len(image_paths),
            'processing_time': processing_time,
            'metadata': metadata
        }
    
    def _extract_with_trocr_single(
        self,
        image_path: str,
        start_time: float
    ) -> Dict[str, Any]:
        """Extract using TrOCR for single image"""
        self._load_trocr()
        
        text, metadata = self.trocr.extract_text(image_path)
        processing_time = time.time() - start_time
        
        return {
            'text': text,
            'confidence': 95.0,
            'method_used': 'tesseract',
            'pages': 1,
            'processing_time': processing_time,
            'metadata': metadata
        }
    
    def _cleanup_temp_images(self, temp_dir: Path):
        """Clean up temporary image files"""
        try:
            if temp_dir.exists():
                import shutil
                shutil.rmtree(temp_dir)
                logger.info("Cleaned up temporary images")
        except Exception as e:
            logger.warning(f"Failed to cleanup temp images: {str(e)}")