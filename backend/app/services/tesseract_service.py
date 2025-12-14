import pytesseract
from PIL import Image
import numpy as np
from typing import Dict, Tuple, List
import logging
import cv2
import os
import shutil
from app.config import get_settings
from app.services.preprocessing import ImagePreprocessor
from app.services.region_detector import RegionDetector

logger = logging.getLogger(__name__)
settings = get_settings()

class TesseractService:
    """
    Enhanced Tesseract OCR service with optimized configuration and region-based processing.
    Uses proven configuration for insurance document extraction.
    """
    
    def __init__(self):
        self.confidence_threshold = settings.tesseract_confidence_threshold
        self.preprocessor = ImagePreprocessor()
        self.region_detector = RegionDetector()
        
        # Try to find Tesseract executable
        tesseract_paths = [
            r"C:\Program Files\tesseract.exe",  # Direct installation
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            r"C:\Tesseract-OCR\tesseract.exe",
            shutil.which("tesseract")  # Check if in PATH
        ]
        
        tesseract_found = False
        for path in tesseract_paths:
            if path and os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                logger.info(f"Tesseract found at: {path}")
                
                # Set TESSDATA_PREFIX to the tessdata directory itself
                tesseract_dir = os.path.dirname(path)
                tessdata_dir = os.path.join(tesseract_dir, 'tessdata')
                
                # Important: Set to the tessdata directory
                os.environ['TESSDATA_PREFIX'] = tessdata_dir + os.sep
                logger.info(f"Set TESSDATA_PREFIX to: {tessdata_dir}")
                
                # Verify files exist
                eng_data = os.path.join(tessdata_dir, 'eng.traineddata')
                osd_data = os.path.join(tessdata_dir, 'osd.traineddata')
                
                if os.path.exists(eng_data):
                    logger.info(f"eng.traineddata found at: {eng_data}")
                else:
                    logger.error(f"eng.traineddata NOT found at: {eng_data}")
                    
                if os.path.exists(osd_data):
                    logger.info(f"osd.traineddata found at: {osd_data}")
                else:
                    logger.warning(f"osd.traineddata NOT found at: {osd_data}")
                
                # Add Tesseract directory to PATH
                if tesseract_dir not in os.environ.get('PATH', ''):
                    os.environ['PATH'] = tesseract_dir + os.pathsep + os.environ.get('PATH', '')
                    logger.info(f"Added {tesseract_dir} to PATH")
                
                tesseract_found = True
                break
        
        if not tesseract_found:
            logger.warning("Tesseract not found. Please install from: https://github.com/UB-Mannheim/tesseract/wiki")
            logger.warning("After installation, restart the OCR backend server")

        # Optimized Tesseract configuration
        # --oem 3: Use default OCR Engine Mode (LSTM + Legacy)
        # --psm 6: Assume a single uniform block of text
        self.tesseract_config = r'--oem 3 --psm 6'

        # Multiple PSM configurations for different content types
        self.psm_configs = {
            'default': r'--oem 3 --psm 6',      # Single uniform block
            'sparse': r'--oem 3 --psm 11',      # Sparse text
            'single_line': r'--oem 3 --psm 7',  # Single text line
            'single_word': r'--oem 3 --psm 8',  # Single word
            'auto': r'--oem 3 --psm 3',         # Auto page segmentation
        }
        
        logger.info(f"TesseractService initialized with config: {self.tesseract_config}")
        
    def extract_text_region_based(
        self, 
        image_path: str, 
        preprocess: bool = True
    ) -> Tuple[str, float, Dict]:
        """
        Extract text by processing each region separately.
        Significantly improves accuracy for complex layouts.
        
        Args:
            image_path: Path to image file
            preprocess: Whether to preprocess
            
        Returns:
            Tuple of (text, confidence, metadata)
        """
        try:
            logger.info(f"Starting region-based OCR for: {image_path}")
            
            # Detect text regions
            regions = self.region_detector.detect_text_regions(image_path)
            
            if not regions:
                logger.warning("No regions detected, falling back to full-page OCR")
                return self.extract_text_with_confidence(image_path, preprocess)
            
            # Read original image
            image = cv2.imread(image_path)
            
            # Process each region
            all_texts = []
            all_confidences = []
            region_details = []
            
            for region in regions:
                # Extract region
                x, y, w, h = region['x'], region['y'], region['width'], region['height']
                region_img = image[y:y+h, x:x+w]
                
                # Choose appropriate PSM based on region properties
                psm_config = self._select_psm_for_region(region)
                
                # Preprocess region if needed
                if preprocess:
                    region_img = self._preprocess_region(region_img)
                
                # Extract text from region
                try:
                    text = pytesseract.image_to_string(
                        region_img, 
                        lang=settings.tesseract_lang,
                        config=psm_config
                    )
                    
                    # Get confidence
                    data = pytesseract.image_to_data(
                        region_img,
                        lang=settings.tesseract_lang,
                        config=psm_config,
                        output_type=pytesseract.Output.DICT
                    )
                    
                    confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                    region_confidence = sum(confidences) / len(confidences) if confidences else 0.0
                    
                    if text.strip():  # Only add non-empty text
                        all_texts.append(text.strip())
                        all_confidences.append(region_confidence)
                        
                        region_details.append({
                            'region_id': region['id'],
                            'text': text.strip()[:50] + '...' if len(text) > 50 else text.strip(),
                            'confidence': round(region_confidence, 2),
                            'words': len(text.split()),
                            'position': {'x': x, 'y': y, 'w': w, 'h': h}
                        })
                    
                except Exception as e:
                    logger.warning(f"Failed to process region {region['id']}: {str(e)}")
                    continue
            
            # Combine results
            combined_text = '\n'.join(all_texts)
            overall_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0
            
            metadata = {
                'word_count': len(combined_text.split()),
                'page_count': 1,
                'average_confidence': round(overall_confidence, 2),
                'min_confidence': round(min(all_confidences), 2) if all_confidences else 0.0,
                'max_confidence': round(max(all_confidences), 2) if all_confidences else 0.0,
                'regions_processed': len(regions),
                'successful_regions': len(all_confidences),
                'text_length': len(combined_text),
                'method': 'region_based',
                'region_details': region_details
            }
            
            logger.info(
                f"Region-based OCR complete: {len(all_confidences)}/{len(regions)} regions, "
                f"confidence={overall_confidence:.2f}%"
            )
            
            return combined_text, overall_confidence, metadata
            
        except Exception as e:
            logger.error(f"Region-based OCR failed: {str(e)}")
            # Fallback to regular extraction
            return self.extract_text_with_confidence(image_path, preprocess)
    
    def _select_psm_for_region(self, region: Dict) -> str:
        """
        Select appropriate PSM mode based on region characteristics.
        
        Different regions need different page segmentation modes:
        - Narrow regions: single line (PSM 7)
        - Small regions: single word (PSM 8)
        - Large regions: auto (PSM 3)
        """
        aspect_ratio = region['aspect_ratio']
        area = region['area']
        
        # Single line (horizontal text)
        if aspect_ratio > 5 and region['height'] < 50:
            return self.psm_configs['single_line']
        
        # Single word
        elif area < 5000:
            return self.psm_configs['single_word']
        
        # Sparse text
        elif area < 20000:
            return self.psm_configs['sparse']
        
        # Large blocks - auto segmentation
        else:
            return self.psm_configs['auto']
    
    def _preprocess_region(self, region_img: np.ndarray) -> np.ndarray:
        """Apply preprocessing to region"""
        # Convert to grayscale if needed
        if len(region_img.shape) == 3:
            gray = cv2.cvtColor(region_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = region_img
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        # Apply adaptive threshold
        thresh = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        return thresh
    
    def extract_text_with_confidence(
        self, 
        image_path: str, 
        preprocess: bool = True
    ) -> Tuple[str, float, Dict]:
        """
        Extract text from image using Tesseract and calculate confidence.
        
        Args:
            image_path: Path to the image file
            preprocess: Whether to preprocess the image
            
        Returns:
            Tuple of (text, confidence, metadata)
        """
        try:
            logger.info(f"Starting Tesseract OCR for: {image_path}")

            # Read image using OpenCV (more reliable than PIL)
            if preprocess:
                pil_image = self.preprocessor.preprocess_for_ocr(image_path)
                # Convert PIL to OpenCV format
                image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            else:
                # Read directly with OpenCV
                image = cv2.imread(image_path)

            if image is None:
                raise ValueError(f"Could not read image from path: {image_path}")
            
            config = self.psm_configs['default']
            
            # Extract text with optimized config
            text = pytesseract.image_to_string(
                image, 
                lang=settings.tesseract_lang,
                config=config
            )
            
            # Get detailed data from Tesseract
            data = pytesseract.image_to_data(
                image, 
                lang=settings.tesseract_lang,
                config=config,
                output_type=pytesseract.Output.DICT
            )
            
            # Calculate confidence (simplified and reliable)
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

            # Count words
            word_count = len([w for w in data['text'] if str(w).strip()])
            
            # Prepare metadata
            metadata = {
                'word_count': word_count,
                'page_count': 1,
                'min_confidence': round(min(confidences)) if confidences else 0.0,
                'max_confidence': round(max(confidences)) if confidences else 0.0,
                'average_confidence': round(avg_confidence, 2),
                'low_confidence_words': sum(1 for c in confidences if c < self.confidence_threshold),
                'total_confidence_values': len(confidences),
                'text_length': len(text),
                'method': 'full_page',
                'config_used': config
            }
            
            logger.info(f"Tesseract extraction: confidence={avg_confidence:.2f}")
            
            return text.strip(), float(avg_confidence), metadata
            
        except Exception as e:
            logger.error(f"Tesseract extraction failed: {str(e)}", exc_info=True)
            raise

    def extract_from_multiple_images(
        self,
        image_paths: List[str],
        preprocess: bool = True,
        use_region_based: bool = True
    ) -> Tuple[str, float, Dict]:
        """
        Extract text from multiple images (PDF pages).
        Combines results and calculates overall confidence.
        
        Args:
            image_paths: List of image file paths
            preprocess: Whether to preprocess images
            
        Returns:
            Tuple of (combined_text, avg_confidence, metadata)
        """
        all_texts = []
        all_confidences = []
        total_words = 0
        page_details = []
        
        logger.info(f"Processing {len(image_paths)} pages with Tesseract (config: {self.tesseract_config})"
                    f"({'region-based' if use_region_based else 'full-page'})")
        
        for page_num, image_path in enumerate(image_paths, start=1):
            try:
                # Use region-based extraction for better accuracy
                if use_region_based:
                    text, confidence, metadata = self.extract_text_region_based(
                        image_path, preprocess
                    )
                else:
                    text, confidence, metadata = self.extract_text_with_confidence(
                        image_path, preprocess
                    )
                
                # Add page marker
                all_texts.append(f"\n--- Page {page_num} ---\n{text}")
                all_confidences.append(confidence)
                total_words += metadata['word_count']

                page_details.append({
                    'page': page_num,
                    'confidence': round(confidence, 2),
                    'words': metadata['word_count'],
                    'text_length': metadata['text_length'],
                    'method': metadata.get('method', 'unknown'),
                    'regions': metadata.get('regions_processed', 0)
                })
                
                logger.info(
                    f"Page {page_num}/{len(image_paths)}: "
                    f"confidence={confidence:.2f}%, words={metadata['word_count']}"
                )
                
            except Exception as e:
                logger.error(f"Failed to process page {page_num}: {str(e)}")
                all_texts.append(f"\n--- Page {page_num} (ERROR) ---\n")
                all_confidences.append(0.0)
        
        # Combine results
        combined_text = "\n".join(all_texts)
        overall_confidence = float(sum(all_confidences) / len(all_confidences)) if all_confidences else 0.0
        
        metadata = {
            'page_count': len(image_paths),
            'total_word_count': total_words,
            'overall_confidence': round(overall_confidence, 2),
            'min_page_confidence': round(min(all_confidences)) if all_confidences else 0.0,
            'max_page_confidence': round(max(all_confidences)) if all_confidences else 0.0,
            'page_details': page_details,
            'successful_pages': sum(1 for c in all_confidences if c > 0),
            'processing_method': 'region_based' if use_region_based else 'full_page'
        }

        logger.info(
            f"Multi-page extraction complete: "
            f"{metadata['successful_pages']}/{len(image_paths)} pages successful, "
            f"overall confidence={overall_confidence:.2f}%"
        )
        
        return combined_text, overall_confidence, metadata
    
    def should_use_trocr(self, confidence: float) -> bool:
        """
        Determine if TrOCR should be used based on confidence score.
        
        Args:
            confidence: Tesseract confidence score
            
        Returns:
            True if TrOCR should be used, False otherwise
        """
        should_use = confidence < self.confidence_threshold
        
        if should_use:
            logger.info(
                f"Confidence {confidence:.2f}% < threshold {self.confidence_threshold}%, "
                f"recommending TrOCR fallback"
            )
        
        return should_use