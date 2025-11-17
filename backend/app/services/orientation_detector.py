import cv2
import numpy as np
from PIL import Image
import pytesseract
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

class OrientationDetector:
    """
    Detects and corrects document orientation (landscape/portrait, rotation).
    Critical for insurance forms that may be scanned in different orientations.
    """
    
    def detect_and_correct_orientation(
        self, 
        image_path: str
    ) -> Tuple[Image.Image, dict]:
        """
        Detect orientation using Tesseract OSD and correct if needed.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Tuple of (corrected_image, orientation_info)
        """
        try:
            image = Image.open(image_path)
            
            # Use Tesseract's OSD (Orientation and Script Detection)
            osd_data = pytesseract.image_to_osd(image, output_type=pytesseract.Output.DICT)
            
            rotation_angle = osd_data.get('rotate', 0)
            orientation_confidence = osd_data.get('orientation_conf', 0)
            
            logger.info(
                f"Detected rotation: {rotation_angle}°, "
                f"confidence: {orientation_confidence:.2f}"
            )
            
            # Correct orientation if needed
            if rotation_angle != 0:
                image = image.rotate(-rotation_angle, expand=True, fillcolor='white')
                logger.info(f"Corrected rotation by {-rotation_angle}°")
            
            orientation_info = {
                'original_rotation': rotation_angle,
                'was_corrected': rotation_angle != 0,
                'confidence': orientation_confidence,
                'script': osd_data.get('script', 'Latin')
            }
            
            return image, orientation_info
            
        except Exception as e:
            logger.warning(f"OSD failed, trying alternative method: {str(e)}")
            return self._detect_with_text_line_analysis(image_path)
    
    def _detect_with_text_line_analysis(
        self, 
        image_path: str
    ) -> Tuple[Image.Image, dict]:
        """
        Alternative orientation detection using text line analysis.
        Fallback when OSD fails.
        """
        image = Image.open(image_path)
        img_cv = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        # Try all 4 orientations and pick best
        best_rotation = 0
        best_score = -1
        
        for angle in [0, 90, 180, 270]:
            rotated = self._rotate_image_cv(img_cv, angle)
            score = self._calculate_text_confidence(rotated)
            
            if score > best_score:
                best_score = score
                best_rotation = angle
        
        if best_rotation != 0:
            image = image.rotate(-best_rotation, expand=True, fillcolor='white')
            logger.info(f"Corrected rotation to {best_rotation}° based on text analysis")
        
        return image, {
            'original_rotation': best_rotation,
            'was_corrected': best_rotation != 0,
            'confidence': best_score,
            'method': 'text_line_analysis'
        }
    
    def _rotate_image_cv(self, image: np.ndarray, angle: int) -> np.ndarray:
        """Rotate image by given angle"""
        if angle == 0:
            return image
        elif angle == 90:
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            return cv2.rotate(image, cv2.ROTATE_180)
        elif angle == 270:
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return image
    
    def _calculate_text_confidence(self, image: np.ndarray) -> float:
        """Calculate text readability score"""
        try:
            pil_img = Image.fromarray(image)
            data = pytesseract.image_to_data(
                pil_img, 
                output_type=pytesseract.Output.DICT
            )
            
            confidences = [
                float(c) for c in data['conf'] 
                if c != '-1' and str(c).replace('.', '').isdigit()
            ]
            
            return np.mean(confidences) if confidences else 0.0
        except:
            return 0.0