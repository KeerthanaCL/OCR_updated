import cv2
import numpy as np
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

class RegionDetector:
    """
    Detects and segments text regions in documents.
    Processes each region separately for better OCR accuracy.
    """
    
    def __init__(self):
        self.min_region_area = 1000  # Minimum area for a valid region
        self.dilation_kernel_size = (30, 30)  # For connecting nearby text
    
    def detect_text_regions(self, image_path: str) -> List[Dict[str, any]]:
        """
        Detect individual text regions in the image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            List of region dictionaries with coordinates and metadata
        """
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Cannot read image: {image_path}")
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply preprocessing
            processed = self._preprocess_for_region_detection(gray)
            
            # Find contours (text regions)
            contours, _ = cv2.findContours(
                processed, 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            # Extract and sort regions
            regions = self._extract_regions(contours, image.shape)
            
            logger.info(f"Detected {len(regions)} text regions")
            
            return regions
            
        except Exception as e:
            logger.error(f"Region detection failed: {str(e)}")
            return []
    
    def _preprocess_for_region_detection(self, gray_image: np.ndarray) -> np.ndarray:
        """Preprocess image for better region detection"""
        
        # Apply bilateral filter to reduce noise while keeping edges
        blurred = cv2.bilateralFilter(gray_image, 9, 75, 75)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            blurred, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 
            11, 
            2
        )
        
        # Dilate to connect nearby text
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, self.dilation_kernel_size)
        dilated = cv2.dilate(thresh, kernel, iterations=2)
        
        return dilated
    
    def _extract_regions(
        self, 
        contours: list, 
        image_shape: Tuple[int, int, int]
    ) -> List[Dict[str, any]]:
        """Extract valid regions from contours"""
        regions = []
        height, width = image_shape[:2]
        
        for idx, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            # Filter small regions
            if area < self.min_region_area:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            
            # Add padding
            padding = 10
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(width - x, w + 2 * padding)
            h = min(height - y, h + 2 * padding)
            
            # Calculate region properties
            aspect_ratio = w / h if h > 0 else 0
            
            regions.append({
                'id': idx,
                'x': x,
                'y': y,
                'width': w,
                'height': h,
                'area': area,
                'aspect_ratio': aspect_ratio,
                'center_x': x + w // 2,
                'center_y': y + h // 2
            })
        
        # Sort regions top-to-bottom, left-to-right
        regions = sorted(regions, key=lambda r: (r['y'], r['x']))
        
        return regions
    
    def extract_region_image(
        self, 
        image_path: str, 
        region: Dict[str, int]
    ) -> np.ndarray:
        """Extract specific region from image"""
        image = cv2.imread(image_path)
        x, y, w, h = region['x'], region['y'], region['width'], region['height']
        return image[y:y+h, x:x+w]
    
    def detect_form_fields(self, image_path: str) -> List[Dict[str, any]]:
        """
        Detect form fields (boxes, lines) in insurance forms.
        Uses horizontal and vertical line detection.
        """
        try:
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply threshold
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            
            # Detect horizontal lines
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
            horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
            
            # Detect vertical lines
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
            vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
            
            # Combine lines
            form_structure = cv2.add(horizontal_lines, vertical_lines)
            
            # Find contours of form fields
            contours, _ = cv2.findContours(form_structure, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            fields = []
            for idx, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if area < 500:  # Skip small artifacts
                    continue
                
                x, y, w, h = cv2.boundingRect(contour)
                fields.append({
                    'id': idx,
                    'x': x,
                    'y': y,
                    'width': w,
                    'height': h,
                    'type': 'form_field'
                })
            
            logger.info(f"Detected {len(fields)} form fields")
            return fields
            
        except Exception as e:
            logger.error(f"Form field detection failed: {str(e)}")
            return []