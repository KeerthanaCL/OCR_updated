import cv2
import numpy as np
from PIL import Image
import pytesseract
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class TableExtractor:
    """
    Extracts tables from medical bills and invoices.
    Detects table structure and extracts data in structured format.
    """
    
    def detect_tables(self, image_path: str) -> List[Dict[str, Any]]:
        """
        Detect and extract tables from image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            List of detected tables with their content
        """
        try:
            # Read image
            img = cv2.imread(image_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply thresholding
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            
            # Detect horizontal and vertical lines
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
            
            horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
            vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
            
            # Combine lines to find table structure
            table_mask = cv2.add(horizontal_lines, vertical_lines)
            
            # Find contours (table cells)
            contours, _ = cv2.findContours(table_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            # Extract tables
            tables = self._extract_table_data(img, contours)
            
            logger.info(f"Detected {len(tables)} tables")
            return tables
            
        except Exception as e:
            logger.error(f"Table detection failed: {str(e)}")
            return []
    
    def _extract_table_data(
        self, 
        image: np.ndarray, 
        contours: list
    ) -> List[Dict[str, Any]]:
        """Extract data from detected table regions"""
        tables = []
        
        # Filter and sort contours by area
        valid_contours = [c for c in contours if cv2.contourArea(c) > 1000]
        valid_contours = sorted(valid_contours, key=lambda c: cv2.boundingRect(c)[1])
        
        for idx, contour in enumerate(valid_contours[:5]):  # Limit to top 5 tables
            x, y, w, h = cv2.boundingRect(contour)
            
            # Extract table region
            table_roi = image[y:y+h, x:x+w]
            
            # OCR on table region
            pil_img = Image.fromarray(cv2.cvtColor(table_roi, cv2.COLOR_BGR2RGB))
            table_text = pytesseract.image_to_string(pil_img)
            
            tables.append({
                'table_id': idx + 1,
                'position': {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)},
                'raw_text': table_text,
                'rows': self._parse_table_rows(table_text)
            })
        
        return tables
    
    def _parse_table_rows(self, text: str) -> List[str]:
        """Parse table text into rows"""
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        return lines