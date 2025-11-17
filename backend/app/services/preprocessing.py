import cv2
import numpy as np
from PIL import Image
from typing import Tuple

class ImagePreprocessor:
    """
    Preprocessing service to enhance image quality for OCR.
    Applies various techniques to improve text extraction accuracy.
    """
    
    @staticmethod
    def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
        """Convert image to grayscale"""
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
    
    @staticmethod
    def apply_thresholding(image: np.ndarray) -> np.ndarray:
        """
        Apply adaptive thresholding to enhance text visibility.
        Uses Otsu's method for automatic threshold determination.
        """
        gray = ImagePreprocessor.convert_to_grayscale(image)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh
    
    @staticmethod
    def denoise_image(image: np.ndarray) -> np.ndarray:
        """Remove noise using Non-local Means Denoising"""
        gray = ImagePreprocessor.convert_to_grayscale(image)
        return cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    
    @staticmethod
    def deskew_image(image: np.ndarray) -> np.ndarray:
        """
        Deskew image by detecting and correcting rotation angle.
        This improves OCR accuracy for scanned documents.
        """
        gray = ImagePreprocessor.convert_to_grayscale(image)
        coords = np.column_stack(np.where(gray > 0))
        angle = cv2.minAreaRect(coords)[-1]
        
        # Correct angle
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        
        # Rotate image
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), 
                                  flags=cv2.INTER_CUBIC, 
                                  borderMode=cv2.BORDER_REPLICATE)
        return rotated
    
    @staticmethod
    def remove_borders(image: np.ndarray) -> np.ndarray:
        """Remove borders from scanned documents"""
        gray = ImagePreprocessor.convert_to_grayscale(image)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get the largest contour (document)
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            return image[y:y+h, x:x+w]
        
        return image
    
    @staticmethod
    def enhance_contrast(image: np.ndarray) -> np.ndarray:
        """Enhance image contrast using CLAHE"""
        gray = ImagePreprocessor.convert_to_grayscale(image)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)
    
    def preprocess_for_ocr(self, image_path: str, aggressive: bool = False) -> Image.Image:
        """
        Main preprocessing pipeline for OCR.
        
        Args:
            image_path: Path to the input image
            aggressive: Whether to apply more aggressive preprocessing
            
        Returns:
            Preprocessed PIL Image
        """
        # Load image
        image = cv2.imread(image_path)
        
        # Apply preprocessing steps
        image = self.remove_borders(image)
        image = self.deskew_image(image)
        image = self.denoise_image(image)
        
        if aggressive:
            image = self.enhance_contrast(image)
        
        image = self.apply_thresholding(image)
        
        # Convert back to PIL Image
        return Image.fromarray(image)