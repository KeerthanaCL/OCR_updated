import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional
import logging
from app.config import get_settings
settings = get_settings()
logger = logging.getLogger(__name__)

class AdvancedImagePreprocessor:
    """
    Advanced preprocessing pipeline for OCR optimization.
    Includes deskewing, denoising, contrast enhancement, and super-resolution.
    """
    
    def __init__(self, enable_super_resolution: bool = False):
        """
        Initialize preprocessor.
        
        Args:
            enable_super_resolution: Enable deep learning super-resolution (slower but better quality)
        """
        self.enable_super_resolution = enable_super_resolution
        self.sr_model = None
        
        # Load super-resolution model lazily
        if self.enable_super_resolution and settings.enable_super_resolution:
            try:
                import cv2.dnn_superres
                self.sr = cv2.dnn_superres.DnnSuperResImpl_create()
                self.sr.readModel(settings.sr_model_path)
                self.sr.setModel("espcn", 4)
                logger.info("Super-resolution model loaded")
            except AttributeError:
                logger.warning("opencv-contrib-python not installed. Super-resolution disabled.")
                self.sr = None
            except Exception as e:
                logger.error(f"Failed to load super-resolution model: {e}")
                self.sr = None
    
    def _load_super_resolution_model(self):
        """Load ESPCN super-resolution model (OpenCV DNN)"""
        try:
            # OpenCV's pre-trained ESPCN model (fast and effective)
            # Download from: https://github.com/Saafke/EDSR_Tensorflow/tree/master/models
            model_path = "./models/ESPCN_x4.pb"  # 4x upscaling
            self.sr_model = cv2.dnn_superres.DnnSuperResImpl_create()
            self.sr_model.readModel(model_path)
            self.sr_model.setModel("espcn", 4)  # ESPCN with 4x scale
            logger.info("Super-resolution model loaded: ESPCN x4")
        except Exception as e:
            logger.warning(f"Failed to load super-resolution model: {e}")
            self.sr_model = None
            self.enable_super_resolution = False
    
    def preprocess_for_ocr(
        self, 
        image_path: str, 
        aggressive: bool = False,
        enhance_quality: bool = True
    ) -> Image.Image:
        """
        Main preprocessing pipeline for OCR.
        
        Args:
            image_path: Path to input image
            aggressive: Apply more aggressive processing for poor quality images
            enhance_quality: Enable deep learning enhancements
            
        Returns:
            Preprocessed PIL Image ready for OCR
        """
        # Load image
        image = cv2.imread(image_path)
        
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        logger.info(f"Starting preprocessing pipeline (aggressive={aggressive}, enhance={enhance_quality})")
        
        # Step 1: Super-resolution for low-quality images
        if enhance_quality and self._is_low_quality(image):
            image = self._apply_super_resolution(image)
        
        # Step 2: Remove borders/margins
        image = self._remove_borders(image)
        
        # Step 3: Deskew (correct rotation)
        image = self._deskew_advanced(image)
        
        # Step 4: Denoise
        image = self._denoise_adaptive(image, aggressive=aggressive)
        
        # Step 5: Contrast enhancement
        image = self._enhance_contrast_adaptive(image)
        
        # Step 6: Binarization (convert to black & white)
        if aggressive:
            image = self._binarize_sauvola(image)
        else:
            image = self._binarize_adaptive(image)
        
        # Convert to PIL Image
        return Image.fromarray(image)
    
    def _is_low_quality(self, image: np.ndarray) -> bool:
        """
        Detect low-quality images that need super-resolution.
        Uses Laplacian variance to measure blur.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Low variance indicates blurry/low-quality image
        is_low_quality = laplacian_var < 100
        
        logger.info(f"Quality check: blur_score={laplacian_var:.2f}, low_quality={is_low_quality}")
        return is_low_quality
    
    def _apply_super_resolution(self, image: np.ndarray) -> np.ndarray:
        """
        Apply deep learning super-resolution to enhance image quality.
        Uses OpenCV's DNN Super Resolution (ESPCN model).
        """
        if not self.enable_super_resolution or self.sr_model is None:
            return image
        
        try:
            logger.info("Applying super-resolution (4x upscaling)...")
            upscaled = self.sr_model.upsample(image)
            logger.info(f"Upscaled: {image.shape} -> {upscaled.shape}")
            return upscaled
        except Exception as e:
            logger.warning(f"Super-resolution failed: {e}, using original")
            return image
    
    def _remove_borders(self, image: np.ndarray) -> np.ndarray:
        """
        Remove black borders and margins from scanned documents.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Threshold to find document area
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return image
        
        # Get largest contour (document)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Add small padding
        padding = 10
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image.shape[1] - x, w + 2 * padding)
        h = min(image.shape[0] - y, h + 2 * padding)
        
        # Crop
        cropped = image[y:y+h, x:x+w]
        logger.info(f"Border removal: {image.shape} -> {cropped.shape}")
        return cropped
    
    def _deskew_advanced(self, image: np.ndarray) -> np.ndarray:
        """
        Advanced deskewing using Hough Transform for better accuracy.
        Detects text lines and corrects rotation.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Detect edges
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Detect lines using Hough Transform
        lines = cv2.HoughLinesP(
            edges, 
            rho=1, 
            theta=np.pi/180, 
            threshold=100,
            minLineLength=100,
            maxLineGap=10
        )
        
        if lines is None or len(lines) == 0:
            logger.info("No lines detected for deskewing, skipping")
            return image
        
        # Calculate angles of detected lines
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            angles.append(angle)
        
        # Get median angle (more robust than mean)
        median_angle = np.median(angles)
        
        # Normalize angle to [-45, 45] range
        if median_angle < -45:
            median_angle = 90 + median_angle
        elif median_angle > 45:
            median_angle = median_angle - 90
        
        # Only rotate if angle is significant (> 0.5 degrees)
        if abs(median_angle) > 0.5:
            logger.info(f"Deskewing: rotating by {median_angle:.2f} degrees")
            
            # Rotate image
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
            
            # Calculate new image size to avoid cropping
            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])
            new_w = int((h * sin) + (w * cos))
            new_h = int((h * cos) + (w * sin))
            
            # Adjust rotation matrix
            M[0, 2] += (new_w / 2) - center[0]
            M[1, 2] += (new_h / 2) - center[1]
            
            rotated = cv2.warpAffine(
                image, M, (new_w, new_h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE
            )
            return rotated
        
        logger.info("Image already aligned, no deskewing needed")
        return image
    
    def _denoise_adaptive(self, image: np.ndarray, aggressive: bool = False) -> np.ndarray:
        """
        Adaptive denoising based on image quality.
        Uses Non-Local Means Denoising for best results.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Calculate noise level
        noise_sigma = self._estimate_noise(gray)
        
        if noise_sigma > 10 or aggressive:
            # High noise - aggressive denoising
            logger.info(f"High noise detected ({noise_sigma:.2f}), applying aggressive denoising")
            denoised = cv2.fastNlMeansDenoising(gray, None, h=15, templateWindowSize=7, searchWindowSize=21)
        elif noise_sigma > 5:
            # Moderate noise - standard denoising
            logger.info(f"Moderate noise detected ({noise_sigma:.2f}), applying standard denoising")
            denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
        else:
            # Low noise - light denoising
            logger.info(f"Low noise detected ({noise_sigma:.2f}), applying light denoising")
            denoised = cv2.GaussianBlur(gray, (3, 3), 0)
        
        return denoised
    
    def _estimate_noise(self, image: np.ndarray) -> float:
        """
        Estimate noise level using median absolute deviation.
        """
        H, W = image.shape
        M = [[1, -2, 1],
             [-2, 4, -2],
             [1, -2, 1]]
        
        sigma = np.sum(np.sum(np.absolute(cv2.filter2D(image, -1, np.array(M)))))
        sigma = sigma * np.sqrt(0.5 * np.pi) / (6 * (W - 2) * (H - 2))
        
        return sigma
    
    def _enhance_contrast_adaptive(self, image: np.ndarray) -> np.ndarray:
        """
        Adaptive contrast enhancement using CLAHE.
        Parameters adjusted based on image statistics.
        """
        # Calculate image histogram statistics
        mean_intensity = np.mean(image)
        std_intensity = np.std(image)
        
        # Adjust CLAHE parameters based on image characteristics
        if std_intensity < 30:
            # Low contrast image - aggressive enhancement
            clip_limit = 3.0
            tile_size = (8, 8)
            logger.info("Low contrast detected, applying aggressive CLAHE")
        else:
            # Normal contrast - standard enhancement
            clip_limit = 2.0
            tile_size = (8, 8)
            logger.info("Normal contrast, applying standard CLAHE")
        
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
        enhanced = clahe.apply(image)
        
        return enhanced
    
    def _binarize_adaptive(self, image: np.ndarray) -> np.ndarray:
        """
        Adaptive thresholding for binarization.
        Better than Otsu for varying lighting conditions.
        """
        binary = cv2.adaptiveThreshold(
            image, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=11,
            C=2
        )
        logger.info("Applied adaptive thresholding")
        return binary
    
    def _binarize_sauvola(self, image: np.ndarray, window_size: int = 25, k: float = 0.2) -> np.ndarray:
        """
        Sauvola binarization - excellent for degraded documents.
        Better than adaptive thresholding for historical/poor quality documents.
        """
        # Calculate local mean and standard deviation
        mean = cv2.boxFilter(image.astype(np.float32), -1, (window_size, window_size))
        sqmean = cv2.boxFilter((image.astype(np.float32))**2, -1, (window_size, window_size))
        std = np.sqrt(sqmean - mean**2)
        
        # Sauvola threshold
        R = 128  # Dynamic range of standard deviation
        threshold = mean * (1 + k * ((std / R) - 1))
        
        # Apply threshold
        binary = np.where(image > threshold, 255, 0).astype(np.uint8)
        logger.info("Applied Sauvola binarization")
        return binary


# Legacy class for backward compatibility
class ImagePreprocessor(AdvancedImagePreprocessor):
    """Backward compatible wrapper"""
    
    @staticmethod
    def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
        """Convert image to grayscale"""
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
    
    @staticmethod
    def apply_thresholding(image: np.ndarray) -> np.ndarray:
        """Apply Otsu thresholding"""
        gray = ImagePreprocessor.convert_to_grayscale(image)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh
    
    @staticmethod
    def denoise_image(image: np.ndarray) -> np.ndarray:
        """Remove noise using Non-local Means Denoising"""
        gray = ImagePreprocessor.convert_to_grayscale(image)
        return cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)