import torch
import torch.nn.functional as F
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import logging
from typing import Tuple, Dict, List
from app.config import get_settings
from app.services.preprocessing import ImagePreprocessor
import numpy as np
import cv2

logger = logging.getLogger(__name__)
settings = get_settings()

class TrOCRService:
    """
    Service for Microsoft TrOCR (Transformer-based OCR).
    Used as fallback when Tesseract confidence is low.
    """
    
    def __init__(self, use_handwritten: bool = False, auto_detect: bool = True):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.preprocessor = ImagePreprocessor()
        self.auto_detect = auto_detect
        
        # Don't load model yet if auto-detect is enabled
        if auto_detect:
            self.model_name = None
            self.processor = None
            self.model = None
            logger.info("TrOCR initialized with auto-detection mode")
        else:
            # Load specific model
            model_name = (
                settings.trocr_handwritten_model if use_handwritten
                else settings.trocr_model_name
            )
            self._load_model(model_name)

    def _load_model(self, model_name: str):
        """Load TrOCR model and processor"""
        if self.model_name == model_name and self.model is not None:
            return  # Already loaded
            
        logger.info(f"Loading TrOCR model: {model_name} on {self.device}")
        self.model_name = model_name
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name).to(self.device)
        logger.info(f"TrOCR model loaded: {model_name}")
    
    def extract_text(
        self, 
        image_path: str,
        preprocess: bool = False
    ) -> Tuple[str, Dict]:
        """
        Extract text using TrOCR transformer model.
        
        Args:
            image_path: Path to the image file
            preprocess: Whether to preprocess (usually False for TrOCR)
            
        Returns:
            Tuple of (extracted_text, metadata)
        """
        try:
            # Load and preprocess image
            if preprocess:
                image = self.preprocessor.preprocess_for_ocr(image_path, aggressive=False)
            else:
                image = Image.open(image_path).convert("RGB")
            
            # Process image
            pixel_values = self.processor(
                image, 
                return_tensors="pt"
            ).pixel_values.to(self.device)
            
            # Generate predictions
            with torch.no_grad():
                generated_ids = self.model.generate(pixel_values)
            
            # Decode text
            generated_text = self.processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True
            )[0]
            
            # Prepare metadata
            metadata = {
                'model_used': self.model.config.name_or_path,
                'device': self.device,
                'page_count': 1
            }
            
            logger.info(f"TrOCR extraction completed: {len(generated_text)} characters")
            
            return generated_text.strip(), metadata
            
        except Exception as e:
            logger.error(f"TrOCR extraction failed: {str(e)}")
            raise

    def extract_from_multiple_images(
        self,
        image_paths: List[str]
    ) -> Tuple[str, Dict]:
        """
        Extract text from multiple images (PDF pages) using TrOCR.
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            Tuple of (combined_text, metadata)
        """
        all_texts = []
        
        logger.info(f"Processing {len(image_paths)} pages with TrOCR")
        
        for page_num, image_path in enumerate(image_paths, start=1):
            try:
                text, _ = self.extract_text(image_path)
                all_texts.append(f"\n--- Page {page_num} ---\n{text}")
                logger.info(f"TrOCR processed page {page_num}")
                
            except Exception as e:
                logger.error(f"Failed to process page {page_num}: {str(e)}")
                all_texts.append(f"\n--- Page {page_num} (ERROR) ---\n")
        
        combined_text = "\n".join(all_texts)
        
        metadata = {
            'model_used': self.model.config.name_or_path,
            'device': self.device,
            'page_count': len(image_paths)
        }
        
        return combined_text, metadata
    
    def extract_from_regions(
        self, 
        image_path: str, 
        regions: list
    ) -> Tuple[str, Dict]:
        """
        Extract text from specific regions of an image.
        Useful for structured documents with multiple text blocks.
        
        Args:
            image_path: Path to the image file
            regions: List of (x, y, w, h) tuples defining regions
            
        Returns:
            Tuple of (combined_text, metadata)
        """
        try:
            # Ensure model is loaded before processing
            if self.auto_detect and self.processor is None:
                is_handwritten = self.detect_handwriting(image_path)
                model_name = (
                    settings.trocr_handwritten_model if is_handwritten
                    else settings.trocr_model_name
                )
                logger.info(f"Auto-detected for regions: {'handwritten' if is_handwritten else 'printed'}")
                self._load_model(model_name)
                
            image = Image.open(image_path).convert('RGB')
            all_texts = []
            
            for region in regions:
                # Crop region
                x, y, w, h = region['x'], region['y'], region['width'], region['height']
                region_img = image.crop((x, y, x + w, y + h))
                
                # Process region
                pixel_values = self.processor(region_img, return_tensors="pt").pixel_values
                pixel_values = pixel_values.to(self.device)
                
                # Generate text
                generated_ids = self.model.generate(pixel_values)
                text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                
                if text.strip():
                    all_texts.append(text)
            
            combined_text = ' '.join(all_texts)
            
            metadata = {
                'regions_processed': len(regions),
                'regions_with_text': len(all_texts),
                'model': 'trocr-region-based'
            }
            
            return combined_text, metadata
            
        except Exception as e:
            logger.error(f"TrOCR region extraction failed: {e}")
            raise

    def extract_text_with_confidence(self, image_path: str) -> Tuple[str, float, Dict]:
        """
        Extract text from full image with confidence calculation.
        Auto-detects handwritten vs printed if enabled.
        
        Returns:
            Tuple of (text, confidence, metadata)
        """
        # Auto-detect and load appropriate model
        if self.auto_detect:
            is_handwritten = self.detect_handwriting(image_path)
            model_name = (
                settings.trocr_handwritten_model if is_handwritten
                else settings.trocr_model_name
            )
            logger.info(f"Auto-detected: {'handwritten' if is_handwritten else 'printed'}, "
                    f"using model: {model_name}")
            self._load_model(model_name)

        # Load image
        image = Image.open(image_path).convert("RGB")
        
        # Prepare inputs
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.to(self.device)
        
        # Generate with output scores
        generated_ids = self.model.generate(
            pixel_values,
            max_length=512,
            return_dict_in_generate=True,
            output_scores=True
        )
        
        # Decode text
        text = self.processor.batch_decode(generated_ids.sequences, skip_special_tokens=True)[0]
        
        # Calculate confidence from token scores
        confidence = self._calculate_confidence(generated_ids.scores)
        
        metadata = {
            'model': self.model_name,
            'device': str(self.device),
            'text_length': len(text),
            'confidence_source': 'token_probabilities',
            'document_type': 'handwritten' if 'handwritten' in self.model_name else 'printed'
        }
        
        return text, confidence, metadata


    def extract_from_regions_with_confidence(self, image_path: str, regions: List) -> Tuple[str, float, Dict]:
        """
        Extract text from detected regions with confidence calculation.
        
        Returns:
            Tuple of (text, confidence, metadata)
        """
        try:
            # Auto-detect and load model BEFORE processing regions
            if self.auto_detect and self.processor is None:
                is_handwritten = self.detect_handwriting(image_path)
                model_name = (
                    settings.trocr_handwritten_model if is_handwritten
                    else settings.trocr_model_name
                )
                logger.info(f"Auto-detected: {'handwritten' if is_handwritten else 'printed'}, "
                        f"using model: {model_name}")
                self._load_model(model_name)

            # Now processor and model are guaranteed to be loaded
            image = Image.open(image_path).convert("RGB")
            
            region_texts = []
            region_confidences = []
            
            for region in regions:
                if isinstance(region, dict):
                    if 'box' in region:
                        # Format 1: {'box': (x, y, w, h)}
                        x, y, w, h = region['box']
                    else:
                        # Format 2: {'x': x, 'y': y, 'width': w, 'height': h}
                        x = region.get('x', 0)
                        y = region.get('y', 0)
                        w = region.get('width', 0)
                        h = region.get('height', 0)
                else:
                    # Format 3: tuple (x, y, w, h)
                    x, y, w, h = region
                
                # Skip invalid regions
                if w <= 0 or h <= 0:
                    continue
                
                # Crop region
                cropped = image.crop((x, y, x + w, y + h))
                
                # Process region
                pixel_values = self.processor(cropped, return_tensors="pt").pixel_values.to(self.device)
                
                # Generate with scores
                generated_ids = self.model.generate(
                    pixel_values,
                    max_length=256,
                    return_dict_in_generate=True,
                    output_scores=True
                )
                
                # Decode
                text = self.processor.batch_decode(generated_ids.sequences, skip_special_tokens=True)[0]
                
                # Calculate confidence
                confidence = self._calculate_confidence(generated_ids.scores)
                
                if text.strip():
                    region_texts.append(text)
                    region_confidences.append(confidence)
            
            # Combine
            combined_text = '\n'.join(region_texts)
            
            # Average confidence weighted by text length
            if region_confidences:
                avg_confidence = sum(region_confidences) / len(region_confidences)
            else:
                avg_confidence = 0.0
            
            metadata = {
                'model': self.model_name,
                'regions_processed': len(regions),
                'regions_with_text': len(region_texts),
                'confidence_per_region': region_confidences,
                'confidence_source': 'token_probabilities'
            }

            logger.info(f"TrOCR region-based: {len(region_texts)}/{len(regions)} regions, avg confidence: {avg_confidence:.2f}%")
            
            return combined_text, avg_confidence, metadata
        except Exception as e:
            logger.error(f"TrOCR region extraction with confidence failed: {e}", exc_info=True)
            raise

    def _calculate_confidence(self, scores: Tuple) -> float:
        """
        Calculate confidence from model output scores (token probabilities).
        
        Uses mean of max softmax probabilities across all generated tokens.
        This gives a measure of how confident the model was for each prediction.
        
        Args:
            scores: Tuple of score tensors from model generation
            
        Returns:
            Confidence score as percentage (0-100)
        """
        if not scores:
            logger.warning("No scores provided, using default confidence")
            return 85.0  # Default fallback
        
        # Convert scores to probabilities
        probabilities = []
        
        for score_tensor in scores:
            # Apply softmax to get probabilities
            probs = F.softmax(score_tensor, dim=-1)
            
            # Get max probability for this token (most confident prediction)
            max_prob = torch.max(probs).item()
            probabilities.append(max_prob)
        
        if not probabilities:
            logger.warning("Empty probabilities list, using default confidence")
            return 85.0
        
        # Calculate mean confidence
        mean_confidence = sum(probabilities) / len(probabilities)
        
        # Convert to percentage
        confidence_percent = mean_confidence * 100
        
        # Clamp between reasonable bounds (TrOCR tends to be confident)
        confidence_percent = max(70.0, min(99.0, confidence_percent))
        logger.debug(f"Calculated TrOCR confidence: {confidence_percent:.2f}% from {len(probabilities)} tokens")
        
        return confidence_percent
    
    def detect_handwriting(self, image_path: str) -> bool:
        """
        Detect if document contains handwritten text using variance analysis.
        
        Handwritten text typically has:
        - Higher stroke width variance
        - More irregular spacing
        - Less uniform baselines
        
        Args:
            image_path: Path to image file
            
        Returns:
            True if handwritten detected, False if printed
        """
        try:
            # Read image
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            
            # Apply threshold to get binary image
            _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Calculate stroke width variance (handwriting has higher variance)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return False  # Default to printed if no contours found
            
            # Calculate metrics
            stroke_widths = []
            heights = []
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 50:  # Skip noise
                    continue
                    
                x, y, w, h = cv2.boundingRect(contour)
                heights.append(h)
                
                # Estimate stroke width from area/perimeter ratio
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    stroke_width = area / perimeter
                    stroke_widths.append(stroke_width)
            
            if not stroke_widths:
                return False
            
            # Calculate variance metrics
            stroke_variance = np.var(stroke_widths)
            height_variance = np.var(heights)
            
            # Thresholds (tune based on your documents)
            # Handwriting typically has higher variance
            is_handwritten = (stroke_variance > 15.0 or height_variance > 50.0)
            
            logger.info(f"Handwriting detection: stroke_var={stroke_variance:.2f}, "
                    f"height_var={height_variance:.2f}, "
                    f"is_handwritten={is_handwritten}")
            
            return is_handwritten
            
        except Exception as e:
            logger.error(f"Handwriting detection failed: {e}")
            return False  # Default to printed on error