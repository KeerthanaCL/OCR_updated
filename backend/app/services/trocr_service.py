import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import logging
from typing import Tuple, Dict, List
from app.config import get_settings
from app.services.preprocessing import ImagePreprocessor

logger = logging.getLogger(__name__)
settings = get_settings()

class TrOCRService:
    """
    Service for Microsoft TrOCR (Transformer-based OCR).
    Used as fallback when Tesseract confidence is low.
    """
    
    def __init__(self, use_handwritten: bool = False):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.preprocessor = ImagePreprocessor()
        
        # Select model based on document type
        model_name = (
            settings.trocr_handwritten_model if use_handwritten 
            else settings.trocr_model_name
        )
        
        logger.info(f"Loading TrOCR model: {model_name} on {self.device}")
        
        # Load model and processor
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name).to(self.device)
        
        logger.info("TrOCR model loaded successfully")
    
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
        image = Image.open(image_path).convert("RGB")
        extracted_texts = []
        
        for idx, (x, y, w, h) in enumerate(regions):
            # Crop region
            region = image.crop((x, y, x + w, y + h))
            
            # Process region
            pixel_values = self.processor(
                region, 
                return_tensors="pt"
            ).pixel_values.to(self.device)
            
            # Generate text
            with torch.no_grad():
                generated_ids = self.model.generate(pixel_values)
            
            text = self.processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True
            )[0]
            
            extracted_texts.append(text)
        
        combined_text = "\n".join(extracted_texts)
        metadata = {
            'regions_processed': len(regions),
            'model_used': self.model.config.name_or_path
        }
        
        return combined_text, metadata