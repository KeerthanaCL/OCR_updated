import boto3
import logging
from typing import Tuple, Dict
import time
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

class AWSTextractService:
    """
    AWS Textract service for high-accuracy OCR fallback.
    Used when Tesseract confidence is below threshold.

    Cost: $0.0015/page (basic) or $0.015/page (with tables)
    """
    
    def __init__(self):
        try:
            self.client = boto3.client(
                'textract',
                region_name=settings.aws_region,
                aws_access_key_id=settings.aws_access_key_id,
                aws_secret_access_key=settings.aws_secret_access_key
            )
            logger.info(f"AWS Textract initialized (region: {settings.aws_region})")
        except Exception as e:
            logger.error(f"Failed to initialize AWS Textract: {e}")
            raise
    
    def extract_text(self, image_path: str) -> Tuple[str, float, Dict]:
        """
        Extract text using AWS Textract (basic text detection).
        
        Cost: $0.0015 per page
        
        Args:
            image_path: Path to image file
        
        Returns:
            Tuple of (text, confidence, metadata)
        """
        start_time = time.time()
        
        try:
            logger.info(f"Calling AWS Textract for: {image_path}")
            
            # Read image bytes
            with open(image_path, 'rb') as document:
                image_bytes = document.read()
            
            # Call Textract DetectDocumentText (cheaper)
            response = self.client.detect_document_text(
                Document={'Bytes': image_bytes}
            )
            
            # Extract text and confidence
            extracted_texts = []
            confidences = []
            
            for block in response['Blocks']:
                if block['BlockType'] == 'LINE':
                    text = block.get('Text', '')
                    confidence = block.get('Confidence', 0)
                    extracted_texts.append(text)
                    confidences.append(confidence)
            
            full_text = '\n'.join(extracted_texts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            processing_time = time.time() - start_time
            
            metadata = {
                'engine': 'aws_textract',
                'api_method': 'detect_document_text',
                'cost_per_page': 0.0015,
                'text_length': len(full_text),
                'word_count': len(full_text.split()),
                'average_confidence': round(avg_confidence, 2),
                'detected_lines': len(extracted_texts),
                'total_blocks': len(response['Blocks']),
                'page_count': 1,
                'processing_time': round(processing_time, 2)
            }
            
            logger.info(
                f"AWS Textract completed: confidence={avg_confidence:.2f}%, "
                f"lines={len(extracted_texts)}, time={processing_time:.2f}s"
            )
            
            return full_text.strip(), avg_confidence, metadata
            
        except Exception as e:
            logger.error(f"AWS Textract failed: {str(e)}", exc_info=True)
            raise
    
    def extract_with_tables(self, image_path: str) -> Tuple[str, float, Dict]:
        """
        Extract text WITH table and form detection.
        
        Cost: $0.015 per page (10x more expensive)
        Use only for complex documents with tables/forms.
        
        Args:
            image_path: Path to image file
        
        Returns:
            Tuple of (text, confidence, metadata)
        """
        start_time = time.time()
        
        try:
            logger.info(f"Calling AWS Textract (with TABLES) for: {image_path}")
            
            with open(image_path, 'rb') as document:
                image_bytes = document.read()
            
            # Use AnalyzeDocument for tables/forms (more expensive)
            response = self.client.analyze_document(
                Document={'Bytes': image_bytes},
                FeatureTypes=['TABLES', 'FORMS']
            )
            
            # Extract text
            extracted_texts = []
            confidences = []
            tables = []
            key_value_pairs = []
            
            for block in response['Blocks']:
                if block['BlockType'] == 'LINE':
                    text = block.get('Text', '')
                    confidence = block.get('Confidence', 0)
                    extracted_texts.append(text)
                    confidences.append(confidence)
                elif block['BlockType'] == 'TABLE':
                    tables.append(block)
                elif block['BlockType'] == 'KEY_VALUE_SET':
                    key_value_pairs.append(block)
            
            full_text = '\n'.join(extracted_texts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            processing_time = time.time() - start_time
            
            metadata = {
                'engine': 'aws_textract_tables',
                'api_method': 'analyze_document',
                'features': ['TABLES', 'FORMS'],
                'cost_per_page': 0.015,
                'text_length': len(full_text),
                'word_count': len(full_text.split()),
                'average_confidence': round(avg_confidence, 2),
                'detected_lines': len(extracted_texts),
                'detected_tables': len(tables),
                'detected_forms': len(key_value_pairs),
                'total_blocks': len(response['Blocks']),
                'page_count': 1,
                'processing_time': round(processing_time, 2)
            }
            
            logger.info(
                f"AWS Textract (tables) completed: confidence={avg_confidence:.2f}%, "
                f"lines={len(extracted_texts)}, tables={len(tables)}, "
                f"forms={len(key_value_pairs)}, time={processing_time:.2f}s"
            )
            
            return full_text.strip(), avg_confidence, metadata
            
        except Exception as e:
            logger.error(f"AWS Textract (tables) failed: {str(e)}", exc_info=True)
            raise