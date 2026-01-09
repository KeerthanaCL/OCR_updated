import logging
from typing import Tuple, Dict, List
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from paddleocr import PaddleOCR
    import paddle
    PADDLE_AVAILABLE = True
except Exception as e:
    logger.warning(f"PaddleOCR not available: {e}")
    PADDLE_AVAILABLE = False

from app.config import get_settings
from app.services.pdf_converter import PDFConverter

settings = get_settings()


class PaddleOCRService:
    """PaddleOCR service with GPU support (PaddleOCR 2.7 + PaddlePaddle 2.6.2)"""
    
    def __init__(self):
        if not PADDLE_AVAILABLE:
            raise RuntimeError("PaddleOCR is not available")
        
        self.ocr = None
        self.confidence_threshold = settings.tesseract_confidence_threshold
        self.pdf_converter = PDFConverter(method="pymupdf", save_images=True)
        
        logger.info("PaddleOCRService initialized (lazy loading)")
    
    def _load_ocr(self):
        """Lazy load PaddleOCR 2.7"""
        if self.ocr is not None:
            return
        
        logger.info("Loading PaddleOCR 2.7 models...")
        
        # Check GPU
        gpu_available = False
        if paddle.device.is_compiled_with_cuda():
            try:
                gpu_count = paddle.device.cuda.device_count()
                if gpu_count > 0:
                    gpu_available = True
                    gpu_name = paddle.device.cuda.get_device_name(0)
                    logger.info(f"🚀 GPU detected: {gpu_name} ({gpu_count} device)")
            except Exception as e:
                logger.warning(f"GPU detection failed: {e}")
        
        # Initialize PaddleOCR 2.7
        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang='en',
            use_gpu=gpu_available,
            show_log=False
        )
        
        if gpu_available:
            logger.info("✅ PaddleOCR loaded with GPU acceleration")
        else:
            logger.warning("⚠️ PaddleOCR loaded in CPU mode")
    
    def _is_pdf(self, file_path: str) -> bool:
        """Check if file is PDF"""
        return Path(file_path).suffix.lower() == '.pdf'
    
    def _convert_pdf_to_images(self, pdf_path: str) -> List[str]:
        """Convert PDF to images"""
        try:
            pdf_dir = Path(pdf_path).parent
            output_dir = pdf_dir / "pdf_pages"
            
            logger.info(f"📄 Converting PDF: {Path(pdf_path).name}")
            image_paths = self.pdf_converter.convert(
                pdf_path=pdf_path,
                output_dir=str(output_dir)
            )
            
            logger.info(f"✅ PDF converted to {len(image_paths)} pages")
            return image_paths
            
        except Exception as e:
            logger.error(f"PDF conversion failed: {e}", exc_info=True)
            raise
    
    def extract_text(self, image_path: str) -> Tuple[str, float, Dict]:
        """
        Extract text using PaddleOCR 2.7.
        
        Returns:
            Tuple of (full_text, avg_confidence, metadata)
        """
        import time
        start_time = time.time()
        
        try:
            self._load_ocr()
            
            # Handle PDF
            if self._is_pdf(image_path):
                logger.info(f"📄 Processing PDF: {Path(image_path).name}")
                image_paths = self._convert_pdf_to_images(image_path)
                
                if not image_paths:
                    raise ValueError("PDF conversion produced no images")
                
                total_pages = len(image_paths)
                logger.info(f"Total pages to process: {total_pages}")

                # PROCESS ALL PAGES
                all_text = []
                all_confidences = []
                all_blocks = []
                page_texts = []
                
                for page_num, page_path in enumerate(image_paths, 1):
                    logger.info(f"🔍 Processing page {page_num}/{total_pages}")
                    
                    # Run PaddleOCR on this page
                    result = self.ocr.ocr(page_path, cls=True)

                    # Collect text for this page
                    page_text_lines = []
                    
                    # Parse results for this page
                    if result and result[0]:
                        for line in result[0]:
                            box = line[0]
                            text_info = line[1]
                            text = text_info[0]
                            confidence = text_info[1] * 100
                            
                            all_text.append(text)
                            all_confidences.append(confidence)
                            page_text_lines.append(text)
                            all_blocks.append({
                                "text": text,
                                "confidence": round(confidence, 2),
                                "box": box,
                                "page": page_num  # Track which page
                            })

                    # Add page separator with extracted text
                    if page_text_lines:
                        page_content = "\n".join(page_text_lines)
                        page_texts.append({
                            "page": page_num,
                            "text": page_content,
                            "word_count": len(page_content.split()),
                            "line_count": len(page_text_lines)
                        })
                
                # Combine all pages with clear page markers
                formatted_pages = []
                for page_info in page_texts:
                    formatted_pages.append(
                        f"\n{'='*80}\n"
                        f"PAGE {page_info['page']}/{total_pages}\n"
                        f"{'='*80}\n"
                        f"{page_info['text']}"
                    )
                
                # Combine all pages
                full_text = "\n".join(all_text)
                avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0
                
                processing_time = time.time() - start_time
                
                metadata = {
                    "engine": "paddleocr_v2.7_gpu" if paddle.device.cuda.device_count() > 0 else "paddleocr_v2.7_cpu",
                    "gpu_enabled": paddle.device.cuda.device_count() > 0,
                    "input_type": "pdf",
                    "pages_in_document": total_pages,
                    "pages_processed": total_pages,  # ✅ All pages!
                    "text_length": len(full_text),
                    "word_count": len(full_text.split()),
                    "average_confidence": round(avg_confidence, 2),
                    "detected_text_blocks": len(all_blocks),
                    "low_confidence_blocks": len([b for b in all_blocks if b["confidence"] < 70]),
                    "blocks": all_blocks[:10],  # First 10 blocks as sample
                    "page_details": page_texts,
                    "processing_time": round(processing_time, 2)
                }
                
                return full_text, avg_confidence, metadata
            
            else:
                # Single image (not PDF)
                actual_image_path = image_path
                
                # Run PaddleOCR 2.7
                logger.info(f"🔍 Running PaddleOCR on: {Path(actual_image_path).name}")
                result = self.ocr.ocr(actual_image_path, cls=True)
                
                # Parse PaddleOCR 2.7 result
                texts: List[str] = []
                confidences: List[float] = []
                blocks: List[Dict] = []
                
                if result and result[0]:
                    for line in result[0]:
                        try:
                            if not line or len(line) < 2:
                                continue
                            
                            box = line[0]
                            text_info = line[1]
                            
                            # Handle different result formats
                            if isinstance(text_info, (tuple, list)) and len(text_info) >= 2:
                                text = str(text_info[0])
                                score = float(text_info[1])
                            else:
                                text = str(text_info)
                                score = 1.0
                            
                            conf = score * 100.0
                            texts.append(text)
                            confidences.append(conf)
                            blocks.append({
                                "text": text,
                                "confidence": round(conf, 2),
                                "box": box
                            })
                        except Exception as e:
                            logger.warning(f"Failed to parse line: {e}")
                            continue
                
                # Format output
                full_text = "\n".join(texts)
                avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
                processing_time = time.time() - start_time
                
                # Check GPU usage
                gpu_used = False
                if paddle.device.is_compiled_with_cuda():
                    try:
                        gpu_used = paddle.device.cuda.device_count() > 0
                    except:
                        pass
                
                logger.info(
                    f"✅ PaddleOCR complete: {avg_conf:.1f}% confidence, "
                    f"{len(blocks)} lines, {len(full_text.split())} words, "
                    f"GPU={gpu_used}, {processing_time:.2f}s"
                )
                
                metadata = {
                    "engine": "paddleocr_v2.7_gpu" if gpu_used else "paddleocr_v2.7_cpu",
                    "gpu_enabled": gpu_used,
                    "input_type": "image",  # ✅ Always "image" in this branch
                    "pages_in_document": 1,  # ✅ Single image = 1 page
                    "pages_processed": 1,    # ✅ Single image = 1 page processed
                    "text_length": len(full_text),
                    "word_count": len(full_text.split()),
                    "average_confidence": round(avg_conf, 2),
                    "detected_text_blocks": len(blocks),
                    "low_confidence_blocks": sum(
                        1 for c in confidences if c < self.confidence_threshold
                    ),
                    "blocks": blocks[:10],
                    "processing_time": round(processing_time, 2)
                }
                
                return full_text, avg_conf, metadata
            
        except Exception as e:
            logger.error(f"PaddleOCR extraction failed: {e}", exc_info=True)
            raise
    
    def extract_text_region_based(
        self, 
        image_path: str, 
        preprocess: bool = True
    ) -> Tuple[str, float, Dict]:
        """Extract with optional preprocessing"""
        if preprocess and not self._is_pdf(image_path):
            from app.services.preprocessing import AdvancedImagePreprocessor
            import tempfile
            
            logger.info("📝 Applying preprocessing...")
            preprocessor = AdvancedImagePreprocessor(enable_super_resolution=False)
            preprocessed_img = preprocessor.preprocess_for_ocr(image_path)
            
            temp_path = Path(tempfile.gettempdir()) / f"preprocessed_{Path(image_path).name}"
            preprocessed_img.save(temp_path)
            
            try:
                result = self.extract_text(str(temp_path))
            finally:
                temp_path.unlink(missing_ok=True)
            
            return result
        else:
            return self.extract_text(image_path)


def is_paddleocr_available() -> bool:
    """Check if PaddleOCR is available"""
    return PADDLE_AVAILABLE
