import time
import logging
from typing import Dict, Any, List
from pathlib import Path
import cv2

from langgraph.graph import StateGraph, END
import asyncio
from app.utils import cancellation_manager

from app.agents.graph_state import ExtractionState
from app.services.tesseract_service import TesseractService
from app.services.aws_textract_service import AWSTextractService
from app.services.pdf_converter import PDFConverter
from app.services.orientation_detector import OrientationDetector
from app.services.region_detector import RegionDetector
from app.utils.metrics import MetricsCollector
from app.database import Document, Extraction
from app.config import get_settings
from app.services.storage import StorageService
from sqlalchemy.orm import Session
from app.utils.sanitisation import sanitize_sensitive_info
import uuid

logger = logging.getLogger(__name__)
settings = get_settings()

def _check_cancel():
    if cancellation_manager.cancel_event.is_set():
        raise asyncio.CancelledError()
        
class LangGraphExtractionAgent:
    """
    LangGraph-based extraction agent with Tesseract → AWS Textract fallback.

    Graph Flow:
    START → analyze_document → extract_with_tesseract → evaluate_result
    → [if confidence < threshold] → fallback_to_textract → END
    → [else] → END

    Strategy:
    1. Try Tesseract first (FREE, fast, 80-90% of documents)
    2. If confidence < threshold → Fallback to AWS Textract (PAID, accurate)

    Cost Optimization:
    - Tesseract: $0/page (handles most documents)
    - AWS Textract: $0.0015/page (only for difficult cases)
    - Expected savings: 80-90% vs using Textract for everything
    """

    
    def __init__(self):
        self.tesseract = TesseractService()
        self.aws_textract = None  # Lazy load only when needed
        self.pdf_converter = PDFConverter(method="pymupdf")
        self.orientation_detector = OrientationDetector()
        self.region_detector = RegionDetector()

        # Build the graph
        self.graph = self._build_graph()

        logger.info("Extraction Agent initialized (Tesseract + AWS Textract fallback)")

    def _load_aws_textract(self):
        """Lazy load AWS Textract (only when Tesseract confidence is low)"""
        if self.aws_textract is None:
            logger.info("Loading AWS Textract fallback service...")
            self.aws_textract = AWSTextractService()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        
        # Create graph
        workflow = StateGraph(ExtractionState)
        
        # Add nodes
        workflow.add_node("analyze_document", self._analyze_document_node)
        workflow.add_node("extract_with_tesseract", self._extract_tesseract_node)
        workflow.add_node("evaluate_result", self._evaluate_result_node)
        workflow.add_node("fallback_to_textract", self._fallback_textract_node)
        
        # Set entry point
        workflow.set_entry_point("analyze_document")
        
        # Add edges
        workflow.add_edge("analyze_document", "extract_with_tesseract")
        workflow.add_edge("extract_with_tesseract", "evaluate_result")
        
        # Conditional edge from evaluate_result
        workflow.add_conditional_edges(
            "evaluate_result",
            self._should_fallback_to_textract,
            {
                "fallback": "fallback_to_textract",
                "success": END
            }
        )
        
        # Edge from fallback back to end
        workflow.add_edge("fallback_to_textract", END)
        
        return workflow.compile()
    
    def _analyze_document_node(self, state: ExtractionState) -> ExtractionState:
        """Node: Analyze document characteristics"""
        _check_cancel()
        logger.info("Analyzing document...")
        
        try:
            img = cv2.imread(state['file_path'], cv2.IMREAD_GRAYSCALE)
            if img is None:
                state['doc_analysis'] = {'analysis_failed': True}
                return state
            
            # Detect regions
            regions = self.region_detector.detect_text_regions(state['file_path'])
            
            # Calculate quality metrics
            blur_score = cv2.Laplacian(img, cv2.CV_64F).var()
            height, width = img.shape
            
            state['doc_analysis'] = {
                'num_regions': len(regions),
                'is_simple_layout': len(regions) <= 3,
                'is_complex_layout': len(regions) > 10,
                'has_tables': len(regions) > 10,
                'quality': 'high' if blur_score > 500 else 'medium' if blur_score > 100 else 'low',
                'blur_score': round(blur_score, 2),
                'image_size': (width, height)
            }
            
            logger.info(
                f"Analysis: {len(regions)} regions, "
                f"quality={state['doc_analysis']['quality']}, "
                f"blur_score={blur_score:.2f}"
            )
            
        except asyncio.CancelledError:
            logger.warning("Extraction cancelled")
            raise
        
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            state['doc_analysis'] = {'analysis_failed': True}
        
        return state
    
    def _extract_tesseract_node(self, state: ExtractionState) -> ExtractionState:
        """Node: Extract text using Tesseract"""
        _check_cancel()
        file_path = state['file_path']
        start_time = time.time()
        
        logger.info("Extracting with Tesseract (FREE)...")
        
        try:
            # Correct orientation if needed
            if state.get('correct_orientation'):
                _check_cancel()
                orientation_result = self._correct_image_orientation(file_path)
                file_path = orientation_result['path']
                state['file_path'] = file_path
                state['orientation_info'] = orientation_result['info']
            
            # Extract with Tesseract (region-based for better accuracy)
            text, confidence, metadata = self.tesseract.extract_text_region_based(
                file_path,
                preprocess=state.get('use_preprocessing', True)
            )
            _check_cancel()
            
            # Update state
            state['extracted_text'] = text
            state['confidence'] = confidence
            state['method_used'] = 'tesseract'
            state['processing_time'] = time.time() - start_time
            state['attempted_methods'].append('tesseract')
            state['metadata'] = metadata
            
            logger.info(
                f"Tesseract completed: confidence={confidence:.2f}%, "
                f"words={len(text.split())}, time={state['processing_time']:.2f}s"
            )
            
        except asyncio.CancelledError:
            logger.warning("Extraction cancelled")
            raise

        except Exception as e:
            logger.error(f"Tesseract extraction failed: {e}")
            state['error'] = str(e)
            state['confidence'] = 0.0
            state['extracted_text'] = ""
        
        return state
    
    def _evaluate_result_node(self, state: ExtractionState) -> ExtractionState:
        """Node: Evaluate if result meets quality goals"""
        
        confidence = state.get('confidence', 0)
        threshold = settings.textract_fallback_threshold
        
        if confidence >= threshold:
            logger.info(f"Tesseract result acceptable: {confidence:.2f}% >= {threshold}%")
            state['success'] = True
            state['needs_fallback'] = False
        else:
            logger.warning(
                f"Tesseract confidence LOW: {confidence:.2f}% < {threshold}% "
                f"→ Will fallback to AWS Textract"
            )
            state['success'] = False
            state['needs_fallback'] = True
        
        return state
    
    def _should_fallback_to_textract(self, state: ExtractionState) -> str:
        """Conditional: Determine if AWS Textract fallback is needed"""
        if not settings.use_aws_textract_fallback:
            logger.info("AWS Textract fallback disabled in config")
            return "success"
        
        if state.get('needs_fallback', False):
            return "fallback"
        
        return "success"
    
    def _fallback_textract_node(self, state: ExtractionState) -> ExtractionState:
        """Node: Fallback to AWS Textract for better accuracy"""
        _check_cancel()
        file_path = state['file_path']
        start_time = time.time()
        tesseract_confidence = state.get('confidence', 0)
        
        logger.info("Falling back to AWS Textract (PAID)...")
        
        try:
            self._load_aws_textract()
            
            # Use AWS Textract
            if settings.textract_enable_tables:
                logger.info("Using AWS Textract with TABLE extraction")
                text, confidence, metadata = self.aws_textract.extract_with_tables(file_path)
            else:
                logger.info("Using AWS Textract (basic text extraction)")
                text, confidence, metadata = self.aws_textract.extract_text(file_path)
            _check_cancel()
            
            # Update state with Textract result
            improvement = confidence - tesseract_confidence
            
            state['extracted_text'] = text
            state['confidence'] = confidence
            state['method_used'] = 'aws_textract_fallback'
            state['processing_time'] += (time.time() - start_time)
            state['attempted_methods'].append('aws_textract')
            state['metadata'] = {
                **metadata,
                'tesseract_confidence': tesseract_confidence,
                'textract_confidence': confidence,
                'improvement': round(improvement, 2)
            }
            state['success'] = True
            
            logger.info(
                f"AWS Textract completed: confidence={confidence:.2f}%, "
                f"improvement={improvement:+.2f}%"
            )
            
        except asyncio.CancelledError:
            logger.warning("Extraction cancelled")
            raise
        
        except Exception as e:
            logger.error(f"AWS Textract fallback failed: {e}")
            # Keep Tesseract result even if Textract fails
            logger.warning("Keeping Tesseract result despite low confidence")
            state['error'] = f"Textract fallback failed: {str(e)}"
            state['success'] = True  # Don't fail completely
        
        return state
    
    def _correct_image_orientation(self, image_path: str) -> Dict[str, Any]:
        """
        Detect and correct image orientation before OCR.
        Critical for landscape insurance forms.
        """
        try:
            corrected_image, orientation_info = self.orientation_detector.detect_and_correct_orientation(image_path)
            
            # Save corrected image if rotation was applied
            if orientation_info['was_corrected']:
                corrected_path = str(Path(image_path).with_suffix('')) + '_corrected.png'
                corrected_image.save(corrected_path)
                logger.info(f"Saved corrected image to {corrected_path}")
                return {'path': corrected_path, 'info': orientation_info}
            
            return {'path': image_path, 'info': orientation_info}

        
        except asyncio.CancelledError:
            logger.warning("Extraction cancelled")
            raise

        except Exception as e:
            logger.warning(f"Orientation correction failed: {str(e)}, using original")
            return {'path': image_path, 'info': {'was_corrected': False, 'error': str(e)}}
        
    def _is_pdf(self, file_path: str) -> bool:
        """Check if file is a PDF"""
        return Path(file_path).suffix.lower() == '.pdf'
        
    async def execute(
        self, document_id: str, db: Session, **kwargs
    ) -> Dict[str, Any]:
        """
        Execute the extraction workflow.
        
        Args:
            document_id: Document ID to process
            db: Database session
            **kwargs: Additional options
        
        Returns:
            Dict with extraction results and metrics
        """
        metrics = MetricsCollector("extraction_agent").start()
        logger.info(f"Starting LangGraph extraction for: {document_id}")
    
        try:
            
            _check_cancel()
            
            storage = StorageService()
            file_path = storage.get_file_path(document_id)
            logger.info(f"File path resolved: {file_path}")
            
            _check_cancel()
            
            # Get document from database
            document = db.query(Document).filter(Document.id == document_id).first()
            if not document:
                raise ValueError(f"Document {document_id} not found")
            
            # Handle PDF conversion
            if self._is_pdf(file_path):
                result = await self._process_pdf(file_path, document_id, **kwargs)
            else:
                result = await self._process_image(file_path, document_id, **kwargs)
                
            # Save to database
            extraction_id = str(uuid.uuid4())
            db_extraction = Extraction(
                id=extraction_id,
                document_id=document_id,
                text=sanitize_sensitive_info(result.get('text', '')),
                confidence=float(result.get('confidence', 0.0)),
                method_used=result.get('method_used', 'multi-page'),
                pages=result.get('pages', 1),
                processing_time=float(result.get('processing_time', 0.0)),
                extraction_metadata=result.get('metadata', {})
            )
            db.add(db_extraction)
            db.commit()
            db.refresh(db_extraction)
            
            metrics.add_custom_metric('document_type', 'pdf')
            metrics.add_custom_metric('total_pages', result.get('pages', 1))
            metrics.add_custom_metric('avg_confidence', result.get('confidence', 0))
            
            final_metrics = metrics.complete(success=result.get('success', False))
            
            return {
                'success': True,
                'extraction_id': extraction_id,
                'text': result.get('text'),
                'confidence': result.get('confidence'),
                'method_used': result.get('method_used'),
                'pages': result.get('pages'),
                'processing_time': result.get('processing_time'),
                'metadata': result.get('metadata'),
                'metrics': final_metrics
            }
            
        except asyncio.CancelledError:
            logger.warning("Extraction cancelled")
            raise
        
        except Exception as e:
            logger.error(f"Extraction failed with exception: {e}", exc_info=True)
            metrics.record_error(str(e), severity='critical')
            final_metrics = metrics.complete(success=False)
            
            return {
                "success": False,
                "extraction_id": None,
                "error": str(e),
                "text": "",
                "confidence": 0.0,
                "method_used": None,
                "pages": 0,
                "processing_time": 0.0,
                "attempted_methods": [],
                "metadata": {},
                "metrics": final_metrics
            }
    
    async def _process_pdf(
        self, 
        pdf_path: str, 
        document_id: str, 
        **kwargs
    ) -> Dict[str, Any]:
        """Process multi-page PDF document"""
        
        start_time = time.time()
        
        # Convert PDF to images
        storage = StorageService()
        document_dir = Path(storage.get_file_path(document_id)).parent
        output_dir = document_dir / "pdf_pages"  # Save in document folder
        _check_cancel()
        image_paths = self.pdf_converter.convert(pdf_path, str(output_dir))
        _check_cancel()
        logger.info(f"PDF converted to {len(image_paths)} pages")
        
        try:
            all_results = []
            
            # Process each page through the graph
            for idx, img_path in enumerate(image_paths, 1):
                logger.info(f"Processing page {idx}/{len(image_paths)}")
                _check_cancel()
                
                # Create state for this page
                page_state: ExtractionState = {
                    'file_path': img_path,
                    'document_id': document_id,
                    'use_preprocessing': kwargs.get('use_preprocessing', True),
                    'correct_orientation': kwargs.get('correct_orientation', True),
                    'doc_analysis': None,
                    'attempted_methods': [],
                    'extracted_text': None,
                    'confidence': 0.0,
                    'method_used': None,
                    'processing_time': 0.0,
                    'success': False,
                    'needs_fallback': False,
                    'error': None,
                    'metadata': {}
                }
                
                # Run graph for this page
                page_result = await self.graph.ainvoke(page_state)
                all_results.append(page_result)
            
            # Combine results from all pages
            combined_text = '\n\n'.join(r.get('extracted_text', '') for r in all_results)
            avg_confidence = sum(r.get('confidence', 0) for r in all_results) / len(all_results)

            # Count method usage
            method_counts = {}
            for r in all_results:
                method = r.get('method_used', 'unknown')
                method_counts[method] = method_counts.get(method, 0) + 1
            
            primary_method = max(method_counts, key=method_counts.get) if method_counts else 'unknown'
            
            return {
                'text': combined_text,
                'confidence': avg_confidence,
                'method_used': f'{primary_method} (multi-page)',
                'pages': len(image_paths),
                'processing_time': time.time() - start_time,
                'attempted_methods': list(set(
                    method for r in all_results 
                    for method in r.get('attempted_methods', [])
                )),
                'success': all([r.get('success') for r in all_results]),
                'metadata': {
                    'page_count': len(image_paths),
                    'page_results': all_results,
                    'pdf_pages_directory': str(output_dir)
                }
            }
        
        # finally:
        #     self._cleanup_temp_images(output_dir)
        
        except asyncio.CancelledError:
            logger.warning("Extraction cancelled")
            raise

        except Exception as e:
            logger.error(f"PDF processing failed: {e}", exc_info=True)
            raise

    async def _process_image(
        self, image_path: str, document_id: str, **kwargs
    ) -> Dict[str, Any]:
        """Process single image through the Tesseract + Textract fallback graph"""
        
        initial_state: ExtractionState = {
            'file_path': image_path,
            'document_id': document_id,
            'use_preprocessing': kwargs.get('use_preprocessing', True),
            'correct_orientation': kwargs.get('correct_orientation', True),
            'doc_analysis': None,
            'attempted_methods': [],
            'extracted_text': None,
            'confidence': 0.0,
            'method_used': None,
            'processing_time': 0.0,
            'needs_fallback': False,
            'success': False,
            'error': None,
            'metadata': {}
        }
        
        # Run the graph
        final_state = await self.graph.ainvoke(initial_state)
        
        if not final_state.get('extracted_text'):
            raise Exception("No text extracted from document")
        
        return {
            'text': final_state['extracted_text'],
            'confidence': final_state['confidence'],
            'method_used': final_state['method_used'],
            'pages': 1,
            'processing_time': final_state['processing_time'],
            'attempted_methods': final_state['attempted_methods'],
            'metadata': final_state.get('metadata', {}),
            'success': final_state.get('success', False)
        }
    
    def _cleanup_temp_images(self, temp_dir: Path):
        """Clean up temporary image files"""
        try:
            if temp_dir.exists():
                import shutil
                shutil.rmtree(temp_dir)
                logger.info("Cleaned up temporary images")
        except Exception as e:
            logger.warning(f"Failed to cleanup temp images: {str(e)}")

    def _get_quality_tier(self, confidence: float) -> str:
        """Categorize confidence into quality tiers"""
        if confidence >= 90:
            return 'excellent'
        elif confidence >= 75:
            return 'good'
        elif confidence >= 60:
            return 'acceptable'
        elif confidence >= 40:
            return 'poor'
        else:
            return 'very_poor'