import time
import logging
from typing import Dict, Any, List
from pathlib import Path
import cv2

from langgraph.graph import StateGraph, END

from app.agents.graph_state import ExtractionState
from app.services.tesseract_service import TesseractService
from app.services.trocr_service import TrOCRService
from app.services.easyocr_service import EasyOCRService
from app.services.pdf_converter import PDFConverter
from app.services.orientation_detector import OrientationDetector
from app.services.region_detector import RegionDetector
from app.utils.metrics import MetricsCollector
from app.database import Document, Extraction
from app.services.storage import StorageService
from sqlalchemy.orm import Session
import concurrent.futures
from functools import partial
import uuid
import asyncio

logger = logging.getLogger(__name__)

class LangGraphExtractionAgent:
    """
    LangGraph-based extraction agent with cyclic workflow.
    
    Graph Flow:
    START → analyze_document → decide_strategy → execute_extraction 
    → evaluate_result → [if low confidence] → retry_with_fallback → execute_extraction
    → [if success] → END
    """
    
    def __init__(self):
        self.tesseract = TesseractService()
        self.easyocr = None
        self.trocr = None  # Lazy loading
        self.pdf_converter = PDFConverter(method="pymupdf")
        self.orientation_detector = OrientationDetector()
        self.region_detector = RegionDetector()

        # Build the graph
        self.graph = self._build_graph()

    def _load_easyocr(self):
        """Lazy load EasyOCR"""
        if self.easyocr is None:
            logger.info("Agent loading EasyOCR tool...")
            self.easyocr = EasyOCRService()
    
    def _load_trocr(self):
        """Lazy load TrOCR model (heavy operation)"""
        if self.trocr is None:
            logger.info("Agent loading TrOCR tool...")
            self.trocr = TrOCRService(auto_detect=True)
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        
        # Create graph
        workflow = StateGraph(ExtractionState)
        
        # Add nodes
        workflow.add_node("analyze_document", self._analyze_document_node)
        workflow.add_node("decide_strategy", self._decide_strategy_node)
        workflow.add_node("execute_extraction", self._execute_extraction_node)
        workflow.add_node("evaluate_result", self._evaluate_result_node)
        workflow.add_node("retry_with_fallback", self._retry_fallback_node)
        
        # Set entry point
        workflow.set_entry_point("analyze_document")
        
        # Add edges
        workflow.add_edge("analyze_document", "decide_strategy")
        workflow.add_edge("decide_strategy", "execute_extraction")
        workflow.add_edge("execute_extraction", "evaluate_result")
        
        # Conditional edge from evaluate_result
        workflow.add_conditional_edges(
            "evaluate_result",
            self._should_retry,
            {
                "retry": "retry_with_fallback",
                "success": END
            }
        )
        
        # Edge from retry back to execution
        workflow.add_edge("retry_with_fallback", "execute_extraction")
        
        return workflow.compile()
    
    def _analyze_document_node(self, state: ExtractionState) -> ExtractionState:
        """Node: Analyze document characteristics"""
        logger.info("Analyzing document...")
        
        try:
            img = cv2.imread(state['file_path'], cv2.IMREAD_GRAYSCALE)
            if img is None:
                state['doc_analysis'] = {'analysis_failed': True}
                return state
            
            regions = self.region_detector.detect_text_regions(state['file_path'])
            blur_score = cv2.Laplacian(img, cv2.CV_64F).var()
            height, width = img.shape
            
            state['doc_analysis'] = {
                'num_regions': len(regions),
                'is_simple_layout': len(regions) <= 3,
                'is_complex_layout': len(regions) > 10,
                'has_tables': len(regions) > 10,
                'quality': 'high' if blur_score > 500 else 'medium' if blur_score > 100 else 'low',
                'image_size': (width, height)
            }
            
            logger.info(f"Analysis: {state['doc_analysis']['num_regions']} regions, "
                       f"quality={state['doc_analysis']['quality']}")
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            state['doc_analysis'] = {'analysis_failed': True}
        
        return state
    
    def _decide_strategy_node(self, state: ExtractionState) -> ExtractionState:
        """Node: Decide extraction strategy based on analysis"""
        
        if state.get('force_trocr'):
            state['chosen_strategy'] = 'trocr'
            logger.info("Strategy: trocr (forced)")
            return state
        
        doc_analysis = state.get('doc_analysis', {})
        
        if doc_analysis.get('analysis_failed'):
            state['chosen_strategy'] = 'tesseract'
            logger.info("Strategy: tesseract (default)")
            return state
        
        # Decision logic
        if doc_analysis.get('quality') == 'low':
            state['chosen_strategy'] = 'trocr'
            logger.info("Strategy: trocr (low quality)")
        elif doc_analysis.get('is_complex_layout'):
            state['chosen_strategy'] = 'tesseract_region'
            logger.info("Strategy: tesseract_region (complex layout)")
        else:
            state['chosen_strategy'] = 'tesseract'
            logger.info("Strategy: tesseract (default)")
        
        return state
    
    def _execute_extraction_node(self, state: ExtractionState) -> ExtractionState:
        """Node: Execute chosen extraction strategy"""
        
        strategy = state['chosen_strategy']
        file_path = state['file_path']
        start_time = time.time()
        
        logger.info(f"Executing strategy: {strategy}")
        
        try:
            # Correct orientation if needed
            if state.get('correct_orientation'):
                orientation_result = self._correct_image_orientation(file_path)
                file_path = orientation_result['path']
                state['file_path'] = file_path
                state['orientation_info'] = orientation_result['info']
            
            # Execute strategy
            if strategy == 'tesseract' or strategy == 'tesseract_region':
                text, confidence, metadata = self.tesseract.extract_text_region_based(file_path)
                method = 'tesseract'
            
            elif strategy == 'easyocr':
                self._load_easyocr()
                text, confidence, metadata = self.easyocr.extract_text(file_path)
                method = 'easyocr'
            
            elif strategy == 'trocr':
                self._load_trocr()
                regions = self.region_detector.detect_text_regions(file_path)
                if regions:
                    text, confidence, metadata = self.trocr.extract_from_regions_with_confidence(file_path, regions)
                else:
                    text, confidence, metadata = self.trocr.extract_text_with_confidence(file_path)
                method = 'trocr'
            
            # Update state
            state['extracted_text'] = text
            state['confidence'] = confidence
            state['method_used'] = method
            state['processing_time'] = time.time() - start_time
            state['attempted_methods'].append(method)
            
            logger.info(f"Extracted: confidence={confidence:.2f}%, method={method}")

            latency_ms = (time.time() - start_time) * 1000
            
        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            state['error'] = str(e)
            state['success'] = False
        
        return state
    
    def _evaluate_result_node(self, state: ExtractionState) -> ExtractionState:
        """Node: Evaluate if result meets quality goals"""
        
        confidence = state.get('confidence', 0)
        target = state.get('target_confidence', 60.0)
        
        if confidence >= target:
            logger.info(f"Target achieved: {confidence:.2f}% >= {target}%")
            state['success'] = True
        else:
            logger.warning(f"Below target: {confidence:.2f}% < {target}%")
            state['success'] = False
        
        return state
    
    def _should_retry(self, state: ExtractionState) -> str:
        """Conditional: Determine if retry is needed"""
        
        if state.get('success'):
            return "success"
        
        if state['retry_count'] >= state['max_retries']:
            logger.warning("Max retries reached")
            state['success'] = False
            return "success"  # End even if not ideal
        
        return "retry"
    
    def _retry_fallback_node(self, state: ExtractionState) -> ExtractionState:
        """Node: Decide fallback strategy for retry"""
        
        state['retry_count'] += 1
        current_method = state.get('method_used')
        attempted = state.get('attempted_methods', [])
        
        logger.info(f"Retry {state['retry_count']}/{state['max_retries']}")
        
        # Fallback chain
        if current_method == 'tesseract' and 'easyocr' not in attempted:
            state['chosen_strategy'] = 'easyocr'
        elif 'trocr' not in attempted:
            state['chosen_strategy'] = 'trocr'
        else:
            logger.warning("No more fallback options")
            state['success'] = False
        
        return state

    def _is_pdf(self, file_path: str) -> bool:
        """Check if file is a PDF"""
        return Path(file_path).suffix.lower() == '.pdf'
    
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
            
        except Exception as e:
            logger.warning(f"Orientation correction failed: {str(e)}, using original")
            return {'path': image_path, 'info': {'was_corrected': False, 'error': str(e)}}
        
    async def execute(
        self, document_id: str, db: Session, **kwargs
    ) -> Dict[str, Any]:
        """
        Execute the LangGraph workflow
        """
        metrics = MetricsCollector("extraction_agent").start()
        logger.info(f"Starting LangGraph extraction for: {document_id}")
    
        try:
            storage = StorageService()
            file_path = storage.get_file_path(document_id)
            logger.info(f"File path resolved: {file_path}")
            
            # Get document from database
            document = db.query(Document).filter(Document.id == document_id).first()
            if not document:
                raise ValueError(f"Document {document_id} not found")
            
            # Rest of your code continues here...
            # Handle PDF conversion
            if self._is_pdf(file_path):
                result = await self._process_pdf(file_path, document_id, **kwargs)
                
                # Save to database
                extraction_id = str(uuid.uuid4())
                db_extraction = Extraction(
                    id=extraction_id,
                    document_id=document_id,
                    text=result.get('text', ''),
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
            
            # Image processing (existing code continues...)
            metrics.add_custom_metric('document_type', 'image')
            
            initial_state: ExtractionState = {
                'file_path': file_path,
                'document_id': document_id,
                'use_preprocessing': kwargs.get('use_preprocessing', True),
                'force_trocr': kwargs.get('force_trocr', False),
                'correct_orientation': kwargs.get('correct_orientation', True),
                'doc_analysis': None,
                'chosen_strategy': None,
                'attempted_methods': [],
                'extracted_text': None,
                'confidence': 0.0,
                'method_used': None,
                'processing_time': 0.0,
                'retry_count': 0,
                'max_retries': 2,
                'target_confidence': 60.0,
                'success': False,
                'error': None,
                'metadata': {}
            }
            
            # Run the graph
            final_state = await self.graph.ainvoke(initial_state)
            
            # Save to database
            extraction_id = str(uuid.uuid4())
            
            extracted_text = final_state.get('extracted_text')
            if not extracted_text:
                raise Exception("No text extracted from document")
            
            db_extraction = Extraction(
                id=extraction_id,
                document_id=document_id,
                text=extracted_text,
                confidence=float(final_state.get('confidence', 0.0)),
                method_used=final_state.get('method_used', 'tesseract'),
                pages=1,
                processing_time=float(final_state.get('processing_time', 0.0)),
                extraction_metadata=final_state.get('metadata', {})
            )
            
            db.add(db_extraction)
            db.commit()
            db.refresh(db_extraction)
            
            # Extract metrics from final state
            confidence = float(final_state.get("confidence", 0))
            target_confidence = float(final_state.get("target_confidence", 70.0))
            attempted_methods = final_state.get("attempted_methods", [])
            retry_count = int(final_state.get("retry_count", 0))
            
            # Add extraction metrics
            metrics.add_custom_metric('confidence_score', confidence)
            metrics.add_custom_metric('target_confidence', target_confidence)
            metrics.add_custom_metric('confidence_gap', target_confidence - confidence)
            metrics.add_custom_metric('ocr_method_used', final_state.get("method_used"))
            metrics.add_custom_metric('attempted_methods', attempted_methods)
            metrics.add_custom_metric('retry_count', retry_count)
            metrics.add_custom_metric('fallback_used', retry_count > 0)
            metrics.add_custom_metric('text_length', len(extracted_text))
            metrics.add_custom_metric('word_count', len(extracted_text.split()))
            metrics.add_custom_metric('quality_tier', self._get_quality_tier(confidence))
            
            # Document analysis metrics
            doc_analysis = final_state.get('doc_analysis', {})
            if doc_analysis and not doc_analysis.get('analysis_failed'):
                metrics.add_custom_metric('document_quality', doc_analysis.get('quality'))
                metrics.add_custom_metric('text_regions', doc_analysis.get('num_regions'))
                metrics.add_custom_metric('layout_complexity',
                    'complex' if doc_analysis.get('is_complex_layout')
                    else 'simple' if doc_analysis.get('is_simple_layout')
                    else 'medium')
            
            # Success criteria metrics
            task_completed = final_state.get("success", False)
            goal_achieved = confidence >= target_confidence
            
            metrics.add_custom_metric('task_completed', task_completed)
            metrics.add_custom_metric('goal_achieved', goal_achieved)
            metrics.add_custom_metric('success_rate', 100.0 if task_completed else 0.0)
            
            # Record errors if any
            if final_state.get('error'):
                metrics.record_error(final_state.get('error'))
            
            # Finalize metrics
            final_metrics = metrics.complete(success=task_completed)
            
            # Return result with metrics
            return {
                "success": True,
                "extraction_id": extraction_id,
                "text": extracted_text,
                "confidence": confidence,
                "method_used": final_state.get("method_used"),
                "pages": 1,
                "processing_time": final_state.get("processing_time"),
                "attempted_methods": attempted_methods,
                "metadata": final_state.get("metadata", {}),
                "metrics": final_metrics
            }
            
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
        output_dir = Path(pdf_path).parent / "temp_images"
        image_paths = self.pdf_converter.convert(pdf_path, str(output_dir))
        
        logger.info(f"PDF converted to {len(image_paths)} pages")
        
        try:
            all_results = []
            
            # Process each page through the graph
            for idx, img_path in enumerate(image_paths, 1):
                logger.info(f"Processing page {idx}/{len(image_paths)}")
                
                # Create state for this page
                page_state: ExtractionState = {
                    'file_path': img_path,
                    'document_id': document_id,
                    'use_preprocessing': kwargs.get('use_preprocessing', True),
                    'force_trocr': kwargs.get('force_trocr', False),
                    'correct_orientation': kwargs.get('correct_orientation', True),
                    'doc_analysis': None,
                    'chosen_strategy': None,
                    'attempted_methods': [],
                    'extracted_text': None,
                    'confidence': 0.0,
                    'method_used': None,
                    'processing_time': 0.0,
                    'retry_count': 0,
                    'max_retries': 2,
                    'target_confidence': 60.0,
                    'success': False,
                    'error': None,
                    'metadata': {}
                }
                
                # Run graph for this page
                page_result = await self.graph.ainvoke(page_state)
                all_results.append(page_result)
            
            # Combine results from all pages
            combined_text = '\n\n'.join(r.get('extracted_text', '') for r in all_results)
            avg_confidence = sum(r.get('confidence', 0) for r in all_results) / len(all_results)
            
            return {
                'text': combined_text,
                'confidence': avg_confidence,
                'method_used': 'multi-page',
                'pages': len(image_paths),
                'processing_time': time.time() - start_time,
                'attempted_methods': list(set(
                    method for r in all_results 
                    for method in r.get('attempted_methods', [])
                )),
                'success': all([r.get('success') for r in all_results]),
                'metadata': {
                    'page_count': len(image_paths),
                    'page_results': all_results
                }
            }
        
        finally:
            self._cleanup_temp_images(output_dir)
    
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
        
    async def execute_parallel_best(
        self, document_id: str, db: Session, **kwargs
    ) -> Dict[str, Any]:
        """
        Execute ALL OCR methods in parallel and select the best result.
        
        Strategy: Run Tesseract, EasyOCR, and TrOCR simultaneously for each page,
        then pick the one with highest confidence.
        
        Args:
            document_id: Document ID to process
            db: Database session
            
        Returns:
            Dict with best extraction results
        """
        metrics = MetricsCollector("extraction_agent_parallel").start()
        logger.info(f"Starting PARALLEL multi-method extraction for: {document_id}")
        
        try:
            # Get file path
            storage = StorageService()
            file_path = storage.get_file_path(document_id)
            logger.info(f"File path resolved: {file_path}")
            
            # Get document
            document = db.query(Document).filter(Document.id == document_id).first()
            if not document:
                raise ValueError(f"Document {document_id} not found")
            
            start_time = time.time()
            
            # Handle PDF
            if self._is_pdf(file_path):
                result = await self._process_pdf_parallel(file_path, document_id, **kwargs)
            else:
                result = await self._process_image_parallel(file_path, document_id, **kwargs)
            
            # Save to database
            extraction_id = str(uuid.uuid4())
            
            db_extraction = Extraction(
                id=extraction_id,
                document_id=document_id,
                text=result['text'],
                confidence=float(result['confidence']),
                method_used=result['method_used'],
                pages=result['pages'],
                processing_time=float(result['processing_time']),
                extraction_metadata=result.get('metadata', {})
            )
            
            db.add(db_extraction)
            db.commit()
            db.refresh(db_extraction)
            
            logger.info(f"Parallel extraction saved: {extraction_id}, best method: {result['method_used']}")
            
            # Add metrics
            metrics.add_custom_metric('best_method', result['method_used'])
            metrics.add_custom_metric('confidence', result['confidence'])
            metrics.add_custom_metric('methods_compared', len(result.get('all_results', [])))
            
            final_metrics = metrics.complete(success=True)
            
            return {
                'success': True,
                'extraction_id': extraction_id,
                'document_id': document_id,
                'text': result['text'],
                'confidence': result['confidence'],
                'method_used': result['method_used'],
                'pages': result['pages'],
                'processing_time': result['processing_time'],
                'metadata': result.get('metadata', {}),
                'all_results': result.get('all_results', []),  # All method results for comparison
                'metrics': final_metrics
            }
            
        except Exception as e:
            logger.error(f"Parallel extraction failed: {str(e)}", exc_info=True)
            metrics.record_error(str(e), severity='critical')
            final_metrics = metrics.complete(success=False)
            
            return {
                'success': False,
                'extraction_id': None,
                'error': str(e),
                'metrics': final_metrics
            }


    async def _process_image_parallel(
        self, image_path: str, document_id: str, **kwargs
    ) -> Dict[str, Any]:
        """
        Run all 3 OCR methods in parallel for a single image.
        """
        start_time = time.time()
        
        logger.info(f"Running 3 OCR methods in parallel...")
        
        # Load all OCR engines
        self._load_easyocr()
        self._load_trocr()

        logger.info("Starting parallel execution:")
        logger.info("   → Tesseract (region-based)")
        logger.info("   → EasyOCR")
        logger.info("   → TrOCR")
        
        # Run in thread pool since OCR operations are blocking
        loop = asyncio.get_event_loop()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            tasks = [
                loop.run_in_executor(executor, partial(self._run_tesseract, image_path)),
                loop.run_in_executor(executor, partial(self._run_easyocr, image_path)),
                loop.run_in_executor(executor, partial(self._run_trocr, image_path))
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        tesseract_result, easyocr_result, trocr_result = results
        
        logger.info("All 3 methods completed, analyzing results...")
        
        # Collect valid results
        all_results = []
        
        if not isinstance(tesseract_result, Exception):
            all_results.append({
                'method': 'tesseract',
                'text': tesseract_result['text'],
                'confidence': tesseract_result['confidence'],
                'metadata': tesseract_result.get('metadata', {})
            })
            logger.info(f"Tesseract: {tesseract_result['confidence']:.2f}% confidence")
        else:
            logger.error(f"Tesseract failed: {tesseract_result}")
        
        if not isinstance(easyocr_result, Exception):
            all_results.append({
                'method': 'easyocr',
                'text': easyocr_result['text'],
                'confidence': easyocr_result['confidence'],
                'metadata': easyocr_result.get('metadata', {})
            })
            logger.info(f"EasyOCR: {easyocr_result['confidence']:.2f}% confidence")
        else:
            logger.error(f"EasyOCR failed: {easyocr_result}")
        
        if not isinstance(trocr_result, Exception):
            all_results.append({
                'method': 'trocr',
                'text': trocr_result['text'],
                'confidence': trocr_result['confidence'],
                'metadata': trocr_result.get('metadata', {})
            })
            logger.info(f"TrOCR: {trocr_result['confidence']:.2f}% confidence")
        else:
            logger.error(f"TrOCR failed: {trocr_result}")
        
        if not all_results:
            raise Exception("All OCR methods failed")
        
        # Select best result
        best = self._select_best_result(all_results)
        
        logger.info(f"Best method: {best['method']} with {best['confidence']:.2f}% confidence")
        
        return {
            'text': best['text'],
            'confidence': best['confidence'],
            'method_used': best['method'],
            'pages': 1,
            'processing_time': time.time() - start_time,
            'metadata': {
                'selection_strategy': 'highest_confidence',
                'methods_tried': len(all_results),
                'all_confidences': {r['method']: r['confidence'] for r in all_results}
            },
            'all_results': all_results
        }


    async def _process_pdf_parallel(
        self, pdf_path: str, document_id: str, **kwargs
    ) -> Dict[str, Any]:
        """
        Process multi-page PDF with parallel method execution per page.
        """
        start_time = time.time()
        
        # Convert PDF to images
        output_dir = Path(pdf_path).parent / "temp_images"
        image_paths = self.pdf_converter.convert(pdf_path, str(output_dir))
        
        logger.info(f"PDF converted to {len(image_paths)} pages")
        
        try:
            page_results = []
            
            # Process each page with parallel methods
            for idx, img_path in enumerate(image_paths, 1):
                logger.info(f"Processing page {idx}/{len(image_paths)} with parallel methods...")
                
                page_result = await self._process_image_parallel(img_path, document_id, **kwargs)
                page_results.append(page_result)
            
            # Combine results
            combined_text = '\n\n'.join(r['text'] for r in page_results)
            avg_confidence = sum(r['confidence'] for r in page_results) / len(page_results)
            
            # Count which method won most often
            method_wins = {}
            for r in page_results:
                method = r['method_used']
                method_wins[method] = method_wins.get(method, 0) + 1
            
            best_overall_method = max(method_wins, key=method_wins.get)
            
            logger.info(f"Method wins: {method_wins}, Overall best: {best_overall_method}")
            
            return {
                'text': combined_text,
                'confidence': avg_confidence,
                'method_used': f'parallel-best ({best_overall_method} won {method_wins[best_overall_method]}/{len(page_results)} pages)',
                'pages': len(image_paths),
                'processing_time': time.time() - start_time,
                'metadata': {
                    'method_wins': method_wins,
                    'page_results': page_results
                },
                'all_results': page_results
            }
            
        finally:
            self._cleanup_temp_images(output_dir)


    def _select_best_result(self, results: List[Dict]) -> Dict:
        """
        Select the best OCR result based on confidence and text quality.
        
        Selection criteria (in order):
        1. Highest confidence
        2. If confidence is close (within 5%), prefer longest text
        3. Prefer Tesseract/EasyOCR over TrOCR (faster for same quality)
        """
        if len(results) == 1:
            return results[0]
        
        # Sort by confidence (descending)
        sorted_results = sorted(results, key=lambda r: r['confidence'], reverse=True)
        
        best = sorted_results[0]
        runner_up = sorted_results[1] if len(sorted_results) > 1 else None
        
        # If confidence is very close (within 5%), prefer longer text
        if runner_up and abs(best['confidence'] - runner_up['confidence']) < 5.0:
            best_len = len(best['text'])
            runner_len = len(runner_up['text'])
            
            if runner_len > best_len * 1.1:  # Runner-up has 10% more text
                logger.info(
                    f"Choosing {runner_up['method']} over {best['method']} "
                    f"(similar confidence but more text: {runner_len} vs {best_len})"
                )
                return runner_up
        
        return best


    def _run_tesseract(self, image_path: str) -> Dict:
        """Run Tesseract OCR"""
        try:
            text, confidence, metadata = self.tesseract.extract_text_region_based(image_path)
            logger.info(f"Tesseract: Completed with {confidence:.2f}% confidence")
            return {
                'text': text,
                'confidence': confidence,
                'metadata': metadata
            }
        except Exception as e:
            raise Exception(f"Tesseract failed: {str(e)}")


    def _run_easyocr(self, image_path: str) -> Dict:
        """Run EasyOCR"""
        try:
            text, confidence, metadata = self.easyocr.extract_text(image_path)
            logger.info(f"EasyOCR: Completed with {confidence:.2f}% confidence")
            return {
                'text': text,
                'confidence': confidence,
                'metadata': metadata
            }
        except Exception as e:
            raise Exception(f"EasyOCR failed: {str(e)}")


    def _run_trocr(self, image_path: str) -> Dict:
        """Run TrOCR"""
        try:
            regions = self.region_detector.detect_text_regions(image_path)
            if regions:
                text, confidence, metadata = self.trocr.extract_from_regions_with_confidence(image_path, regions)
            else:
                text, confidence, metadata = self.trocr.extract_text_with_confidence(image_path)

            logger.info(f"TrOCR: Completed with {confidence:.2f}% confidence")
            return {
                'text': text,
                'confidence': confidence,  # TrOCR doesn't provide confidence, assume high
                'metadata': metadata
            }
        except Exception as e:
            raise Exception(f"TrOCR failed: {str(e)}")