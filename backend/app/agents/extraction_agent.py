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
            self.trocr = TrOCRService()
    
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
                    text, metadata = self.trocr.extract_from_regions(file_path, regions)
                else:
                    text, metadata = self.trocr.extract_text(file_path)
                confidence = 95.0
                method = 'trocr'
            
            # Update state
            state['extracted_text'] = text
            state['confidence'] = confidence
            state['method_used'] = method
            state['processing_time'] = time.time() - start_time
            state['attempted_methods'].append(method)
            
            logger.info(f"Extracted: confidence={confidence:.2f}%, method={method}")
            
        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            state['error'] = str(e)
            state['success'] = False
        
        return state
    
    def _evaluate_result_node(self, state: ExtractionState) -> ExtractionState:
        """Node: Evaluate if result meets quality goals"""
        
        confidence = state.get('confidence', 0)
        target = state.get('target_confidence', 70.0)
        
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
        self, file_path: str, document_id: str, **kwargs
    ) -> Dict[str, Any]:
        """
        Execute the LangGraph workflow
        """
        logger.info(f"Starting LangGraph extraction for: {file_path}")
    
        # Handle PDF conversion
        if self._is_pdf(file_path):
            return await self._process_pdf(file_path, document_id, **kwargs)
        
        # Initialize state
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
            'target_confidence': 70.0,
            'success': False,
            'error': None,
            'metadata': {}
        }
        
        # Run the graph
        final_state = await self.graph.ainvoke(initial_state)
        
        # Return result
        return {
            'text': final_state.get('extracted_text', ''),
            'confidence': final_state.get('confidence', 0.0),
            'method_used': final_state.get('method_used'),
            'pages': 1,
            'processing_time': final_state.get('processing_time'),
            'attempted_methods': final_state.get('attempted_methods', []),
            'success': final_state.get('success'),
            'metadata': final_state.get('metadata', {})
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
                    'target_confidence': 75.0,
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