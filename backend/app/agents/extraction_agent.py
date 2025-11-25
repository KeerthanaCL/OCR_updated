import time
import logging
from typing import Dict, Any, List
from pathlib import Path
import cv2

from app.agents.base_agent import BaseAgent
from app.services.tesseract_service import TesseractService
from app.services.trocr_service import TrOCRService
from app.services.easyocr_service import EasyOCRService
from app.services.pdf_converter import PDFConverter
from app.services.orientation_detector import OrientationDetector
from app.services.region_detector import RegionDetector

logger = logging.getLogger(__name__)

class ExtractionAgent(BaseAgent):
    """
    Intelligent agent for document text extraction.
    
    Agent Capabilities:
    - Autonomous decision-making on OCR method selection
    - Document analysis and strategy planning
    - Adaptive fallback with confidence-based retry
    - Learning from execution history
    - Self-healing and error recovery
    
    Goals:
    - Achieve target confidence (default: 85%)
    - Minimize processing time
    - Maximize text extraction accuracy
    """
    
    def __init__(self):
        super().__init__("ExtractionAgent")

        self.tesseract = TesseractService()
        self.easyocr = None
        self.trocr = None  # Lazy loading
        self.pdf_converter = PDFConverter(method="pymupdf")
        self.orientation_detector = OrientationDetector()
        self.region_detector = RegionDetector()

        # Agent Goals
        self.goals = {
            'target_confidence': 75.0,
            'max_processing_time': 180,  # 3 minutes
            'min_acceptable_confidence': 60.0,
            'prefer_speed': False  # vs accuracy
        }
        
        # Agent State - Performance Tracking
        self.state['performance_metrics'] = {
            'tesseract': {'success_rate': 0.0, 'avg_confidence': 0.0, 'avg_time': 0.0},
            'easyocr': {'success_rate': 0.0, 'avg_confidence': 0.0, 'avg_time': 0.0},
            'trocr': {'success_rate': 0.0, 'avg_confidence': 0.0, 'avg_time': 0.0}
        }
        
        logger.info(f"{self.agent_name} initialized with goals: {self.goals}")

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
        
    def analyze_document(self, file_path: str) -> Dict[str, Any]:
        """
        Agent analyzes document to inform extraction strategy.
        
        Analysis includes:
        - Layout complexity
        - Image quality
        - Document type characteristics
        """
        try:
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return {'analysis_failed': True}
            
            # Detect regions
            regions = self.region_detector.detect_text_regions(file_path)
            
            # Assess image quality
            blur_score = cv2.Laplacian(img, cv2.CV_64F).var()
            
            # Calculate layout complexity
            height, width = img.shape
            region_density = len(regions) / (height * width) if regions else 0
            
            analysis = {
                'num_regions': len(regions),
                'is_simple_layout': len(regions) <= 3,
                'is_complex_layout': len(regions) > 10,
                'has_tables': len(regions) > 10,
                'is_multi_column': any(r['width'] < width * 0.6 for r in regions) if regions else False,
                'quality_score': blur_score,
                'quality': 'high' if blur_score > 500 else 'medium' if blur_score > 100 else 'low',
                'region_density': region_density,
                'image_size': (width, height)
            }
            
            logger.info(f"📊 Document Analysis: {analysis['num_regions']} regions, "
                       f"quality={analysis['quality']}, complexity={'high' if analysis['is_complex_layout'] else 'low'}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Document analysis failed: {e}")
            return {'analysis_failed': True}
        
    def decide_strategy(self, file_path: str, doc_analysis: Dict[str, Any]) -> str:
        """
        Agent makes intelligent decision on extraction strategy.
        
        Decision factors:
        1. Document characteristics
        2. Historical performance on similar documents
        3. Time constraints
        4. Quality requirements
        """
        
        # If analysis failed, use default
        if doc_analysis.get('analysis_failed'):
            logger.info("🎯 Strategy: tesseract (default - analysis failed)")
            return 'tesseract'
        
        # Check learned patterns
        similar_patterns = self._find_similar_documents(doc_analysis)
        if similar_patterns:
            best_method = max(similar_patterns, key=lambda x: x.get('confidence', 0))
            logger.info(f"🧠 Agent learned: {best_method['method']} worked best for similar docs")
            return best_method['method']
        
        # Decision tree based on document characteristics
        if doc_analysis['quality'] == 'low':
            logger.info("🎯 Strategy: trocr (low quality document)")
            return 'trocr'
        
        elif doc_analysis['is_simple_layout'] and doc_analysis['quality'] == 'high':
            logger.info("🎯 Strategy: tesseract (simple layout, high quality)")
            return 'tesseract'
        
        elif doc_analysis['is_complex_layout'] or doc_analysis['has_tables']:
            logger.info("🎯 Strategy: tesseract_region (complex layout)")
            return 'tesseract_region'
        
        elif doc_analysis['is_multi_column']:
            logger.info("🎯 Strategy: tesseract_region (multi-column)")
            return 'tesseract_region'
        
        # Check if EasyOCR has been performing well
        easyocr_metrics = self.state['performance_metrics'].get('easyocr', {})
        if easyocr_metrics.get('success_rate', 0) > 0.85:
            logger.info("🎯 Strategy: easyocr (high historical success rate)")
            return 'easyocr'
        
        # Default
        logger.info("🎯 Strategy: tesseract (default)")
        return 'tesseract'
    
    def _find_similar_documents(self, current_analysis: Dict) -> List[Dict]:
        """Find similar documents in execution history"""
        similar = []
        
        for execution in self.state['execution_history']:
            if 'document_analysis' not in execution.get('result', {}):
                continue
            
            past_analysis = execution['result']['document_analysis']
            
            # Simple similarity check
            if (past_analysis.get('quality') == current_analysis.get('quality') and
                past_analysis.get('is_complex_layout') == current_analysis.get('is_complex_layout')):
                
                similar.append({
                    'method': execution['result'].get('method_used'),
                    'confidence': execution['result'].get('confidence', 0)
                })
        
        return similar
    
    def execute_strategy(self, strategy: str, file_path: str, start_time: float) -> Dict[str, Any]:
        """Execute chosen extraction strategy"""
        
        if strategy == 'tesseract':
            text, confidence, metadata = self.tesseract.extract_text_with_confidence(
                file_path, preprocess=True
            )
            return {
                'text': text,
                'confidence': confidence,
                'method_used': 'tesseract',
                'metadata': metadata,
                'processing_time': time.time() - start_time
            }
        
        elif strategy == 'tesseract_region':
            # Use region-based for complex layouts
            text, confidence, metadata = self.tesseract.extract_text_region_based(
                file_path, preprocess=True
            )
            return {
                'text': text,
                'confidence': confidence,
                'method_used': 'tesseract',
                'metadata': metadata,
                'processing_time': time.time() - start_time
            }
        
        elif strategy == 'easyocr':
            self._load_easyocr()
            text, confidence, metadata = self.easyocr.extract_text(file_path)
            return {
                'text': text,
                'confidence': confidence,
                'method_used': 'easyocr',
                'metadata': metadata,
                'processing_time': time.time() - start_time
            }
        
        elif strategy == 'trocr':
            self._load_trocr()
            regions = self.region_detector.detect_text_regions(file_path)
            if regions:
                text, metadata = self.trocr.extract_from_regions(file_path, regions)
            else:
                text, metadata = self.trocr.extract_text(file_path)
            return {
                'text': text,
                'confidence': 95.0,
                'method_used': 'trocr',
                'metadata': metadata,
                'processing_time': time.time() - start_time
            }
    
    def adaptive_fallback(self, file_path: str, initial_result: Dict, start_time: float) -> Dict[str, Any]:
        """
        Agent autonomously tries alternative strategies if initial attempt fails.
        """
        time_elapsed = time.time() - start_time
        time_remaining = self.goals['max_processing_time'] - time_elapsed
        
        if time_remaining < 10:
            logger.warning("Time constraint reached, accepting current result")
            return initial_result
        
        current_result = initial_result
        attempted_methods = [initial_result['method_used']]
        
        # Define fallback chain
        fallback_chain = {
            'tesseract': 'easyocr',
            'easyocr': 'trocr',
            'trocr': None  # No more fallbacks
        }
        
        # Keep trying fallbacks until target met or chain exhausted
        while current_result['confidence'] < self.goals['target_confidence']:
            current_method = current_result['method_used']
            next_method = fallback_chain.get(current_method)
            
            # No more fallback options
            if next_method is None or next_method in attempted_methods:
                logger.info(f"Exhausted all fallback options, best confidence: {current_result['confidence']:.2f}%")
                break
            
            # Check time constraint
            time_elapsed = time.time() - start_time
            time_remaining = self.goals['max_processing_time'] - time_elapsed
            if time_remaining < 10:
                logger.warning("Time constraint, stopping fallback chain")
                break
            
            # Try next fallback
            logger.info(f"Confidence {current_result['confidence']:.2f}% still below target {self.goals['target_confidence']}%, trying {next_method}")
            
            try:
                fallback_result = self.execute_strategy(next_method, file_path, start_time)
                attempted_methods.append(next_method)
                
                # Use fallback if it's better
                if fallback_result['confidence'] > current_result['confidence']:
                    logger.info(f"{next_method} improved confidence: {current_result['confidence']:.2f}% → {fallback_result['confidence']:.2f}%")
                    current_result = fallback_result
                else:
                    logger.info(f"{next_method} didn't improve ({fallback_result['confidence']:.2f}%), keeping previous result")
                    # Still update current to continue chain
                    current_result = fallback_result if fallback_result['confidence'] >= self.goals['min_acceptable_confidence'] else current_result
                    
            except Exception as e:
                logger.error(f"Fallback to {next_method} failed: {e}")
                break
        
        # Final check
        if current_result['confidence'] >= self.goals['target_confidence']:
            logger.info(f"Target confidence achieved: {current_result['confidence']:.2f}%")
        
        return current_result
    
    def execute(
        self, 
        file_path: str,
        use_preprocessing: bool = True,
        force_trocr: bool = False,
        use_easyocr: bool = False,
        correct_orientation: bool = True
    ) -> Dict[str, Any]:
        """
        Main agent execution with intelligent decision-making and adaptation.
        
        Workflow:
        1. Analyze document
        2. Decide extraction strategy
        3. Execute strategy
        4. Evaluate result against goals
        5. Adaptive fallback if needed
        6. Learn from execution
        7. Return result
        """
        start_time = time.time()

        logger.info(f"{self.agent_name} starting execution for: {file_path}")
        
        try:
            # Handle PDF conversion
            if Path(file_path).suffix.lower() == '.pdf':
                return self._process_pdf(file_path, use_preprocessing, force_trocr, 
                                        use_easyocr, correct_orientation, start_time)
            
            # Step 1: Analyze Document
            doc_analysis = self.analyze_document(file_path)
            
            # Step 2: Decide Strategy (unless forced)
            if force_trocr:
                strategy = 'trocr'
                logger.info("Strategy forced: trocr")
            elif use_easyocr:
                strategy = 'easyocr'
                logger.info("Strategy forced: easyocr")
            else:
                strategy = self.decide_strategy(file_path, doc_analysis)
            
            # Step 3: Correct orientation if needed
            if correct_orientation:
                orientation_result = self._correct_image_orientation(file_path)
                file_path = orientation_result['path']
            
            # Step 4: Execute Strategy
            result = self.execute_strategy(strategy, file_path, start_time)
            result['document_analysis'] = doc_analysis
            
            # Step 5: Evaluate against goals
            if result['confidence'] < self.goals['target_confidence']:
                logger.warning(f"Confidence {result['confidence']:.2f}% below target {self.goals['target_confidence']}%")
                result = self.adaptive_fallback(file_path, result, start_time)
            
            # Step 6: Learn from this execution
            success = result['confidence'] >= self.goals['min_acceptable_confidence']
            self.record_execution('text_extraction', result, success)
            
            # Learn pattern
            pattern_key = f"{doc_analysis.get('quality', 'unknown')}_{doc_analysis.get('is_complex_layout', False)}"
            self.learn_from_execution(pattern_key, {
                'method': result['method_used'],
                'confidence': result['confidence']
            })
            
            # Update performance metrics
            self._update_performance_metrics(result)
            
            logger.info(f"{self.agent_name} completed: {result['method_used']}, "
                       f"confidence={result['confidence']:.2f}%, time={result['processing_time']:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"{self.agent_name} execution failed: {str(e)}")
            self.record_execution('text_extraction', {'error': str(e)}, False)
            raise

    def _process_pdf(
        self,
        pdf_path: str,
        use_preprocessing: bool,
        force_trocr: bool,
        use_easyocr: bool,
        correct_orientation: bool,
        start_time: float
    ) -> Dict[str, Any]:
        """Process PDF document"""
        
        # Convert PDF to images
        output_dir = Path(pdf_path).parent / "temp_images"
        image_paths = self.pdf_converter.convert(pdf_path, str(output_dir))
        
        logger.info(f"PDF converted to {len(image_paths)} images")

        try:
            # Process each page with agent decision-making
            all_results = []
            
            for idx, img_path in enumerate(image_paths, 1):
                logger.info(f"Processing page {idx}/{len(image_paths)}")
                
                # Analyze page
                doc_analysis = self.analyze_document(img_path)

                # Decide strategy for this page
                if force_trocr:
                    strategy = 'trocr'
                elif use_easyocr:
                    strategy = 'easyocr'
                else:
                    strategy = self.decide_strategy(img_path, doc_analysis)

                # Correct orientation
                if correct_orientation:
                    orientation_result = self._correct_image_orientation(img_path)
                    img_path = orientation_result['path']

                # Execute strategy (with page-specific timing)
                page_start = time.time()
                page_result = self.execute_strategy(strategy, img_path, page_start)
                page_result['document_analysis'] = doc_analysis

                # Apply adaptive fallback if confidence is low
                if page_result['confidence'] < self.goals['target_confidence']:
                    logger.warning(f"Page {idx} confidence {page_result['confidence']:.2f}% below target, trying fallback")
                    page_result = self.adaptive_fallback(img_path, page_result, page_start)
                all_results.append(page_result)
            
            # Combine results
            combined_text = '\n\n'.join(r['text'] for r in all_results)
            avg_confidence = sum(r['confidence'] for r in all_results) / len(all_results)
            
            final_result = {
                'text': combined_text,
                'confidence': avg_confidence,
                'method_used': 'multi-page',
                'pages': len(image_paths),
                'processing_time': time.time() - start_time,
                'metadata': {'page_results': all_results}
            }

            success = avg_confidence >= self.goals['min_acceptable_confidence']
            self.record_execution('pdf_extraction', final_result, success)
            self._update_performance_metrics(final_result)

            return final_result
        
        finally:
            self._cleanup_temp_images(output_dir)
    
    def _extract_with_trocr_multi(
        self,
        image_paths: List[str],
        start_time: float
    ) -> Dict[str, Any]:
        """Extract using TrOCR for multiple images"""
        self._load_trocr()
        
        all_texts = []
        total_regions = 0
        
        # Process each page with region detection
        for idx, image_path in enumerate(image_paths, 1):
            regions = self.region_detector.detect_text_regions(image_path)
            
            if regions and len(regions) > 0:
                logger.info(f"Page {idx}/{len(image_paths)}: {len(regions)} regions detected")
                page_text, _ = self.trocr.extract_from_regions(image_path, regions)
                total_regions += len(regions)
            else:
                logger.info(f"Page {idx}/{len(image_paths)}: No regions, using full-page")
                page_text, _ = self.trocr.extract_text(image_path)
            
            all_texts.append(page_text)
        
        # Combine all pages
        full_text = '\n\n'.join(all_texts)
        processing_time = time.time() - start_time
        
        metadata = {
            'page_count': len(image_paths),
            'total_regions': total_regions,
            'extraction_mode': 'region-based',
            'avg_regions_per_page': total_regions / len(image_paths) if image_paths else 0
        }
        
        return {
            'text': full_text,
            'confidence': 95.0,
            'method_used': 'trocr',
            'pages': metadata['page_count'],
            'processing_time': processing_time,
            'metadata': metadata
        }
    
    def _extract_with_easyocr_single(
        self, image_path: str, start_time: float
    ) -> Dict[str, Any]:
        """Extract using EasyOCR for single image"""
        self._load_easyocr()
        
        text, confidence, metadata = self.easyocr.extract_text(image_path)
        processing_time = time.time() - start_time
        
        return {
            'text': text,
            'confidence': confidence,
            'method_used': 'easyocr',
            'pages': 1,
            'processing_time': processing_time,
            'metadata': metadata
        }
    
    def _extract_with_easyocr_multi(
        self, image_paths: List[str], start_time: float
    ) -> Dict[str, Any]:
        """Extract using EasyOCR for multiple images"""
        self._load_easyocr()
        
        text, confidence, metadata = self.easyocr.extract_from_multiple_images(image_paths)
        processing_time = time.time() - start_time
        
        return {
            'text': text,
            'confidence': confidence,
            'method_used': 'easyocr',
            'pages': len(image_paths),
            'processing_time': processing_time,
            'metadata': metadata
        }
    
    def _extract_with_trocr_single(
        self,
        image_path: str,
        start_time: float
    ) -> Dict[str, Any]:
        """Extract using TrOCR for single image"""
        self._load_trocr()
        
        # Detect text regions
        regions = self.region_detector.detect_text_regions(image_path)
        
        if regions and len(regions) > 0:
            # Use region-based extraction
            logger.info(f"Using region-based TrOCR: {len(regions)} regions detected")
            text, metadata = self.trocr.extract_from_regions(image_path, regions)
            metadata['regions_processed'] = len(regions)
            metadata['extraction_mode'] = 'region-based'
        else:
            # Fallback to full-page extraction
            logger.info("No regions detected, using full-page TrOCR")
            text, metadata = self.trocr.extract_text(image_path)
            metadata['extraction_mode'] = 'full-page'
        processing_time = time.time() - start_time
        
        return {
            'text': text,
            'confidence': 95.0,
            'method_used': 'trocr',
            'pages': 1,
            'processing_time': processing_time,
            'metadata': metadata
        }
    
    def _update_performance_metrics(self, result: Dict[str, Any]):
        """Update running performance metrics for each OCR method"""
        method = result['method_used']
        
        if method not in self.state['performance_metrics']:
            self.state['performance_metrics'][method] = {
                'success_rate': 0.0,
                'avg_confidence': 0.0,
                'avg_time': 0.0,
                'count': 0
            }
        
        metrics = self.state['performance_metrics'][method]
        count = metrics['count']
        
        # Update running averages
        metrics['avg_confidence'] = (metrics['avg_confidence'] * count + result['confidence']) / (count + 1)
        metrics['avg_time'] = (metrics['avg_time'] * count + result['processing_time']) / (count + 1)
        metrics['count'] = count + 1
        
        # Calculate success rate
        method_executions = [e for e in self.state['execution_history'] 
                           if e.get('result', {}).get('method_used') == method]
        if method_executions:
            successful = sum(1 for e in method_executions if e.get('success', False))
            metrics['success_rate'] = successful / len(method_executions)
    
    def _cleanup_temp_images(self, temp_dir: Path):
        """Clean up temporary image files"""
        try:
            if temp_dir.exists():
                import shutil
                shutil.rmtree(temp_dir)
                logger.info("Cleaned up temporary images")
        except Exception as e:
            logger.warning(f"Failed to cleanup temp images: {str(e)}")