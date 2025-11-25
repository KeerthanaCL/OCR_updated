import logging
from typing import Dict, Any

from app.agents.base_agent import BaseAgent
from app.agents.extraction_agent import ExtractionAgent
from app.agents.segment_agent import SegmentAgent

logger = logging.getLogger(__name__)


class OrchestratorAgent(BaseAgent):
    """
    Master orchestrator agent coordinating extraction and segmentation agents.
    
    Responsibilities:
    - Coordinate multi-agent workflow
    - Monitor overall pipeline health
    - Make high-level decisions
    - Aggregate results from sub-agents
    """
    
    def __init__(self):
        super().__init__("OrchestratorAgent")
        
        # Initialize sub-agents
        self.extraction_agent = ExtractionAgent()
        self.segmentation_agent = SegmentAgent()
        
        logger.info(f"{self.agent_name} initialized with sub-agents")
    
    async def process_document(
        self,
        file_path: str,
        use_preprocessing: bool = True,
        correct_orientation: bool = True,
        force_trocr: bool = False,
        use_easyocr: bool = False,
        extract_appeals_first: bool = True
    ) -> Dict[str, Any]:
        """
        Orchestrate complete document processing pipeline.
        
        Workflow:
        1. ExtractionAgent extracts text
        2. Validate extraction
        3. SegmentationAgent segments text
        4. Validate segmentation
        5. Aggregate and return results
        """
        logger.info(f"🎭 {self.agent_name} starting pipeline for: {file_path}")
        
        try:
            # Step 1: Text Extraction
            logger.info("Step 1: Text Extraction")
            extraction_result = self.extraction_agent.execute(
                file_path=file_path,
                use_preprocessing=use_preprocessing,
                correct_orientation=correct_orientation,
                force_trocr=force_trocr,
                use_easyocr=use_easyocr
            )
            
            extracted_text = extraction_result.get('text', '')
            
            # Validate extraction
            if not extracted_text or len(extracted_text.strip()) < 50:
                logger.warning("Insufficient text extracted")
                return {
                    'success': False,
                    'error': 'Insufficient text extracted',
                    'extraction': extraction_result
                }
            
            logger.info(f"Extraction complete: {len(extracted_text)} chars, "
                       f"confidence={extraction_result.get('confidence'):.2f}%")
            
            # Step 2: Text Segmentation
            logger.info("Step 2: Text Segmentation")
            segmentation_result = await self.segmentation_agent.extract_all_segments(
                text=extracted_text,
                extract_appeals_first=extract_appeals_first
            )
            
            overall_success = segmentation_result.get('overall_success', False)
            
            logger.info(f"Segmentation complete: success={overall_success}")
            
            # Step 3: Aggregate results
            final_result = {
                'success': True,
                'extraction': extraction_result,
                'segmentation': segmentation_result,
                'metadata': {
                    'file_path': file_path,
                    'ocr_method': extraction_result.get('method_used'),
                    'confidence': extraction_result.get('confidence'),
                    'processing_time': extraction_result.get('processing_time'),
                    'text_length': len(extracted_text),
                    'segmentation_success': overall_success
                }
            }
            
            # Record execution
            self.record_execution('process_document', final_result, overall_success)
            
            logger.info(f"{self.agent_name} pipeline complete!")
            
            return final_result
            
        except Exception as e:
            logger.error(f"{self.agent_name} pipeline failed: {str(e)}")
            self.record_execution('process_document', {'error': str(e)}, False)
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_agent_stats(self) -> Dict[str, Any]:
        """Get statistics from all agents"""
        return {
            'orchestrator': {
                'success_rate': self.get_success_rate(),
                'total_executions': self.state['total_executions']
            },
            'extraction_agent': {
                'success_rate': self.extraction_agent.get_success_rate(),
                'total_executions': self.extraction_agent.state['total_executions'],
                'performance_metrics': self.extraction_agent.state['performance_metrics']
            },
            'segmentation_agent': {
                'success_rate': self.segmentation_agent.get_success_rate(),
                'total_executions': self.segmentation_agent.state['total_executions']
            }
        }