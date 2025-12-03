import logging
from typing import Dict, Any

from langgraph.graph import StateGraph, END

from app.agents.graph_state import DocumentProcessingState
from app.agents.extraction_agent import LangGraphExtractionAgent
from app.agents.segment_agent import LangGraphSegmentationAgent

logger = logging.getLogger(__name__)


class LangGraphOrchestratorAgent:
    """
    Master orchestrator using LangGraph.
    
    Graph Flow:
    START → extraction → check_extraction → segmentation → finalize → END
    """
    
    def __init__(self):
        # Initialize sub-agents
        self.extraction_agent = LangGraphExtractionAgent()
        self.segmentation_agent = LangGraphSegmentationAgent()
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build orchestration workflow"""
        
        workflow = StateGraph(DocumentProcessingState)
        
        workflow.add_node("extraction", self._extraction_node)
        workflow.add_node("check_extraction", self._check_extraction_node)
        workflow.add_node("segmentation", self._segmentation_node)
        workflow.add_node("finalize", self._finalize_node)
        
        workflow.set_entry_point("extraction")
        workflow.add_edge("extraction", "check_extraction")
        
        workflow.add_conditional_edges(
            "check_extraction",
            lambda s: "continue" if s['extraction_complete'] else "fail",
            {"continue": "segmentation", "fail": "finalize"}
        )
        
        workflow.add_edge("segmentation", "finalize")
        workflow.add_edge("finalize", END)
        
        return workflow.compile()
    
    async def _extraction_node(self, state: DocumentProcessingState) -> DocumentProcessingState:
        """Node: Run extraction agent"""
        logger.info("Running extraction agent...")
        
        result = await self.extraction_agent.execute(
            file_path=state['file_path'],
            document_id=state['document_id'],
            use_preprocessing=state.get('use_preprocessing', True),
            force_trocr=state.get('force_trocr', False)
        )
        
        state['extraction_result'] = result
        state['extraction_complete'] = result.get('success', False)
        
        return state
    
    def _check_extraction_node(self, state: DocumentProcessingState) -> DocumentProcessingState:
        """Node: Validate extraction succeeded"""
        
        if not state['extraction_complete']:
            logger.error("Extraction failed")
            state['success'] = False
            state['error'] = "Text extraction failed"
        else:
            logger.info("Extraction successful")
        
        return state
    
    async def _segmentation_node(self, state: DocumentProcessingState) -> DocumentProcessingState:
        """Node: Run segmentation agent"""
        logger.info("Running segmentation agent...")
        
        extracted_text = state['extraction_result'].get('text', '')
        
        result = await self.segmentation_agent.execute(
            text=extracted_text,
            document_id=state['document_id']
        )
        
        state['segmentation_result'] = result
        state['segmentation_complete'] = True
        
        return state
    
    def _finalize_node(self, state: DocumentProcessingState) -> DocumentProcessingState:
        """Node: Finalize pipeline results"""
        
        state['success'] = (
            state.get('extraction_complete', False) and
            state.get('segmentation_complete', False)
        )
        
        logger.info(f"Pipeline complete: success={state['success']}")
        
        return state
    
    async def process_document(self, file_path: str, document_id: str, **kwargs) -> Dict[str, Any]:
        """
        Execute complete pipeline
        """
        initial_state: DocumentProcessingState = {
            'file_path': file_path,
            'document_id': document_id,
            'use_preprocessing': kwargs.get('use_preprocessing', True),
            'force_trocr': kwargs.get('force_trocr', False),
            'extraction_state': None,
            'extraction_complete': False,
            'segmentation_state': None,
            'segmentation_complete': False,
            'success': False,
            'extraction_result': None,
            'segmentation_result': None,
            'error': None
        }
        
        logger.info("Starting LangGraph orchestration")
        
        final_state = await self.graph.ainvoke(initial_state)
        
        return {
            'success': final_state['success'],
            'extraction': final_state.get('extraction_result'),
            'segmentation': final_state.get('segmentation_result'),
            'error': final_state.get('error')
        }