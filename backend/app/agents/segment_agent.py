import logging
from typing import Dict, Any
import time

from langgraph.graph import StateGraph, END

from app.agents.graph_state import SegmentationState
from app.services.appeals_service import AppealsExtractionService
from app.services.openai_extraction_service import OpenAIExtractionService

logger = logging.getLogger(__name__)

class LangGraphSegmentationAgent:
    """
    LangGraph-based segmentation agent with validation and retry.
    
    Graph Flow:
    START → extract_appeals → extract_references → validate_references
    → [if invalid & retries left] → extract_references
    → extract_medical → validate_medical → [retry if needed]
    → extract_legal → validate_legal → [retry if needed]
    → finalize → END
    """
    
    def __init__(self):
        self.appeals_service = AppealsExtractionService()
        self.openai_service = OpenAIExtractionService()  # If using AI for extraction
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build segmentation workflow graph"""
        
        workflow = StateGraph(SegmentationState)
        
        # Add nodes
        workflow.add_node("extract_appeals", self._extract_appeals_node)
        workflow.add_node("extract_references", self._extract_references_node)
        workflow.add_node("validate_references", self._validate_references_node)
        workflow.add_node("extract_medical", self._extract_medical_node)
        workflow.add_node("validate_medical", self._validate_medical_node)
        workflow.add_node("extract_legal", self._extract_legal_node)
        workflow.add_node("validate_legal", self._validate_legal_node)
        workflow.add_node("finalize", self._finalize_node)
        
        # Set entry
        workflow.set_entry_point("extract_appeals")
        
        # Linear flow with conditional retries
        workflow.add_edge("extract_appeals", "extract_references")
        workflow.add_edge("extract_references", "validate_references")
        
        workflow.add_conditional_edges(
            "validate_references",
            lambda s: "retry" if not s['references_valid'] and s['references_retry_count'] < s['max_retries'] else "continue",
            {"retry": "extract_references", "continue": "extract_medical"}
        )
        
        workflow.add_edge("extract_medical", "validate_medical")
        
        workflow.add_conditional_edges(
            "validate_medical",
            lambda s: "retry" if not s['medical_valid'] and s['medical_retry_count'] < s['max_retries'] else "continue",
            {"retry": "extract_medical", "continue": "extract_legal"}
        )
        
        workflow.add_edge("extract_legal", "validate_legal")
        workflow.add_edge("validate_legal", "finalize")
        workflow.add_edge("finalize", END)
        
        return workflow.compile()
    
    def _extract_appeals_node(self, state: SegmentationState) -> SegmentationState:
        """Node: Extract appeals section if requested"""
        
        if not state.get('extract_appeals_first'):
            return state
        
        try:
            appeals_text, found = self.appeals_service.extract_appeals_section(state['text'])
            if found:
                state['text'] = appeals_text
                logger.info(f"Using appeals section: {len(appeals_text)} chars")
        except Exception as e:
            logger.warning(f"Appeals extraction failed: {e}")
        
        return state
    
    async def _extract_references_node(self, state: SegmentationState) -> SegmentationState:
        """
        Node: Extract references
        """
        # Skip if not requested
        if 'references' not in state.get('segments_to_extract', ['references']):
            logger.info("Skipping references extraction (not requested)")
            state['references_valid'] = True  # Mark as valid to skip validation
            return state
        
        logger.info("Extracting references...")
        
        try:
            result = await self.openai_service.extract_references(state['text'])
            state['references'] = result.dict()
        except Exception as e:
            logger.error(f"Reference extraction failed: {e}")
            state['references'] = {'success': False, 'error': str(e)}
        
        return state
    
    async def _extract_medical_node(self, state: SegmentationState) -> SegmentationState:
        """Node: Extract medical context"""
        # Skip if not requested
        if 'medical' not in state.get('segments_to_extract', ['medical']):
            logger.info("Skipping medical extraction (not requested)")
            state['medical_valid'] = True
            return state
        
        logger.info("Extracting medical context...")
        
        try:
            result = await self.openai_service.extract_medical_context(state['text'])
            state['medical'] = result.dict()
        except Exception as e:
            logger.error(f"Medical extraction failed: {e}")
            state['medical'] = {'success': False, 'error': str(e)}
        
        return state
        
    async def _extract_legal_node(self, state: SegmentationState) -> SegmentationState:
        """Node: Extract legal context"""
        # Skip if not requested
        if 'legal' not in state.get('segments_to_extract', ['legal']):
            logger.info("Skipping legal extraction (not requested)")
            state['legal_valid'] = True
            return state
        
        logger.info("Extracting legal context...")
        
        try:
            result = await self.openai_service.extract_legal_context(state['text'])
            state['legal'] = result.dict()
        except Exception as e:
            logger.error(f"Legal extraction failed: {e}")
            state['legal'] = {'success': False, 'error': str(e)}
        
        return state
        
    def _validate_references_node(self, state: SegmentationState) -> SegmentationState:
        """Node: Validate references"""
        
        refs = state.get('references', {})
        ref_count = len(refs.get('patient_details', [])) + len(refs.get('research_papers', []))
        
        state['references_valid'] = ref_count >= 3
        
        if not state['references_valid']:
            state['references_retry_count'] += 1
            logger.warning(f"References validation failed, retry {state['references_retry_count']}")
        else:
            logger.info("References validated")
        
        return state
    
    def _validate_medical_node(self, state: SegmentationState) -> SegmentationState:
        """Node: Validate medical extraction"""
        
        med = state.get('medical', {})
        condition_count = len(med.get('conditions', []))
        
        state['medical_valid'] = condition_count >= 1
        
        if not state['medical_valid'] and state['medical_retry_count'] < state['max_retries']:
            state['medical_retry_count'] += 1  # Use specific counter
            logger.warning("Medical validation failed")
        else:
            logger.info("Medical validated")
        
        return state
    
    def _validate_legal_node(self, state: SegmentationState) -> SegmentationState:
        """Node: Validate legal extraction"""
        
        legal = state.get('legal', {})
        state['legal_valid'] = legal.get('success', False)
        
        if not state['legal_valid'] and state['legal_retry_count'] < state['max_retries']:
            state['legal_retry_count'] += 1  # Use specific counter
            logger.warning(f"Legal validation failed, retry {state['legal_retry_count']}")
        else:
            logger.info("Legal validated")
        
        return state
        
    def extract_appeals_section(self, text: str) -> Dict[str, Any]:
        """
        Extract appeals section from full document text.
        Useful for pre-processing before segment extraction.
        """
        try:
            appeals_text, found = self.appeals_service.extract_appeals_section(text)
            return {
                'success': found,
                'appeals_text': appeals_text,
                'found': found
            }
        except Exception as e:
            logger.error(f"Appeals section extraction failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'found': False
            }
    
    def _finalize_node(self, state: SegmentationState) -> SegmentationState:
        """Node: Finalize results"""
        
        state['overall_success'] = (
            state.get('references_valid', False) and
            state.get('medical_valid', False) and
            state.get('legal_valid', False)
        )
        
        state['success'] = True
        logger.info(f"Segmentation complete: overall_success={state['overall_success']}")
        
        return state
    
    async def execute(self, text: str, document_id: str, segments_to_extract: list = None, **kwargs) -> Dict[str, Any]:
        """
        Execute segmentation workflow

        Args:
        segments_to_extract: List of segments to extract ['references', 'medical', 'legal']
                           If None, extracts all segments (default behavior)
        """
        # Default to extracting all if not specified
        if segments_to_extract is None:
            segments_to_extract = ['references', 'medical', 'legal']

        initial_state: SegmentationState = {
            'text': text,
            'document_id': document_id,
            'extract_appeals_first': kwargs.get('extract_appeals_first', True),
            'segments_to_extract': segments_to_extract,
            'references': None,
            'medical': None,
            'legal': None,
            'references_retry_count': 0,  
            'medical_retry_count': 0,     
            'legal_retry_count': 0, 
            'max_retries': 2,
            'references_valid': False,
            'medical_valid': False,
            'legal_valid': False,
            'success': False,
            'overall_success': False,
            'error': None
        }
        
        logger.info("Starting LangGraph segmentation")
        
        final_state = await self.graph.ainvoke(initial_state)
        
        return {
            'references': final_state.get('references'),
            'medical': final_state.get('medical'),
            'legal': final_state.get('legal'),
            'overall_success': final_state.get('overall_success')
        }