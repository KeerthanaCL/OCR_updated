from typing import TypedDict, Optional, Dict, Any, List, Annotated
from operator import add


class ExtractionState(TypedDict):
    """State for document extraction workflow"""
    
    # Input
    file_path: str
    document_id: str
    use_preprocessing: bool
    force_trocr: bool
    correct_orientation: bool
    
    # Document Analysis
    doc_analysis: Optional[Dict[str, Any]]
    
    # Extraction Strategy
    chosen_strategy: Optional[str]
    attempted_methods: Annotated[List[str], add]  # Accumulates methods tried
    
    # Current Result
    extracted_text: Optional[str]
    confidence: float
    method_used: Optional[str]
    processing_time: float
    
    # Agent State
    retry_count: int
    max_retries: int
    target_confidence: float
    
    # Final Output
    success: bool
    error: Optional[str]
    metadata: Dict[str, Any]


class SegmentationState(TypedDict):
    """State for text segmentation workflow"""
    
    # Input
    text: str
    document_id: str
    extract_appeals_first: bool
    segments_to_extract: list
    
    # Segmentation Results
    references: Optional[Dict[str, Any]]
    medical: Optional[Dict[str, Any]]
    legal: Optional[Dict[str, Any]]
    
    # Agent State - SEPARATE COUNTERS
    references_retry_count: int  
    medical_retry_count: int     
    legal_retry_count: int       
    max_retries: int
    
    # Validation
    references_valid: bool
    medical_valid: bool
    legal_valid: bool
    
    # Final Output
    success: bool
    overall_success: bool
    error: Optional[str]


class DocumentProcessingState(TypedDict):
    """Complete document processing pipeline state"""
    
    # Input
    file_path: str
    document_id: str
    use_preprocessing: bool
    force_trocr: bool
    
    # Extraction State (nested)
    extraction_state: Optional[ExtractionState]
    extraction_complete: bool
    
    # Segmentation State (nested)
    segmentation_state: Optional[SegmentationState]
    segmentation_complete: bool
    
    # Final Output
    success: bool
    extraction_result: Optional[Dict[str, Any]]
    segmentation_result: Optional[Dict[str, Any]]
    error: Optional[str]