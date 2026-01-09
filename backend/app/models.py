from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

class OCRMethod(str, Enum):
    """OCR methods used for extraction"""
    TESSERACT = "tesseract"
    AWS_TEXTRACT = "aws_textract"
    TESSERACT_TEXTRACT_FALLBACK = "tesseract_textract_fallback"  # Most common
    PADDLEOCR = "paddleocr"  # Optional (for testing)

class DocumentStatus(str, Enum):
    """Document processing status"""
    UPLOADED = "uploaded"
    EXTRACTING = "extracting"
    EXTRACTED = "extracted"
    PARSING = "parsing"
    COMPLETED = "completed"
    FAILED = "failed"

class JobStatusEnum(str, Enum):
    """Job processing status for orchestrator"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class TextUpload(BaseModel):
    """Model for uploading text documents"""
    content: str  

class UploadResponse(BaseModel):
    """Response model for document upload"""
    document_id: str
    filename: str
    file_size: int
    status: DocumentStatus
    message: str

class ExtractionRequest(BaseModel):
    """Request model for text extraction"""
    document_id: str
    use_preprocessing: bool = True

class ExtractionResult(BaseModel):
    """Result model for text extraction"""
    document_id: str
    text: str
    confidence: float
    method_used: OCRMethod
    pages: int
    processing_time: float
    metadata: Dict[str, Any] = {}

class ParsingRequest(BaseModel):
    """Request model for text parsing"""
    document_id: str
    extraction_id: str
    parsing_rules: Optional[Dict[str, Any]] = None

class ParsedField(BaseModel):
    """Individual parsed field"""
    field_name: str
    field_value: str
    confidence: float
    position: Optional[Dict[str, int]] = None

class ParsingResult(BaseModel):
    """Result model for parsed data"""
    document_id: str
    extraction_id: str
    fields: List[ParsedField]
    parsing_method: str
    processing_time: float

class JobStatus(BaseModel):
    """Status of an agent job"""
    job_id: str
    document_id: str
    status: DocumentStatus
    current_step: str
    progress: float
    results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: datetime
    updated_at: datetime

class JobStatusResponse(BaseModel):
    """Response model for job status queries"""
    job_id: str
    document_id: str
    status: str
    created_at: str
    updated_at: str
    results: Dict[str, Any] = {}
    error: Optional[str] = None
    metadata: Dict[str, Any] = {}

class SegmentExtractionRequest(BaseModel):
    """Request for extracting a specific segment type"""
    document_id: Optional[str] = None  # Extract from document
    text: Optional[str] = None  # Or provide text directly
    extract_appeals_first: bool = True  # First extract appeals section

class ReferenceItem(BaseModel):
    """Individual reference found in document"""
    reference_type: str  # "research_paper", "online_resource", "patient_detail", "contact_info", etc.
    content: str
    location: Optional[str] = None
    priority: str = "low"  # "high", "medium", "low"

class ReferencesExtractionResponse(BaseModel):
    """Response from references extraction"""
    success: bool
    research_papers: List[ReferenceItem] = []  # High priority
    online_resources: List[ReferenceItem] = []  # High priority
    summary: str
    confidence: float
    raw_text_analyzed: str
    processing_time: float
    usage: Optional[Dict[str, int]] = None

class MedicalCondition(BaseModel):
    """Medical condition mentioned"""
    condition: str
    diagnosis_date: Optional[str] = None
    severity: Optional[str] = None
    treatment: Optional[str] = None

class Medication(BaseModel):
    """Medication mentioned"""
    name: str
    dosage: Optional[str] = None
    frequency: Optional[str] = None
    purpose: Optional[str] = None

class MedicalContextResponse(BaseModel):
    """Response from medical context extraction"""
    success: bool
    patient_name: Optional[str] = None
    conditions: List[MedicalCondition]
    medications: List[Medication]
    medical_history: str
    medical_necessity_argument: Optional[str] = None
    providers_mentioned: List[str] = []
    confidence: float
    raw_text_analyzed: str
    processing_time: float
    usage: Optional[Dict[str, int]] = None

class LegalClaim(BaseModel):
    """Legal claim or right mentioned"""
    claim_type: str
    description: str
    relevant_statute: Optional[str] = None
    deadline: Optional[str] = None

class LegalContextResponse(BaseModel):
    """Response from legal context extraction"""
    success: bool
    legal_claims: List[LegalClaim]
    appeal_type: Optional[str] = None
    legal_summary: str
    statutes_cited: List[str] = []
    deadlines_mentioned: List[str] = []
    confidence: float
    raw_text_analyzed: str
    processing_time: float
    usage: Optional[Dict[str, int]] = None

# ============= AI DETECTION MODELS =============
class AIDetectionRequest(BaseModel):
    """Request for AI detection"""
    extraction_id: str

class AIDetectionResponse(BaseModel):
    """Response from AI detection"""
    success: bool
    extraction_id: str
    is_ai_generated: Optional[bool] = None
    confidence: float
    detection_method: str
    flags: List[str] = []
    summary: str
    processing_time: float
    error: Optional[str] = None

# ============= HORIZON MODELS =============
class HorizonRequest(BaseModel):
    """Request for horizon segment extraction"""
    extraction_id: str

class SegmentValidation(BaseModel):
    """Validation result for a segment"""
    is_complete: bool
    confidence: float
    quality_issues: List[str] = []
    completeness_score: float
    missing_fields: List[str] = []

class HorizonSegmentResponse(BaseModel):
    """Response for a single horizon segment"""
    success: bool
    extraction_id: str
    segment_type: str  # 'references', 'medical', 'legal'
    data: Dict[str, Any]
    validation: SegmentValidation
    processing_time: float
    error: Optional[str] = None

# ============= ORCHESTRATION MODELS =============
class ProcessRequest(BaseModel):
    """Request to start orchestrated processing"""
    document_id: Optional[str] = None  # For complete processing (extraction + analyses)
    extraction_id: Optional[str] = None  # For analysis-only (text already extracted)

class ProcessResponse(BaseModel):
    """Response from orchestrator"""
    job_id: str
    status: str
    message: str
    started_at: str

class JobStatusResponse(BaseModel):
    """Job status with progressive results"""
    job_id: str
    extraction_id: str
    status: str  # 'processing', 'complete', 'failed'
    progress: int  # 0-100
    
    # Progressive results (null until available)
    ai_detection: Optional[Dict[str, Any]] = None
    references: Optional[Dict[str, Any]] = None
    medical: Optional[Dict[str, Any]] = None
    legal: Optional[Dict[str, Any]] = None
    
    started_at: str
    completed_at: Optional[str] = None
    error: Optional[str] = None