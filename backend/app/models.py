from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

class OCRMethod(str, Enum):
    """OCR methods used for extraction"""
    TESSERACT = "tesseract"
    TROCR = "trocr"
    EASYOCR = "easyocr"
    HYBRID = "hybrid"

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
    force_trocr: bool = False

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
    verified: bool = False
    priority: str = "low"  # "high", "medium", "low"
    validation_notes: Optional[str] = None  # Why it's verified/unverified

class ReferencesExtractionResponse(BaseModel):
    """Response from references extraction"""
    success: bool
    research_papers: List[ReferenceItem] = []  # High priority
    online_resources: List[ReferenceItem] = []  # High priority
    patient_details: List[ReferenceItem] = []  # Medium priority
    other_references: List[ReferenceItem] = []  # Low priority
    summary: str
    confidence: float
    validation_summary: str  # Overall validation assessment
    raw_text_analyzed: str
    processing_time: float

class MedicalCondition(BaseModel):
    """Medical condition mentioned"""
    condition: str
    diagnosis_date: Optional[str] = None
    severity: Optional[str] = None
    treatment: Optional[str] = None
    is_valid_condition: bool = True  # Validated as real medical condition
    validation_notes: Optional[str] = None  # Medical validation feedback

class Medication(BaseModel):
    """Medication mentioned"""
    name: str
    dosage: Optional[str] = None
    frequency: Optional[str] = None
    purpose: Optional[str] = None
    is_valid_medication: bool = True  # Validated as real medication
    is_correct_dosage: bool = True  # Dosage is within normal range
    validation_notes: Optional[str] = None  # Pharmacy validation feedback

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
    validation_summary: str  # Overall medical validation
    medical_accuracy_score: float  # 0-1 score for medical accuracy
    raw_text_analyzed: str
    processing_time: float

class LegalClaim(BaseModel):
    """Legal claim or right mentioned"""
    claim_type: str
    description: str
    relevant_statute: Optional[str] = None
    deadline: Optional[str] = None
    is_valid_claim: bool = True  # Validated as legitimate legal claim
    statute_accuracy: bool = True  # Statute citation is accurate
    validation_notes: Optional[str] = None  # Legal validation feedback

class LegalContextResponse(BaseModel):
    """Response from legal context extraction"""
    success: bool
    legal_claims: List[LegalClaim]
    appeal_type: Optional[str] = None
    legal_summary: str
    statutes_cited: List[str] = []
    deadlines_mentioned: List[str] = []
    confidence: float
    validation_summary: str  # Overall legal validation
    legal_accuracy_score: float  # 0-1 score for legal accuracy
    raw_text_analyzed: str
    processing_time: float