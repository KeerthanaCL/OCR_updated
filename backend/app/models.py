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

class InsuranceClaimData(BaseModel):
    """Structured insurance claim data"""
    # Personal Information
    patient_name: Optional[str] = None
    policy_number: Optional[str] = None
    claim_number: Optional[str] = None
    member_id: Optional[str] = None
    
    # Contact Information
    address: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    
    # Medical Information
    diagnosis: Optional[str] = None
    diagnosis_codes: Optional[List[str]] = []
    procedure_codes: Optional[List[str]] = []
    date_of_service: Optional[str] = None
    
    # Financial Information
    total_amount: Optional[float] = None
    claimed_amount: Optional[float] = None
    deductible: Optional[float] = None
    copay: Optional[float] = None
    
    # Provider Information
    provider_name: Optional[str] = None
    provider_npi: Optional[str] = None
    
    # Bill Items
    bill_items: Optional[List[Dict[str, Any]]] = []
    
    # Metadata
    has_signature: bool = False
    orientation_corrected: bool = False
    tables_found: int = 0