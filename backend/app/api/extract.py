from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from app.models import (
    ExtractionRequest, 
    DocumentStatus,
    OCRMethod
)
from app.agents.orchestrator import OrchestratorAgent
from app.services.storage import StorageService
from app.database import get_db, Document, Extraction, AppealsExtraction, AppealsSegment
import uuid
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["extraction"])

# Global orchestrator instance
orchestrator = OrchestratorAgent()

@router.post("/extract")
async def extract_and_segment_document(
    request: ExtractionRequest,
    db: Session = Depends(get_db)
):
    """
    Complete document processing pipeline:
    1. Retrieve document using document_id
    2. Extract text via OCR (Tesseract/EasyOCR/TrOCR)
    3. Segment text into references, medical, and legal contexts
    4. Save all results to database
    
    Args:
        request: ExtractionRequest with document_id and processing options
        
    Returns:
        Complete extraction and segmentation results
    """
    try:
        # Step 1: Get document from database
        document = db.query(Document).filter(Document.id == request.document_id).first()
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Update document status
        document.status = DocumentStatus.EXTRACTING.value
        db.commit()
        
        # Step 2: Get file path
        storage = StorageService()
        file_path = storage.get_file_path(request.document_id)
        
        logger.info(f"Processing document: {document.filename} (ID: {request.document_id})")
        
        # Step 3: Execute complete processing pipeline via orchestrator
        result = await orchestrator.process_document(
            file_path=file_path,
            use_preprocessing=request.use_preprocessing,
            force_trocr=request.force_trocr
        )
        
        if not result.get('success'):
            document.status = DocumentStatus.FAILED.value
            db.commit()
            raise HTTPException(
                status_code=500, 
                detail=result.get('error', 'Processing failed')
            )
        
        extraction_data = result['extraction']
        segmentation_data = result['segmentation']
        
        # Step 4: Save extraction results to database
        extraction_id = str(uuid.uuid4())
        
        # Handle OCR method enum conversion
        method_used_str = extraction_data['method_used']
        try:
            method_used_enum = OCRMethod(method_used_str)
        except ValueError:
            method_used_enum = OCRMethod.TESSERACT
        
        db_extraction = Extraction(
            id=extraction_id,
            document_id=request.document_id,
            text=extraction_data['text'],
            confidence=extraction_data['confidence'],
            method_used=method_used_str,
            pages=extraction_data['pages'],
            processing_time=extraction_data['processing_time'],
            extraction_metadata=extraction_data.get('metadata', {})
        )
        db.add(db_extraction)
        
        # Step 5: Save appeals extraction metadata
        appeals_extraction_id = str(uuid.uuid4())
        db_appeals_extraction = AppealsExtraction(
            id=appeals_extraction_id,
            document_id=request.document_id,
            extraction_id=extraction_id,
            appeals_text=extraction_data['text'],  # Full extracted text
            appeals_found=True,
            total_confidence=extraction_data['confidence'],
            processing_time=extraction_data['processing_time'],
            appeals_metadata=result.get('metadata', {})
        )
        db.add(db_appeals_extraction)
        
        # Step 6: Save segmented data (references, medical, legal)
        segments_to_save = [
            {
                'type': 'references',
                'data': segmentation_data.get('references', {})
            },
            {
                'type': 'medical',
                'data': segmentation_data.get('medical', {})
            },
            {
                'type': 'legal',
                'data': segmentation_data.get('legal', {})
            }
        ]
        
        for segment in segments_to_save:
            segment_id = str(uuid.uuid4())
            db_segment = AppealsSegment(
                id=segment_id,
                appeals_extraction_id=appeals_extraction_id,
                segment_type=segment['type'],
                content=str(segment['data']),  # Convert to JSON string
                confidence=segment['data'].get('confidence', 0.0) if isinstance(segment['data'], dict) else 0.0
            )
            db.add(db_segment)
        
        # Step 7: Update document status to completed
        document.status = DocumentStatus.COMPLETED.value
        db.commit()
        
        logger.info(f"✅ Document processing completed: {extraction_id}")
        
        # Step 8: Return complete results
        return {
            'success': True,
            'document_id': request.document_id,
            'extraction_id': extraction_id,
            'extraction': {
                'text': extraction_data['text'],
                'confidence': extraction_data['confidence'],
                'method_used': method_used_enum.value,
                'pages': extraction_data['pages'],
                'processing_time': extraction_data['processing_time']
            },
            'segmentation': segmentation_data,
            'metadata': result.get('metadata', {})
        }
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Document file not found on storage")
    except Exception as e:
        logger.error(f"Extraction failed: {str(e)}", exc_info=True)
        # Update status to failed
        if 'document' in locals():
            document.status = DocumentStatus.FAILED.value
            db.commit()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/extract/status/{document_id}")
async def get_extraction_status(document_id: str, db: Session = Depends(get_db)):
    """
    Get extraction status for a document.
    
    Args:
        document_id: Document ID
        
    Returns:
        Status and results if available
    """
    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Get latest extraction for this document
    extraction = db.query(Extraction).filter(
        Extraction.document_id == document_id
    ).order_by(Extraction.created_at.desc()).first()
    
    response = {
        'document_id': document_id,
        'status': document.status,
        'filename': document.filename,
        'created_at': document.created_at.isoformat(),
        'updated_at': document.updated_at.isoformat()
    }
    
    if extraction:
        response['extraction'] = {
            'extraction_id': extraction.id,
            'confidence': extraction.confidence,
            'method_used': extraction.method_used,
            'pages': extraction.pages,
            'processing_time': extraction.processing_time,
            'text_length': len(extraction.text) if extraction.text else 0
        }
    
    return response