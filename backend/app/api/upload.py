from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from sqlalchemy.orm import Session
from app.models import UploadResponse, DocumentStatus
from app.services.storage import StorageService
from app.database import get_db, Document
from app.config import get_settings
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["upload"])
settings = get_settings()

@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Upload a document for processing.
    
    Supports: PDF, PNG, JPG, JPEG, TIFF
    Max size: 10MB (configurable)
    """
    # Validate file type
    allowed_extensions = {'.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.tif'}
    file_ext = '.' + file.filename.split('.')[-1].lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"File type not supported. Allowed: {', '.join(allowed_extensions)}"
        )
    
    try:
        # Save file
        storage = StorageService()
        document_id, file_path, file_size = await storage.save_upload_file(file)
        
        # # Check file size
        # if file_size > settings.max_file_size:
        #     storage.delete_document(document_id)
        #     raise HTTPException(
        #         status_code=400,
        #         detail=f"File too large. Max size: {settings.max_file_size / (1024*1024):.1f} MB"
        #     )
        
        # Save to database
        db_document = Document(
            id=document_id,
            filename=file.filename,
            file_path=file_path,
            file_size=file_size,
            status=DocumentStatus.UPLOADED.value
        )
        db.add(db_document)
        db.commit()
        
        logger.info(f"Uploaded: {file.filename} ({file_size/(1024*1024):.2f}MB)")
        logger.info(f"Document uploaded: {document_id} ({file.filename})")
        
        return UploadResponse(
            document_id=document_id,
            filename=file.filename,
            file_size=file_size,
            status=DocumentStatus.UPLOADED,
            message=f"{'PDF' if file_ext == '.pdf' else 'Image'} uploaded successfully. Ready for extraction."
        )
        
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))