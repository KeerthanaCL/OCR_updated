from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
import uuid
import logging

from app.utils import cancellation_manager
from app.database import get_db, Document, Extraction
from app.models import UploadResponse, DocumentStatus

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["upload"])


class TextUpload(BaseModel):
    content: str = Field(..., min_length=1, max_length=5000)


@router.post("/upload-text", response_model=UploadResponse)
async def upload_text(
    payload: TextUpload,
    db: Session = Depends(get_db)
):
        # STEP 3: Block future requests if cancelled
    if not cancellation_manager.accept_requests:
        raise HTTPException(
            status_code=503,
            detail="Processing has been cancelled by user"
        )

    try:
        text = payload.content.strip()
        if not text:
            raise HTTPException(status_code=400, detail="Text cannot be empty")

        document_id = str(uuid.uuid4())
        extraction_id = str(uuid.uuid4())

        # 1️⃣ Create Document
        document = Document(
            id=document_id,
            filename="pasted_text.txt",
            file_path=None,
            file_size=len(text.encode("utf-8")),
            status=DocumentStatus.COMPLETED.value
        )

        # 2️⃣ Create Extraction (CRITICAL)
        extraction = Extraction(
            id=extraction_id,
            document_id=document_id,
            text=text,
            confidence=1.0,
            method_used="paste",
            pages=1,
            processing_time=0.0,
            extraction_metadata={"source": "paste"}
        )

        db.add(document)
        db.add(extraction)
        db.commit()

        logger.info(f"Pasted text uploaded with extraction_id={extraction_id}")

        return UploadResponse(
            document_id=document_id,
            filename="Pasted Text",
            file_size=len(text),
            status=DocumentStatus.COMPLETED,
            message="Text uploaded and extracted successfully"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Text upload failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to upload text")
