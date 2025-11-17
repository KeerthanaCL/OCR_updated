import os
import shutil
import uuid
import aiofiles
from pathlib import Path
from fastapi import UploadFile
from typing import Tuple
from app.config import get_settings

settings = get_settings()

class StorageService:
    """Service for handling file storage operations"""
    
    def __init__(self):
        self.upload_dir = Path(settings.upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
    
    async def save_upload_file(self, file: UploadFile) -> Tuple[str, str, int]:
        """
        Save uploaded file to disk.
        
        Args:
            file: FastAPI UploadFile object
            
        Returns:
            Tuple of (document_id, file_path, file_size)
        """
        # Generate unique document ID
        document_id = str(uuid.uuid4())
        
        # Create document directory
        doc_dir = self.upload_dir / document_id
        doc_dir.mkdir(parents=True, exist_ok=True)
        
        # Save file
        file_path = doc_dir / file.filename
        
        file_size = 0
        async with aiofiles.open(file_path, 'wb') as f:
            while chunk := await file.read(1024 * 1024):  # Read 1MB chunks
                await f.write(chunk)
                file_size += len(chunk)
        
        return document_id, str(file_path), file_size
    
    def get_file_path(self, document_id: str) -> str:
        """Get file path for a document"""
        doc_dir = self.upload_dir / document_id
        if not doc_dir.exists():
            raise FileNotFoundError(f"Document {document_id} not found")
        
        # Get first file in directory
        files = list(doc_dir.iterdir())
        if not files:
            raise FileNotFoundError(f"No files found for document {document_id}")
        
        return str(files[0])
    
    def delete_document(self, document_id: str) -> bool:
        """Delete document and all associated files"""
        doc_dir = self.upload_dir / document_id
        if doc_dir.exists():
            shutil.rmtree(doc_dir)
            return True
        return False