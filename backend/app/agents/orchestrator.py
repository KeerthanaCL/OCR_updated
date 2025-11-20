import uuid
import time
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
from app.agents.extraction_agent import ExtractionAgent
from app.models import OCRMethod, JobStatusEnum
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class Job:
    """
    Represents a document processing job.
    Tracks status, results, and metadata throughout the pipeline.
    """
    
    def __init__(self, job_id: str, document_id: str):
        self.job_id = job_id
        self.document_id = document_id
        self.status = JobStatusEnum.PENDING
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        self.results = {}
        self.error = None
        self.current_step = "initialized" 
        self.progress = 0.0
        self.metadata = {
            "steps_completed": [],
            "steps_failed": []
        }
    
    def update_status(self, status: JobStatusEnum, error: Optional[str] = None):
        """Update job status and timestamp."""
        self.status = status
        self.updated_at = datetime.utcnow()
        if error:
            self.error = error
    
    def add_step(self, step_name: str, success: bool = True):
        """Track completed or failed steps."""
        self.current_step = step_name

        if success:
            self.metadata["steps_completed"].append({
                "step": step_name,
                "timestamp": datetime.utcnow().isoformat()
            })

            total_steps = 4  # ocr_start, ocr_complete, parsing_start, parsing_complete
            completed = len(self.metadata["steps_completed"])
            self.progress = min((completed / total_steps) * 100, 100)

        else:
            self.metadata["steps_failed"].append({
                "step": step_name,
                "timestamp": datetime.utcnow().isoformat()
            })
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert job to dictionary."""
        return {
            "job_id": self.job_id,
            "document_id": self.document_id,
            "status": self.status.value if isinstance(self.status, JobStatusEnum) else self.status,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "current_step": self.current_step, 
            "progress": self.progress,
            "results": self.results,
            "error": self.error,
            "metadata": self.metadata
        }


class AgentOrchestrator:
    """
    Main orchestrator for document processing pipeline.
    Coordinates extraction, parsing, and appeals processing.
    
    Pipeline Stages:
    1. OCR Extraction (Tesseract/TrOCR/EasyOCR)
    2. Optional: General Parsing (claims, invoices, etc.)
    3. Optional: Appeals Extraction and Validation
    """
    
    def __init__(self):
        self.extraction_agent = ExtractionAgent()
        self.jobs: Dict[str, Job] = {}  # In-memory job tracking
        logger.info("AgentOrchestrator initialized")
    
    def create_job(self, document_id: str) -> str:
        """
        Create a new processing job.
        
        Args:
            document_id: Unique document identifier
            
        Returns:
            job_id: Unique job identifier
        """
        job_id = str(uuid.uuid4())
        job = Job(job_id=job_id, document_id=document_id)
        self.jobs[job_id] = job
        
        logger.info(f"Created job {job_id} for document {document_id}")
        return job_id
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get current status of a job.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Job status dictionary or None if not found
        """
        job = self.jobs.get(job_id)
        if not job:
            logger.warning(f"Job {job_id} not found")
            return None
        
        return job.to_dict()
    
    def execute_full_pipeline(
        self,
        job_id: str,
        file_path: str,
        use_preprocessing: bool = True,
        force_trocr: bool = False,
        use_easyocr: bool = False,
        correct_orientation: bool = True,
        apply_parsing: bool = False
    ) -> Dict[str, Any]:
        """
        Execute complete OCR extraction pipeline.
        
        Args:
            job_id: Job identifier
            file_path: Path to document file
            use_preprocessing: Apply image preprocessing
            force_trocr: Force TrOCR usage (skip Tesseract)
            use_easyocr: Use EasyOCR as fallback
            correct_orientation: Auto-correct document orientation
            apply_parsing: Apply structured parsing (claims, invoices)
            
        Returns:
            Dictionary with extraction results
        """
        job = self.jobs.get(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")
        
        try:
            # Update job status to processing
            job.update_status(JobStatusEnum.PROCESSING)
            logger.info(f"Starting extraction pipeline for job {job_id}")
            
            # Step 1: OCR Extraction
            job.add_step("ocr_extraction_started")
            extraction_results = self._execute_extraction(
                file_path=file_path,
                use_preprocessing=use_preprocessing,
                force_trocr=force_trocr,
                use_easyocr=use_easyocr,
                correct_orientation=correct_orientation
            )
            
            job.results['extraction'] = extraction_results
            job.add_step("ocr_extraction_completed")
            
            # Step 2: Optional Parsing
            if apply_parsing:
                job.add_step("parsing_started")
                parsing_results = self._execute_parsing(extraction_results)
                job.results['parsing'] = parsing_results
                job.add_step("parsing_completed")
            
            # Update final status
            job.update_status(JobStatusEnum.COMPLETED)
            logger.info(f"Job {job_id} completed successfully")
            
            # Return extraction results for compatibility
            return extraction_results
            
        except Exception as e:
            logger.error(f"Job {job_id} failed: {e}")
            job.update_status(JobStatusEnum.FAILED, error=str(e))
            job.add_step("pipeline_execution", success=False)
            raise
    
    def _execute_extraction(
        self,
        file_path: str,
        use_preprocessing: bool,
        force_trocr: bool,
        use_easyocr: bool,
        correct_orientation: bool
    ) -> Dict[str, Any]:
        """
        Execute OCR extraction using ExtractionAgent.
        
        Returns:
            Dictionary with text, confidence, method_used, pages, processing_time
        """
        logger.info(f"Extracting text from {file_path}")
        
        start_time = time.time()
        
        try:
            # Call extraction agent
            results = self.extraction_agent.execute(
                file_path=file_path,
                use_preprocessing=use_preprocessing,
                force_trocr=force_trocr,
                use_easyocr=use_easyocr,
                correct_orientation=correct_orientation
            )
            
            processing_time = time.time() - start_time
            
            # Ensure consistent return format
            return {
                'text': results.get('text', ''),
                'confidence': results.get('confidence', 0.0),
                'method_used': results.get('method_used', OCRMethod.TESSERACT),
                'pages': results.get('pages', 1),
                'processing_time': processing_time,
                'metadata': results.get('metadata', {})
            }
            
        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            raise Exception(f"OCR extraction failed: {str(e)}")
    
    def _execute_parsing(self, extraction_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute structured parsing on extracted text.
        
        This is for general document parsing (claims, invoices, forms).
        For appeals-specific processing, use AppealsAgent separately.
        
        Args:
            extraction_results: Results from OCR extraction
            
        Returns:
            Dictionary with parsed structured data
        """
        logger.info("Executing structured parsing")
        
        text = extraction_results.get('text', '')
        
        if not text:
            logger.warning("No text available for parsing")
            return {
                'success': False,
                'error': 'No text extracted',
                'parsed_data': {}
            }
        
        try:
            # Placeholder for parsing logic
            # In production, this would call specialized parsers
            # based on document type (claim form, invoice, etc.)
            
            parsed_data = {
                'document_type': 'unknown',
                'fields_extracted': 0,
                'raw_text_length': len(text),
                'parsing_method': 'basic'
            }
            
            logger.info("Parsing completed")
            return {
                'success': True,
                'parsed_data': parsed_data
            }
            
        except Exception as e:
            logger.error(f"Parsing failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'parsed_data': {}
            }
    
    def execute_appeals_pipeline(
        self,
        job_id: str,
        extracted_text: str,
        validate_with_openai: bool = True
    ) -> Dict[str, Any]:
        """
        Execute appeals-specific processing pipeline.
        
        This should be called AFTER execute_full_pipeline() has completed OCR.
        
        Args:
            job_id: Job identifier
            extracted_text: OCR-extracted text from document
            validate_with_openai: Whether to validate with OpenAI
            
        Returns:
            Dictionary with appeals extraction and validation results
        """
        job = self.jobs.get(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")
        
        try:
            logger.info(f"Starting appeals pipeline for job {job_id}")
            job.add_step("appeals_processing_started")
            
            # Import AppealsAgent here to avoid circular imports
            from app.agents.appeals_agent import AppealsAgent
            
            appeals_agent = AppealsAgent()
            
            # Process appeals (this is async, but we'll handle it in the calling code)
            # For now, return a placeholder that indicates async processing needed
            
            logger.info("Appeals pipeline initiated")
            job.add_step("appeals_processing_initiated")
            
            return {
                'status': 'appeals_processing_initiated',
                'message': 'Use AppealsAgent.process_appeals() for async processing'
            }
            
        except Exception as e:
            logger.error(f"Appeals pipeline failed: {e}")
            job.add_step("appeals_processing", success=False)
            raise
    
    def get_extraction_results(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get extraction results for a completed job.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Extraction results or None
        """
        job = self.jobs.get(job_id)
        if not job:
            return None
        
        return job.results.get('extraction')
    
    def get_parsing_results(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get parsing results for a completed job.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Parsing results or None
        """
        job = self.jobs.get(job_id)
        if not job:
            return None
        
        return job.results.get('parsing')
    
    def cleanup_job(self, job_id: str) -> bool:
        """
        Remove job from memory after processing is complete.
        
        Args:
            job_id: Job identifier
            
        Returns:
            True if job was removed, False if not found
        """
        if job_id in self.jobs:
            del self.jobs[job_id]
            logger.info(f"Cleaned up job {job_id}")
            return True
        return False
    
    def get_all_jobs(self) -> List[Dict[str, Any]]:
        """
        Get status of all jobs (for monitoring/debugging).
        
        Returns:
            List of job status dictionaries
        """
        return [job.to_dict() for job in self.jobs.values()]
    
    def get_active_jobs_count(self) -> int:
        """
        Get count of currently processing jobs.
        
        Returns:
            Number of active jobs
        """
        return sum(
            1 for job in self.jobs.values() 
            if job.status == JobStatusEnum.PROCESSING
        )
    
    def get_job_statistics(self) -> Dict[str, Any]:
        """
        Get overall job statistics.
        
        Returns:
            Dictionary with job statistics
        """
        total_jobs = len(self.jobs)
        completed = sum(1 for job in self.jobs.values() if job.status == JobStatusEnum.COMPLETED)
        failed = sum(1 for job in self.jobs.values() if job.status == JobStatusEnum.FAILED)
        processing = sum(1 for job in self.jobs.values() if job.status == JobStatusEnum.PROCESSING)
        pending = sum(1 for job in self.jobs.values() if job.status == JobStatusEnum.PENDING)
        
        return {
            'total_jobs': total_jobs,
            'completed': completed,
            'failed': failed,
            'processing': processing,
            'pending': pending,
            'success_rate': (completed / total_jobs * 100) if total_jobs > 0 else 0
        }