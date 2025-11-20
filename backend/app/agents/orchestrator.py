import uuid
from datetime import datetime
from typing import Dict, Any, Optional
import logging

from app.agents.extraction_agent import ExtractionAgent
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

class JobStatusEnum:
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class Job:
    """Tracks the state and data for a single processing job."""
    def __init__(self, job_id: str, document_id: str):
        self.job_id = job_id
        self.document_id = document_id
        self.status = JobStatusEnum.PENDING
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        self.results = {}
        self.error = None
        self.metadata = {
            "steps_completed": [],
            "steps_failed": []
        }
        self.current_step = None

    def update_status(self, status: str, error: Optional[str] = None):
        self.status = status
        self.updated_at = datetime.utcnow()
        if error:
            self.error = error

    def add_step(self, step_name: str, success: bool = True):
        self.current_step = step_name
        entry = {
            "step": step_name,
            "timestamp": datetime.utcnow().isoformat()
        }
        if success:
            self.metadata["steps_completed"].append(entry)
        else:
            self.metadata["steps_failed"].append(entry)
    
    def to_dict(self):
        return {
            "job_id": self.job_id,
            "document_id": self.document_id,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "results": self.results,
            "error": self.error,
            "metadata": self.metadata,
            "current_step": self.current_step
        }


class AgentOrchestrator:
    """
    Orchestrates the complete document processing pipeline,
    from OCR to segment extraction and (optionally) validation.
    """
    def __init__(self):
        self.extraction_agent = ExtractionAgent()
        self.jobs: Dict[str, Job] = {}
        logger.info("AgentOrchestrator initialized")

    def create_job(self, document_id: str) -> str:
        """Start a new processing job."""
        job_id = str(uuid.uuid4())
        job = Job(job_id, document_id)
        self.jobs[job_id] = job
        logger.info(f"Created job {job_id} for document {document_id}")
        return job_id

    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        job = self.jobs.get(job_id)
        if not job:
            logger.warning(f"Job {job_id} not found")
            return None
        return job.to_dict()

    def execute_extraction_pipeline(
        self,
        job_id: str,
        file_path: str,
        use_preprocessing: bool = True,
        force_trocr: bool = False,
        correct_orientation: bool = True,
        apply_parsing: bool = False
    ) -> Dict[str, Any]:
        """Executes full OCR extraction pipeline and stores result."""
        job = self.jobs.get(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")

        try:
            job.update_status(JobStatusEnum.PROCESSING)
            job.add_step("ocr_extraction_started")
            logger.info(f"Starting OCR for job {job_id} ({file_path})")

            extraction_result = self.extraction_agent.execute(
                file_path=file_path,
                use_preprocessing=use_preprocessing,
                force_trocr=force_trocr,
                use_easyocr=False,
                correct_orientation=correct_orientation
            )
            job.results['extraction'] = extraction_result
            job.add_step("ocr_extraction_completed")
            job.update_status(JobStatusEnum.COMPLETED)
            logger.info(f"Job {job_id} extraction successfully finished")
            return extraction_result

        except Exception as e:
            logger.error(f"Job {job_id} failed: {e}")
            job.update_status(JobStatusEnum.FAILED, error=str(e))
            job.add_step("pipeline_execution_failed", success=False)
            raise

    def execute_segment_extraction(
        self,
        job_id: str,
        text: str,
        segment_service
    ) -> Dict[str, Any]:
        """
        Execute Gemini/OpenAI segment extraction pipeline for references, medical, and legal.
        `segment_service` is your OpenAIExtractionService with Gemini backend.
        """
        job = self.jobs.get(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")
        results = {}

        try:
            job.update_status(JobStatusEnum.PROCESSING)
            job.add_step("segment_extraction_started")

            # References
            references = segment_service.extract_references(text)
            results['references'] = references

            # Medical
            medical = segment_service.extract_medical_context(text)
            results['medical'] = medical

            # Legal
            legal = segment_service.extract_legal_context(text)
            results['legal'] = legal

            job.results['segments'] = results
            job.add_step("segment_extraction_completed")
            job.update_status(JobStatusEnum.COMPLETED)
            logger.info(f"Job {job_id} segment extraction successfully finished")
            return results

        except Exception as e:
            logger.error(f"Job {job_id} segment extraction failed: {e}")
            job.update_status(JobStatusEnum.FAILED, error=str(e))
            job.add_step("segment_extraction_failed", success=False)
            raise

    def get_job_statistics(self) -> Dict[str, Any]:
        total_jobs = len(self.jobs)
        completed = sum(1 for job in self.jobs.values() if job.status == JobStatusEnum.COMPLETED)
        failed = sum(1 for job in self.jobs.values() if job.status == JobStatusEnum.FAILED)
        processing = sum(1 for job in self.jobs.values() if job.status == JobStatusEnum.PROCESSING)
        pending = sum(1 for job in self.jobs.values() if job.status == JobStatusEnum.PENDING)
        return {
            "total_jobs": total_jobs,
            "completed": completed,
            "failed": failed,
            "processing": processing,
            "pending": pending,
            "success_rate": (completed / total_jobs * 100) if total_jobs else 0
        }