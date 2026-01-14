"""
Orchestrator Agent
Coordinates extraction and all analyses (AI Detection + Horizon segments)
"""
import logging
import asyncio
import uuid
from datetime import datetime
from typing import Dict, Any
from sqlalchemy.orm import Session
from app.utils import cancellation_manager

from app.agents.extraction_agent import LangGraphExtractionAgent
from app.agents.horizon_agent import HorizonAgent
from app.services.ai_detection_service import get_ai_detection_service
from app.database import get_db, Extraction, OrchestrationJob
from pydantic import BaseModel

def convert_pydantic_to_dict(obj: Any) -> Any:
    """
    Recursively convert Pydantic models to dicts for JSON serialization.
    Handles: Pydantic models, lists, nested dicts, primitives.
    """
    if obj is None:
        return None
    
    # If it's a Pydantic model, convert it
    if isinstance(obj, BaseModel):
        return obj.model_dump() if hasattr(obj, 'model_dump') else obj.dict()
    
    # If it's a list, convert each item
    if isinstance(obj, list):
        return [convert_pydantic_to_dict(item) for item in obj]
    
    # If it's a dict, convert each value recursively
    if isinstance(obj, dict):
        return {key: convert_pydantic_to_dict(value) for key, value in obj.items()}
    
    # For primitives (str, int, float, bool), return as-is
    return obj

logger = logging.getLogger(__name__)

class OrchestratorAgent:
    """
    Master coordinator for document processing workflow.
    Manages extraction and parallel analysis.
    """
    
    def __init__(self):
        self.extraction_agent = LangGraphExtractionAgent()
        self.horizon_agent = HorizonAgent()
        self.ai_detection_service = get_ai_detection_service()
        logger.info("Orchestrator Agent initialized")
    
    async def extract_only(self, document_id: str, db: Session) -> Dict:
        """
        Phase 1: Extract text only (for Page 1)
        
        Args:
            document_id: Document ID to process
            db: Database session
            
        Returns:
            Dict with extraction_id
        """
        try:
            logger.info(f"Starting extraction for document: {document_id}")
            
            # Run extraction agent
            result = await self.extraction_agent.execute(document_id, db)
            
            extraction_id = result.get('extraction_id')
            
            logger.info(f"Extraction complete: {extraction_id}")
            
            return {
                "success": True,
                "extraction_id": extraction_id,
                "status": "complete"
            }
            
        except Exception as e:
            logger.error(f"Extraction failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }
    
    async def process_full_analysis(self, extraction_id: str, db: Session) -> str:
        job_id = str(uuid.uuid4())

        job = OrchestrationJob(
            id=job_id,
            extraction_id=extraction_id,
            status="processing",
            progress=0,
            started_at=datetime.utcnow()
        )
        db.add(job)
        db.commit()

        async def pipeline():
            try:
                if cancellation_manager.cancel_event.is_set():
                    raise asyncio.CancelledError()

                await self._run_full_analysis(job_id, extraction_id, db)

            except asyncio.CancelledError:
                logger.warning(f"[CANCELLED] Job {job_id} (analysis)")
                self._mark_job_cancelled(job_id, db)
                raise

            finally:
                await cancellation_manager.unregister(job_id)

        task = asyncio.create_task(pipeline())
        await cancellation_manager.register(job_id, task)

        logger.info(f"Started orchestrated processing: job_id={job_id}")
        return job_id

    
    async def process_complete(self, document_id: str, db: Session) -> str:
        job_id = str(uuid.uuid4())

        job = OrchestrationJob(
            id=job_id,
            extraction_id="pending",
            status="processing",
            progress=0,
            started_at=datetime.utcnow()
        )
        db.add(job)
        db.commit()

        async def pipeline():
            try:
                if cancellation_manager.cancel_event.is_set():
                    raise asyncio.CancelledError()

                await self._run_complete_processing(job_id, document_id, db)

            except asyncio.CancelledError:
                logger.warning(f"[CANCELLED] Job {job_id} (complete)")
                self._mark_job_cancelled(job_id, db)
                raise

            finally:
                await cancellation_manager.unregister(job_id)

        task = asyncio.create_task(pipeline())
        await cancellation_manager.register(job_id, task)

        logger.info(f"Started complete processing: job_id={job_id}")
        return job_id



    async def _run_complete_processing(self, job_id: str, document_id: str, db: Session):
        """
        Background task: Run extraction + all 4 analyses.
        """
        if cancellation_manager.cancel_event.is_set():
            raise asyncio.CancelledError()

        try:
            logger.info(f"[Job {job_id}] Phase 1: Running OCR extraction...")
            
            # Phase 1: Extract text
            self._update_job_progress(job_id, 10, db)
            extraction_result = await self.extraction_agent.execute(document_id, db)
            
            if not extraction_result.get('success'):
                raise Exception(f"Extraction failed: {extraction_result.get('error')}")
            
            extraction_id = extraction_result['extraction_id']
            text = extraction_result['text']
            
            # Update job with extraction_id
            job = db.query(OrchestrationJob).filter(OrchestrationJob.id == job_id).first()
            job.extraction_id = extraction_id
            db.commit()
            
            logger.info(f"[Job {job_id}] Phase 2: Running parallel analyses...")
            self._update_job_progress(job_id, 30, db)

            if cancellation_manager.cancel_event.is_set():
                raise asyncio.CancelledError()

            # Wrap AI detection to handle service unavailability
            async def safe_ai_detection():
                try:
                    return await self.ai_detection_service.detect(text)
                except Exception as e:
                    logger.warning(f"[Job {job_id}] AI Detection service unavailable: {e}")
                    return {"success": False, "error": f"Service unavailable: {str(e)}", "data": None}
            
            if cancellation_manager.cancel_event.is_set():
                raise asyncio.CancelledError()

            # Phase 2: Run 4 analyses in parallel
            results = await asyncio.gather(
                safe_ai_detection(),
                self.horizon_agent.extract_references(text),
                self.horizon_agent.extract_medical(text),
                self.horizon_agent.extract_legal(text),
                return_exceptions=True
            )
            
            ai_detection, references, medical, legal = results
            
            # Convert Pydantic models to dicts before saving
            job.ai_detection_result = convert_pydantic_to_dict(
            ai_detection if not isinstance(ai_detection, Exception) else {"error": str(ai_detection)}
            )
            job.references_result = convert_pydantic_to_dict(
                references if not isinstance(references, Exception) else {"error": str(references)}
            )
            job.medical_result = convert_pydantic_to_dict(
                medical if not isinstance(medical, Exception) else {"error": str(medical)}
            )
            job.legal_result = convert_pydantic_to_dict(
                legal if not isinstance(legal, Exception) else {"error": str(legal)}
            )
            job.status = "complete"
            job.progress = 100
            job.completed_at = datetime.utcnow()
            db.commit()
            
            logger.info(f"[Job {job_id}] Complete processing finished successfully")
            
        except Exception as e:
            logger.error(f"[Job {job_id}] Complete processing failed: {e}", exc_info=True)
            
            try:
                db.rollback()  # Clear the failed transaction
                
                # Now update job status to failed
                job = db.query(OrchestrationJob).filter(OrchestrationJob.id == job_id).first()
                if job:
                    job.status = "failed"
                    job.error_message = str(e)
                    job.completed_at = datetime.utcnow()
                    db.commit()
                    logger.info(f"[Job {job_id}] Status updated to 'failed'")
            except Exception as update_error:
                logger.error(f"[Job {job_id}] Failed to update job status: {update_error}")
                db.rollback()
    
    async def _run_full_analysis(self, job_id: str, extraction_id: str, db: Session):
        """
        Background task: Run all 4 analyses in parallel
        """
        if cancellation_manager.cancel_event.is_set():
            raise asyncio.CancelledError()

        try:
            # Get extracted text
            extraction = db.query(Extraction).filter(Extraction.id == extraction_id).first()
            if not extraction:
                raise ValueError(f"Extraction {extraction_id} not found")
            
            text = extraction.text
            
            logger.info(f"Running parallel analyses for job {job_id}")
            
            # Update progress
            self._update_job_progress(job_id, 10, db)
            
            if cancellation_manager.cancel_event.is_set():
                raise asyncio.CancelledError()

            # Run 4 analyses in parallel
            results = await asyncio.gather(
                self.ai_detection_service.detect(text),
                self.horizon_agent.extract_references(text),
                self.horizon_agent.extract_medical(text),
                self.horizon_agent.extract_legal(text),
                return_exceptions=True
            )
            
            ai_detection, references, medical, legal = results
            
            # Update job with results
            job.ai_detection_result = convert_pydantic_to_dict(
                ai_detection if not isinstance(ai_detection, Exception) else {"error": str(ai_detection)}
            )
            job.references_result = convert_pydantic_to_dict(
                references if not isinstance(references, Exception) else {"error": str(references)}
            )
            job.medical_result = convert_pydantic_to_dict(
                medical if not isinstance(medical, Exception) else {"error": str(medical)}
            )
            job.legal_result = convert_pydantic_to_dict(
                legal if not isinstance(legal, Exception) else {"error": str(legal)}
            )
            job.status = "complete"
            job.progress = 100
            job.completed_at = datetime.utcnow()
            
            db.commit()
            
            logger.info(f"Orchestrated processing complete: job_id={job_id}")
            
        except Exception as e:
            logger.error(f"Orchestrated processing failed: {e}", exc_info=True)
            
            # Mark job as failed
            try:
                db.rollback()
                
                # Mark job as failed
                job = db.query(OrchestrationJob).filter(OrchestrationJob.id == job_id).first()
                if job:
                    job.status = "failed"
                    job.error_message = str(e)
                    job.completed_at = datetime.utcnow()
                    db.commit()
                    logger.info(f"[Job {job_id}] Status updated to 'failed'")
            except Exception as update_error:
                logger.error(f"[Job {job_id}] Failed to update job status: {update_error}")
                db.rollback()
    
    def _mark_job_cancelled(self, job_id: str, db: Session):
        try:
            db.rollback()
            job = db.query(OrchestrationJob).filter(OrchestrationJob.id == job_id).first()
            if job:
                job.status = "cancelled"
                job.completed_at = datetime.utcnow()
                db.commit()
        except Exception as e:
            logger.error(f"[Job {job_id}] Failed to mark cancelled: {e}")
            db.rollback()

    
    def _update_job_progress(self, job_id: str, progress: int, db: Session):
        """Update job progress"""
        job = db.query(OrchestrationJob).filter(OrchestrationJob.id == job_id).first()
        if job:
            job.progress = progress
            db.commit()
    
    def get_job_status(self, job_id: str, db: Session) -> Dict:
        """
        Get current job status and results
        
        Args:
            job_id: Job ID to query
            db: Database session
            
        Returns:
            Dict with status and available results
        """
        job = db.query(OrchestrationJob).filter(OrchestrationJob.id == job_id).first()
        
        if not job:
            return {"error": "Job not found"}
        
        return {
            "job_id": job.id,
            "extraction_id": job.extraction_id,
            "status": job.status,
            "progress": job.progress,
            "ai_detection": job.ai_detection_result,
            "references": job.references_result,
            "medical": job.medical_result,
            "legal": job.legal_result,
            "started_at": job.started_at.isoformat(),
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "error": job.error_message
        }