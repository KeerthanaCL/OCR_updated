import logging
import uuid
from typing import Dict, Any
from datetime import datetime
from app.agents.extraction_agent import ExtractionAgent
from app.agents.insurance_parsing_agent import InsuranceClaimParsingAgent
from app.services.table_extractor import TableExtractor
from app.models import DocumentStatus, JobStatus

logger = logging.getLogger(__name__)

class AgentOrchestrator:
    """
    Main orchestrator that coordinates extraction and parsing agents.
    Implements an agent-based workflow pattern.
    """
    
    def __init__(self):
        self.extraction_agent = ExtractionAgent()
        self.parsing_agent = InsuranceClaimParsingAgent()
        self.table_extractor = TableExtractor()
        self.jobs: Dict[str, JobStatus] = {}
    
    def create_job(self, document_id: str) -> str:
        """Create a new processing job"""
        job_id = str(uuid.uuid4())
        
        self.jobs[job_id] = JobStatus(
            job_id=job_id,
            document_id=document_id,
            status=DocumentStatus.UPLOADED,
            current_step="initialized",
            progress=0.0,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        return job_id
    
    def get_job_status(self, job_id: str) -> JobStatus:
        """Get current job status"""
        return self.jobs.get(job_id)
    
    def execute_full_pipeline(
        self,
        job_id: str,
        file_path: str,
        use_preprocessing: bool = True,
        force_trocr: bool = False,
        apply_parsing: bool = True,
        parsing_rules: Dict[str, Any] = None,
        extract_tables: bool = True
    ) -> Dict[str, Any]:
        """
        Execute full insurance claim processing pipeline.
        
        Workflow:
        1. Extract tables (medical bills)
        2. Extract text with orientation correction
        3. Parse with insurance-specific logic
        4. Combine results
        
        Args:
            job_id: Job identifier
            file_path: Path to document
            use_preprocessing: Enable image preprocessing
            force_trocr: Force TrOCR usage
            apply_parsing: Whether to parse extracted text
            parsing_rules: Custom parsing rules
            
        Returns:
            Complete pipeline results
        """
        try:
            # Step 0: Extract tables if needed
            table_data = None
            if extract_tables:
                self._update_job(job_id, DocumentStatus.EXTRACTING, "extracting_tables", 0.1)
                logger.info(f"Job {job_id}: Extracting tables")
                
                try:
                    # For PDF, extract from first converted image
                    from pathlib import Path
                    if Path(file_path).suffix.lower() == '.pdf':
                        # Convert first page for table detection
                        from app.services.pdf_converter import PDFConverter
                        converter = PDFConverter()
                        temp_dir = Path(file_path).parent / "temp_table_extraction"
                        img_paths = converter.extract_specific_pages(file_path, str(temp_dir), [1])
                        table_data = self.table_extractor.detect_tables(img_paths[0])
                    else:
                        table_data = self.table_extractor.detect_tables(file_path)
                    
                    logger.info(f"Extracted {len(table_data) if table_data else 0} tables")
                except Exception as e:
                    logger.warning(f"Table extraction failed: {str(e)}")

            # Step 1: Start extraction
            self._update_job(job_id, DocumentStatus.EXTRACTING, "extracting_text", 0.3)
            
            logger.info(f"Job {job_id}: Starting extraction with orientation correction")
            extraction_result = self.extraction_agent.execute(
                file_path,
                use_preprocessing=use_preprocessing,
                force_trocr=force_trocr,
                correct_orientation=True
            )
            
            # Step 2: Update progress
            self._update_job(job_id, DocumentStatus.EXTRACTED, "extraction_complete", 0.6)
            
            results = {
                'extraction': extraction_result
            }
            
            # Step 3: Parse with insurance-specific logic
            if apply_parsing:
                self._update_job(job_id, DocumentStatus.PARSING, "parsing_insurance_claim", 0.8)
                
                logger.info(f"Job {job_id}: Parsing insurance claim data")
                parsing_result = self.parsing_agent.execute(
                    extraction_result['text'],
                    table_data=table_data,
                    parsing_rules=parsing_rules
                )
                
                results['parsing'] = parsing_result

                if table_data:
                    results['tables'] = table_data
            
            # Step 4: Complete
            self._update_job(
                job_id, 
                DocumentStatus.COMPLETED, 
                "completed", 
                1.0, 
                results
            )
            
            logger.info(f"Job {job_id}: Insurance claim processing completed")
            
            return results
            
        except Exception as e:
            logger.error(f"Job {job_id}: Pipeline failed - {str(e)}")
            self._update_job(
                job_id,
                DocumentStatus.FAILED,
                "failed",
                0.0,
                error=str(e)
            )
            raise
    
    def _update_job(
        self,
        job_id: str,
        status: DocumentStatus,
        step: str,
        progress: float,
        results: Dict[str, Any] = None,
        error: str = None
    ):
        """Update job status"""
        if job_id in self.jobs:
            job = self.jobs[job_id]
            job.status = status
            job.current_step = step
            job.progress = progress
            job.updated_at = datetime.utcnow()
            
            if results:
                job.results = results
            if error:
                job.error = error