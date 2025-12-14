from openai import OpenAI
import google.generativeai as genai
import logging
import time
from google.api_core.exceptions import ResourceExhausted
from typing import Dict, List, Optional
from app.config import get_settings
from app.models import (
    ReferenceItem,
    ReferencesExtractionResponse,
    MedicalCondition,
    Medication,
    MedicalContextResponse,
    LegalClaim,
    LegalContextResponse
)
import asyncio
import json
import os
import re
from app.utils.rate_limiter import get_gemini_rate_limiter

logger = logging.getLogger(__name__)
settings = get_settings()

class OpenAIExtractionService:
    """
    Service for extracting document segments using OpenAI.
    More accurate than pattern matching, understands context.
    """
    
    def __init__(self):
        # self.client = OpenAI(api_key=settings.openai_api_key)
        # self.model = settings.openai_model
        # self.temperature = settings.openai_temperature
        # self.max_tokens = settings.openai_max_tokens
        # self.timeout = settings.openai_timeout
        # Configure Gemini with API key
        genai.configure(api_key=settings.google_api_key)
        
        # Initialize model with JSON response format
        self.model = genai.GenerativeModel(
            settings.gemini_model,
            generation_config={
                "response_mime_type": "application/json",
                "temperature": 0.1,
                "max_output_tokens": 4000
            }
        )
        
        logger.info(f"✅ Gemini initialized with model: {settings.gemini_model}")
        # Initialize rate limiter (adjust RPM based on your API tier)
        self.max_retries = 3
        self.initial_retry_delay = 120.0  # Start with 60 seconds
        self.max_retry_delay = 300.0  # Max 5 minutes
        self.rate_limiter = get_gemini_rate_limiter(rpm=settings.gemini_rpm)

    async def _call_with_retry(self, prompt: str, operation_name: str) -> dict:
        """
        Call Gemini API with exponential backoff retry logic.
        
        Args:
            prompt: The prompt to send
            operation_name: Name of operation for logging
            
        Returns:
            Parsed JSON response
            
        Raises:
            ResourceExhausted: If all retries exhausted
        """
        retry_count = 0
        last_exception = None
        total_wait_time = 0.0
        
        while retry_count <= self.max_retries:
            try:
                wait_time = await self.rate_limiter.acquire()
                if wait_time > 0:
                    logger.info(f"Rate limiter delayed {operation_name} by {wait_time:.1f}s")
                logger.info(f"Calling Gemini for {operation_name}... (attempt {retry_count + 1}/{self.max_retries + 1})")
                
                response = self.model.generate_content(prompt)
                result = json.loads(response.text)
                
                if total_wait_time > 0:
                    logger.info(
                        f"✅ Gemini {operation_name} succeeded after {retry_count} retries "
                        f"and {total_wait_time:.1f}s total wait time"
                    )
                else:
                    logger.info(f"✅ Gemini {operation_name} succeeded on first attempt")
                
                return result
                
            except ResourceExhausted as e:
                last_exception = e
                retry_count += 1
                
                # Extract retry delay from error if available
                error_str = str(e)
                suggested_delay = 60.0  # Default
                
                match = re.search(r'retry in ([\d.]+)s', error_str)
                if match:
                    suggested_delay = float(match.group(1))
                
                # Check if daily quota is completely exhausted (limit: 0)
                if "limit: 0" in error_str and "PerDay" in error_str:
                    logger.error(
                        f"DAILY quota exhausted for {operation_name}. "
                        f"Free tier resets at midnight Pacific Time. "
                        f"\nOptions:"
                        f"\n   1. Wait until midnight PT for quota reset"
                        f"\n   2. Enable billing at https://aistudio.google.com/"
                        f"\n   3. Switch to gemini-1.5-flash model in config.py"
                    )
                    # Don't retry for daily quota - no point waiting
                    raise
                
                # Check if we should retry
                if retry_count > self.max_retries:
                    logger.error(
                        f"Gemini {operation_name} failed after {self.max_retries} retries "
                        f"and {total_wait_time:.1f}s total wait time"
                    )
                    raise
                
                # Calculate wait time (use suggested delay, capped at max)
                wait_time = min(suggested_delay, self.max_retry_delay)
                total_wait_time += wait_time
                
                logger.warning(
                    f"Gemini quota exceeded for {operation_name}. "
                    f"Waiting {wait_time:.1f}s before retry {retry_count}/{self.max_retries} "
                    f"(total wait so far: {total_wait_time:.1f}s)..."
                )
                
                # Wait the suggested time
                await asyncio.sleep(wait_time)
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse Gemini response for {operation_name}: {e}")
                # Return empty result instead of crashing
                return {}
                
            except Exception as e:
                logger.error(
                    f"Unexpected error in Gemini {operation_name}: {e}", 
                    exc_info=True
                )
                raise
        
        # If we get here, all retries failed
        raise last_exception
    
    async def extract_references(self, text: str) -> ReferencesExtractionResponse:
        """Extract ONLY research papers and online resources.
        Do NOT extract patient information.
        """
        start_time = time.time()
        
        prompt = f"""You are a medical research expert. Extract ONLY scientific references and online resources from this insurance appeals document.

    **EXTRACT THESE (references):**
    1. Research papers with DOI, PMID, or journal citations
    2. Medical websites and URLs (https://, www.)
    3. Clinical guidelines and protocols

    **DO NOT EXTRACT (not references):**
    - Patient names, initials, or identifiers (like "SZ", "7s")
    - Member IDs, case numbers, medical record numbers
    - Phone numbers, addresses, dates of birth
    - Provider names, company names
    - Any personal health information (PHI)

    Document text:
    {text}

    Return JSON with this EXACT structure:
    {{
    "research_papers": [
        {{
        "reference_type": "research_paper",
        "content": "Full citation with authors, title, journal, year, DOI/PMID",
        "location": "Where found in document",
        "priority": "high"
        }}
    ],
    "online_resources": [
        {{
        "reference_type": "online_resource",
        "content": "Full URL (must start with http:// or https:// or www.)",
        "location": "Where found",
        "priority": "high"
        }}
    ],
    "summary": "Brief summary of scientific references found",
    "confidence": 0.85
    }}

    CRITICAL RULES:
    1. ONLY include actual research papers (with DOI/PMID) or URLs
    2. DO NOT include patient initials, names, or IDs
    3. DO NOT include company names like "Appeals Team" or "Claimable"
    4. If no research papers found, return empty array []
    5. Return ONLY valid JSON, no markdown"""

        try:
            # logger.info("Calling OpenAI for reference extraction...")

            # response = await self.client.chat.completions.create(
            #     model=self.model,
            #     messages=[
            #         {"role": "system", "content": "You are an expert at extracting references from insurance documents. Return valid JSON only."},
            #         {"role": "user", "content": prompt}
            #     ],
            #     temperature=self.temperature,
            #     max_tokens=self.max_tokens,
            #     timeout=self.timeout,
            #     response_format={"type": "json_object"}
            # )
            
            # result = json.loads(response.choices[0].message.content)

            logger.info("Calling Gemini for reference extraction")
            
            # Use retry logic with automatic waiting
            result = await self._call_with_retry(prompt, "reference_extraction")

            processing_time = time.time() - start_time

            # Capture token usage from Gemini response
            usage_metadata = None
            
            # Parse categorized references
            research_papers = [ReferenceItem(**ref) for ref in result.get("research_papers", [])]
            online_resources = [ReferenceItem(**ref) for ref in result.get("online_resources", [])]
            
            total_refs = len(research_papers) + len(online_resources)
            
            logger.info(
                f"Extracted {total_refs} scientific references "
                f"(Papers: {len(research_papers)}, URLs: {len(online_resources)}) "
                f"in {processing_time:.2f}s"
            )
            
            return ReferencesExtractionResponse(
                success=True,
                research_papers=research_papers,
                online_resources=online_resources,
                patient_details=[],
                other_references=[],
                summary=result.get("summary", ""),
                confidence=float(result.get("confidence", 0.8)),
                raw_text_analyzed=text[:500] + "...",
                processing_time=processing_time,
                usage=usage_metadata
            )
            
        except Exception as e:
            logger.error(f"Gemini daily quota exhausted: {e}")
            return ReferencesExtractionResponse(
                success=False,
                research_papers=[],
                online_resources=[],
                patient_details=[],
                other_references=[],
                summary=f"Extraction failed: {str(e)}",
                confidence=0.0,
                raw_text_analyzed=text[:500] + "...",
                processing_time=time.time() - start_time
            )
    
    async def extract_medical_context(self, text: str) -> MedicalContextResponse:
        """
        Extract medical context including conditions, medications, and medical history.
        No validation - pure extraction.
        """
        start_time = time.time()
        
        prompt = f"""You are a medical expert. Extract all medical information from this insurance appeals document.

Extract the following:
- Patient information
- Medical conditions with diagnosis dates, severity, and treatments
- Medications with dosages, frequency, and purpose
- Medical history summary
- Medical necessity argument (why treatment is needed)
- Healthcare providers mentioned

Document text:
{text}

Return JSON with this EXACT structure:
{{
  "patient_name": "Patient name or null",
  "conditions": [
    {{
      "condition": "Medical condition name",
      "diagnosis_date": "Date or null",
      "severity": "mild|moderate|severe or null",
      "treatment": "Current treatment or null"
    }}
  ],
  "medications": [
    {{
      "name": "Medication name",
      "dosage": "Dosage or null",
      "frequency": "Frequency or null",
      "purpose": "Why prescribed or null"
    }}
  ],
  "medical_history": "Comprehensive summary of patient's medical background",
  "medical_necessity_argument": "Why the patient needs the denied treatment/medication",
  "providers_mentioned": ["List of doctors/providers mentioned"],
  "confidence": 0.85
}}

IMPORTANT:
- Extract all information as-is from the document
- Do not validate or verify medical information
- Return ONLY valid JSON, no markdown"""

        try:
            logger.info("Calling Gemini for medical extraction")

            # response = await self.client.chat.completions.create(
            #     model=self.model,
            #     messages=[
            #         {"role": "system", "content": "You are a medical documentation expert. Extract medical information accurately. Return valid JSON only."},
            #         {"role": "user", "content": prompt}
            #     ],
            #     temperature=self.temperature,
            #     max_tokens=self.max_tokens,
            #     timeout=self.timeout,
            #     response_format={"type": "json_object"}
            # )
            
            # result = json.loads(response.choices[0].message.content)
            result = await self._call_with_retry(prompt, "medical_extraction")
            processing_time = time.time() - start_time

            # Capture token usage
            usage_metadata = None
            conditions = [
                MedicalCondition(**cond) for cond in result.get("conditions", [])
            ]
            
            medications = [
                Medication(**med) for med in result.get("medications", [])
            ]
            
            logger.info(
                f"Extracted medical context: {len(conditions)} conditions, "
                f"{len(medications)} medications in {processing_time:.2f}s"
            )
            
            return MedicalContextResponse(
                success=True,
                patient_name=result.get("patient_name"),
                conditions=conditions,
                medications=medications,
                medical_history=result.get("medical_history", ""),
                medical_necessity_argument=result.get("medical_necessity_argument"),
                providers_mentioned=result.get("providers_mentioned", []),
                confidence=float(result.get("confidence", 0.8)),
                raw_text_analyzed=text[:500] + "...",
                processing_time=processing_time,
                usage=usage_metadata
            )
            
        except Exception as e:
            logger.error(f"Gemini medical extraction failed: {e}", exc_info=True)
            return MedicalContextResponse(
                success=False,
                patient_name=None,
                conditions=[],
                medications=[],
                medical_history=f"Extraction failed: {str(e)}",
                medical_necessity_argument=None,
                providers_mentioned=[],
                medical_accuracy_score=0.0,
                confidence=0.0,
                raw_text_analyzed=text[:500] + "...",
                processing_time=time.time() - start_time
            )
    
    async def extract_legal_context(self, text: str) -> LegalContextResponse:
        """
        Extract legal context including claims, statutes, and deadlines.
        No validation - pure extraction.
        """
        start_time = time.time()
        
        prompt = f"""You are a legal expert specializing in insurance appeals. Extract all legal information from this appeals document.

Extract the following:
- Type of appeal
- Legal claims being made
- Statutes or regulations cited
- Deadlines mentioned
- Legal arguments summary

Document text:
{text}

Return JSON with this EXACT structure:
{{
  "appeal_type": "Type of appeal (e.g., insurance denial appeal, medical necessity appeal)",
  "legal_claims": [
    {{
      "claim_type": "Type of claim",
      "description": "Detailed description of the claim",
      "relevant_statute": "Statute or regulation cited or null",
      "deadline": "Deadline mentioned or null"
    }}
  ],
  "legal_summary": "Summary of the legal arguments being made",
  "statutes_cited": ["List of statutes/regulations cited"],
  "deadlines_mentioned": ["List of deadlines mentioned"],
  "confidence": 0.85
}}

IMPORTANT:
- Extract all information as-is from the document
- Do not validate or verify legal information
- Return ONLY valid JSON, no markdown"""

        try:
            # response = await self.client.chat.completions.create(
            #     model=self.model,
            #     messages=[
            #         {"role": "system", "content": "You are an insurance law expert. Extract legal information accurately. Return valid JSON only."},
            #         {"role": "user", "content": prompt}
            #     ],
            #     temperature=self.temperature,
            #     max_tokens=self.max_tokens,
            #     timeout=self.timeout,
            #     response_format={"type": "json_object"}
            # )
            
            # result = json.loads(response.choices[0].message.content)
            logger.info("Calling Gemini for legal extraction...")
            
            result = await self._call_with_retry(prompt, "legal_extraction")
            processing_time = time.time() - start_time

            # Capture token usage
            usage_metadata = None
            legal_claims = [
                LegalClaim(**claim) for claim in result.get("legal_claims", [])
            ]
            
            logger.info(
                f"Extracted legal context: {len(legal_claims)} claims, "
                f"{len(result.get('statutes_cited', []))} statutes in {processing_time:.2f}s"
            )
            
            return LegalContextResponse(
                success=True,
                legal_claims=legal_claims,
                appeal_type=result.get("appeal_type"),
                legal_summary=result.get("legal_summary", ""),
                statutes_cited=result.get("statutes_cited", []),
                deadlines_mentioned=result.get("deadlines_mentioned", []),
                confidence=float(result.get("confidence", 0.8)),
                raw_text_analyzed=text[:500] + "...",
                processing_time=processing_time,
                usage=usage_metadata
            )
            
        except Exception as e:
            logger.error(f"Gemini legal extraction failed: {e}", exc_info=True)
            return LegalContextResponse(
                success=False,
                legal_claims=[],
                appeal_type=None,
                legal_summary=f"Extraction failed: {str(e)}",
                statutes_cited=[],
                deadlines_mentioned=[],
                confidence=0.0,
                raw_text_analyzed=text[:500] + "...",
                processing_time=time.time() - start_time
            )