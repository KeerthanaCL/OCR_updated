from openai import OpenAI
import google.generativeai as genai
import logging
import time
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
import json
import os

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

    async def extract_references(self, text: str) -> ReferencesExtractionResponse:
        """
        Extract all references from appeals text using OpenAI.
        References include: case numbers, contact info, provider names, etc.
        """
        start_time = time.time()
        
        prompt = f"""You are a document analysis expert. Extract ALL references from this insurance appeals document.

References include:
- Case/Reference/Claim numbers
- Phone numbers and fax numbers
- Mailing addresses and P.O. Boxes
- Provider names with credentials (doctors, pharmacists)
- Reviewer names
- Insurance company names and affiliates
- Any cited documents or forms

Document text:
{text}

Return a JSON object with:
{{
    "references": [
        {{
            "reference_type": "case_number|phone|address|provider|company|other",
            "content": "the actual reference text",
            "location": "brief context where found",
            "verified": true if it looks valid
        }}
    ],
    "summary": "brief summary of what references were found",
    "confidence": 0.0-1.0 (how confident you are)
}}"""

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

            logger.info("Calling Gemini for reference extraction...")
            
            # Call Gemini API
            response = self.model.generate_content(prompt)
            
            # Parse JSON response
            result = json.loads(response.text)

            processing_time = time.time() - start_time
            
            references = [
                ReferenceItem(**ref) for ref in result.get("references", [])
            ]
            
            logger.info(f"Extracted {len(references)} references in {processing_time:.2f}s")
            
            return ReferencesExtractionResponse(
                success=True,
                references=references,
                summary=result.get("summary", ""),
                confidence=float(result.get("confidence", 0.8)),
                raw_text_analyzed=text[:500] + "...",  # First 500 chars
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Gemini references extraction failed: {e}", exc_info=True)
            return ReferencesExtractionResponse(
                success=False,
                references=[],
                summary=f"Extraction failed: {str(e)}",
                confidence=0.0,
                raw_text_analyzed=text[:500] + "...",
                processing_time=time.time() - start_time
            )
    
    async def extract_medical_context(self, text: str) -> MedicalContextResponse:
        """
        Extract medical context from appeals text using OpenAI.
        Includes conditions, medications, history, necessity arguments.
        """
        start_time = time.time()
        
        prompt = f"""You are a medical documentation expert. Extract ALL medical information from this insurance appeals document.

Extract:
1. Patient name
2. Medical conditions/diagnoses (with dates if mentioned)
3. Medications (name, dosage, frequency, purpose)
4. Medical history and family history
5. The medical necessity argument (why patient needs treatment)
6. Healthcare providers mentioned

Document text:
{text}

Return a JSON object with:
{{
    "patient_name": "name if found",
    "conditions": [
        {{
            "condition": "condition name",
            "diagnosis_date": "date if mentioned",
            "severity": "mild|moderate|severe if mentioned",
            "treatment": "current treatment if mentioned"
        }}
    ],
    "medications": [
        {{
            "name": "medication name",
            "dosage": "dosage if mentioned",
            "frequency": "frequency if mentioned",
            "purpose": "why prescribed"
        }}
    ],
    "medical_history": "comprehensive summary of patient's medical background",
    "medical_necessity_argument": "why the patient needs the denied treatment/medication",
    "providers_mentioned": ["list of doctors/providers mentioned"],
    "confidence": 0.0-1.0
}}"""

        try:
            logger.info("Calling Gemini for medical extraction...")

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
            response = self.model.generate_content(prompt)
            result = json.loads(response.text)
            processing_time = time.time() - start_time
            
            conditions = [
                MedicalCondition(**cond) for cond in result.get("conditions", [])
            ]
            
            medications = [
                Medication(**med) for med in result.get("medications", [])
            ]
            
            logger.info(f"Extracted medical context: {len(conditions)} conditions, {len(medications)} medications in {processing_time:.2f}s")
            
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
                processing_time=processing_time
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
                confidence=0.0,
                raw_text_analyzed=text[:500] + "...",
                processing_time=time.time() - start_time
            )
    
    async def extract_legal_context(self, text: str) -> LegalContextResponse:
        """
        Extract legal context from appeals text using OpenAI.
        Includes legal claims, rights, statutes, deadlines.
        """
        start_time = time.time()
        
        prompt = f"""You are a legal document expert specializing in insurance law. Extract ALL legal information from this insurance appeals document.

Extract:
1. Type of appeal (standard, expedited, external review)
2. Legal rights and claims mentioned
3. Statutes, regulations, and acts cited (ERISA, PPACA, etc.)
4. Deadlines and timeframes
5. Legal procedures described
6. Legal arguments being made

Document text:
{text}

Return a JSON object with:
{{
    "legal_claims": [
        {{
            "claim_type": "appeal_right|erisa_right|ppaca_right|denial_challenge|other",
            "description": "what legal claim is being made",
            "relevant_statute": "statute/regulation if cited",
            "deadline": "deadline if mentioned"
        }}
    ],
    "appeal_type": "standard|expedited|external_review|other",
    "legal_summary": "summary of legal arguments and rights",
    "statutes_cited": ["list of laws/regulations mentioned"],
    "deadlines_mentioned": ["list of all deadlines"],
    "confidence": 0.0-1.0
}}"""

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
            
            response = self.model.generate_content(prompt)
            result = json.loads(response.text)
            processing_time = time.time() - start_time
            
            legal_claims = [
                LegalClaim(**claim) for claim in result.get("legal_claims", [])
            ]
            
            logger.info(f"Extracted legal context: {len(legal_claims)} claims in {processing_time:.2f}s")
            
            return LegalContextResponse(
                success=True,
                legal_claims=legal_claims,
                appeal_type=result.get("appeal_type"),
                legal_summary=result.get("legal_summary", ""),
                statutes_cited=result.get("statutes_cited", []),
                deadlines_mentioned=result.get("deadlines_mentioned", []),
                confidence=float(result.get("confidence", 0.8)),
                raw_text_analyzed=text[:500] + "...",
                processing_time=processing_time
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