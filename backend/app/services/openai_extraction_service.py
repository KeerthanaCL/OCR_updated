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
        HIGH PRIORITY: Research papers, online resources
        MEDIUM PRIORITY: Patient details
        LOW PRIORITY: Contact info, addresses
        """
        start_time = time.time()
        
        prompt = f"""You are a medical research and document analysis expert. Extract and VALIDATE ALL references from this insurance appeals document.

PRIORITIZATION:
1. HIGH PRIORITY (research_papers): Medical research papers, clinical studies, journal articles with DOI/PMID
2. HIGH PRIORITY (online_resources): Medical websites, clinical guidelines, treatment protocols with URLs
3. MEDIUM PRIORITY (patient_details): Patient name, DOB, member ID, case numbers
4. LOW PRIORITY (other_references): Phone numbers, addresses, insurance company info, provider credentials

VALIDATION TASKS:
- For research papers: Verify if citation format looks legitimate (author, year, journal)
- For online resources: Check if URL format is valid and domain looks medical/authoritative
- For patient details: Verify if format matches expected patterns (e.g., member ID format)
- For medications/conditions mentioned: Flag if they appear incorrect or misspelled

Document text:
{text[:4000]}

Return JSON with this EXACT structure:
{{
    "research_papers": [
        {{
            "reference_type": "research_paper",
            "content": "Full citation text",
            "location": "Where found in document",
            "verified": true/false,
            "priority": "high",
            "validation_notes": "Why verified or concerns found"
        }}
    ],
    "online_resources": [
        {{
            "reference_type": "online_resource",
            "content": "URL or website reference",
            "location": "Where found",
            "verified": true/false,
            "priority": "high",
            "validation_notes": "URL validation status"
        }}
    ],
    "patient_details": [
        {{
            "reference_type": "patient_name|member_id|case_number|dob",
            "content": "Patient detail value",
            "location": "Where found",
            "verified": true,
            "priority": "medium",
            "validation_notes": "Format validation"
        }}
    ],
    "other_references": [
        {{
            "reference_type": "phone|address|provider|company",
            "content": "Reference value",
            "location": "Where found",
            "verified": true,
            "priority": "low",
            "validation_notes": "Additional notes"
        }}
    ],
    "summary": "Brief summary of all references found",
    "validation_summary": "Overall assessment of reference quality and completeness",
    "confidence": 0.85
}}

IMPORTANT: 
- Only include references actually found in the document
- Mark verified=true only if reference looks legitimate
- Provide specific validation_notes for each reference
- Return ONLY valid JSON, no markdown"""

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

            logger.info("Calling Gemini for reference extraction + validation...")
            
            # Call Gemini API
            response = self.model.generate_content(prompt)
            
            # Parse JSON response
            result = json.loads(response.text)

            processing_time = time.time() - start_time
            
            # Parse categorized references
            research_papers = [ReferenceItem(**ref) for ref in result.get("research_papers", [])]
            online_resources = [ReferenceItem(**ref) for ref in result.get("online_resources", [])]
            patient_details = [ReferenceItem(**ref) for ref in result.get("patient_details", [])]
            other_references = [ReferenceItem(**ref) for ref in result.get("other_references", [])]
            
            total_refs = len(research_papers) + len(online_resources) + len(patient_details) + len(other_references)
            
            logger.info(f"✅ Extracted {total_refs} references (Papers: {len(research_papers)}, Online: {len(online_resources)}, Patient: {len(patient_details)}, Other: {len(other_references)}) in {processing_time:.2f}s")
            
            return ReferencesExtractionResponse(
                success=True,
                research_papers=research_papers,
                online_resources=online_resources,
                patient_details=patient_details,
                other_references=other_references,
                summary=result.get("summary", ""),
                validation_summary=result.get("validation_summary", ""),
                confidence=float(result.get("confidence", 0.8)),
                raw_text_analyzed=text[:500] + "...",
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Gemini references extraction failed: {e}", exc_info=True)
            return ReferencesExtractionResponse(
                success=False,
                research_papers=[],
                online_resources=[],
                patient_details=[],
                other_references=[],
                summary=f"Extraction failed: {str(e)}",
                validation_summary="Validation could not be performed due to extraction failure",
                confidence=0.0,
                raw_text_analyzed=text[:500] + "...",
                processing_time=time.time() - start_time
            )
    
    async def extract_medical_context(self, text: str) -> MedicalContextResponse:
        """
        Extract and VALIDATE medical context including conditions and medications.
        Validates: Real medical conditions, legitimate medications, correct dosages.
        """
        start_time = time.time()
        
        prompt = f"""You are a medical expert with knowledge of conditions, medications, and clinical standards. Extract and VALIDATE all medical information from this insurance appeals document.

VALIDATION REQUIREMENTS:

CONDITIONS:
- Verify each condition is a real, recognized medical diagnosis
- Check if condition name is spelled correctly
- Assess if severity/treatment mentioned makes medical sense
- Flag any conditions that seem incorrect or misspelled

MEDICATIONS:
- Verify each medication is a real, FDA-approved drug (check against your knowledge)
- Validate dosage is within normal therapeutic range for that medication
- Check if frequency (e.g., "twice daily") is appropriate for the medication
- Flag any medications with incorrect names, dosages, or suspicious details

MEDICAL HISTORY:
- Assess if medical history is coherent and medically logical
- Check if family history correlates with patient's conditions
- Verify if medical necessity argument is medically sound

Document text:
{text[:4000]}

Return JSON with this EXACT structure:
{{
    "patient_name": "Patient name or null",
    "conditions": [
        {{
            "condition": "Medical condition name",
            "diagnosis_date": "Date or null",
            "severity": "mild|moderate|severe or null",
            "treatment": "Current treatment or null",
            "is_valid_condition": true/false,
            "validation_notes": "Medical validation: Is this a real condition? Spelling correct? Makes clinical sense?"
        }}
    ],
    "medications": [
        {{
            "name": "Medication name",
            "dosage": "Dosage or null",
            "frequency": "Frequency or null",
            "purpose": "Why prescribed or null",
            "is_valid_medication": true/false,
            "is_correct_dosage": true/false,
            "validation_notes": "Pharmacy validation: Real medication? Dosage within normal range? Appropriate frequency?"
        }}
    ],
    "medical_history": "Comprehensive summary of patient's medical background",
    "medical_necessity_argument": "Why the patient needs the denied treatment/medication",
    "providers_mentioned": ["List of doctors/providers mentioned"],
    "validation_summary": "Overall medical validation: Are all conditions real? All medications legitimate? Dosages correct? Medical history coherent?",
    "medical_accuracy_score": 0.9,
    "confidence": 0.85
}}

IMPORTANT:
- Mark is_valid_condition=false if condition seems incorrect/misspelled
- Mark is_valid_medication=false if medication name is wrong
- Mark is_correct_dosage=false if dosage is outside therapeutic range
- Provide detailed validation_notes explaining your assessment
- medical_accuracy_score should be 0-1 based on overall medical accuracy
- Return ONLY valid JSON, no markdown"""

        try:
            logger.info("Calling Gemini for medical extraction + validation...")

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
            
            # Count validation issues
            invalid_conditions = sum(1 for c in conditions if not c.is_valid_condition)
            invalid_meds = sum(1 for m in medications if not m.is_valid_medication)
            incorrect_dosages = sum(1 for m in medications if not m.is_correct_dosage)
            
            logger.info(f"✅ Extracted medical context: {len(conditions)} conditions ({invalid_conditions} flagged), {len(medications)} medications ({invalid_meds} invalid, {incorrect_dosages} dosage issues) in {processing_time:.2f}s")
            
            return MedicalContextResponse(
                success=True,
                patient_name=result.get("patient_name"),
                conditions=conditions,
                medications=medications,
                medical_history=result.get("medical_history", ""),
                medical_necessity_argument=result.get("medical_necessity_argument"),
                providers_mentioned=result.get("providers_mentioned", []),
                validation_summary=result.get("validation_summary", ""),
                medical_accuracy_score=float(result.get("medical_accuracy_score", 0.8)),
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
                validation_summary="Validation could not be performed",
                medical_accuracy_score=0.0,
                confidence=0.0,
                raw_text_analyzed=text[:500] + "...",
                processing_time=time.time() - start_time
            )
    
    async def extract_legal_context(self, text: str) -> LegalContextResponse:
        """
        Extract and VALIDATE legal context including claims, statutes, and procedures.
        Validates: Legitimate legal claims, accurate statute citations, proper procedures.
        """
        start_time = time.time()
        
        prompt = f"""You are a legal expert specializing in insurance law, ERISA, PPACA, and healthcare regulations. Extract and VALIDATE all legal information from this insurance appeals document.

VALIDATION REQUIREMENTS:

LEGAL CLAIMS:
- Verify each claim is a legitimate legal right (e.g., appeal rights, ERISA protections)
- Check if claim description is legally accurate
- Assess if claim is applicable to insurance appeals

STATUTES & REGULATIONS:
- Validate statute citations are real (e.g., ERISA Section 502, PPACA provisions)
- Check if statute citation format is correct
- Verify statute is relevant to the claim being made

DEADLINES:
- Check if deadlines mentioned are standard for appeals (e.g., 180 days, 72 hours)
- Verify deadline format is clear and specific

LEGAL PROCEDURES:
- Assess if described procedures are standard for insurance appeals
- Check if legal arguments are coherent and properly structured

Document text:
{text[:4000]}

Return JSON with this EXACT structure:
{{
    "legal_claims": [
        {{
            "claim_type": "appeal_right|erisa_right|ppaca_right|denial_challenge|other",
            "description": "What legal claim is being made",
            "relevant_statute": "Statute citation or null",
            "deadline": "Deadline or null",
            "is_valid_claim": true/false,
            "statute_accuracy": true/false,
            "validation_notes": "Legal validation: Is this a real legal right? Statute citation correct? Applicable to insurance appeals?"
        }}
    ],
    "appeal_type": "standard|expedited|external_review or null",
    "legal_summary": "Summary of legal arguments and rights asserted",
    "statutes_cited": ["List of all statutes/regulations mentioned"],
    "deadlines_mentioned": ["List of all deadlines with context"],
    "validation_summary": "Overall legal validation: Are claims legitimate? Statutes accurate? Procedures proper?",
    "legal_accuracy_score": 0.9,
    "confidence": 0.85
}}

IMPORTANT:
- Mark is_valid_claim=false if claim is not a recognized legal right
- Mark statute_accuracy=false if statute citation is incorrect or doesn't exist
- Provide detailed validation_notes explaining legal assessment
- legal_accuracy_score should be 0-1 based on overall legal accuracy
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
            
            response = self.model.generate_content(prompt)
            result = json.loads(response.text)
            processing_time = time.time() - start_time
            
            legal_claims = [
                LegalClaim(**claim) for claim in result.get("legal_claims", [])
            ]
            
            # Count validation issues
            invalid_claims = sum(1 for c in legal_claims if not c.is_valid_claim)
            inaccurate_statutes = sum(1 for c in legal_claims if not c.statute_accuracy)
            
            logger.info(f"Extracted legal context: {len(legal_claims)} claims ({invalid_claims} invalid, {inaccurate_statutes} statute errors) in {processing_time:.2f}s")
            
            return LegalContextResponse(
                success=True,
                legal_claims=legal_claims,
                appeal_type=result.get("appeal_type"),
                legal_summary=result.get("legal_summary", ""),
                statutes_cited=result.get("statutes_cited", []),
                deadlines_mentioned=result.get("deadlines_mentioned", []),
                validation_summary=result.get("validation_summary", ""),
                legal_accuracy_score=float(result.get("legal_accuracy_score", 0.8)),
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
                validation_summary="Validation could not be performed",
                legal_accuracy_score=0.0,
                confidence=0.0,
                raw_text_analyzed=text[:500] + "...",
                processing_time=time.time() - start_time
            )