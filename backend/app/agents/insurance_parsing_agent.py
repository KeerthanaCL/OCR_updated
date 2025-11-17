import re
import time
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from app.models import ParsedField

logger = logging.getLogger(__name__)

class InsuranceClaimParsingAgent:
    """
    Specialized parsing agent for insurance claim forms.
    Extracts medical claim-specific fields, addresses, dates, amounts, and diagnoses.
    """
    
    def __init__(self):
        # Insurance-specific patterns
        self.patterns = {
            # Personal Information
            'patient_name': r'(?:patient\s+name|insured\s+name|name\s+of\s+patient)[\s:]*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
            'policy_number': r'(?:policy\s+(?:no|number|#)|member\s+id)[\s:]*([A-Z0-9\-]+)',
            'claim_number': r'(?:claim\s+(?:no|number|#))[\s:]*([A-Z0-9\-]+)',
            'id_number': r'(?:ID\s+(?:no|number)|identification)[\s:]*([A-Z0-9\-]+)',
            
            # Contact Information
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'(?:phone|mobile|tel)[\s:]*(\+?[\d\s\-\(\)]{10,})',
            'phone_general': r'\b(?:\+?1[-.]?)?\(?([0-9]{3})\)?[-.]?([0-9]{3})[-.]?([0-9]{4})\b',
            
            # Address (Enhanced pattern)
            'address': r'(?:address)[\s:]*([0-9]+\s+[A-Za-z0-9\s,\.]+(?:street|st|avenue|ave|road|rd|drive|dr|lane|ln|boulevard|blvd)[A-Za-z0-9\s,\.]*)',
            'zipcode': r'\b\d{5}(?:-\d{4})?\b',
            'state': r'\b(?:AL|AK|AZ|AR|CA|CO|CT|DE|FL|GA|HI|ID|IL|IN|IA|KS|KY|LA|ME|MD|MA|MI|MN|MS|MO|MT|NE|NV|NH|NJ|NM|NY|NC|ND|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|VT|VA|WA|WV|WI|WY)\b',
            
            # Medical Information
            'diagnosis': r'(?:diagnosis|condition|illness|disease)[\s:]*([A-Za-z0-9\s,\.\-]+)',
            'diagnosis_code': r'(?:ICD-10|diagnosis\s+code)[\s:]*([A-Z]\d{2}(?:\.\d{1,2})?)',
            'procedure_code': r'(?:CPT|procedure\s+code)[\s:]*(\d{5})',
            'date_of_service': r'(?:date\s+of\s+service|service\s+date|DOS)[\s:]*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})',
            'date_of_admission': r'(?:date\s+of\s+admission|admission\s+date)[\s:]*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})',
            'date_of_discharge': r'(?:date\s+of\s+discharge|discharge\s+date)[\s:]*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})',
            
            # Financial Information
            'total_amount': r'(?:total\s+(?:amount|charge|cost|bill))[\s:]*\$?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
            'claimed_amount': r'(?:claimed\s+amount|amount\s+claimed)[\s:]*\$?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
            'deductible': r'(?:deductible)[\s:]*\$?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
            'copay': r'(?:co-?pay|copayment)[\s:]*\$?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
            'amount': r'\$\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
            
            # Provider Information
            'provider_name': r'(?:provider\s+name|facility\s+name|hospital)[\s:]*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
            'provider_npi': r'(?:NPI|provider\s+id)[\s:]*(\d{10})',
            'tax_id': r'(?:tax\s+id|EIN)[\s:]*(\d{2}-\d{7})',
            
            # Dates (General)
            'date': r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b',
            'date_filed': r'(?:date\s+filed|submission\s+date)[\s:]*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})',
            
            # Signatures
            'signature_present': r'(?:signature|signed\s+by)[\s:]*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        }

        # Medical terms dictionary for context
        self.medical_terms = [
            'surgery', 'consultation', 'emergency', 'outpatient', 'inpatient',
            'prescription', 'medication', 'therapy', 'treatment', 'examination',
            'radiology', 'laboratory', 'test', 'scan', 'MRI', 'CT', 'X-ray',
            'fracture', 'diabetes', 'hypertension', 'infection', 'chronic'
        ]
    
    def execute(
        self, 
        text: str,
        table_data: Optional[List[Dict]] = None,
        parsing_rules: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Execute comprehensive insurance claim parsing.
        
        Args:
            text: Extracted OCR text
            table_data: Optional table data from medical bills
            parsing_rules: Optional custom parsing rules
            
        Returns:
            Structured parsing results
        """
        start_time = time.time()
        
        try:
            logger.info("Starting insurance claim parsing")

            # Extract all fields
            fields = []
            
            # 1. Extract standard fields
            for field_name, pattern in self.patterns.items():
                matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
                if matches:
                    # Handle tuple matches (like phone numbers)
                    if isinstance(matches[0], tuple):
                        matches = [''.join(m) if isinstance(m, tuple) else m for m in matches]
                    
                    for idx, match in enumerate(matches):
                        fields.append(ParsedField(
                            field_name=f"{field_name}_{idx + 1}" if len(matches) > 1 else field_name,
                            field_value=str(match).strip(),
                            confidence=self._calculate_confidence(field_name, match),  # Pattern-based confidence
                            position=None
                        ))
            
            # 2. Extract address with enhanced logic
            address_fields = self._extract_address(text)
            fields.extend(address_fields)
            
            # 3. Extract medical bill items from tables
            if table_data:
                bill_items = self._parse_medical_bill_table(table_data)
                fields.extend(bill_items)
            
            # 4. Extract medical context
            medical_context = self._extract_medical_context(text)
            if medical_context:
                fields.append(ParsedField(
                    field_name='medical_context',
                    field_value=medical_context,
                    confidence=0.8,
                    position=None
                ))
            
            # 5. Detect signature regions (heuristic)
            signature_detected = self._detect_signature_keywords(text)
            if signature_detected:
                fields.append(ParsedField(
                    field_name='signature_status',
                    field_value='Signature section detected',
                    confidence=0.75,
                    position=None
                ))

            # 6. Apply custom rules if provided
            if parsing_rules:
                custom_fields = self._apply_custom_rules(text, parsing_rules)
                fields.extend(custom_fields)

            # 7. Deduplicate and prioritize fields
            fields = self._deduplicate_fields(fields)
            
            processing_time = time.time() - start_time
            
            return {
                'fields': fields,
                'parsing_method': 'insurance_claim_specialized',
                'processing_time': processing_time,
                'total_fields': len(fields),
                'field_categories': self._categorize_fields(fields)
            }
            
        except Exception as e:
            logger.error(f"Insurance parsing failed: {str(e)}")
            raise

    def _extract_address(self, text: str) -> List[ParsedField]:
        """Extract complete address with components"""
        fields = []
        
        # Find address block
        address_pattern = r'(?:address|location)[\s:]*\n?((?:[0-9]+\s+[A-Za-z0-9\s,\.]+\n?)+)'
        matches = re.findall(address_pattern, text, re.IGNORECASE | re.MULTILINE)
        
        for match in matches:
            # Extract components
            street = re.search(r'[0-9]+\s+[A-Za-z0-9\s,\.]+(?:street|st|avenue|ave|road|rd)', match, re.IGNORECASE)
            city = re.search(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*),\s*[A-Z]{2}', match)
            state = re.search(r'\b([A-Z]{2})\b', match)
            zipcode = re.search(r'\b(\d{5}(?:-\d{4})?)\b', match)
            
            if street:
                fields.append(ParsedField(
                    field_name='street_address',
                    field_value=street.group(0),
                    confidence=0.9,
                    position=None
                ))
            
            if city:
                fields.append(ParsedField(
                    field_name='city',
                    field_value=city.group(1),
                    confidence=0.9,
                    position=None
                ))
            
            if state:
                fields.append(ParsedField(
                    field_name='state',
                    field_value=state.group(1),
                    confidence=0.95,
                    position=None
                ))
            
            if zipcode:
                fields.append(ParsedField(
                    field_name='zipcode',
                    field_value=zipcode.group(1),
                    confidence=0.95,
                    position=None
                ))
        
        return fields
    
    def _parse_medical_bill_table(self, table_data: List[Dict]) -> List[ParsedField]:
        """Parse medical bill line items from table data"""
        fields = []
        
        for table in table_data:
            rows = table.get('rows', [])
            
            # Look for amount patterns in each row
            for idx, row in enumerate(rows):
                # Extract service description and amount
                amount_match = re.search(r'\$?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)', row)
                
                if amount_match:
                    # Service description is the text before the amount
                    service = row[:amount_match.start()].strip()
                    amount = amount_match.group(1)
                    
                    if service:
                        fields.append(ParsedField(
                            field_name=f'bill_item_{idx + 1}_service',
                            field_value=service,
                            confidence=0.85,
                            position={'table_id': table['table_id'], 'row': idx}
                        ))
                        
                        fields.append(ParsedField(
                            field_name=f'bill_item_{idx + 1}_amount',
                            field_value=f"${amount}",
                            confidence=0.9,
                            position={'table_id': table['table_id'], 'row': idx}
                        ))
        
        return fields
    
    def _extract_medical_context(self, text: str) -> Optional[str]:
        """Extract medical context and relevant terms"""
        found_terms = []
        
        for term in self.medical_terms:
            if re.search(rf'\b{term}\b', text, re.IGNORECASE):
                found_terms.append(term)
        
        if found_terms:
            return ', '.join(found_terms[:5])  # Top 5 terms
        
        return None
    
    def _detect_signature_keywords(self, text: str) -> bool:
        """Detect if signature section is present"""
        signature_keywords = [
            'signature', 'signed', 'authorized', 'signatory',
            'signed by', 'patient signature', 'provider signature'
        ]
        
        for keyword in signature_keywords:
            if re.search(rf'\b{keyword}\b', text, re.IGNORECASE):
                return True
        
        return False
    
    def _calculate_confidence(self, field_name: str, value: str) -> float:
        """Calculate confidence score based on field type and value"""
        # High confidence for structured fields
        high_confidence_fields = ['policy_number', 'claim_number', 'zipcode', 'state', 'procedure_code']
        if field_name in high_confidence_fields:
            return 0.95
        
        # Medium confidence for names and addresses
        medium_confidence_fields = ['patient_name', 'provider_name', 'address']
        if field_name in medium_confidence_fields:
            return 0.85
        
        # Lower confidence for amounts (can be ambiguous)
        if 'amount' in field_name:
            return 0.8
        
        return 0.75
    
    def _apply_custom_rules(
        self, 
        text: str, 
        rules: Dict[str, Any]
    ) -> List[ParsedField]:
        """Apply custom parsing rules"""
        fields = []
        
        for field_name, rule in rules.items():
            if 'pattern' in rule:
                pattern = rule['pattern']
                matches = re.findall(pattern, text, re.IGNORECASE)
                
                for match in matches:
                    fields.append(ParsedField(
                        field_name=field_name,
                        field_value=str(match),
                        confidence=rule.get('confidence', 0.75),
                        position=None
                    ))
        
        return fields
    
    def _deduplicate_fields(self, fields: List[ParsedField]) -> List[ParsedField]:
        """Remove duplicate fields, keeping highest confidence"""
        seen = {}
        
        for field in fields:
            key = f"{field.field_name}_{field.field_value}"
            
            if key not in seen or field.confidence > seen[key].confidence:
                seen[key] = field
        
        return list(seen.values())
    
    def _categorize_fields(self, fields: List[ParsedField]) -> Dict[str, int]:
        """Categorize extracted fields"""
        categories = {
            'personal_info': 0,
            'medical_info': 0,
            'financial_info': 0,
            'provider_info': 0,
            'dates': 0,
            'other': 0
        }
        
        for field in fields:
            name = field.field_name.lower()
            
            if any(k in name for k in ['name', 'id', 'address', 'phone', 'email']):
                categories['personal_info'] += 1
            elif any(k in name for k in ['diagnosis', 'procedure', 'medical', 'service']):
                categories['medical_info'] += 1
            elif any(k in name for k in ['amount', 'deductible', 'copay', 'bill']):
                categories['financial_info'] += 1
            elif any(k in name for k in ['provider', 'facility', 'npi']):
                categories['provider_info'] += 1
            elif 'date' in name:
                categories['dates'] += 1
            else:
                categories['other'] += 1
        
        return categories