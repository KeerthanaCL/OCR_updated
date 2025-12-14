"""
AI Detection Service
Wrapper for external team's AI detection API
"""
import httpx
import logging
import time
from typing import Dict, Optional
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

class AIDetectionService:
    """
    Service to call external AI detection API.
    Flexible wrapper that can adapt to their API format.
    """
    
    def __init__(self):
        self.api_url = settings.ai_detection_api_url
        self.api_key = settings.ai_detection_api_key
        self.timeout = 30.0
        
        logger.info(f"AI Detection Service initialized: {self.api_url}")
    
    async def detect(self, text: str) -> Dict:
        """
        Call external AI detection API
        
        Args:
            text: Extracted text to analyze
            
        Returns:
            Dict with detection results
        """
        start_time = time.time()
        
        try:
            logger.info("Calling external AI Detection API...")
            
            # Prepare request (adjust based on their API format)
            payload = {
                "text": text,
                "options": {
                    "detailed_analysis": True
                }
            }
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # Make API call
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    self.api_url,
                    json=payload,
                    headers=headers
                )
                
                response.raise_for_status()
                result = response.json()
            
            processing_time = time.time() - start_time
            
            # Parse response (adjust based on their API format)
            return {
                "success": True,
                "is_ai_generated": result.get("is_ai_generated", False),
                "confidence": float(result.get("confidence", 0.0)),
                "detection_method": result.get("method", "unknown"),
                "flags": result.get("flags", []),
                "summary": result.get("summary", "Analysis complete"),
                "processing_time": processing_time,
                "raw_response": result
            }
            
        except httpx.TimeoutException:
            logger.error("AI Detection API timeout")
            return self._error_response(
                "API timeout - request took too long",
                time.time() - start_time
            )
        
        except httpx.HTTPStatusError as e:
            logger.error(f"AI Detection API error: {e.response.status_code}")
            return self._error_response(
                f"API error: {e.response.status_code}",
                time.time() - start_time
            )
        
        except Exception as e:
            logger.error(f"AI Detection failed: {e}", exc_info=True)
            return self._error_response(
                str(e),
                time.time() - start_time
            )
    
    def _error_response(self, error_message: str, processing_time: float) -> Dict:
        """Return error response format"""
        return {
            "success": False,
            "is_ai_generated": None,
            "confidence": 0.0,
            "detection_method": "error",
            "flags": [],
            "summary": f"Detection failed: {error_message}",
            "processing_time": processing_time,
            "error": error_message
        }

# Global instance
_ai_detection_service: Optional[AIDetectionService] = None

def get_ai_detection_service() -> AIDetectionService:
    """Get or create AI detection service instance"""
    global _ai_detection_service
    if _ai_detection_service is None:
        _ai_detection_service = AIDetectionService()
    return _ai_detection_service