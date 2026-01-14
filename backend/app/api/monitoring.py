from fastapi import APIRouter
from app.utils.rate_limiter import get_gemini_rate_limiter
from app.utils import cancellation_manager
import logging


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["monitoring"])

@router.post("/cancel")
async def cancel_all_processing():
    """
    Cancel all running jobs and block future processing requests.
    Called when the frontend Cancel button is pressed.
    """
    await cancellation_manager.cancel_all()

    logger.warning("🚨 Global cancellation triggered")

    return {
        "status": "cancelled",
        "message": "All running jobs cancelled and future requests blocked"
    }
    
@router.post("/reset")
async def reset_processing():
    await cancellation_manager.reset()
    return {
        "status": "ready",
        "message": "Processing reset. New jobs allowed."
    }

@router.get("/rate-limit-stats")
async def get_rate_limit_stats():
    """Get current rate limiter statistics."""
    limiter = get_gemini_rate_limiter()
    return limiter.get_stats()


@router.post("/rate-limit-reset")
async def reset_rate_limit_stats():
    """Reset rate limiter statistics."""
    limiter = get_gemini_rate_limiter()
    limiter.reset_stats()
    return {"message": "Rate limiter statistics reset"}