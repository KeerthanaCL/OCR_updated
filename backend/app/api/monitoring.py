from fastapi import APIRouter
from app.utils.rate_limiter import get_gemini_rate_limiter

router = APIRouter(prefix="/monitoring", tags=["Monitoring"])


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