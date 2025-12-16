"""
Rate Limiter for API calls
Ensures compliance with API rate limits (5-15 requests per minute)
"""

import asyncio
import time
import logging
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Token bucket rate limiter for API calls.
    
    Supports:
    - Configurable requests per minute (RPM)
    - Automatic delay insertion between calls
    - Burst handling with token bucket
    - Request tracking and statistics
    """
    
    def __init__(self, requests_per_minute: int = 10, burst_size: Optional[int] = None):
        """
        Initialize rate limiter.
        
        Args:
            requests_per_minute: Maximum requests allowed per minute (default: 10)
            burst_size: Maximum burst size (default: same as RPM)
        """
        self.rpm = requests_per_minute
        self.burst_size = burst_size or requests_per_minute
        
        # Token bucket
        self.tokens = self.burst_size
        self.last_update = time.time()
        
        # Minimum delay between requests (in seconds)
        self.min_delay = 60.0 / requests_per_minute
        
        # Statistics
        self.total_requests = 0
        self.total_delays = 0
        self.total_delay_time = 0.0
        
        # Lock for thread safety
        self._lock = asyncio.Lock()
        
        logger.info(f"🔒 Rate limiter initialized: {requests_per_minute} RPM, "
                   f"min delay: {self.min_delay:.2f}s, burst: {self.burst_size}")
    
    async def acquire(self) -> float:
        """
        Acquire permission to make an API call.
        Blocks if rate limit would be exceeded.
        
        Returns:
            float: Time waited in seconds
        """
        async with self._lock:
            now = time.time()
            
            # Refill tokens based on time elapsed
            time_passed = now - self.last_update
            self.tokens = min(
                self.burst_size,
                self.tokens + (time_passed * self.rpm / 60.0)
            )
            self.last_update = now
            
            # Check if we have tokens
            if self.tokens < 1.0:
                # Calculate wait time needed
                tokens_needed = 1.0 - self.tokens
                wait_time = (tokens_needed * 60.0) / self.rpm
                
                logger.warning(f"⏳ Rate limit: waiting {wait_time:.2f}s "
                             f"(tokens: {self.tokens:.2f}/{self.burst_size})")
                
                # Wait for tokens to refill
                await asyncio.sleep(wait_time)
                
                # Update tracking
                self.total_delays += 1
                self.total_delay_time += wait_time
                
                # Recalculate tokens after wait
                now = time.time()
                time_passed = now - self.last_update
                self.tokens = min(
                    self.burst_size,
                    self.tokens + (time_passed * self.rpm / 60.0)
                )
                self.last_update = now
                
                wait_time_total = wait_time
            else:
                wait_time_total = 0.0
            
            # Consume one token
            self.tokens -= 1.0
            self.total_requests += 1
            
            logger.debug(f"✅ Rate limit passed: request #{self.total_requests}, "
                        f"tokens remaining: {self.tokens:.2f}/{self.burst_size}")
            
            return wait_time_total
    
    def get_stats(self) -> dict:
        """Get rate limiter statistics."""
        return {
            "total_requests": self.total_requests,
            "total_delays": self.total_delays,
            "total_delay_time": self.total_delay_time,
            "average_delay": (
                self.total_delay_time / self.total_delays 
                if self.total_delays > 0 else 0.0
            ),
            "current_tokens": self.tokens,
            "max_tokens": self.burst_size,
            "rpm_limit": self.rpm,
            "min_delay": self.min_delay
        }
    
    def reset_stats(self):
        """Reset statistics."""
        self.total_requests = 0
        self.total_delays = 0
        self.total_delay_time = 0.0
        logger.info("📊 Rate limiter statistics reset")


# Global rate limiter instances
_gemini_rate_limiter: Optional[RateLimiter] = None


def get_gemini_rate_limiter(rpm: int = 10) -> RateLimiter:
    """
    Get or create the global Gemini API rate limiter.
    
    Args:
        rpm: Requests per minute limit (default: 10, adjust based on your tier)
    
    Returns:
        RateLimiter instance
    """
    global _gemini_rate_limiter
    
    if _gemini_rate_limiter is None:
        _gemini_rate_limiter = RateLimiter(requests_per_minute=rpm)
        logger.info(f"🔒 Created global Gemini rate limiter: {rpm} RPM")
    
    return _gemini_rate_limiter


def set_gemini_rate_limit(rpm: int):
    """
    Set a new rate limit for Gemini API.
    
    Args:
        rpm: New requests per minute limit
    """
    global _gemini_rate_limiter
    _gemini_rate_limiter = RateLimiter(requests_per_minute=rpm)
    logger.info(f"🔒 Updated Gemini rate limit to {rpm} RPM")