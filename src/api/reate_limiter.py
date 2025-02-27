"""Rate limiter implementation for API clients."""
import asyncio
import logging
import time
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Rate limiter to control the rate of API requests.

    Attributes:
        requests_per_second: Maximum requests per second
        requests_per_minute: Maximum requests per minute
    """

    def __init__(self, requests_per_second: int = 5, requests_per_minute: int = 100):
        """
        Initialize the rate limiter.

        Args:
            requests_per_second: Maximum requests per second
            requests_per_minute: Maximum requests per minute
        """
        self.requests_per_second = requests_per_second
        self.requests_per_minute = requests_per_minute

        self._second_window_requests = 0
        self._minute_window_requests = 0
        self._last_second_reset = time.time()
        self._last_minute_reset = time.time()
        self._lock = asyncio.Lock()

    async def acquire(self) -> bool:
        """
        Acquire permission to make a request.

        Returns:
            True if permission is granted, False otherwise
        """
        async with self._lock:
            current_time = time.time()

            # Reset counters if needed
            if current_time - self._last_second_reset >= 1:
                self._second_window_requests = 0
                self._last_second_reset = current_time

            if current_time - self._last_minute_reset >= 60:
                self._minute_window_requests = 0
                self._last_minute_reset = current_time

            # Check if we're within limits
            if (self._second_window_requests >= self.requests_per_second or
                    self._minute_window_requests >= self.requests_per_minute):
                return False

            # Increment counters
            self._second_window_requests += 1
            self._minute_window_requests += 1
            return True

    async def wait_until_available(self, timeout: Optional[float] = None) -> bool:
        """
        Wait until a request slot becomes available.

        Args:
            timeout: Maximum time to wait in seconds (None for no timeout)

        Returns:
            True if a slot became available, False if timeout was reached
        """
        start_time = time.time()
        backoff = 0.1  # Start with 100ms backoff

        while True:
            if await self.acquire():
                return True

            # Check timeout
            if timeout is not None and time.time() - start_time > timeout:
                logger.warning("Timeout reached while waiting for rate limit slot")
                return False

            # Wait with exponential backoff
            await asyncio.sleep(backoff)
            backoff = min(backoff * 1.5, 5)  # Cap at 5 seconds

    def get_stats(self) -> Dict[str, Any]:
        """
        Get current rate limiter statistics.

        Returns:
            Dictionary with current usage statistics
        """
        return {
            "second_window_requests": self._second_window_requests,
            "minute_window_requests": self._minute_window_requests,
            "requests_per_second_limit": self.requests_per_second,
            "requests_per_minute_limit": self.requests_per_minute,
            "second_window_usage": self._second_window_requests / self.requests_per_second,
            "minute_window_usage": self._minute_window_requests / self.requests_per_minute
        }
