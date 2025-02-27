# src/api/rate_limiter.py
import asyncio
import time
from datetime import datetime
from typing import Dict, Optional, Tuple
from loguru import logger


class RateLimiter:
    """
    Implements rate limiting for API requests.
    Enforces configurable limits on requests per second/minute and handles backoff when approaching limits.
    """

    def __init__(
            self,
            requests_per_second: int = 5,
            requests_per_minute: int = 300,
            max_backoff_time: float = 60.0,
            initial_backoff_time: float = 1.0,
            backoff_factor: float = 2.0
    ):
        """
        Initialize the rate limiter with the specified limits.

        Args:
            requests_per_second: Maximum number of requests allowed per second.
            requests_per_minute: Maximum number of requests allowed per minute.
            max_backoff_time: Maximum time to wait during backoff in seconds.
            initial_backoff_time: Initial backoff time in seconds.
            backoff_factor: Factor by which backoff time increases with each attempt.
        """
        self.requests_per_second = requests_per_second
        self.requests_per_minute = requests_per_minute
        self.max_backoff_time = max_backoff_time
        self.initial_backoff_time = initial_backoff_time
        self.backoff_factor = backoff_factor

        # Track request timestamps
        self.second_tracker: Dict[int, int] = {}  # {second: count}
        self.minute_tracker: Dict[int, int] = {}  # {minute: count}

        # Lock for thread safety
        self._lock = asyncio.Lock()

        # Tracking for metrics
        self.total_requests = 0
        self.delayed_requests = 0

    async def _clean_old_entries(self) -> None:
        """Clean up old entries from the trackers."""
        current_time = int(time.time())
        current_minute = current_time // 60

        # Clean second tracker (keep last 5 seconds)
        second_keys = list(self.second_tracker.keys())
        for second in second_keys:
            if current_time - second > 5:
                del self.second_tracker[second]

        # Clean minute tracker (keep last 5 minutes)
        minute_keys = list(self.minute_tracker.keys())
        for minute in minute_keys:
            if current_minute - minute > 5:
                del self.minute_tracker[minute]

    async def _get_current_rates(self) -> Tuple[int, int]:
        """Get the current request rates for the last second and minute."""
        current_time = int(time.time())
        current_second = current_time
        current_minute = current_time // 60

        # Calculate current rates
        requests_this_second = self.second_tracker.get(current_second, 0)
        requests_this_minute = sum(
            count for timestamp, count in self.minute_tracker.items()
            if current_minute - 1 <= timestamp <= current_minute
        )

        return requests_this_second, requests_this_minute

    async def _record_request(self) -> None:
        """Record a new request in the trackers."""
        current_time = int(time.time())
        current_second = current_time
        current_minute = current_time // 60

        # Update trackers
        self.second_tracker[current_second] = self.second_tracker.get(current_second, 0) + 1
        self.minute_tracker[current_minute] = self.minute_tracker.get(current_minute, 0) + 1

        # Update metrics
        self.total_requests += 1

    async def _calculate_backoff_time(self,
                                      second_rate: int,
                                      minute_rate: int,
                                      attempt: int = 0) -> float:
        """
        Calculate how long to wait before the next request.

        Args:
            second_rate: Current requests per second.
            minute_rate: Current requests per minute.
            attempt: Current retry attempt number.

        Returns:
            Time to wait in seconds.
        """
        # Check if we're approaching any limits
        second_ratio = second_rate / self.requests_per_second
        minute_ratio = minute_rate / self.requests_per_minute

        # If either ratio is high, apply backoff
        if second_ratio > 0.8 or minute_ratio > 0.8:
            # Calculate backoff using exponential backoff
            backoff_time = min(
                self.max_backoff_time,
                self.initial_backoff_time * (self.backoff_factor ** attempt)
            )

            # Increase backoff if we're very close to limits
            if second_ratio > 0.95 or minute_ratio > 0.95:
                backoff_time *= 2

            return backoff_time

        return 0.0  # No backoff needed

    async def acquire(self, endpoint: Optional[str] = None) -> None:
        """
        Acquire permission to make an API request. Will wait if necessary to respect rate limits.

        Args:
            endpoint: Optional endpoint string for more fine-grained rate limiting.
        """
        async with self._lock:
            await self._clean_old_entries()

            # Get current rates and calculate backoff
            second_rate, minute_rate = await self._get_current_rates()

            # Try to acquire permission, with exponential backoff
            attempt = 0
            while True:
                second_rate, minute_rate = await self._get_current_rates()

                # If we're below limits, we can proceed
                if second_rate < self.requests_per_second and minute_rate < self.requests_per_minute:
                    await self._record_request()
                    return

                # Otherwise, wait with backoff
                backoff_time = await self._calculate_backoff_time(second_rate, minute_rate, attempt)

                if backoff_time > 0:
                    self.delayed_requests += 1
                    logger.warning(
                        f"Rate limit approaching: {second_rate}/{self.requests_per_second} req/s, "
                        f"{minute_rate}/{self.requests_per_minute} req/min. "
                        f"Backing off for {backoff_time:.2f}s"
                    )
                    await asyncio.sleep(backoff_time)
                    attempt += 1
                else:
                    # If no backoff but still at limit, use a small delay
                    await asyncio.sleep(0.1)

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the rate limiter usage."""
        return {
            "total_requests": self.total_requests,
            "delayed_requests": self.delayed_requests,
            "delay_percentage": round((self.delayed_requests / max(1, self.total_requests)) * 100, 2),
            "current_time": datetime.now().isoformat()
        }
