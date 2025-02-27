"""Tests for the rate limiter."""
import pytest
import asyncio
import time
from unittest.mock import patch, Mock

from src.api.rate_limiter import RateLimiter


@pytest.fixture
def rate_limiter():
    """Create a rate limiter with a small window for testing."""
    return RateLimiter(requests_per_second=5, requests_per_minute=10)


@pytest.mark.asyncio
async def test_acquire_within_limits(rate_limiter):
    """Test acquiring permission when within limits."""
    # Should be able to acquire up to the per-second limit
    for _ in range(rate_limiter.requests_per_second):
        assert await rate_limiter.acquire() is True

    # The next one should fail (exceeded per-second limit)
    assert await rate_limiter.acquire() is False


@pytest.mark.asyncio
async def test_wait_until_available(rate_limiter):
    """Test waiting until a request slot is available."""
    # Use up all slots
    for _ in range(rate_limiter.requests_per_second):
        assert await rate_limiter.acquire() is True

    # The next acquire should fail
    assert await rate_limiter.acquire() is False

    # Patch time to simulate passage of time
    original_time = time.time

    try:
        # Mock time.time to return a value 1 second in the future
        with patch('time.time', return_value=original_time() + 1):
            # Should be able to acquire again after the second window resets
            assert await rate_limiter.wait_until_available(timeout=0.1) is True
    finally:
        time.time = original_time


@pytest.mark.asyncio
async def test_wait_until_available_timeout(rate_limiter):
    """Test timeout while waiting for a request slot."""
    # Use up all slots
    for _ in range(rate_limiter.requests_per_second):
        assert await rate_limiter.acquire() is True

    # Patch sleep to avoid actually waiting
    with patch('asyncio.sleep', return_value=None):
        # Wait with a tiny timeout, should fail
        assert await rate_limiter.wait_until_available(timeout=0.001) is False


@pytest.mark.asyncio
async def test_per_minute_limit(rate_limiter):
    """Test per-minute rate limiting."""
    # Use up all per-minute slots
    for _ in range(rate_limiter.requests_per_minute):
        # Reset per-second counter after each second
        if _ % rate_limiter.requests_per_second == 0 and _ > 0:
            with patch('time.time', return_value=time.time() + 1):
                rate_limiter._second_window_requests = 0
                rate_limiter._last_second_reset = time.time()

        assert await rate_limiter.acquire() is True

    # The next one should fail (exceeded per-minute limit)
    assert await rate_limiter.acquire() is False

    # Patch time to simulate passage of time
    original_time = time.time

    try:
        # Mock time.time to return a value 60 seconds in the future
        with patch('time.time', return_value=original_time() + 60):
            # Should be able to acquire again after the minute window resets
            assert await rate_limiter.wait_until_available(timeout=0.1) is True
    finally:
        time.time = original_time


@pytest.mark.asyncio
async def test_get_stats(rate_limiter):
    """Test getting rate limiter statistics."""
    # Acquire some slots
    for _ in range(3):
        await rate_limiter.acquire()

    # Get stats
    stats = rate_limiter.get_stats()

    # Assert stats are correct
    assert stats["second_window_requests"] == 3
    assert stats["minute_window_requests"] == 3
    assert stats["requests_per_second_limit"] == 5
    assert stats["requests_per_minute_limit"] == 10
    assert stats["second_window_usage"] == 3 / 5
    assert stats["minute_window_usage"] == 3 / 10
