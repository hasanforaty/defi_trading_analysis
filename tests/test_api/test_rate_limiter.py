# tests/test_api/test_rate_limiter.py
import pytest
import asyncio
import time
from datetime import datetime

from src.api.rate_limiter import RateLimiter


@pytest.fixture
def rate_limiter():
    # Configure a rate limiter with strict limits for testing
    return RateLimiter(
        requests_per_second=5,
        requests_per_minute=10,
        max_backoff_time=2.0,
        initial_backoff_time=0.1,
        backoff_factor=2.0
    )


@pytest.mark.asyncio
async def test_rate_limiter_basic(rate_limiter):
    # Test basic acquisition
    start_time = time.time()

    # These should all be allowed
    for _ in range(5):
        await rate_limiter.acquire()

    end_time = time.time()

    # All 5 requests should be allowed quickly
    assert end_time - start_time < 0.5, "Rate limiter should allow requests up to the limit"


@pytest.mark.asyncio
async def test_rate_limiter_second_limit(rate_limiter):
    # Test second-based rate limiting
    start_time = time.time()

    # First 5 should be quick
    for _ in range(5):
        await rate_limiter.acquire()

    # This one should be delayed
    await rate_limiter.acquire()

    end_time = time.time()

    # The 6th request should be delayed
    assert end_time - start_time >= 0.1, "Rate limiter should delay requests beyond the per-second limit"


@pytest.mark.asyncio
async def test_rate_limiter_minute_limit(rate_limiter):
    # Test minute-based rate limiting
    start_time = time.time()

    # Make 10 requests to hit the minute limit
    for _ in range(10):
        await rate_limiter.acquire()

    # This one should be delayed significantly
    await rate_limiter.acquire()

    end_time = time.time()

    # The 11th request should be delayed
    assert end_time - start_time >= 0.1, "Rate limiter should delay requests beyond the per-minute limit"


@pytest.mark.asyncio
async def test_rate_limiter_stats(rate_limiter):
    # Test statistics tracking
    await rate_limiter.acquire()
    await rate_limiter.acquire()

    stats = rate_limiter.get_stats()

    assert stats["total_requests"] == 2
    assert "delayed_requests" in stats
    assert "delay_percentage" in stats
    assert "current_time" in stats


@pytest.mark.asyncio
async def test_rate_limiter_concurrent(rate_limiter):
    # Test concurrent requests
    async def make_request(id):
        await rate_limiter.acquire()
        return id

    # Launch 10 concurrent requests
    tasks = [make_request(i) for i in range(10)]
    results = await asyncio.gather(*tasks)

    # All tasks should complete
    assert len(results) == 10
    assert set(results) == set(range(10))
