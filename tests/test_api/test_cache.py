# tests/test_api/test_cache.py
import pytest
import asyncio
from datetime import datetime, timedelta

from src.api.cache import CacheManager


@pytest.fixture
async def cache_manager():
    # Use in-memory cache for testing
    cache = CacheManager(use_redis=False)
    yield cache
    await cache.clear_all()


@pytest.mark.asyncio
async def test_cache_set_get(cache_manager):
    # Test basic set and get
    await cache_manager.set("test_key", "test_value")
    value = await cache_manager.get("test_key")
    assert value == "test_value"


@pytest.mark.asyncio
async def test_cache_ttl(cache_manager):
    # Test that TTL works
    await cache_manager.set("ttl_key", "ttl_value", ttl=1)

    # Value should exist immediately
    assert await cache_manager.get("ttl_key") == "ttl_value"

    # Wait for TTL to expire
    await asyncio.sleep(1.1)

    # Value should be gone
    assert await cache_manager.get("ttl_key") is None


@pytest.mark.asyncio
async def test_cache_delete(cache_manager):
    # Test delete functionality
    await cache_manager.set("delete_key", "delete_value")
    assert await cache_manager.get("delete_key") == "delete_value"

    # Delete the key
    success = await cache_manager.delete("delete_key")
    assert success is True

    # Value should be gone
    assert await cache_manager.get("delete_key") is None

    # Deleting non-existent key should return False
    success = await cache_manager.delete("nonexistent_key")
    assert success is False


@pytest.mark.asyncio
async def test_cache_complex_values(cache_manager):
    # Test with complex values
    complex_value = {
        "number": 42,
        "string": "hello",
        "list": [1, 2, 3],
        "nested": {"a": 1, "b": 2}
    }

    await cache_manager.set("complex_key", complex_value)
    retrieved = await cache_manager.get("complex_key")

    assert retrieved == complex_value


@pytest.mark.asyncio
async def test_cache_clear_all(cache_manager):
    # Test clearing all cache
    await cache_manager.set("key1", "value1")
    await cache_manager.set("key2", "value2")

    assert await cache_manager.get("key1") == "value1"
    assert await cache_manager.get("key2") == "value2"

    # Clear all cache
    success = await cache_manager.clear_all()
    assert success is True

    # All values should be gone
    assert await cache_manager.get("key1") is None
    assert await cache_manager.get("key2") is None


@pytest.mark.asyncio
async def test_cache_invalidate_pattern(cache_manager):
    # Test invalidating keys by pattern
    await cache_manager.set("prefix:key1", "value1")
    await cache_manager.set("prefix:key2", "value2")
    await cache_manager.set("other:key3", "value3")

    # Invalidate by pattern
    count = await cache_manager.invalidate_pattern("prefix:*")

    # Two keys should be invalidated
    assert count == 2

    # Check that the right keys were invalidated
    assert await cache_manager.get("prefix:key1") is None
    assert await cache_manager.get("prefix:key2") is None
    assert await cache_manager.get("other:key3") == "value3"
