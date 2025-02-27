"""Tests for the cache manager."""
import pytest
import asyncio
from unittest.mock import patch, MagicMock

from src.api.cache import CacheManager, MemoryCache, RedisCache


@pytest.fixture
def memory_cache():
    """Create a memory cache instance."""
    return MemoryCache()


@pytest.fixture
def cache_manager():
    """Create a cache manager instance with memory cache."""
    with patch('src.api.cache.REDIS_AVAILABLE', False):
        return CacheManager(prefix="test")


@pytest.mark.asyncio
async def test_memory_cache_set_get(memory_cache):
    """Test setting and getting values in memory cache."""
    # Set a value
    await memory_cache.set("test_key", "test_value")

    # Get the value
    value = await memory_cache.get("test_key")

    # Assert the value matches
    assert value == "test_value"


@pytest.mark.asyncio
async def test_memory_cache_ttl(memory_cache):
    """Test TTL functionality in memory cache."""
    # Set a value with a TTL of 0.1 seconds
    await memory_cache.set("test_key", "test_value", ttl=0.1)

    # Get the value immediately (should exist)
    value = await memory_cache.get("test_key")
    assert value == "test_value"

    # Wait for the TTL to expire
    await asyncio.sleep(0.2)

    # Get the value again (should be None)
    value = await memory_cache.get("test_key")
    assert value is None


@pytest.mark.asyncio
async def test_memory_cache_delete(memory_cache):
    """Test deleting values from memory cache."""
    # Set a value
    await memory_cache.set("test_key", "test_value")

    # Delete the value
    await memory_cache.delete("test_key")

    # Get the value (should be None)
    value = await memory_cache.get("test_key")
    assert value is None


@pytest.mark.asyncio
async def test_memory_cache_clear(memory_cache):
    """Test clearing the memory cache."""
    # Set multiple values
    await memory_cache.set("test_key1", "test_value1")
    await memory_cache.set("test_key2", "test_value2")

    # Clear the cache
    await memory_cache.clear()

    # Get the values (should be None)
    value1 = await memory_cache.get("test_key1")
    value2 = await memory_cache.get("test_key2")
    assert value1 is None
    assert value2 is None


@pytest.mark.asyncio
async def test_cache_manager_format_key(cache_manager):
    """Test key formatting in cache manager."""
    # Format a key
    key = cache_manager._format_key("test_key")

    # Assert the key is correctly formatted
    assert key == "test:test_key"


@pytest.mark.asyncio
async def test_cache_manager_get_set(cache_manager):
    """Test setting and getting values through cache manager."""
    # Set a value
    await cache_manager.set("test_key", "test_value")

    # Get the value
    value = await cache_manager.get("test_key")

    # Assert the value matches
    assert value == "test_value"


@pytest.mark.asyncio
async def test_cache_manager_delete(cache_manager):
    """Test deleting values through cache manager."""
    # Set a value
    await cache_manager.set("test_key", "test_value")

    # Delete the value
    await cache_manager.delete("test_key")

    # Get the value (should be None)
    value = await cache_manager.get("test_key")
    assert value is None


@pytest.mark.asyncio
async def test_cache_manager_clear(cache_manager):
    """Test clearing through cache manager."""
    # Set multiple values
    await cache_manager.set("test_key1", "test_value1")
    await cache_manager.set("test_key2", "test_value2")

    # Clear the cache
    await cache_manager.clear()

    # Get the values (should be None)
    value1 = await cache_manager.get("test_key1")
    value2 = await cache_manager.get("test_key2")
    assert value1 is None
    assert value2 is None


@pytest.mark.asyncio
async def test_cache_manager_cached_decorator(cache_manager):
    """Test the cached decorator functionality."""
    # Create a mock function that counts calls
    call_count = 0

    async def test_function(arg1, arg2):
        nonlocal call_count
        call_count += 1
        return f"{arg1}_{arg2}"

    # Call the function through the cached decorator
    result1 = await cache_manager.cached(
        "test_func", test_function, "arg1", arg2="arg2", ttl=10
    )
    assert result1 == "arg1_arg2"
    assert call_count == 1

    # Call again with the same arguments (should use cache)
    result2 = await cache_manager.cached(
        "test_func", test_function, "arg1", arg2="arg2", ttl=10
    )
    assert result2 == "arg1_arg2"
    assert call_count == 1  # Still 1 because it used the cache

    # Call with different arguments (should call the function again)
    result3 = await cache_manager.cached(
        "test_func", test_function, "arg1", arg2="arg3", ttl=10
    )
    assert result3 == "arg1_arg3"
    assert call_count == 2


@pytest.mark.asyncio
async def test_redis_cache():
    """Test Redis cache functionality with mocks."""
    # Create a mock Redis client
    mock_redis = MagicMock()
    mock_redis.get = MagicMock(return_value=b'{"key": "value"}')
    mock_redis.setex = MagicMock()
    mock_redis.set = MagicMock()
    mock_redis.delete = MagicMock()
    mock_redis.keys = MagicMock(return_value=["test_key1", "test_key2"])

    # Create a Redis cache with the mock client
    redis_cache = RedisCache("redis://localhost:6379")
    redis_cache.redis = mock_redis

    # Test get
    value = await redis_cache.get("test_key")
    assert value == {"key": "value"}
    mock_redis.get.assert_called_once_with("test_key")

    # Test set with TTL
    await redis_cache.set("test_key", {"key": "value"}, ttl=10)
    mock_redis.setex.assert_called_once()

    # Test set without TTL
    await redis_cache.set("test_key", {"key": "value"})
    mock_redis.set.assert_called_once()

    # Test delete
    await redis_cache.delete("test_key")
    mock_redis.delete.assert_called_once_with("test_key")

    # Test clear
    await redis_cache.clear()
    mock_redis.keys.assert_called_once_with("dextools:*")
    mock_redis.delete.assert_called()
