"""Cache management for API responses."""
import json
import logging
import asyncio
from typing import Any, Dict, Optional, Union, Callable
from datetime import datetime, timedelta
import time

from config import settings

try:
    import redis.asyncio as redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from config.settings import get_settings

logger = logging.getLogger(__name__)


class MemoryCache:
    """Simple in-memory cache implementation."""

    def __init__(self):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[Any]:
        """Get a value from the cache."""
        async with self._lock:
            cache_item = self._cache.get(key)
            if cache_item is None:
                return None

            # Check if item is expired
            if cache_item.get('expiry') and time.time() > cache_item['expiry']:
                del self._cache[key]
                return None

            return cache_item['value']

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set a value in the cache with optional TTL in seconds."""
        async with self._lock:
            cache_item: Dict[str, Any] = {'value': value}
            if ttl:
                cache_item['expiry'] = time.time() + ttl
            self._cache[key] = cache_item

    async def delete(self, key: str) -> None:
        """Delete a key from the cache."""
        async with self._lock:
            if key in self._cache:
                del self._cache[key]

    async def clear(self) -> None:
        """Clear all items from the cache."""
        async with self._lock:
            self._cache.clear()


class RedisCache:
    """Redis-backed cache implementation."""

    def __init__(self, redis_url: str):
        self.redis = redis.from_url(redis_url)

    async def get(self, key: str) -> Optional[Any]:
        """Get a value from Redis cache."""
        value = await self.redis.get(key)
        if value:
            return json.loads(value)
        return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set a value in Redis cache with optional TTL in seconds."""
        serialized = json.dumps(value)
        if ttl:
            await self.redis.setex(key, ttl, serialized)
        else:
            await self.redis.set(key, serialized)

    async def delete(self, key: str) -> None:
        """Delete a key from Redis cache."""
        await self.redis.delete(key)

    async def clear(self) -> None:
        """Clear all items from Redis cache with a specific prefix."""
        # This is a simplified implementation. In production, you might
        # want to use a more sophisticated approach to only clear your app's keys.
        all_keys = await self.redis.keys('dextools:*')
        if all_keys:
            await self.redis.delete(*all_keys)


class CacheManager:
    """Cache manager that can work with either Redis or in-memory cache."""

    def __init__(self, prefix: str = "dextools"):
        self.prefix = prefix

        # Try to use Redis if it's available and configured
        if REDIS_AVAILABLE and hasattr(settings, 'REDIS_URL') and settings.REDIS_URL:
            logger.info("Using Redis cache")
            self.cache = RedisCache(settings.REDIS_URL)
        else:
            logger.info("Using in-memory cache")
            self.cache = MemoryCache()

    def _format_key(self, key: str) -> str:
        """Format a cache key with the prefix."""
        return f"{self.prefix}:{key}"

    async def get(self, key: str) -> Optional[Any]:
        """Get a value from the cache."""
        full_key = self._format_key(key)
        return await self.cache.get(full_key)

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set a value in the cache with optional TTL in seconds."""
        full_key = self._format_key(key)
        await self.cache.set(full_key, value, ttl)

    async def delete(self, key: str) -> None:
        """Delete a key from the cache."""
        full_key = self._format_key(key)
        await self.cache.delete(full_key)

    async def clear(self) -> None:
        """Clear all items from the cache."""
        await self.cache.clear()

    async def cached(self, key_prefix: str, func: Callable, *args, ttl: Optional[int] = 300, **kwargs) -> Any:
        """
        Decorator-like function to cache the results of a function call.

        Args:
            key_prefix: Prefix for the cache key
            func: Function to call if cache misses
            ttl: Time-to-live for the cache entry in seconds
            *args, **kwargs: Arguments to pass to the function

        Returns:
            The result of the function call, either from cache or fresh
        """
        # Create a cache key from the function name and arguments
        key_parts = [key_prefix]
        key_parts.extend([str(arg) for arg in args])
        key_parts.extend([f"{k}={v}" for k, v in kwargs.items()])
        cache_key = ":".join(key_parts)

        # Try to get from cache
        cached_value = await self.get(cache_key)
        if cached_value is not None:
            logger.debug(f"Cache hit for {cache_key}")
            return cached_value

        # Call the function and cache the result
        logger.debug(f"Cache miss for {cache_key}, calling function")
        result = await func(*args, **kwargs)
        await self.set(cache_key, result, ttl)
        return result
