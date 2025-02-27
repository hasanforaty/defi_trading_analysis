# src/api/cache.py
import json
import time
import asyncio
from typing import Any, Dict, Optional, TypeVar, Generic, Callable, Union, List
from datetime import datetime, timedelta
from loguru import logger
import redis.asyncio as redis

from config.settings import settings

T = TypeVar('T')


class CacheManager:
    """
    Cache manager that handles caching API responses and other data.
    Supports both Redis and in-memory caching.
    """

    def __init__(self, use_redis: bool = True, default_ttl: int = 300):
        """
        Initialize the cache manager.

        Args:
            use_redis: Whether to use Redis or in-memory cache.
            default_ttl: Default time-to-live in seconds for cached items.
        """
        self.use_redis = use_redis and settings.redis_url is not None
        self.default_ttl = default_ttl
        self._memory_cache: Dict[str, Dict[str, Any]] = {}
        self._redis_client: Optional[redis.Redis] = None
        self._lock = asyncio.Lock()

        # Initialize Redis if enabled
        if self.use_redis:
            try:
                self._redis_client = redis.from_url(settings.redis_url)
                logger.info("Redis cache initialized")
            except Exception as e:
                self.use_redis = False
                logger.warning(f"Failed to initialize Redis. Falling back to in-memory cache. Error: {e}")

    async def _set_memory_cache(self, key: str, value: Any, ttl: int) -> None:
        """Set a value in the in-memory cache with TTL."""
        expiry = time.time() + ttl
        self._memory_cache[key] = {
            'value': value,
            'expiry': expiry
        }

    async def _get_memory_cache(self, key: str) -> Optional[Any]:
        """Get a value from the in-memory cache, respecting TTL."""
        if key not in self._memory_cache:
            return None

        cache_item = self._memory_cache[key]
        current_time = time.time()

        # Check if the item is expired
        if current_time > cache_item['expiry']:
            del self._memory_cache[key]
            return None

        return cache_item['value']

    async def _clean_memory_cache(self) -> None:
        """Clean expired items from the in-memory cache."""
        current_time = time.time()
        keys_to_remove = []

        for key, item in self._memory_cache.items():
            if current_time > item['expiry']:
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del self._memory_cache[key]

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set a value in the cache.

        Args:
            key: The cache key.
            value: The value to cache.
            ttl: Time-to-live in seconds. If None, uses the default TTL.
        """
        if ttl is None:
            ttl = self.default_ttl

        # Serialize the value if it's not a simple type
        if not isinstance(value, (str, int, float, bool)) or value is None:
            value = json.dumps(value)

        async with self._lock:
            if self.use_redis and self._redis_client:
                try:
                    await self._redis_client.setex(key, ttl, value)
                    logger.debug(f"Set key '{key}' in Redis cache with TTL {ttl}s")
                except Exception as e:
                    logger.error(f"Error setting Redis cache for key '{key}': {e}")
                    await self._set_memory_cache(key, value, ttl)
            else:
                await self._set_memory_cache(key, value, ttl)
                logger.debug(f"Set key '{key}' in memory cache with TTL {ttl}s")

    async def get(self, key: str, default: Any = None) -> Any:
        """
        Get a value from the cache.

        Args:
            key: The cache key.
            default: Default value to return if the key doesn't exist.

        Returns:
            The cached value or the default value if not found.
        """
        async with self._lock:
            if self.use_redis and self._redis_client:
                try:
                    value = await self._redis_client.get(key)
                    if value is None:
                        return default

                    # Try to deserialize if it's a string
                    if isinstance(value, bytes):
                        value = value.decode('utf-8')

                    try:
                        return json.loads(value)
                    except (json.JSONDecodeError, TypeError):
                        return value

                except Exception as e:
                    logger.error(f"Error getting Redis cache for key '{key}': {e}")
                    # Fallback to memory cache
                    value = await self._get_memory_cache(key)
                    return value if value is not None else default
            else:
                value = await self._get_memory_cache(key)
                return value if value is not None else default

    async def delete(self, key: str) -> bool:
        """
        Delete a key from the cache.

        Args:
            key: The cache key to delete.

        Returns:
            True if the key was deleted, False otherwise.
        """
        async with self._lock:
            if self.use_redis and self._redis_client:
                try:
                    result = await self._redis_client.delete(key)
                    success = result > 0
                except Exception as e:
                    logger.error(f"Error deleting Redis cache for key '{key}': {e}")
                    success = False

                # Also delete from memory cache if it exists
                if key in self._memory_cache:
                    del self._memory_cache[key]
                    success = True

                return success
            else:
                if key in self._memory_cache:
                    del self._memory_cache[key]
                    return True
                return False

    async def clear_all(self) -> bool:
        """
        Clear all cached data.

        Returns:
            True if the operation was successful, False otherwise.
        """
        async with self._lock:
            if self.use_redis and self._redis_client:
                try:
                    await self._redis_client.flushdb()
                    logger.info("Redis cache cleared")
                except Exception as e:
                    logger.error(f"Error clearing Redis cache: {e}")

            # Clear memory cache
            self._memory_cache.clear()
            logger.info("Memory cache cleared")
            return True

    async def invalidate_pattern(self, pattern: str) -> int:
        """
        Delete all keys matching a pattern.

        Args:
            pattern: The pattern to match (e.g., "user:*").

        Returns:
            Number of keys deleted.
        """
        count = 0
        async with self._lock:
            if self.use_redis and self._redis_client:
                try:
                    # Get all keys matching the pattern
                    keys = await self._redis_client.keys(pattern)
                    if keys:
                        count = await self._redis_client.delete(*keys)
                        logger.info(f"Deleted {count} keys matching pattern '{pattern}' from Redis")
                except Exception as e:
                    logger.error(f"Error invalidating Redis keys with pattern '{pattern}': {e}")

            # Also invalidate memory cache with simple pattern matching
            # This is a simplified version of Redis pattern matching
            memory_keys = list(self._memory_cache.keys())
            for key in memory_keys:
                if self._match_pattern(key, pattern):
                    del self._memory_cache[key]
                    count += 1

            if count:
                logger.info(f"Deleted {count} keys matching pattern '{pattern}' from memory cache")

            return count

    def _match_pattern(self, key: str, pattern: str) -> bool:
        """
        Simple implementation of Redis-like pattern matching.
        Supports only the * wildcard.

        Args:
            key: The key to check.
            pattern: The pattern to match against.

        Returns:
            True if the key matches the pattern, False otherwise.
        """
        # Replace * with .* for regex
        import re
        regex_pattern = pattern.replace("*", ".*")
        return bool(re.match(f"^{regex_pattern}$", key))

    async def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the cache.

        Returns:
            Dictionary with cache statistics.
        """
        async with self._lock:
            stats = {
                "type": "redis" if self.use_redis else "memory",
                "memory_cache_size": len(self._memory_cache),
                "default_ttl": self.default_ttl,
            }

            if self.use_redis and self._redis_client:
                try:
                    info = await self._redis_client.info()
                    stats.update({
                        "redis_used_memory": info.get("used_memory_human", "N/A"),
                        "redis_total_keys": info.get("db0", {}).get("keys", 0),
                        "redis_uptime": info.get("uptime_in_seconds", 0),
                    })
                except Exception as e:
                    logger.error(f"Error getting Redis info: {e}")

            return stats

    async def close(self) -> None:
        """Close connections and clean up resources."""
        if self.use_redis and self._redis_client:
            await self._redis_client.close()
            logger.info("Redis connection closed")
