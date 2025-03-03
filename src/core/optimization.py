import asyncio
import functools
import time
from typing import Dict, Any, Callable, Optional, List, Tuple, Set, TypeVar, cast
import inspect
from dataclasses import dataclass, field
import concurrent.futures
from contextlib import asynccontextmanager
from functools import wraps
import pickle
import hashlib
import os
import logging
from loguru import logger

T = TypeVar('T')
R = TypeVar('R')


class Cache:
    """
    Simple in-memory cache implementation with TTL support.
    """

    def __init__(self, max_size: int = 1000, default_ttl: int = 300):
        """
        Initialize the cache.

        Args:
            max_size: Maximum number of items in the cache
            default_ttl: Default time-to-live in seconds
        """
        self._cache: Dict[str, Tuple[Any, float]] = {}
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._hits = 0
        self._misses = 0
        self._lock = asyncio.Lock()
        logger.info(f"Cache initialized with max_size={max_size}, default_ttl={default_ttl}s")

    async def get(self, key: str, default: Any = None) -> Any:
        """
        Get a value from the cache.

        Args:
            key: Cache key
            default: Default value if key is not in cache

        Returns:
            Any: Cached value or default
        """
        async with self._lock:
            if key in self._cache:
                value, expiry = self._cache[key]

                # Check if expired
                if expiry < time.time():
                    # Expired
                    del self._cache[key]
                    self._misses += 1
                    return default

                # Not expired
                self._hits += 1
                return value

            # Not in cache
            self._misses += 1
            return default

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set a value in the cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (None for default)
        """
        ttl = ttl if ttl is not None else self._default_ttl
        expiry = time.time() + ttl

        async with self._lock:
            # If at max size, remove oldest item
            if len(self._cache) >= self._max_size and key not in self._cache:
                # Find oldest item (lowest expiry)
                oldest_key = min(self._cache.items(), key=lambda item: item[1][1])[0]
                del self._cache[oldest_key]

            # Set value
            self._cache[key] = (value, expiry)

    async def delete(self, key: str) -> bool:
        """
        Delete a key from the cache.

        Args:
            key: Cache key

        Returns:
            bool: True if key was deleted, False if not found
        """
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    async def clear(self) -> None:
        """Clear the cache."""
        async with self._lock:
            self._cache.clear()
            logger.debug("Cache cleared")

    async def expire_all(self) -> int:
        """
        Remove all expired items from the cache.

        Returns:
            int: Number of items removed
        """
        now = time.time()
        count = 0

        async with self._lock:
            # Find expired keys
            expired_keys = [key for key, (_, expiry) in self._cache.items() if expiry < now]

            # Remove expired keys
            for key in expired_keys:
                del self._cache[key]
                count += 1

        return count

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dict[str, Any]: Cache statistics
        """
        hit_rate = 0
        total = self._hits + self._misses
        if total > 0:
            hit_rate = (self._hits / total) * 100

        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "default_ttl": self._default_ttl
        }


# Global cache instance
_global_cache: Optional[Cache] = None


def get_cache() -> Cache:
    """
    Get the global cache instance.

    Returns:
        Cache: Global cache instance
    """
    global _global_cache
    if _global_cache is None:
        _global_cache = Cache()
    return _global_cache


def cached(ttl: Optional[int] = None, key_prefix: Optional[str] = None):
    """
    Decorator for caching function results.

    Args:
        ttl: Time-to-live in seconds for cached results (None for default)
        key_prefix: Optional prefix for cache keys

    Returns:
        Decorator function
    """

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get cache
            cache = get_cache()

            # Generate cache key
            key_parts = [key_prefix or func.__qualname__]

            # Add args and kwargs to key
            for arg in args:
                try:
                    key_parts.append(str(arg))
                except Exception:
                    # If cannot be converted to string, use type
                    key_parts.append(type(arg).__name__)

            # Sort kwargs for consistent key
            for k, v in sorted(kwargs.items()):
                try:
                    key_parts.append(f"{k}={v}")
                except Exception:
                    key_parts.append(f"{k}={type(v).__name__}")

            # Generate final key
            cache_key = hashlib.md5("_".join(key_parts).encode()).hexdigest()

            # Try to get from cache
            result = await cache.get(cache_key)
            if result is not None:
                logger.debug(f"Cache hit for {func.__qualname__}")
                return result

            # Not in cache, call function
            logger.debug(f"Cache miss for {func.__qualname__}")
            result = await func(*args, **kwargs)

            # Store in cache
            await cache.set(cache_key, result, ttl)

            return result

        return wrapper

    return decorator


class ParallelExecutor:
    """
    Manages parallel execution of tasks using concurrent.futures.
    """

    def __init__(self, max_workers: Optional[int] = None,
                 worker_type: str = "thread"):
        """
        Initialize the parallel executor.

        Args:
            max_workers: Maximum number of workers (None for default)
            worker_type: Type of worker ("thread" or "process")
        """
        self._max_workers = max_workers
        self._worker_type = worker_type
        self._executor = None

        logger.info(f"ParallelExecutor initialized with max_workers={max_workers}, type={worker_type}")

    def _get_executor(self):
        """
        Get the appropriate executor based on worker type.

        Returns:
            Executor: concurrent.futures executor
        """
        if self._worker_type.lower() == "thread":
            return concurrent.futures.ThreadPoolExecutor(max_workers=self._max_workers)
        elif self._worker_type.lower() == "process":
            return concurrent.futures.ProcessPoolExecutor(max_workers=self._max_workers)
        else:
            raise ValueError(f"Invalid worker type: {self._worker_type}")

    async def map(self, func: Callable[[T], R], items: List[T]) -> List[R]:
        """
        Execute a function on a list of items in parallel.

        Args:
            func: Function to execute
            items: List of items to process

        Returns:
            List[R]: List of results
        """
        loop = asyncio.get_event_loop()

        with self._get_executor() as executor:
            # Submit all tasks to the executor
            future = loop.run_in_executor(
                None,  # Use default executor
                lambda: list(executor.map(func, items))
            )

            # Wait for all tasks to complete
            results = await future

        return results

    async def execute(self, tasks: List[Tuple[Callable, Tuple, Dict]]) -> List[Any]:
        """
        Execute a list of tasks in parallel.

        Args:
            tasks: List of (function, args, kwargs) tuples

        Returns:
            List[Any]: List of results
        """
        loop = asyncio.get_event_loop()
        results = []

        with self._get_executor() as executor:
            # Create futures for all tasks
            futures = []
            for func, args, kwargs in tasks:
                future = executor.submit(func, *args, **kwargs)
                futures.append(future)

            # Wait for all futures to complete
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Task execution failed: {str(e)}")
                    results.append(None)

        return results


@dataclass
class RateLimiter:
    """
    Rate limiter for controlling API request rates.
    """

    requests_per_second: float
    max_burst: int = 1
    tokens: float = field(default=0)
    last_refill: float = field(default_factory=time.time)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def acquire(self, tokens: float = 1.0) -> bool:
        """
        Acquire tokens from the rate limiter.

        Args:
            tokens: Number of tokens to acquire

        Returns:
            bool: True if tokens were acquired, False otherwise
        """
        async with self.lock:
            # Refill tokens based on time passed
            current_time = time.time()
            time_passed = current_time - self.last_refill
            self.tokens = min(self.max_burst, self.tokens + time_passed * self.requests_per_second)
            self.last_refill = current_time

            # Check if we have enough tokens
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True

            return False

    async def wait(self, tokens: float = 1.0) -> None:
        """
        Wait until tokens are available.

        Args:
            tokens: Number of tokens to acquire
        """
        while True:
            async with self.lock:
                # Refill tokens based on time passed
                current_time = time.time()
                time_passed = current_time - self.last_refill
                self.tokens = min(self.max_burst, self.tokens + time_passed * self.requests_per_second)

                # Check if we have enough tokens
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    self.last_refill = current_time
                    return

                # Calculate wait time
                wait_time = (tokens - self.tokens) / self.requests_per_second

            # Wait outside the lock
            await asyncio.sleep(wait_time)


class ResourceManager:
    """
    Manages and optimizes resource usage.
    """

    def __init__(self):
        """Initialize the resource manager."""
        self._rate_limiters: Dict[str, RateLimiter] = {}
        self._executor = ParallelExecutor()
        self._cache = get_cache()

    def get_rate_limiter(self, name: str, requests_per_second: float = 1.0,
                         max_burst: int = 1) -> RateLimiter:
        """
        Get or create a rate limiter.

        Args:
            name: Name of the rate limiter
            requests_per_second: Maximum requests per second
            max_burst: Maximum token burst

        Returns:
            RateLimiter: Rate limiter instance
        """
        if name not in self._rate_limiters:
            self._rate_limiters[name] = RateLimiter(
                requests_per_second=requests_per_second,
                max_burst=max_burst
            )
        return self._rate_limiters[name]

    async def execute_parallel(self, func: Callable[[T], R], items: List[T]) -> List[R]:
        """
        Execute a function on a list of items in parallel.

        Args:
            func: Function to execute
            items: List of items to process

        Returns:
            List[R]: List of results
        """
        return await self._executor.map(func, items)

    async def throttle(self, name: str, func: Callable, *args, **kwargs) -> Any:
        """
        Execute a function with rate limiting.

        Args:
            name: Name of the rate limiter
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Any: Function result
        """
        # Get rate limiter
        rate_limiter = self.get_rate_limiter(name)

        # Wait for tokens
        await rate_limiter.wait()

        # Execute function
        if inspect.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            return func(*args, **kwargs)

    @asynccontextmanager
    async def batch_context(self, batch_size: int = 100):
        """
        Context manager for batching operations.

        Args:
            batch_size: Maximum batch size

        Yields:
            BatchContext: Batch context
        """
        batch = []

        class BatchContext:
            def __init__(self, batch, batch_size):
                self.batch = batch
                self.batch_size = batch_size

            async def add(self, item):
                self.batch.append(item)
                if len(self.batch) >= self.batch_size:
                    await self.flush()

            async def flush(self):
                if not self.batch:
                    return
                items = self.batch.copy()
                self.batch.clear()
                return items

        context = BatchContext(batch, batch_size)
        try:
            yield context
        finally:
            await context.flush()


# Global resource manager instance
_resource_manager: Optional[ResourceManager] = None


def get_resource_manager() -> ResourceManager:
    """
    Get the global resource manager instance.

    Returns:
        ResourceManager: Global resource manager instance
    """
    global _resource_manager
    if _resource_manager is None:
        _resource_manager = ResourceManager()
    return _resource_manager


# Memory profiling decorator
def memory_profile(func):
    """
    Decorator for memory profiling.

    Args:
        func: Function to profile

    Returns:
        Wrapped function
    """

    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            import psutil
            import os
        except ImportError:
            logger.warning("psutil not installed, memory profiling disabled")
            return await func(*args, **kwargs)

        # Get memory usage before
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB

        # Call function
        start_time = time.time()
        result = await func(*args, **kwargs)
        duration = time.time() - start_time

        # Get memory usage after
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_diff = memory_after - memory_before

        logger.info(f"Memory profile for {func.__qualname__}: "
                    f"before={memory_before:.2f}MB, after={memory_after:.2f}MB, "
                    f"diff={memory_diff:.2f}MB, duration={duration:.2f}s")

        return result

    return wrapper


# Performance timing decorator
def timed(func):
    """
    Decorator for timing function execution.

    Args:
        func: Function to time

    Returns:
        Wrapped function
    """

    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        duration = time.time() - start_time

        logger.debug(f"Function {func.__qualname__} took {duration:.4f} seconds")

        return result

    return wrapper


class PerformanceMonitor:
    """
    Monitor and track performance metrics for a specific operation.
    Helps analyze execution time, resource usage, and operation counts.
    """

    def __init__(self, name: str):
        """
        Initialize a performance monitor.

        Args:
            name: Name of the monitored operation
        """
        self.name = name
        self.start_time = None
        self.end_time = None
        self.is_running = False
        self.operation_counts = {}
        self.custom_metrics = {}
        self.checkpoints = []

    def start(self) -> None:
        """Start the performance monitoring."""
        if self.is_running:
            logger.warning(f"Performance monitor '{self.name}' already running")
            return

        self.start_time = time.time()
        self.is_running = True
        self.operation_counts = {}
        self.custom_metrics = {}
        self.checkpoints = []

        logger.debug(f"Performance monitor '{self.name}' started")

    def stop(self) -> None:
        """Stop the performance monitoring."""
        if not self.is_running:
            logger.warning(f"Performance monitor '{self.name}' not running")
            return

        self.end_time = time.time()
        self.is_running = False

        logger.debug(f"Performance monitor '{self.name}' stopped, duration={self.get_execution_time():.4f}s")

    def checkpoint(self, name: str) -> float:
        """
        Record a checkpoint with the current time.

        Args:
            name: Name of the checkpoint

        Returns:
            float: Time elapsed since start in seconds
        """
        if not self.is_running:
            logger.warning(f"Performance monitor '{self.name}' not running, checkpoint ignored")
            return 0.0

        current_time = time.time()
        elapsed = current_time - self.start_time

        self.checkpoints.append({
            "name": name,
            "timestamp": current_time,
            "elapsed": elapsed
        })

        return elapsed

    def increment_counter(self, name: str, count: int = 1) -> int:
        """
        Increment an operation counter.

        Args:
            name: Name of the counter
            count: Amount to increment by

        Returns:
            int: New counter value
        """
        if name not in self.operation_counts:
            self.operation_counts[name] = 0

        self.operation_counts[name] += count
        return self.operation_counts[name]

    def set_metric(self, name: str, value: Any) -> None:
        """
        Set a custom metric value.

        Args:
            name: Name of the metric
            value: Metric value
        """
        self.custom_metrics[name] = value

    def get_execution_time(self) -> float:
        """
        Get the total execution time in seconds.

        Returns:
            float: Execution time in seconds, or 0 if not stopped
        """
        if self.start_time is None:
            return 0.0

        end = self.end_time if self.end_time is not None else time.time()
        return end - self.start_time

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get all performance metrics.

        Returns:
            Dict[str, Any]: All performance metrics
        """
        execution_time = self.get_execution_time()

        metrics = {
            "name": self.name,
            "execution_time": execution_time,
            "is_running": self.is_running,
            "operation_counts": self.operation_counts,
            "custom_metrics": self.custom_metrics
        }

        if self.checkpoints:
            metrics["checkpoints"] = self.checkpoints

        # Calculate operations per second if we have counts
        for op_name, count in self.operation_counts.items():
            if execution_time > 0:
                metrics[f"{op_name}_per_second"] = count / execution_time

        return metrics
