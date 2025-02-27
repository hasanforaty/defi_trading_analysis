# src/api/dextools.py
import asyncio
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Tuple, cast
import aiohttp
from loguru import logger

from config.settings import settings
from src.api.exceptions import (
    AuthenticationError,
    DexToolsApiError,
    InvalidRequestError,
    NetworkError,
    RateLimitExceededError,
    ResourceNotFoundError,
    UnexpectedResponseError,
)
from src.api.rate_limiter import RateLimiter
from src.api.cache import CacheManager
from src.models.pair import Pair
from src.models.token import Token
from src.models.transaction import Transaction


class DexToolsApiClient:
    """
    Client for interacting with the DexTools API.
    Handles authentication, request formation, and response parsing.
    """

    # Base URL for the DexTools API
    BASE_URL = "https://public-api.dextools.io/standard"

    def __init__(
            self,
            api_key: Optional[str] = None,
            cache_manager: Optional[CacheManager] = None,
            rate_limiter: Optional[RateLimiter] = None,
            max_retries: int = 3,
            retry_delay: float = 1.0,
            timeout: float = 30.0,
    ):
        """
        Initialize the DexTools API client.

        Args:
            api_key: The API key for DexTools API.
            cache_manager: Instance of CacheManager for caching responses.
            rate_limiter: Instance of RateLimiter for rate limiting requests.
            max_retries: Maximum number of retries for failed requests.
            retry_delay: Initial delay between retries in seconds.
            timeout: Timeout for API requests in seconds.
        """
        self.api_key = api_key or settings.dextools_api_key
        if not self.api_key:
            raise ValueError("DexTools API key is required")

        self.cache_manager = cache_manager or CacheManager()
        self.rate_limiter = rate_limiter or RateLimiter()
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        self.session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create an aiohttp session."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout),
                headers=self._get_headers(),
            )
        return self.session

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for API requests."""
        return {
            "X-API-KEY": self.api_key,
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    async def _make_request(
            self,
            method: str,
            endpoint: str,
            params: Optional[Dict[str, Any]] = None,
            data: Optional[Dict[str, Any]] = None,
            retry_count: int = 0,
    ) -> Dict[str, Any]:
        """
        Make an HTTP request to the DexTools API.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            params: Query parameters
            data: Request body
            retry_count: Current retry attempt

        Returns:
            Parsed JSON response

        Raises:
            Various DexToolsApiError subclasses based on the error type
        """
        url = f"{self.BASE_URL}{endpoint}"
        cache_key = f"dextools:{method}:{endpoint}:{json.dumps(params or {})}:{json.dumps(data or {})}"

        # Try to get from cache first
        cached_response = await self.cache_manager.get(cache_key)
        if cached_response is not None:
            logger.debug(f"Cache hit for {url}")
            return cached_response

        # Apply rate limiting before making the request
        await self.rate_limiter.acquire(endpoint)

        try:
            session = await self._get_session()

            logger.debug(f"Making {method} request to {url} with params={params}")

            async with session.request(
                    method=method,
                    url=url,
                    params=params,
                    json=data
            ) as response:
                # Handle different response status codes
                if response.status == 200:
                    resp_data = await response.json()

                    # Cache the successful response
                    ttl = self._determine_cache_ttl(endpoint)
                    await self.cache_manager.set(cache_key, resp_data, ttl)

                    return resp_data

                elif response.status == 400:
                    raise InvalidRequestError(
                        f"Invalid request: {await response.text()}",
                        status_code=response.status
                    )

                elif response.status == 401 or response.status == 403:
                    raise AuthenticationError(
                        f"Authentication error: {await response.text()}",
                        status_code=response.status
                    )

                elif response.status == 404:
                    raise ResourceNotFoundError(
                        f"Resource not found: {await response.text()}",
                        status_code=response.status
                    )

                elif response.status == 429:
                    raise RateLimitExceededError(
                        f"Rate limit exceeded: {await response.text()}",
                        status_code=response.status
                    )

                else:
                    raise DexToolsApiError(
                        f"API error ({response.status}): {await response.text()}",
                        status_code=response.status
                    )

        except (RateLimitExceededError, DexToolsApiError) as e:
            # For rate limit errors or server errors, retry with backoff
            if retry_count < self.max_retries:
                backoff_time = self.retry_delay * (2 ** retry_count)
                logger.warning(
                    f"Request failed with {e.__class__.__name__}, retrying in {backoff_time:.2f}s "
                    f"(attempt {retry_count + 1}/{self.max_retries})"
                )
                await asyncio.sleep(backoff_time)
                return await self._make_request(method, endpoint, params, data, retry_count + 1)
            raise

        except aiohttp.ClientError as e:
            logger.error(f"Network error: {e}")
            if retry_count < self.max_retries:
                backoff_time = self.retry_delay * (2 ** retry_count)
                logger.warning(
                    f"Network error occurred, retrying in {backoff_time:.2f}s "
                    f"(attempt {retry_count + 1}/{self.max_retries})"
                )
                await asyncio.sleep(backoff_time)
                return await self._make_request(method, endpoint, params, data, retry_count + 1)
            raise NetworkError(f"Network error after {self.max_retries} retries: {e}")

        except Exception as e:
            logger.error(f"Unexpected error in API request: {e}")
            raise DexToolsApiError(f"Unexpected error: {e}")

    def _determine_cache_ttl(self, endpoint: str) -> int:
        """
        Determine the appropriate cache TTL based on the endpoint.

        Args:
            endpoint: The API endpoint

        Returns:
            Cache TTL in seconds
        """
        # Different endpoints may have different cache TTLs
        if "/price" in endpoint:
            # Price data changes frequently
            return 60  # 1 minute
        elif "/transactions" in endpoint:
            # Transaction data should be relatively fresh
            return 300  # 5 minutes
        elif "/liquidity" in endpoint:
            # Liquidity data changes less frequently
            return 600  # 10 minutes
        elif "/pair" in endpoint or "/token" in endpoint:
            # Static data can be cached longer
            return 3600  # 1 hour
        else:
            # Default cache TTL
            return 1800  # 30 minutes

    async def get_pair_info(self, chain: str, pair_address: str) -> Dict[str, Any]:
        """
        Get information about a specific trading pair.

        Args:
            chain: The blockchain ID (e.g., "ether" for Ethereum)
            pair_address: The address of the trading pair

        Returns:
            Pair information
        """
        endpoint = f"/v2/pool/{chain}/{pair_address}"
        response = await self._make_request("GET", endpoint)

        logger.info(f"Retrieved pair info for {pair_address} on {chain}")
        return response

    async def get_transactions(
            self,
            chain: str,
            pair_address: str,
            from_timestamp: Optional[int] = None,
            to_timestamp: Optional[int] = None,
            page: int = 0,
            page_size: int = 100
    ) -> Dict[str, Any]:
        """
        Get transaction history for a specific trading pair.

        Args:
            chain: The blockchain ID (e.g., "ether" for Ethereum)
            pair_address: The address of the trading pair
            from_timestamp: Start timestamp (Unix time in seconds)
            to_timestamp: End timestamp (Unix time in seconds)
            page: Page number (0-indexed)
            page_size: Number of results per page

        Returns:
            Transaction data
        """
        endpoint = f"/v2/pool/{chain}/{pair_address}/swaps"

        params: Dict[str, Any] = {
            "page": page,
            "pageSize": page_size,
        }

        if from_timestamp is not None:
            params["from"] = from_timestamp

        if to_timestamp is not None:
            params["to"] = to_timestamp

        response = await self._make_request("GET", endpoint, params=params)

        logger.info(f"Retrieved transactions for {pair_address} on {chain} (page {page})")
        return response

    async def get_pair_liquidity(self, chain: str, pair_address: str) -> Dict[str, Any]:
        """
        Get liquidity information for a specific trading pair.

        Args:
            chain: The blockchain ID (e.g., "ether" for Ethereum)
            pair_address: The address of the trading pair

        Returns:
            Liquidity information
        """
        endpoint = f"/v2/pool/{chain}/{pair_address}/liquidity"
        response = await self._make_request("GET", endpoint)

        logger.info(f"Retrieved liquidity info for {pair_address} on {chain}")
        return response

    async def get_pair_price(self, chain: str, pair_address: str) -> Dict[str, Any]:
        """
        Get price information for a specific trading pair.

        Args:
            chain: The blockchain ID (e.g., "ether" for Ethereum)
            pair_address: The address of the trading pair

        Returns:
            Price information
        """
        endpoint = f"/v2/pool/{chain}/{pair_address}/price"
        response = await self._make_request("GET", endpoint)

        logger.info(f"Retrieved price info for {pair_address} on {chain}")
        return response

    async def get_token_info(self, chain: str, token_address: str) -> Dict[str, Any]:
        """
        Get information about a specific token.

        Args:
            chain: The blockchain ID (e.g., "ether" for Ethereum)
            token_address: The address of the token

        Returns:
            Token information
        """
        endpoint = f"/v2/token/{chain}/{token_address}"
        response = await self._make_request("GET", endpoint)

        logger.info(f"Retrieved token info for {token_address} on {chain}")
        return response

    async def close(self) -> None:
        """Close the client session."""
        if self.session and not self.session.closed:
            await self.session.close()
            logger.info("DexTools API client session closed")
