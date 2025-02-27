"""DexTools API client implementation."""
import asyncio
import logging
import json
from typing import Dict, List, Optional, Union, Any
import time
from datetime import datetime, timezone
import backoff

import aiohttp
from aiohttp.client_exceptions import ClientError, ClientResponseError

from config.settings import Settings
from src.api.exceptions import (
    APIError, AuthenticationError, RateLimitError,
    ResourceNotFoundError, BadRequestError, ServerError,
    NetworkError, MaxRetryError
)
from src.api.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)


class DexToolsApiClient:
    """
    Client for interacting with the DexTools API.

    This client provides methods to access DexTools API endpoints
    for obtaining pair information, transactions, and liquidity data.
    """

    BASE_URL = "https://public-api.dextools.io/standard"

    def __init__(
            self,
            api_key: Optional[str] = None,
            requests_per_second: int = 5,
            requests_per_minute: int = 100,
            max_retries: int = 3,
            session: Optional[aiohttp.ClientSession] = None
    ):
        """
        Initialize the DexTools API client.

        Args:
            api_key: DexTools API key, defaults to settings.DEXTOOLS_API_KEY
            requests_per_second: Maximum requests per second
            requests_per_minute: Maximum requests per minute
            max_retries: Maximum number of retries for failed requests
            session: Aiohttp session to use, will create one if not provided
        """
        self.api_key = api_key or Settings.dextools_api
        self.session = session
        self.max_retries = max_retries
        self.rate_limiter = RateLimiter(
            requests_per_second=requests_per_second,
            requests_per_minute=requests_per_minute
        )

    async def __aenter__(self):
        """Enter async context."""
        if self.session is None:
            self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context."""
        if self.session is not None:
            await self.session.close()
            self.session = None

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for API requests."""
        return {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

    @backoff.on_exception(
        backoff.expo,
        (NetworkError, ServerError),
        max_tries=3,
        giveup=lambda e: isinstance(e, (AuthenticationError, ResourceNotFoundError, BadRequestError))
    )
    async def _make_request(
            self,
            method: str,
            endpoint: str,
            params: Optional[Dict[str, Any]] = None,
            data: Optional[Dict[str, Any]] = None,
            timeout: int = 30
    ) -> Dict[str, Any]:
        """
        Make a request to the DexTools API.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            params: Query parameters
            data: Request body data
            timeout: Request timeout in seconds

        Returns:
            API response as a dictionary

        Raises:
            AuthenticationError: If the API key is invalid
            RateLimitError: If the rate limit is exceeded
            ResourceNotFoundError: If the requested resource is not found
            BadRequestError: If the request is invalid
            ServerError: If the server returns an error
            NetworkError: If there's a network error
            MaxRetryError: If the maximum retry count is exceeded
        """
        if self.session is None:
            self.session = aiohttp.ClientSession()

        # Wait for rate limit slot
        if not await self.rate_limiter.wait_until_available(timeout=10):
            logger.warning("Rate limit slot not available after waiting")
            raise RateLimitError("Rate limit exceeded, couldn't acquire slot")

        url = f"{self.BASE_URL}{endpoint}"
        headers = self._get_headers()

        try:
            logger.debug(f"Making {method} request to {url}")
            async with self.session.request(
                    method=method,
                    url=url,
                    params=params,
                    json=data,
                    headers=headers,
                    timeout=timeout
            ) as response:
                # Check for error responses
                if response.status == 403:
                    logger.error("Authentication failed")
                    raise AuthenticationError()

                elif response.status == 429:
                    retry_after = int(response.headers.get("Retry-After", 60))
                    logger.warning(f"Rate limit exceeded, retry after {retry_after}s")
                    raise RateLimitError(retry_after=retry_after)

                elif response.status == 404:
                    logger.error(f"Resource not found: {url}")
                    raise ResourceNotFoundError(url)

                elif response.status == 400:
                    error_text = await response.text()
                    logger.error(f"Bad request: {error_text}")
                    raise BadRequestError(f"Invalid request: {error_text}")

                elif response.status >= 500:
                    error_text = await response.text()
                    logger.error(f"Server error ({response.status}): {error_text}")
                    raise ServerError(f"Server error: {error_text}", response.status)

                # Ensure successful response
                response.raise_for_status()

                # Parse JSON response
                return await response.json()

        except aiohttp.ClientResponseError as e:
            logger.error(f"Response error: {e}")
            raise
        except aiohttp.ClientError as e:
            logger.error(f"Request error: {e}")
            raise NetworkError(f"Network error: {str(e)}")
        except asyncio.TimeoutError:
            logger.error("Request timed out")
            raise NetworkError("Request timed out")

    async def get_pair_info(self, chain: str, pair_address: str) -> Dict[str, Any]:
        """
        Get information about a trading pair.

        Args:
            chain: Blockchain identifier (e.g., 'ether' for Ethereum)
            pair_address: Trading pair contract address

        Returns:
            Dictionary containing pair information
        """
        endpoint = f"/v2/pool/{chain}/{pair_address}"
        return await self._make_request("GET", endpoint)

    async def get_transactions(
            self,
            chain: str,
            pair_address: str,
            from_timestamp: Optional[Union[int, datetime]] = None,
            to_timestamp: Optional[Union[int, datetime]] = None,
            page: int = 0,
            page_size: int = 100
    ) -> Dict[str, Any]:
        """
        Get transactions for a trading pair.

        Args:
            chain: Blockchain identifier
            pair_address: Trading pair contract address
            from_timestamp: Start timestamp (Unix timestamp or datetime)
            to_timestamp: End timestamp (Unix timestamp or datetime)
            page: Page number (starts at 0)
            page_size: Number of results per page

        Returns:
            Dictionary containing transaction data
        """
        endpoint = f"/v2/pool/{chain}/{pair_address}/transactions"
        params = {
            "page": page,
            "pageSize": page_size
        }

        # Add timestamp filters if provided
        if from_timestamp:
            if isinstance(from_timestamp, datetime):
                # Convert to ISO string
                params["from"] = from_timestamp.isoformat()
            else:
                params["from"] = from_timestamp

        if to_timestamp:
            if isinstance(to_timestamp, datetime):
                # Convert to ISO string
                params["to"] = to_timestamp.isoformat()
            else:
                params["to"] = to_timestamp

        return await self._make_request("GET", endpoint, params=params)

    async def get_pair_liquidity(self, chain: str, pair_address: str) -> Dict[str, Any]:
        """
        Get liquidity information for a trading pair.

        Args:
            chain: Blockchain identifier
            pair_address: Trading pair contract address

        Returns:
            Dictionary containing liquidity information
        """
        endpoint = f"/v2/pool/{chain}/{pair_address}/liquidity"
        return await self._make_request("GET", endpoint)

    async def get_pair_price(self, chain: str, pair_address: str) -> Dict[str, Any]:
        """
        Get price information for a trading pair.

        Args:
            chain: Blockchain identifier
            pair_address: Trading pair contract address

        Returns:
            Dictionary containing price information
        """
        endpoint = f"/v2/pool/{chain}/{pair_address}/price"
        return await self._make_request("GET", endpoint)
