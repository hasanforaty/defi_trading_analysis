"""Tests for the DexTools API client."""
import pytest
import json
from datetime import datetime, timedelta
import asyncio
from unittest.mock import patch, MagicMock

import aiohttp
from aiohttp.client_exceptions import ClientResponseError

from src.api.dextools import DexToolsApiClient
from src.api.exceptions import (
    AuthenticationError, RateLimitError, ResourceNotFoundError,
    BadRequestError, ServerError, NetworkError
)

# Test constants
API_KEY = "test_api_key"
CHAIN = "ether"
PAIR_ADDRESS = "0xa29fe6ef9592b5d408cca961d0fb9b1faf497d6d"


@pytest.fixture
def mock_response():
    """Create a mock response object."""

    class MockResponse:
        def __init__(self, data, status=200, headers=None):
            self.data = data
            self.status = status
            self.headers = headers or {}

        async def json(self):
            return self.data

        async def text(self):
            return json.dumps(self.data)

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

        def raise_for_status(self):
            if self.status >= 400:
                raise ClientResponseError(
                    request_info=MagicMock(),
                    history=None,
                    status=self.status,
                    message=f"Error: {self.status}",
                    headers=self.headers
                )

    return MockResponse


@pytest.fixture
async def api_client():
    """Create a DexTools API client for testing."""
    async with DexToolsApiClient(api_key=API_KEY) as client:
        yield client


@pytest.mark.asyncio
async def test_get_pair_info_success(api_client, mock_response):
    """Test successful retrieval of pair information."""
    # Mock data
    mock_data = {
        "exchangeName": "Uniswap",
        "exchangeFactory": "0x1234...",
        "creationTime": "2021-01-01T00:00:00.000Z",
        "creationBlock": 12345678,
        "mainToken": {
            "address": "0xabcd...",
            "symbol": "TOKEN",
            "name": "Token Name"
        },
        "sideToken": {
            "address": "0xdef0...",
            "symbol": "ETH",
            "name": "Ethereum"
        },
        "fee": 0.3
    }

    # Patch the session request method to return our mock response
    with patch.object(api_client.session, "request",
                      return_value=mock_response(mock_data)):
        result = await api_client.get_pair_info(CHAIN, PAIR_ADDRESS)

        # Assert the result matches our mock data
        assert result == mock_data


@pytest.mark.asyncio
async def test_get_transactions_success(api_client, mock_response):
    """Test successful retrieval of transactions."""
    # Mock data
    mock_data = {
        "totalPages": 10,
        "page": 0,
        "pageSize": 100,
        "results": [
            {
                "txHash": "0x1234...",
                "blockNumber": 12345678,
                "timestamp": "2021-01-01T00:00:00.000Z",
                "type": "buy",
                "token0Amount": 100,
                "token1Amount": 1,
                "walletAddress": "0xabcd...",
                "priceImpact": 0.1
            }
        ]
    }

    # Patch the session request method to return our mock response
    with patch.object(api_client.session, "request",
                      return_value=mock_response(mock_data)):
        result = await api_client.get_transactions(
            CHAIN, PAIR_ADDRESS, page=0, page_size=100
        )

        # Assert the result matches our mock data
        assert result == mock_data


@pytest.mark.asyncio
async def test_get_pair_liquidity_success(api_client, mock_response):
    """Test successful retrieval of pair liquidity."""
    # Mock data
    mock_data = {
        "liquidity": 1000000,
        "reserves": {
            "mainToken": 10000,
            "sideToken": 100
        }
    }

    # Patch the session request method to return our mock response
    with patch.object(api_client.session, "request",
                      return_value=mock_response(mock_data)):
        result = await api_client.get_pair_liquidity(CHAIN, PAIR_ADDRESS)

        # Assert the result matches our mock data
        assert result == mock_data


@pytest.mark.asyncio
async def test_authentication_error(api_client, mock_response):
    """Test authentication error handling."""
    # Patch the session request method to return a 403 response
    with patch.object(api_client.session, "request",
                      return_value=mock_response({}, status=403)):
        with pytest.raises(AuthenticationError):
            await api_client.get_pair_info(CHAIN, PAIR_ADDRESS)


@pytest.mark.asyncio
async def test_rate_limit_error(api_client, mock_response):
    """Test rate limit error handling."""
    # Patch the session request method to return a 429 response
    with patch.object(api_client.session, "request",
                      return_value=mock_response(
                          {}, status=429, headers={"Retry-After": "60"}
                      )):
        with pytest.raises(RateLimitError) as exc_info:
            await api_client.get_pair_info(CHAIN, PAIR_ADDRESS)

        # Assert the retry_after value was correctly extracted
        assert exc_info.value.retry_after == 60


@pytest.mark.asyncio
async def test_resource_not_found_error(api_client, mock_response):
    """Test resource not found error handling."""
    # Patch the session request method to return a 404 response
    with patch.object(api_client.session, "request",
                      return_value=mock_response({}, status=404)):
        with pytest.raises(ResourceNotFoundError):
            await api_client.get_pair_info(CHAIN, PAIR_ADDRESS)


@pytest.mark.asyncio
async def test_bad_request_error(api_client, mock_response):
    """Test bad request error handling."""
    # Patch the session request method to return a 400 response
    with patch.object(api_client.session, "request",
                      return_value=mock_response(
                          {"error": "Invalid parameter"}, status=400
                      )):
        with pytest.raises(BadRequestError):
            await api_client.get_pair_info(CHAIN, PAIR_ADDRESS)


@pytest.mark.asyncio
async def test_server_error(api_client, mock_response):
    """Test server error handling."""
    # Patch the session request method to return a 500 response
    with patch.object(api_client.session, "request",
                      return_value=mock_response({}, status=500)):
        with pytest.raises(ServerError):
            await api_client.get_pair_info(CHAIN, PAIR_ADDRESS)


@pytest.mark.asyncio
async def test_network_error(api_client):
    """Test network error handling."""
    # Patch the session request method to raise a ClientError
    with patch.object(api_client.session, "request",
                      side_effect=aiohttp.ClientError("Connection error")):
        with pytest.raises(NetworkError):
            await api_client.get_pair_info(CHAIN, PAIR_ADDRESS)


@pytest.mark.asyncio
async def test_retry_on_network_error(api_client):
    """Test retrying on network errors."""
    # Create a side effect that raises an error the first two times
    # but returns a valid response the third time
    mock_response_obj = MagicMock()
    mock_response_obj.status = 200
    mock_response_obj.json = MagicMock(return_value={"success": True})
    mock_response_obj.__aenter__ = MagicMock(return_value=mock_response_obj)
    mock_response_obj.__aexit__ = MagicMock(return_value=None)
    mock_response_obj.raise_for_status = MagicMock()

    side_effects = [
        aiohttp.ClientError("Connection error"),
        aiohttp.ClientError("Connection error"),
        mock_response_obj
    ]

    with patch.object(api_client.session, "request", side_effect=side_effects):
        # This should succeed after retries
        result = await api_client.get_pair_info(CHAIN, PAIR_ADDRESS)
        assert result == {"success": True}


@pytest.mark.asyncio
async def test_max_retries_exceeded(api_client):
    """Test max retries exceeded scenario."""
    # Patch the session request method to always raise a ClientError
    with patch.object(api_client.session, "request",
                      side_effect=aiohttp.ClientError("Connection error")):
        # This should fail after max_retries (default: 3)
        with pytest.raises(NetworkError):
            await api_client.get_pair_info(CHAIN, PAIR_ADDRESS)
