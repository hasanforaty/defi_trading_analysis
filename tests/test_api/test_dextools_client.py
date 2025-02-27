# tests/test_api/test_dextools_client.py
import pytest
import asyncio
from unittest.mock import patch, MagicMock
import aiohttp
from aiohttp import ClientSession, ClientResponse
from aiohttp.web_exceptions import HTTPException

from src.api.dextools import DexToolsApiClient
from src.api.exceptions import (
    AuthenticationError,
    DexToolsApiError,
    InvalidRequestError,
    NetworkError,
    RateLimitExceededError,
    ResourceNotFoundError,
)


class MockResponse:
    def __init__(self, status, json_data):
        self.status = status
        self._json_data = json_data

    async def json(self):
        return self._json_data

    async def text(self):
        return str(self._json_data)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


@pytest.fixture
def mock_session():
    """Create a mock aiohttp session for testing."""
    session = MagicMock(spec=ClientSession)
    return session


@pytest.fixture
async def dextools_client(mock_session):
    """Create a DexTools API client with a mocked session for testing."""
    client = DexToolsApiClient(api_key="test_api_key")
    client.session = mock_session
    yield client
    await client.close()


@pytest.mark.asyncio
async def test_get_pair_info_success(dextools_client, mock_session):
    # Mock successful response
    mock_response = MockResponse(200, {"mainToken": {"symbol": "TEST"}, "exchangeName": "TestDEX"})
    mock_session.request.return_value = mock_response

    # Test the API call
    result = await dextools_client.get_pair_info("ether", "0x1234")

    # Verify response
    assert result["mainToken"]["symbol"] == "TEST"
    assert result["exchangeName"] == "TestDEX"
    # tests/test_api/test_dextools_client.py (continued)

    # Verify correct request
    mock_session.request.assert_called_once_with(
        method="GET",
        url="https://public-api.dextools.io/standard/v2/pool/ether/0x1234",
        headers={"X-API-KEY": "test_api_key"},
        params=None,
        timeout=30
    )


@pytest.mark.asyncio
async def test_get_transactions_success(dextools_client, mock_session):
    # Mock successful response
    mock_data = {
        "results": [
            {"txHash": "0xabc123", "type": "buy", "valueUsd": 1000},
            {"txHash": "0xdef456", "type": "sell", "valueUsd": 500}
        ],
        "totalPages": 5,
        "page": 0,
        "pageSize": 10
    }
    mock_response = MockResponse(200, mock_data)
    mock_session.request.return_value = mock_response

    # Test the API call
    result = await dextools_client.get_transactions("ether", "0x1234", page=0, page_size=10)

    # Verify response
    assert len(result["results"]) == 2
    assert result["results"][0]["txHash"] == "0xabc123"
    assert result["totalPages"] == 5

    # Verify correct request
    mock_session.request.assert_called_once_with(
        method="GET",
        url="https://public-api.dextools.io/standard/v2/pool/ether/0x1234/transactions",
        headers={"X-API-KEY": "test_api_key"},
        params={"page": 0, "pageSize": 10},
        timeout=30
    )


@pytest.mark.asyncio
async def test_get_pair_liquidity_success(dextools_client, mock_session):
    # Mock successful response
    mock_data = {
        "liquidity": 500000,
        "reserves": {
            "mainToken": 1000.5,
            "sideToken": 250.75
        }
    }
    mock_response = MockResponse(200, mock_data)
    mock_session.request.return_value = mock_response

    # Test the API call
    result = await dextools_client.get_pair_liquidity("ether", "0x1234")

    # Verify response
    assert result["liquidity"] == 500000
    assert result["reserves"]["mainToken"] == 1000.5

    # Verify correct request
    mock_session.request.assert_called_once_with(
        method="GET",
        url="https://public-api.dextools.io/standard/v2/pool/ether/0x1234/liquidity",
        headers={"X-API-KEY": "test_api_key"},
        params=None,
        timeout=30
    )


@pytest.mark.asyncio
async def test_get_pair_price_success(dextools_client, mock_session):
    # Mock successful response
    mock_data = {
        "priceUsd": 1.25,
        "priceNative": 0.0005,
        "volume24h": 1500000,
        "txCount24h": 350,
        "priceChange": {
            "m5": 0.02,
            "h1": 0.05,
            "h6": -0.01,
            "h24": 0.08
        }
    }
    mock_response = MockResponse(200, mock_data)
    mock_session.request.return_value = mock_response

    # Test the API call
    result = await dextools_client.get_pair_price("ether", "0x1234")

    # Verify response
    assert result["priceUsd"] == 1.25
    assert result["volume24h"] == 1500000
    assert result["priceChange"]["h24"] == 0.08

    # Verify correct request
    mock_session.request.assert_called_once_with(
        method="GET",
        url="https://public-api.dextools.io/standard/v2/pool/ether/0x1234/price",
        headers={"X-API-KEY": "test_api_key"},
        params=None,
        timeout=30
    )


@pytest.mark.asyncio
async def test_error_handling_authentication(dextools_client, mock_session):
    # Mock an authentication error
    mock_response = MockResponse(403, {"error": "Invalid API key"})
    mock_session.request.return_value = mock_response

    # Test for correct exception
    with pytest.raises(AuthenticationError):
        await dextools_client.get_pair_info("ether", "0x1234")


@pytest.mark.asyncio
async def test_error_handling_rate_limit(dextools_client, mock_session):
    # Mock a rate limit error
    mock_response = MockResponse(429, {"error": "Rate limit exceeded"})
    mock_session.request.return_value = mock_response

    # Test for correct exception
    with pytest.raises(RateLimitExceededError):
        await dextools_client.get_pair_info("ether", "0x1234")


@pytest.mark.asyncio
async def test_error_handling_not_found(dextools_client, mock_session):
    # Mock a not found error
    mock_response = MockResponse(404, {"error": "Resource not found"})
    mock_session.request.return_value = mock_response

    # Test for correct exception
    with pytest.raises(ResourceNotFoundError):
        await dextools_client.get_pair_info("ether", "0x1234")


@pytest.mark.asyncio
async def test_error_handling_bad_request(dextools_client, mock_session):
    # Mock a bad request error
    mock_response = MockResponse(400, {"error": "Bad request"})
    mock_session.request.return_value = mock_response

    # Test for correct exception
    with pytest.raises(InvalidRequestError):
        await dextools_client.get_pair_info("ether", "0x1234")


@pytest.mark.asyncio
async def test_error_handling_server_error(dextools_client, mock_session):
    # Mock a server error
    mock_response = MockResponse(500, {"error": "Internal server error"})
    mock_session.request.return_value = mock_response

    # Test for correct exception
    with pytest.raises(DexToolsApiError):
        await dextools_client.get_pair_info("ether", "0x1234")


@pytest.mark.asyncio
async def test_retry_on_network_error(dextools_client, mock_session):
    # Setup mock to raise an exception first, then succeed
    network_error = aiohttp.ClientConnectionError("Connection error")
    success_response = MockResponse(200, {"mainToken": {"symbol": "TEST"}})

    # On first call, raise an error, on second return success
    mock_session.request.side_effect = [network_error, success_response]

    # Test the API call - should succeed after retry
    result = await dextools_client.get_pair_info("ether", "0x1234")

    # Verify response
    assert result["mainToken"]["symbol"] == "TEST"

    # Verify correct request was made twice
    assert mock_session.request.call_count == 2


@pytest.mark.asyncio
async def test_max_retries_exceeded(dextools_client, mock_session):
    # Setup mock to always raise a network error
    network_error = aiohttp.ClientConnectionError("Connection error")
    mock_session.request.side_effect = network_error

    # Configure client with fewer retries for faster test
    dextools_client.max_retries = 2

    # Test for correct exception after retries
    with pytest.raises(NetworkError):
        await dextools_client.get_pair_info("ether", "0x1234")

    # Verify request was called correct number of times (initial + retries)
    assert mock_session.request.call_count == 3
