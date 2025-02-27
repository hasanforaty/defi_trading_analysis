# src/api/exceptions.py
from typing import Optional, Dict, Any


class DexToolsApiError(Exception):
    """Base exception for DexTools API errors."""

    def __init__(self, message: str, status_code: Optional[int] = None, response_data: Optional[Dict[str, Any]] = None):
        self.message = message
        self.status_code = status_code
        self.response_data = response_data
        super().__init__(self.message)


class RateLimitExceededError(DexToolsApiError):
    """Exception raised when the API rate limit is exceeded."""
    pass


class AuthenticationError(DexToolsApiError):
    """Exception raised when there's an authentication error with the API."""
    pass


class NetworkError(DexToolsApiError):
    """Exception raised when there's a network error."""
    pass


class InvalidRequestError(DexToolsApiError):
    """Exception raised when the request to the API is invalid."""
    pass


class ResourceNotFoundError(DexToolsApiError):
    """Exception raised when the requested resource is not found."""
    pass


class UnexpectedResponseError(DexToolsApiError):
    """Exception raised when the API response doesn't match the expected format."""
    pass
