"""Custom exceptions for API-related errors."""
from typing import Optional


class APIError(Exception):
    """Base class for all API-related errors."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


class AuthenticationError(APIError):
    """Raised when API authentication fails."""

    def __init__(self, message: str = "Authentication failed, check your API key"):
        super().__init__(message, 403)


class RateLimitError(APIError):
    """Raised when API rate limit is exceeded."""

    def __init__(self, message: str = "Rate limit exceeded", retry_after: Optional[int] = None):
        self.retry_after = retry_after
        super().__init__(message, 429)


class ResourceNotFoundError(APIError):
    """Raised when the requested resource is not found."""

    def __init__(self, resource: str):
        super().__init__(f"Resource not found: {resource}", 404)


class BadRequestError(APIError):
    """Raised when the request is malformed or invalid."""

    def __init__(self, message: str = "Invalid request parameters"):
        super().__init__(message, 400)


class ServerError(APIError):
    """Raised when the server returns an error."""

    def __init__(self, message: str = "Server error", status_code: int = 500):
        super().__init__(message, status_code)


class NetworkError(APIError):
    """Raised when there's a network-related error."""

    def __init__(self, message: str = "Network error"):
        super().__init__(message)


class MaxRetryError(APIError):
    """Raised when maximum retry attempts are exceeded."""

    def __init__(self, message: str = "Maximum retry attempts exceeded"):
        super().__init__(message)
