import sys
import traceback
import asyncio
from enum import Enum, auto
from typing import Dict, Any, Optional, Callable, Awaitable, List, Tuple, Type, Union
from functools import wraps
import inspect
from dataclasses import dataclass
from contextlib import asynccontextmanager
from loguru import logger

from src.core.events import EventBus, EventType, Event


class ErrorSeverity(Enum):
    """Enum representing the severity of an error."""
    DEBUG = auto()
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()


class AppError(Exception):
    """Base class for application errors."""

    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.ERROR,
                 details: Optional[Dict[str, Any]] = None):
        """
        Initialize the error.

        Args:
            message: Error message
            severity: Error severity
            details: Additional error details
        """
        super().__init__(message)
        self.message = message
        self.severity = severity
        self.details = details or {}
        self.traceback = traceback.format_exc()

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the error to a dictionary.

        Returns:
            Dict[str, Any]: Error as dictionary
        """
        return {
            "message": self.message,
            "severity": self.severity.name,
            "details": self.details,
            "type": self.__class__.__name__,
            "traceback": self.traceback
        }


class ValidationError(AppError):
    """Error raised when validation fails."""

    def __init__(self, message: str, errors: Dict[str, str],
                 severity: ErrorSeverity = ErrorSeverity.WARNING):
        """
        Initialize the validation error.

        Args:
            message: Error message
            errors: Dictionary of field-specific errors
            severity: Error severity
        """
        super().__init__(message, severity, {"errors": errors})
        self.errors = errors


class DatabaseError(AppError):
    """Error raised when a database operation fails."""

    def __init__(self, message: str, query: Optional[str] = None,
                 severity: ErrorSeverity = ErrorSeverity.ERROR):
        """
        Initialize the database error.

        Args:
            message: Error message
            query: Optional SQL query that failed
            severity: Error severity
        """
        details = {}
        if query:
            details["query"] = query
        super().__init__(message, severity, details)


class ApiError(AppError):
    """Error raised when an API call fails."""

    def __init__(self, message: str, status_code: int = 500,
                 response: Optional[Dict[str, Any]] = None,
                 severity: ErrorSeverity = ErrorSeverity.ERROR):
        """
        Initialize the API error.

        Args:
            message: Error message
            status_code: HTTP status code
            response: API response details
            severity: Error severity
        """
        details = {
            "status_code": status_code
        }
        if response:
            details["response"] = response
        super().__init__(message, severity, details)
        self.status_code = status_code
        self.response = response


class WorkflowError(AppError):
    """Error raised when a workflow operation fails."""

    def __init__(self, message: str, workflow_id: str, step: Optional[str] = None,
                 severity: ErrorSeverity = ErrorSeverity.ERROR):
        """
        Initialize the workflow error.

        Args:
            message: Error message
            workflow_id: ID of the workflow where the error occurred
            step: Optional workflow step name
            severity: Error severity
        """
        details = {
            "workflow_id": workflow_id
        }
        if step:
            details["step"] = step
        super().__init__(message, severity, details)
        self.workflow_id = workflow_id
        self.step = step


class ErrorHandler:
    """
    Centralized error handling system for the application.
    Captures, logs, and manages errors across different modules.
    """

    def __init__(self, event_bus: Optional[EventBus] = None):
        """
        Initialize the error handler.

        Args:
            event_bus: Optional event bus for error events
        """
        self._event_bus = event_bus
        self._error_callbacks: Dict[Type[Exception], List[Callable[[Exception], Awaitable[None]]]] = {}
        self._global_callbacks: List[Callable[[Exception], Awaitable[None]]] = []
        logger.info("ErrorHandler initialized")

    async def handle_error(self, error: Exception,
                           context: Optional[Dict[str, Any]] = None) -> None:
        """
        Handle an error.

        Args:
            error: Exception that occurred
            context: Optional context information
        """
        # Convert to AppError if it's not one already
        if not isinstance(error, AppError):
            app_error = self._convert_to_app_error(error)
        else:
            app_error = error

        # Log the error
        self._log_error(app_error, context)

        # Publish event if event bus is available
        if self._event_bus:
            error_type = getattr(EventType, f"SYSTEM_{app_error.severity.name}")
            await self._event_bus.publish(error_type, {
                "error": app_error.to_dict(),
                "context": context or {}
            })

        # Call error-specific callbacks
        error_type = type(error)
        callbacks = []

        # Get callbacks for this error type
        for error_class, class_callbacks in self._error_callbacks.items():
            if isinstance(error, error_class):
                callbacks.extend(class_callbacks)

        # Call all matching callbacks
        for callback in callbacks:
            try:
                await callback(error)
            except Exception as e:
                logger.error(f"Error in error callback: {str(e)}")

        # Call global callbacks
        for callback in self._global_callbacks:
            try:
                await callback(error)
            except Exception as e:
                logger.error(f"Error in global error callback: {str(e)}")

    def _convert_to_app_error(self, error: Exception) -> AppError:
        """
        Convert a standard exception to AppError.

        Args:
            error: Exception to convert

        Returns:
            AppError: Converted error
        """
        # Determine severity based on error type
        severity = ErrorSeverity.ERROR

        # Convert SQLAlchemy errors
        if "sqlalchemy" in error.__class__.__module__:
            return DatabaseError(str(error), severity=severity)

        # Convert validation errors
        if "validation" in error.__class__.__name__.lower():
            return ValidationError(str(error), {}, severity=ErrorSeverity.WARNING)

        # For other errors, create a generic AppError
        return AppError(str(error), severity=severity)

    def _log_error(self, error: AppError, context: Optional[Dict[str, Any]] = None) -> None:
        """
        Log an error with appropriate severity.

        Args:
            error: Error to log
            context: Optional context information
        """
        log_message = f"{error.__class__.__name__}: {error.message}"
        if context:
            log_message += f" | Context: {context}"

        # Log with appropriate level
        if error.severity == ErrorSeverity.DEBUG:
            logger.debug(log_message)
        elif error.severity == ErrorSeverity.INFO:
            logger.info(log_message)
        elif error.severity == ErrorSeverity.WARNING:
            logger.warning(log_message)
        elif error.severity == ErrorSeverity.ERROR:
            logger.error(log_message)
        elif error.severity == ErrorSeverity.CRITICAL:
            logger.critical(log_message)

    def register_error_callback(self, error_type: Type[Exception],
                                callback: Callable[[Exception], Awaitable[None]]) -> None:
        """
        Register a callback for a specific error type.

        Args:
            error_type: Type of error to handle
            callback: Async callback function
        """
        if error_type not in self._error_callbacks:
            self._error_callbacks[error_type] = []

        self._error_callbacks[error_type].append(callback)
        logger.debug(f"Registered callback for {error_type.__name__}")

    def register_global_callback(self, callback: Callable[[Exception], Awaitable[None]]) -> None:
        """
        Register a global error callback.

        Args:
            callback: Async callback function
        """
        self._global_callbacks.append(callback)
        logger.debug("Registered global error callback")

    def unregister_error_callback(self, error_type: Type[Exception],
                                  callback: Callable[[Exception], Awaitable[None]]) -> bool:
        """
        Unregister a callback for a specific error type.

        Args:
            error_type: Type of error
            callback: Callback function to remove

        Returns:
            bool: True if unregistered successfully, False otherwise
        """
        if error_type in self._error_callbacks and callback in self._error_callbacks[error_type]:
            self._error_callbacks[error_type].remove(callback)
            logger.debug(f"Unregistered callback for {error_type.__name__}")
            return True

        return False

    def unregister_global_callback(self, callback: Callable[[Exception], Awaitable[None]]) -> bool:
        """
        Unregister a global error callback.

        Args:
            callback: Callback function to remove

        Returns:
            bool: True if unregistered successfully, False otherwise
        """
        if callback in self._global_callbacks:
            self._global_callbacks.remove(callback)
            logger.debug("Unregistered global error callback")
            return True

        return False


# Global error handler instance
_error_handler: Optional[ErrorHandler] = None


def get_error_handler(event_bus: Optional[EventBus] = None) -> ErrorHandler:
    """
    Get the global error handler instance.

    Args:
        event_bus: Optional event bus for error events

    Returns:
        ErrorHandler: Global error handler instance
    """
    global _error_handler
    if _error_handler is None:
        _error_handler = ErrorHandler(event_bus)
    return _error_handler


# Error handling decorator
def handle_errors(error_handler: Optional[ErrorHandler] = None):
    """
    Decorator for handling errors in functions.

    Args:
        error_handler: Optional error handler to use

    Returns:
        Decorator function
    """

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            handler = error_handler or get_error_handler()

            try:
                return await func(*args, **kwargs)
            except Exception as e:
                # Add context from function
                context = {
                    "function": func.__qualname__,
                    "module": func.__module__,
                    "args": str(args),
                    "kwargs": str(kwargs)
                }

                # Handle the error
                await handler.handle_error(e, context)

                # Re-raise the error
                raise

        return wrapper

    return decorator


# Context manager for error handling
@asynccontextmanager
async def error_context(context: Dict[str, Any], error_handler: Optional[ErrorHandler] = None):
    """
    Context manager for handling errors with context.

    Args:
        context: Context information for errors
        error_handler: Optional error handler to use

    Yields:
        None
    """
    handler = error_handler or get_error_handler()

    try:
        yield
    except Exception as e:
        # Handle the error with context
        await handler.handle_error(e, context)

        # Re-raise the error
        raise

