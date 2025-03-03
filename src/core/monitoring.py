# src/core/monitoring.py
import asyncio
import time
import json
import os
import sys
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Any, Optional, List, Set, Callable, Awaitable, Union
from dataclasses import dataclass, field
import logging
import platform
import psutil
from loguru import logger
import traceback
import socket
import uuid

from src.core.events import EventBus, EventType, Event
from src.core.error_handling import ErrorSeverity, AppError


class LogLevel(str, Enum):
    """Log levels."""
    TRACE = "TRACE"
    DEBUG = "DEBUG"
    INFO = "INFO"
    SUCCESS = "SUCCESS"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class MetricType(str, Enum):
    """Types of metrics that can be collected."""
    COUNTER = "counter"  # Simple incrementing counter
    GAUGE = "gauge"  # Value that can go up and down
    HISTOGRAM = "histogram"  # Distribution of values
    SUMMARY = "summary"  # Similar to histogram but with additional calculations


@dataclass
class Metric:
    """Represents a monitoring metric."""
    name: str
    type: MetricType
    description: str
    value: Any = 0
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metric to dictionary."""
        return {
            "name": self.name,
            "type": self.type.value,
            "description": self.description,
            "value": self.value,
            "labels": self.labels,
            "timestamp": self.timestamp,
            "datetime": datetime.fromtimestamp(self.timestamp).isoformat()
        }


class LogFormatter:
    """Formats log messages for different outputs."""

    @staticmethod
    def format_console(record: Dict[str, Any]) -> str:
        """Format log record for console output."""
        time_format = "%Y-%m-%d %H:%M:%S"
        timestamp = datetime.fromtimestamp(record["time"]).strftime(time_format)
        level = record["level"].name
        message = record["message"]

        # Add extra fields if present
        extra = ""
        if "extra" in record and record["extra"]:
            extra_items = [f"{k}={v}" for k, v in record["extra"].items()]
            extra = " | " + " | ".join(extra_items)

        return f"[{timestamp}] [{level}] {message}{extra}"

    @staticmethod
    def format_json(record: Dict[str, Any]) -> str:
        """Format log record as JSON."""
        # Create a dictionary with base fields
        log_dict = {
            "timestamp": datetime.fromtimestamp(record["time"]).isoformat(),
            "level": record["level"].name,
            "message": record["message"],
            "logger": record["name"]
        }

        # Add exception info if present
        if "exception" in record and record["exception"]:
            log_dict["exception"] = record["exception"]

        # Add extra fields
        if "extra" in record and record["extra"]:
            log_dict.update(record["extra"])

        return json.dumps(log_dict)


class MonitoringSystem:
    """
    Centralized monitoring system for the application.
    Collects metrics, logs, and health data.
    """

    def __init__(self, event_bus: Optional[EventBus] = None,
                 app_name: str = "defi_analyzer", log_level: LogLevel = LogLevel.INFO):
        """
        Initialize the monitoring system.

        Args:
            event_bus: Optional event bus for monitoring events
            app_name: Name of the application
            log_level: Default log level
        """
        self._event_bus = event_bus
        self._app_name = app_name
        self._log_level = log_level
        self._metrics: Dict[str, Metric] = {}
        self._health_checks: Dict[str, Callable[[], Awaitable[bool]]] = {}
        self._startup_time = time.time()
        self._instance_id = str(uuid.uuid4())
        self._log_handlers = []

        # Set up logging
        self._configure_logging()

        logger.info(f"MonitoringSystem initialized for {app_name}, instance {self._instance_id}")

    def _configure_logging(self) -> None:
        """Configure loguru for application logging."""
        # Remove default handler
        logger.remove()

        # Add console handler
        console_handler_id = logger.add(
            sys.stderr,
            level=self._log_level.value,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}"
        )
        self._log_handlers.append(console_handler_id)

        # Add file handler for INFO and above
        log_path = os.path.join("logs", self._app_name)
        os.makedirs(log_path, exist_ok=True)

        file_handler_id = logger.add(
            os.path.join(log_path, f"{self._app_name}.log"),
            level="INFO",
            rotation="10 MB",
            retention="1 week",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}"
        )
        self._log_handlers.append(file_handler_id)

        # Add error file handler
        error_handler_id = logger.add(
            os.path.join(log_path, f"{self._app_name}_error.log"),
            level="ERROR",
            rotation="10 MB",
            retention="1 month",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message} | {exception}"
        )
        self._log_handlers.append(error_handler_id)

    def set_log_level(self, level: LogLevel) -> None:
        """
        Set the log level.

        Args:
            level: New log level
        """
        self._log_level = level
        logger.info(f"Log level set to {level.value}")

    async def log(self, level: LogLevel, message: str,
                  extra: Optional[Dict[str, Any]] = None) -> None:
        """
        Log a message.

        Args:
            level: Log level
            message: Log message
            extra: Optional extra fields
        """
        # Log message with loguru
        if extra:
            logger_func = getattr(logger.bind(**extra), level.value.lower())
        else:
            logger_func = getattr(logger, level.value.lower())

        logger_func(message)

        # Publish log event if event bus is available
        if self._event_bus:
            event_data = {
                "level": level.value,
                "message": message,
                "extra": extra or {}
            }

            # Use a different event type based on log level for filtering
            if level in [LogLevel.ERROR, LogLevel.CRITICAL]:
                event_type = EventType.SYSTEM_ERROR
            else:
                event_type = EventType.SYSTEM_INFO

            await self._event_bus.publish(event_type, event_data, "monitoring")

    async def counter(self, name: str, increment: float = 1.0,
                      labels: Optional[Dict[str, str]] = None) -> None:
        """
        Increment a counter metric.

        Args:
            name: Metric name
            increment: Value to increment by
            labels: Optional labels for the metric
        """
        labels = labels or {}

        # Create a unique key for the metric including labels
        key = self._get_metric_key(name, labels)

        # Create metric if it doesn't exist
        if key not in self._metrics:
            self._metrics[key] = Metric(
                name=name,
                type=MetricType.COUNTER,
                description=f"Counter for {name}",
                value=0,
                labels=labels
            )

        # Increment the counter
        self._metrics[key].value += increment
        self._metrics[key].timestamp = time.time()

        # Publish metric event if event bus is available
        if self._event_bus:
            await self._event_bus.publish(
                EventType.SYSTEM_INFO,
                {
                    "metric": self._metrics[key].to_dict()
                },
                "monitoring"
            )

    async def gauge(self, name: str, value: float,
                    labels: Optional[Dict[str, str]] = None) -> None:
        """
        Set a gauge metric.

        Args:
            name: Metric name
            value: Gauge value
            labels: Optional labels for the metric
        """
        labels = labels or {}

        # Create a unique key for the metric including labels
        key = self._get_metric_key(name, labels)

        # Create or update gauge
        if key not in self._metrics:
            self._metrics[key] = Metric(
                name=name,
                type=MetricType.GAUGE,
                description=f"Gauge for {name}",
                value=value,
                labels=labels
            )
        else:
            self._metrics[key].value = value
            self._metrics[key].timestamp = time.time()

        # Publish metric event if event bus is available
        if self._event_bus:
            await self._event_bus.publish(
                EventType.SYSTEM_INFO,
                {
                    "metric": self._metrics[key].to_dict()
                },
                "monitoring"
            )

    async def histogram(self, name: str, value: float,
                        labels: Optional[Dict[str, str]] = None) -> None:
        """
        Record a value in a histogram metric.

        Args:
            name: Metric name
            value: Value to record
            labels: Optional labels for the metric
        """
        labels = labels or {}

        # Create a unique key for the metric including labels
        key = self._get_metric_key(name, labels)

        # Create or update histogram
        if key not in self._metrics:
            self._metrics[key] = Metric(
                name=name,
                type=MetricType.HISTOGRAM,
                description=f"Histogram for {name}",
                value={"values": [value], "count": 1, "sum": value},
                labels=labels
            )
        else:
            hist_data = self._metrics[key].value
            hist_data["values"].append(value)
            hist_data["count"] += 1
            hist_data["sum"] += value
            self._metrics[key].timestamp = time.time()

        # Publish metric event if event bus is available
        if self._event_bus:
            await self._event_bus.publish(
                EventType.SYSTEM_INFO,
                {
                    "metric": self._metrics[key].to_dict()
                },
                "monitoring"
            )

    def _get_metric_key(self, name: str, labels: Dict[str, str]) -> str:
        """
        Generate a unique key for a metric based on name and labels.

        Args:
            name: Metric name
            labels: Metric labels

        Returns:
            str: Unique metric key
        """
        if not labels:
            return name

        # Sort labels by key for consistent ordering
        sorted_labels = sorted(labels.items())
        labels_str = ",".join(f"{k}={v}" for k, v in sorted_labels)

        return f"{name}{{{labels_str}}}"

    def get_metrics(self) -> List[Dict[str, Any]]:
        """
        Get all metrics.

        Returns:
            List[Dict[str, Any]]: List of metrics as dictionaries
        """
        return [metric.to_dict() for metric in self._metrics.values()]

    def get_metric(self, name: str, labels: Optional[Dict[str, str]] = None) -> Optional[Dict[str, Any]]:
        """
        Get a specific metric.

        Args:
            name: Metric name
            labels: Optional metric labels

        Returns:
            Optional[Dict[str, Any]]: Metric as dictionary, or None if not found
        """
        key = self._get_metric_key(name, labels or {})
        metric = self._metrics.get(key)

        if metric:
            return metric.to_dict()

        return None

    def register_health_check(self, name: str, check_func: Callable[[], Awaitable[bool]]) -> None:
        """
        Register a health check function.

        Args:
            name: Health check name
            check_func: Async function that returns True if healthy, False otherwise
        """
        self._health_checks[name] = check_func
        logger.info(f"Registered health check: {name}")

    async def run_health_checks(self) -> Dict[str, Any]:
        """
        Run all registered health checks.

        Returns:
            Dict[str, Any]: Health check results
        """
        results = {}
        overall_status = "healthy"

        # Run each health check
        for name, check_func in self._health_checks.items():
            try:
                start_time = time.time()
                status = await check_func()
                duration = time.time() - start_time

                results[name] = {
                    "status": "up" if status else "down",
                    "duration_ms": round(duration * 1000, 2)
                }

                if not status:
                    overall_status = "unhealthy"

            except Exception as e:
                results[name] = {
                    "status": "error",
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }
                overall_status = "unhealthy"

        # Add system info
        system_info = self.get_system_info()

        return {
            "status": overall_status,
            "timestamp": datetime.now().isoformat(),
            "instance_id": self._instance_id,
            "uptime_seconds": round(time.time() - self._startup_time),
            "checks": results,
            "system": system_info
        }

    def get_system_info(self) -> Dict[str, Any]:
        """
        Get system information.

        Returns:
            Dict[str, Any]: System information
        """
        try:
            # Get basic system info
            system_info = {
                "hostname": socket.gethostname(),
                "platform": platform.system(),
                "platform_version": platform.version(),
                "python_version": platform.python_version(),
                "cpu_count": os.cpu_count()
            }

            # Add process info
            process = psutil.Process()
            memory_info = process.memory_info()

            process_info = {
                "pid": process.pid,
                "memory_rss_mb": round(memory_info.rss / (1024 * 1024), 2),
                "cpu_percent": process.cpu_percent(),
                "threads": process.num_threads(),
                "open_files": len(process.open_files())
            }

            # Add system resources
            system_resources = {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage('/').percent
            }

            return {
                "info": system_info,
                "process": process_info,
                "resources": system_resources
            }

        except Exception as e:
            logger.warning(f"Error getting system info: {str(e)}")
            return {"error": str(e)}


# Global monitoring system instance
_monitoring_system: Optional[MonitoringSystem] = None


def get_monitoring_system(event_bus: Optional[EventBus] = None,
                          app_name: str = "defi_analyzer") -> MonitoringSystem:
    """
    Get the global monitoring system instance.

    Args:
        event_bus: Optional event bus for monitoring events
        app_name: Name of the application

    Returns:
        MonitoringSystem: Global monitoring system instance
    """
    global _monitoring_system
    if _monitoring_system is None:
        _monitoring_system = MonitoringSystem(event_bus, app_name)
    return _monitoring_system


# Performance monitoring decorator
def monitor_performance(name: Optional[str] = None, labels: Optional[Dict[str, str]] = None):
    """
    Decorator for monitoring function performance.

    Args:
        name: Optional metric name (defaults to function name)
        labels: Optional metric labels

    Returns:
        Decorator function
    """

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            monitoring = get_monitoring_system()
            metric_name = name or f"function.{func.__qualname__}"
            metric_labels = labels or {}

            start_time = time.time()

            try:
                result = await func(*args, **kwargs)

                # Record successful execution
                duration = time.time() - start_time
                await monitoring.histogram(f"{metric_name}.duration", duration, metric_labels)
                await monitoring.counter(f"{metric_name}.calls", 1, metric_labels)

                return result

            except Exception as e:
                # Record failed execution
                duration = time.time() - start_time
                error_labels = {**metric_labels, "error": type(e).__name__}

                await monitoring.histogram(f"{metric_name}.duration", duration, metric_labels)
                await monitoring.counter(f"{metric_name}.errors", 1, error_labels)

                # Re-raise the exception
                raise

        return wrapper

    return decorator

