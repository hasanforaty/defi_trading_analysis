import asyncio
import uuid
from enum import Enum
from typing import Dict, List, Any, Optional, Callable, Awaitable, Set
from dataclasses import dataclass, field
from datetime import datetime
import time
import json
from loguru import logger


class EventType(str, Enum):
    """Enum representing types of system events."""
    # Workflow events
    WORKFLOW_STARTED = "workflow.started"
    WORKFLOW_COMPLETED = "workflow.completed"
    WORKFLOW_FAILED = "workflow.failed"
    WORKFLOW_CANCELED = "workflow.canceled"
    WORKFLOW_PAUSED = "workflow.paused"
    WORKFLOW_RESUMED = "workflow.resumed"
    WORKFLOW_STEP_STARTED = "workflow.step.started"
    WORKFLOW_STEP_COMPLETED = "workflow.step.completed"
    WORKFLOW_STEP_FAILED = "workflow.step.failed"

    # Add these to the EventType Enum in events.py
    MULTI_ANALYSIS_STARTED = "analysis.multi.started"
    MULTI_ANALYSIS_COMPLETED = "analysis.multi.completed"
    MULTI_ANALYSIS_FAILED = "analysis.multi.failed"

    # Analysis events
    ANALYSIS_STARTED = "analysis.started"
    ANALYSIS_COMPLETED = "analysis.completed"
    ANALYSIS_FAILED = "analysis.failed"
    THRESHOLD_CALCULATED = "analysis.threshold.calculated"
    WAVE_DETECTED = "analysis.wave.detected"
    PATTERN_DETECTED = "analysis.pattern.detected"

    # Data events
    DATA_FETCHED = "data.fetched"
    DATA_UPDATED = "data.updated"

    # Report events
    REPORT_GENERATED = "report.generated"
    REPORT_EXPORTED = "report.exported"

    # System events
    SYSTEM_STARTUP = "system.startup"
    SYSTEM_SHUTDOWN = "system.shutdown"
    DATABASE_ERROR = "system.database.error"
    API_ERROR = "system.api.error"

    # Custom events
    CUSTOM = "custom"


@dataclass
class Event:
    """Represents a system event."""
    id: str
    type: EventType
    data: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    source: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {
            "id": self.id,
            "type": self.type.value,
            "data": self.data,
            "timestamp": self.timestamp,
            "datetime": datetime.fromtimestamp(self.timestamp).isoformat(),
            "source": self.source
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Event':
        """Create event from dictionary."""
        return cls(
            id=data["id"],
            type=EventType(data["type"]),
            data=data["data"],
            timestamp=data["timestamp"],
            source=data.get("source")
        )


class EventBus:
    """
    Implements an event-driven architecture for the application.
    Provides publish-subscribe functionality for system events.
    """

    def __init__(self):
        """Initialize the event bus."""
        self._subscribers: Dict[EventType, List[Callable[[Event], Awaitable[None]]]] = {}
        self._wildcard_subscribers: List[Callable[[Event], Awaitable[None]]] = []
        self._event_history: List[Event] = []
        self._max_history_size = 100
        self._lock = asyncio.Lock()
        self._active = True

        # Initialize event types
        for event_type in EventType:
            self._subscribers[event_type] = []

        logger.info("EventBus initialized")

    async def publish(self, event_type: EventType, data: Dict[str, Any], source: str = None) -> str:
        """
        Publish an event to the bus.

        Args:
            event_type: Type of the event
            data: Event data
            source: Optional source identifier

        Returns:
            str: Event ID
        """
        # Create event
        event_id = str(uuid.uuid4())
        event = Event(id=event_id, type=event_type, data=data, source=source)

        # Add to history
        async with self._lock:
            self._event_history.append(event)

            # Trim history if needed
            if len(self._event_history) > self._max_history_size:
                self._event_history = self._event_history[-self._max_history_size:]

        # Notify subscribers
        if not self._active:
            logger.warning(f"Event {event_type} published while EventBus is inactive")
            return event_id

        # Get subscribers for this event type
        event_subscribers = self._subscribers.get(event_type, [])

        # Notify all relevant subscribers
        tasks = []
        for subscriber in event_subscribers:
            tasks.append(asyncio.create_task(self._notify_subscriber(subscriber, event)))

        # Notify wildcard subscribers
        for subscriber in self._wildcard_subscribers:
            tasks.append(asyncio.create_task(self._notify_subscriber(subscriber, event)))

        # Wait for all notifications to complete
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        logger.debug(f"Event {event_type} published: {event_id}")
        return event_id

    async def _notify_subscriber(self, subscriber: Callable[[Event], Awaitable[None]], event: Event) -> None:
        """
        Notify a subscriber about an event.

        Args:
            subscriber: Subscriber callback
            event: Event to notify about
        """
        try:
            await subscriber(event)
        except Exception as e:
            logger.error(f"Error notifying subscriber about event {event.type}: {str(e)}")

    def subscribe(self, event_type: EventType, callback: Callable[[Event], Awaitable[None]]) -> None:
        """
        Subscribe to events of a specific type.

        Args:
            event_type: Type of the event to subscribe to
            callback: Async callback function to be called when event occurs
        """
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []

        self._subscribers[event_type].append(callback)
        logger.debug(f"Subscribed to event type: {event_type}")

    def unsubscribe(self, event_type: EventType, callback: Callable[[Event], Awaitable[None]]) -> bool:
        """
        Unsubscribe from events of a specific type.

        Args:
            event_type: Type of the event to unsubscribe from
            callback: Callback function to remove

        Returns:
            bool: True if unsubscribed successfully, False otherwise
        """
        if event_type not in self._subscribers:
            return False

        if callback in self._subscribers[event_type]:
            self._subscribers[event_type].remove(callback)
            logger.debug(f"Unsubscribed from event type: {event_type}")
            return True

        return False

    def subscribe_all(self, callback: Callable[[Event], Awaitable[None]]) -> None:
        """
        Subscribe to all events (wildcard subscription).

        Args:
            callback: Async callback function to be called when any event occurs
        """
        self._wildcard_subscribers.append(callback)
        logger.debug("Added wildcard event subscriber")

    def unsubscribe_all(self, callback: Callable[[Event], Awaitable[None]]) -> bool:
        """
        Unsubscribe from all events.

        Args:
            callback: Callback function to remove

        Returns:
            bool: True if unsubscribed successfully, False otherwise
        """
        if callback in self._wildcard_subscribers:
            self._wildcard_subscribers.remove(callback)
            logger.debug("Removed wildcard event subscriber")
            return True

        return False

    def get_subscriber_count(self, event_type: Optional[EventType] = None) -> int:
        """
        Get the number of subscribers for an event type.

        Args:
            event_type: Optional event type to get count for

        Returns:
            int: Number of subscribers
        """
        if event_type is None:
            # Count all subscribers including wildcards
            total = len(self._wildcard_subscribers)
            for subscribers in self._subscribers.values():
                total += len(subscribers)
            return total

        return len(self._subscribers.get(event_type, []))

    def get_recent_events(self, count: int = 10, event_type: Optional[EventType] = None) -> List[Dict[str, Any]]:
        """
        Get recent events from history.

        Args:
            count: Number of events to return
            event_type: Optional filter by event type

        Returns:
            List[Dict[str, Any]]: List of recent events as dictionaries
        """
        events = self._event_history

        # Filter by event type if specified
        if event_type is not None:
            events = [e for e in events if e.type == event_type]

        # Sort by timestamp (newest first) and limit to count
        events = sorted(events, key=lambda e: e.timestamp, reverse=True)[:count]

        # Convert to dictionaries
        return [e.to_dict() for e in events]

    def clear_history(self) -> None:
        """Clear the event history."""
        self._event_history = []
        logger.debug("Event history cleared")

    def deactivate(self) -> None:
        """Deactivate the event bus (stop publishing events)."""
        self._active = False
        logger.info("EventBus deactivated")

    def activate(self) -> None:
        """Activate the event bus."""
        self._active = True
        logger.info("EventBus activated")

    async def close(self) -> None:
        """Close the event bus and clean up resources."""
        self.deactivate()
        self.clear_history()
        logger.info("EventBus closed")


# Decorator for event handlers
def event_handler(event_type: EventType = None):
    """
    Decorator for event handler methods.

    Args:
        event_type: Type of event to handle

    Returns:
        Decorator function
    """

    def decorator(func):
        func._event_handler = True
        func._event_type = event_type
        return func

    return decorator

