# Core Components README

This README provides an overview of the core components in our DeFi Trading Pattern Analysis Tool. These components form the foundation of the system architecture, enabling dependency management, workflow execution, event handling, error management, and performance optimization.

## Core Components Overview

### 1. ApplicationContext (`context.py`)

The ApplicationContext serves as the central dependency injection container for the application, managing component lifecycles and providing service location capabilities.

**Key Features:**
- Component registration and lifecycle management
- Singleton and transient component support
- Dependency resolution and injection
- Database connection management via UnitOfWork pattern

**Usage Example:**
```python
# Get the application context
context = get_context()

# Register a component
context.register_singleton("my_service", lambda: MyService())

# Get a component
my_service = context.get_component("my_service")
```

### 2. EventBus (`events.py`)

The EventBus implements an event-driven architecture for the application, providing publish-subscribe functionality for system events.

**Key Features:**
- Event publishing and subscription
- Event history tracking
- Wildcard subscriptions
- Event filtering and querying

**Event Types:**
- Workflow events (started, completed, failed, etc.)
- Analysis events (started, completed, pattern detected, etc.)
- Data events (fetched, updated)
- Report events (generated, exported)
- System events (startup, shutdown, errors)

**Usage Example:**
```python
# Get the event bus
event_bus = context.get_component("event_bus")

# Subscribe to an event
async def on_pattern_detected(event):
    print(f"Pattern detected: {event.data}")

event_bus.subscribe(EventType.PATTERN_DETECTED, on_pattern_detected)

# Publish an event
await event_bus.publish(
    EventType.PATTERN_DETECTED,
    {"pattern": "double_bottom", "confidence": 0.95}
)
```

### 3. ErrorHandler (`error_handling.py`)

The ErrorHandler provides a centralized error handling system for the application, capturing, logging, and managing errors across different modules.

**Key Features:**
- Error severity levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- Specialized error types (ValidationError, DatabaseError, ApiError, WorkflowError)
- Error callbacks for specific error types
- Integration with EventBus for error events
- Decorator for automatic error handling

**Usage Example:**
```python
# Get the error handler
error_handler = get_error_handler()

# Register an error callback
async def on_database_error(error):
    print(f"Database error occurred: {error}")

error_handler.register_error_callback(DatabaseError, on_database_error)

# Handle an error
try:
    # Some operation
    pass
except Exception as e:
    await error_handler.handle_error(e, {"context": "data_fetch"})
```

### 4. Optimization Utilities (`optimization.py`)

The optimization module provides utilities for improving application performance, including caching, parallel execution, and rate limiting.

**Key Features:**
- In-memory cache with TTL support
- Parallel execution of tasks using threads or processes
- Rate limiting for API requests
- Resource management for controlling concurrency
- Decorators for caching, timing, and memory profiling

**Usage Example:**
```python
# Use the cache decorator
@cached(ttl=300, key_prefix="user_data")
async def get_user_data(user_id):
    # Expensive operation
    return data

# Use parallel execution
executor = ParallelExecutor(max_workers=4)
results = await executor.map(process_item, items)

# Use rate limiting
rate_limiter = RateLimiter(rate=10, period=1.0)
async with rate_limiter:
    # Rate-limited operation
    await api_call()
```

### 5. WorkflowEngine (`workflow.py`)

The WorkflowEngine enables defining and executing analysis workflows, managing dependencies between steps and providing error handling.

**Key Features:**
- Workflow registration and execution
- Step dependency management
- Retry logic for failed steps
- Progress tracking
- Workflow cancellation and pausing
- Checkpointing for resuming workflows

**Usage Example:**
```python
# Create workflow steps
steps = [
    WorkflowStep(
        name="fetch_data",
        func=fetch_data_func,
        dependencies=[]
    ),
    WorkflowStep(
        name="analyze_data",
        func=analyze_data_func,
        dependencies=["fetch_data"]
    ),
    WorkflowStep(
        name="generate_report",
        func=generate_report_func,
        dependencies=["analyze_data"]
    )
]

# Register workflow
workflow_engine = get_workflow_engine()
workflow_engine.register_workflow("data_analysis", steps)

# Execute workflow
workflow_id = await workflow_engine.execute_workflow(
    "data_analysis",
    parameters={"pair_id": "BTC-USD"}
)

# Get workflow status
status = await workflow_engine.get_workflow_status(workflow_id)
```

## Integration Points

These core components are designed to work together seamlessly:

1. **ApplicationContext** provides the foundation for dependency injection
2. **WorkflowEngine** uses the EventBus to publish workflow events
3. **ErrorHandler** integrates with the EventBus for error events
4. **Optimization utilities** can be used throughout the application

## Best Practices

1. **Dependency Injection**: Use the ApplicationContext for managing dependencies
2. **Event-Driven Architecture**: Use the EventBus for loose coupling between components
3. **Error Handling**: Use the ErrorHandler for consistent error management
4. **Performance Optimization**: Use caching and parallel execution for performance-critical operations
5. **Workflow Definition**: Define clear workflows with proper dependencies

## Extending the Core Components

To extend the core components:

1. **New Component Types**: Register new component types with the ApplicationContext
2. **Custom Event Types**: Add new event types to the EventType enum
3. **Specialized Error Types**: Create new error types by extending AppError
4. **Custom Workflow Steps**: Create new workflow steps for specific analysis tasks