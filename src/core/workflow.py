# src/core/workflow.py
import asyncio
import time
import uuid
from enum import Enum
from typing import Dict, List, Any, Optional, Callable, Awaitable, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
import traceback
from loguru import logger

from src.core.context import get_context
from src.core.events import EventBus, EventType
from abc import ABC, abstractmethod


class WorkflowStatus(str, Enum):
    """Enum representing the status of a workflow."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"
    PAUSED = "paused"


@dataclass
class WorkflowStep:
    """Represents a step in a workflow."""
    name: str
    func: Callable[..., Awaitable[Any]]
    dependencies: List[str] = field(default_factory=list)
    retry_count: int = 0
    retry_delay: float = 1.0
    timeout: Optional[float] = None
    weight: float = 1.0  # Used for progress calculation
    result: Any = None
    status: WorkflowStatus = WorkflowStatus.PENDING
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    error: Optional[Exception] = None


@dataclass
class WorkflowContext:
    """Context object for workflow execution."""
    workflow_id: str
    parameters: Dict[str, Any]
    results: Dict[str, Any] = field(default_factory=dict)
    errors: Dict[str, str] = field(default_factory=dict)
    status: WorkflowStatus = WorkflowStatus.PENDING
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    current_step: Optional[str] = None
    checkpoint_data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class WorkflowEngine:
    """
    Engine for defining and executing analysis workflows.
    Manages dependencies between steps and provides error handling.
    """

    def __init__(self):
        """Initialize the workflow engine."""
        self._workflows: Dict[str, Dict[str, WorkflowStep]] = {}
        self._active_contexts: Dict[str, WorkflowContext] = {}
        self._completed_contexts: Dict[str, WorkflowContext] = {}
        self._context_locks: Dict[str, asyncio.Lock] = {}
        self._max_completed_contexts = 100

        # Get event bus from application context
        app_context = get_context()
        if app_context.has_component("event_bus"):
            self._event_bus = app_context.get_component("event_bus")
        else:
            # Create a new event bus if not available
            self._event_bus = EventBus()

        logger.info("WorkflowEngine initialized")

    def register_workflow(self, name: str, steps: List[WorkflowStep]) -> None:
        """
        Register a workflow with the engine.

        Args:
            name: Name of the workflow
            steps: List of workflow steps
        """
        if name in self._workflows:
            raise ValueError(f"Workflow '{name}' is already registered")

        # Validate dependencies
        step_names = {step.name for step in steps}
        for step in steps:
            for dep in step.dependencies:
                if dep not in step_names:
                    raise ValueError(f"Step '{step.name}' depends on undefined step '{dep}'")

        # Convert list to dictionary for easy lookup
        self._workflows[name] = {step.name: step for step in steps}
        logger.info(f"Workflow '{name}' registered with {len(steps)} steps")

    async def execute_workflow(self,
                               workflow_name: str,
                               parameters: Dict[str, Any] = None,
                               workflow_id: str = None) -> str:
        """
        Execute a workflow asynchronously.

        Args:
            workflow_name: Name of the workflow to execute
            parameters: Parameters to pass to the workflow
            workflow_id: Optional custom ID for the workflow

        Returns:
            str: Workflow execution ID
        """
        if workflow_name not in self._workflows:
            raise ValueError(f"Workflow '{workflow_name}' is not registered")

        # Generate a workflow ID if not provided
        if workflow_id is None:
            workflow_id = str(uuid.uuid4())

        # Create workflow context
        context = WorkflowContext(
            workflow_id=workflow_id,
            parameters=parameters or {},
            status=WorkflowStatus.PENDING
        )

        # Create lock for this context
        self._context_locks[workflow_id] = asyncio.Lock()

        # Store context
        self._active_contexts[workflow_id] = context

        # Start workflow execution in background
        asyncio.create_task(self._execute_workflow_internal(workflow_name, context))

        # Emit event for workflow started
        await self._event_bus.publish(
            EventType.WORKFLOW_STARTED,
            {
                "workflow_id": workflow_id,
                "workflow_name": workflow_name,
                "parameters": parameters
            }
        )

        logger.info(f"Workflow '{workflow_name}' started with ID: {workflow_id}")
        return workflow_id

    async def _execute_workflow_internal(self, workflow_name: str, context: WorkflowContext) -> None:
        """
        Internal method to execute a workflow.

        Args:
            workflow_name: Name of the workflow to execute
            context: Workflow context
        """
        workflow_steps = self._workflows[workflow_name]

        # Set start time
        context.start_time = time.time()
        context.status = WorkflowStatus.RUNNING

        try:
            # Determine execution order based on dependencies
            execution_order = self._determine_execution_order(workflow_steps)

            # Execute steps in order
            for step_name in execution_order:
                # Check if workflow was canceled
                if context.status == WorkflowStatus.CANCELED:
                    logger.info(f"Workflow {context.workflow_id} was canceled")
                    break

                # Update current step
                async with self._context_locks[context.workflow_id]:
                    context.current_step = step_name

                step = workflow_steps[step_name]

                # Emit event for step started
                await self._event_bus.publish(
                    EventType.WORKFLOW_STEP_STARTED,
                    {
                        "workflow_id": context.workflow_id,
                        "workflow_name": workflow_name,
                        "step_name": step_name
                    }
                )

                # Execute step with retry logic
                result, success = await self._execute_step_with_retry(step, context)

                if success:
                    # Store result
                    async with self._context_locks[context.workflow_id]:
                        context.results[step_name] = result

                    # Emit event for step completed
                    await self._event_bus.publish(
                        EventType.WORKFLOW_STEP_COMPLETED,
                        {
                            "workflow_id": context.workflow_id,
                            "workflow_name": workflow_name,
                            "step_name": step_name,
                            "success": True
                        }
                    )
                else:
                    # Step failed after retries
                    error_msg = f"Step '{step_name}' failed after {step.retry_count} retries"

                    async with self._context_locks[context.workflow_id]:
                        context.errors[step_name] = error_msg if step.error is None else str(step.error)
                        context.status = WorkflowStatus.FAILED

                    # Emit event for step failed
                    await self._event_bus.publish(
                        EventType.WORKFLOW_STEP_FAILED,
                        {
                            "workflow_id": context.workflow_id,
                            "workflow_name": workflow_name,
                            "step_name": step_name,
                            "error": error_msg
                        }
                    )

                    logger.error(f"Workflow {context.workflow_id} failed at step '{step_name}': {error_msg}")
                    break

            # Workflow completed successfully if not already failed or canceled
            if context.status == WorkflowStatus.RUNNING:
                async with self._context_locks[context.workflow_id]:
                    context.status = WorkflowStatus.COMPLETED

        except Exception as e:
            # Handle unexpected exceptions
            error_msg = f"Unexpected error in workflow: {str(e)}"
            logger.exception(error_msg)

            async with self._context_locks[context.workflow_id]:
                context.status = WorkflowStatus.FAILED
                context.errors["workflow"] = error_msg

            # Emit event for workflow failed
            await self._event_bus.publish(
                EventType.WORKFLOW_FAILED,
                {
                    "workflow_id": context.workflow_id,
                    "workflow_name": workflow_name,
                    "error": error_msg
                }
            )
        finally:
            # Set end time and move to completed contexts
            context.end_time = time.time()

            # Clean up context lock
            if context.workflow_id in self._context_locks:
                del self._context_locks[context.workflow_id]

            # Move from active to completed contexts
            async with self._context_locks.get(context.workflow_id, asyncio.Lock()):
                if context.workflow_id in self._active_contexts:
                    del self._active_contexts[context.workflow_id]
                self._completed_contexts[context.workflow_id] = context

            # Limit the number of completed contexts
            if len(self._completed_contexts) > self._max_completed_contexts:
                oldest_id = min(self._completed_contexts.keys(),
                                key=lambda k: self._completed_contexts[k].end_time or 0)
                del self._completed_contexts[oldest_id]

            # Emit event for workflow completed
            await self._event_bus.publish(
                EventType.WORKFLOW_COMPLETED,
                {
                    "workflow_id": context.workflow_id,
                    "workflow_name": workflow_name,
                    "status": context.status.value,
                    "duration": (context.end_time or 0) - (context.start_time or 0)
                }
            )

            logger.info(f"Workflow {context.workflow_id} {context.status.value} in "
                        f"{(context.end_time or 0) - (context.start_time or 0):.2f} seconds")

    async def _execute_step_with_retry(self,
                                       step: WorkflowStep,
                                       context: WorkflowContext) -> Tuple[Any, bool]:
        """
        Execute a workflow step with retry logic.

        Args:
            step: Workflow step to execute
            context: Workflow context

        Returns:
            Tuple[Any, bool]: (result, success flag)
        """
        step.start_time = time.time()
        step.status = WorkflowStatus.RUNNING

        retry_count = 0
        result = None
        success = False

        while retry_count <= step.retry_count:
            try:
                # Execute step function with parameters and step dependencies
                params = {
                    **context.parameters,
                    "context": context,
                    "step_results": {dep: context.results.get(dep) for dep in step.dependencies}
                }

                if step.timeout is not None:
                    # Run with timeout
                    result = await asyncio.wait_for(step.func(**params), timeout=step.timeout)
                else:
                    # Run without timeout
                    result = await step.func(**params)

                # Step succeeded
                step.result = result
                step.status = WorkflowStatus.COMPLETED
                success = True
                break

            except asyncio.TimeoutError:
                retry_count += 1
                error_msg = f"Step '{step.name}' timed out after {step.timeout} seconds"
                step.error = asyncio.TimeoutError(error_msg)

                logger.warning(f"{error_msg} (attempt {retry_count}/{step.retry_count + 1})")

                if retry_count <= step.retry_count:
                    # Wait before retry
                    await asyncio.sleep(step.retry_delay)

            except Exception as e:
                retry_count += 1
                step.error = e

                logger.warning(f"Step '{step.name}' failed: {str(e)} "
                               f"(attempt {retry_count}/{step.retry_count + 1})")
                logger.debug(f"Exception: {traceback.format_exc()}")

                if retry_count <= step.retry_count:
                    # Wait before retry
                    await asyncio.sleep(step.retry_delay)

        step.end_time = time.time()

        if not success:
            step.status = WorkflowStatus.FAILED

        return result, success

    def _determine_execution_order(self, steps: Dict[str, WorkflowStep]) -> List[str]:
        """
        Determine the execution order of steps based on dependencies.
        Uses topological sorting algorithm.

        Args:
            steps: Dictionary of workflow steps

        Returns:
            List[str]: Ordered list of step names
        """
        # Build dependency graph
        graph: Dict[str, Set[str]] = {step_name: set(step.dependencies) for step_name, step in steps.items()}

        # Find steps with no dependencies
        no_deps = [name for name, deps in graph.items() if not deps]

        # Topological sort
        result = []
        while no_deps:
            # Take a step with no dependencies
            step_name = no_deps.pop(0)
            result.append(step_name)

            # Remove this step from dependencies of other steps
            for name, deps in graph.items():
                if step_name in deps:
                    deps.remove(step_name)
                    # If no more dependencies, add to no_deps
                    if not deps and name not in result and name not in no_deps:
                        no_deps.append(name)

        # Check for cycles
        if len(result) != len(steps):
            unprocessed = set(steps.keys()) - set(result)
            raise ValueError(f"Cycle detected in workflow dependencies. Unprocessed steps: {unprocessed}")

        return result

    async def cancel_workflow(self, workflow_id: str) -> bool:
        """
        Cancel a running workflow.

        Args:
            workflow_id: ID of the workflow to cancel

        Returns:
            bool: True if workflow was canceled, False if not found or already completed
        """
        if workflow_id not in self._active_contexts:
            return False

        context = self._active_contexts[workflow_id]

        # Don't cancel if already completed
        if context.status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED]:
            return False

        async with self._context_locks.get(workflow_id, asyncio.Lock()):
            context.status = WorkflowStatus.CANCELED
            logger.info(f"Workflow {workflow_id} canceled")

            # Emit event for workflow canceled
            await self._event_bus.publish(
                EventType.WORKFLOW_CANCELED,
                {
                    "workflow_id": workflow_id
                }
            )

        return True

    async def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the status of a workflow.

        Args:
            workflow_id: ID of the workflow

        Returns:
            Optional[Dict[str, Any]]: Workflow status information or None if not found
        """
        context = self._active_contexts.get(workflow_id) or self._completed_contexts.get(workflow_id)

        if context is None:
            return None

        # Calculate progress
        progress = 0
        if workflow_id in self._active_contexts:
            workflow_name = context.metadata.get("workflow_name")
            if workflow_name in self._workflows:
                steps = self._workflows[workflow_name]
                step_count = len(steps)

                if step_count > 0:
                    completed_steps = len([s for s in context.results.keys() if s in steps])
                    progress = (completed_steps / step_count) * 100

        # Calculate duration
        if context.start_time:
            if context.end_time:
                duration = context.end_time - context.start_time
            else:
                duration = time.time() - context.start_time
        else:
            duration = 0

        # Build status response
        return {
            "workflow_id": context.workflow_id,
            "status": context.status.value,
            "progress": progress,
            "current_step": context.current_step,
            "start_time": datetime.fromtimestamp(context.start_time).isoformat() if context.start_time else None,
            "end_time": datetime.fromtimestamp(context.end_time).isoformat() if context.end_time else None,
            "duration": duration,
            "error_count": len(context.errors),
            "errors": context.errors,
            "metadata": context.metadata
        }

    async def get_workflow_result(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the complete result of a workflow.

        Args:
            workflow_id: ID of the workflow

        Returns:
            Optional[Dict[str, Any]]: Complete workflow results or None if not found
        """
        context = self._active_contexts.get(workflow_id) or self._completed_contexts.get(workflow_id)

        if context is None:
            return None

        # Don't return results for incomplete workflows
        if context.status != WorkflowStatus.COMPLETED:
            return {
                "status": context.status.value,
                "errors": context.errors
            }

        return {
            "workflow_id": context.workflow_id,
            "status": context.status.value,
            "results": context.results,
            "errors": context.errors,
            "metadata": context.metadata,
            "duration": (context.end_time or 0) - (context.start_time or 0)
        }

    async def checkpoint_workflow(self, workflow_id: str, checkpoint_data: Dict[str, Any]) -> bool:
        """
        Save checkpoint data for a workflow to allow resuming.

        Args:
            workflow_id: ID of the workflow
            checkpoint_data: Data to save for resuming

        Returns:
            bool: True if checkpoint was saved, False if workflow not found
        """
        if workflow_id not in self._active_contexts:
            return False

        context = self._active_contexts[workflow_id]

        async with self._context_locks.get(workflow_id, asyncio.Lock()):
            context.checkpoint_data = checkpoint_data

        return True

    async def resume_workflow(self, workflow_id: str) -> bool:
        """
        Resume a paused workflow.

        Args:
            workflow_id: ID of the workflow to resume

        Returns:
            bool: True if workflow was resumed, False if not found or not paused
        """
        if workflow_id not in self._active_contexts:
            return False

        context = self._active_contexts[workflow_id]

        # Only resume paused workflows
        if context.status != WorkflowStatus.PAUSED:
            return False

        async with self._context_locks.get(workflow_id, asyncio.Lock()):
            context.status = WorkflowStatus.RUNNING

            # Emit event for workflow resumed
            await self._event_bus.publish(
                EventType.WORKFLOW_RESUMED,
                {
                    "workflow_id": workflow_id
                }
            )

        return True

    async def pause_workflow(self, workflow_id: str) -> bool:
        """
        Pause a running workflow.

        Args:
            workflow_id: ID of the workflow to pause

        Returns:
            bool: True if workflow was paused, False if not found or not running
        """
        if workflow_id not in self._active_contexts:
            return False

        context = self._active_contexts[workflow_id]

        # Only pause running workflows
        if context.status != WorkflowStatus.RUNNING:
            return False

        async with self._context_locks.get(workflow_id, asyncio.Lock()):
            context.status = WorkflowStatus.PAUSED

            # Emit event for workflow paused
            await self._event_bus.publish(
                EventType.WORKFLOW_PAUSED,
                {
                    "workflow_id": workflow_id
                }
            )

        return True

    def list_workflows(self, status_filter: Optional[WorkflowStatus] = None) -> List[Dict[str, Any]]:
        """
        List all workflows with optional filtering.

        Args:
            status_filter: Optional filter by workflow status

        Returns:
            List[Dict[str, Any]]: List of workflow information
        """
        all_contexts = {**self._active_contexts, **self._completed_contexts}

        result = []
        for workflow_id, context in all_contexts.items():
            # Apply status filter if specified
            if status_filter is not None and context.status != status_filter:
                continue

            # Calculate duration
            if context.start_time:
                if context.end_time:
                    duration = context.end_time - context.start_time
                else:
                    duration = time.time() - context.start_time
            else:
                duration = 0

            # Build workflow info
            workflow_info = {
                "workflow_id": workflow_id,
                "status": context.status.value,
                "current_step": context.current_step,
                "start_time": datetime.fromtimestamp(context.start_time).isoformat() if context.start_time else None,
                "duration": duration,
                "error_count": len(context.errors)
            }
            result.append(workflow_info)

        return result


class BaseWorkflow(ABC):
    """
    Abstract base class for workflow implementations.
    Provides a standard interface for defining and executing workflows.
    """

    def __init__(self, name: str):
        """Initialize the workflow with a name."""
        self.name = name
        self.steps: List[WorkflowStep] = []
        self.analysis_results = {}
        self._engine = None

    @property
    def engine(self) -> WorkflowEngine:
        """Get the workflow engine, creating it if necessary."""
        if self._engine is None:
            app_context = get_context()
            if app_context.has_component("workflow_engine"):
                self._engine = app_context.get_component("workflow_engine")
            else:
                self._engine = WorkflowEngine()
        return self._engine

    @abstractmethod
    async def initialize(self, parameters: Dict[str, Any]) -> None:
        """
        Initialize the workflow with parameters.

        Args:
            parameters: Dictionary of parameters for the workflow
        """
        pass

    async def execute(self, parameters: Dict[str, Any] = None) -> str:
        """
        Execute the workflow with the given parameters.

        Args:
            parameters: Dictionary of parameters for the workflow

        Returns:
            str: Workflow execution ID
        """
        # Initialize the workflow
        await self.initialize(parameters or {})

        # Register the workflow with the engine
        self.engine.register_workflow(self.name, self.steps)

        # Execute the workflow
        workflow_id = await self.engine.execute_workflow(
            workflow_name=self.name,
            parameters=parameters
        )

        return workflow_id

    async def get_results(self) -> Dict[str, Any]:
        """
        Get the results of the workflow.

        Returns:
            Dict[str, Any]: Workflow results
        """
        return self.analysis_results
