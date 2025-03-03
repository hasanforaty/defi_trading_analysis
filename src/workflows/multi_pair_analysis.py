from datetime import datetime, timedelta
from typing import Dict, Any, List, Set
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

from src.core.workflow import WorkflowStep, BaseWorkflow
from src.core.context import get_context
from src.core.events import EventType
from src.core.optimization import get_resource_manager
from src.analyzers.models import AnalysisType, ComprehensiveResult
from src.data.unit_of_work import UnitOfWork
from src.workflows.pair_analysis import PairAnalysisWorkflow
from config.settings import get_settings

logger = logging.getLogger(__name__)


class MultiPairAnalysisWorkflow(BaseWorkflow):
    """
    Workflow for analyzing multiple trading pairs in parallel.
    Orchestrates multiple PairAnalysisWorkflow instances.
    """

    def __init__(self, name: str = "multi_pair_analysis"):
        super().__init__(name)
        self.context = get_context()

    async def initialize(self, parameters: Dict[str, Any]) -> None:
        """Initialize the workflow with analysis parameters."""
        # Get pair IDs to analyze
        self.pair_ids = parameters.get("pair_ids", [])
        if not self.pair_ids and parameters.get("top_pairs_count"):
            # Fetch top pairs if pair_ids not provided but top_pairs_count is
            self.pair_ids = await self._get_top_pairs(parameters.get("top_pairs_count"))

        if not self.pair_ids:
            raise ValueError("Either pair_ids or top_pairs_count must be provided")

        # Analysis configuration
        self.analysis_types = parameters.get("analysis_types", [AnalysisType.COMPREHENSIVE])
        self.start_time = parameters.get("start_time")
        self.end_time = parameters.get("end_time")

        if not self.start_time:
            self.start_time = datetime.utcnow() - timedelta(days=7)  # Default to last 7 days

        if not self.end_time:
            self.end_time = datetime.utcnow()

        # Get configurations that will be passed to each pair analysis
        self.threshold_config = parameters.get("threshold_config")
        self.wave_config = parameters.get("wave_config")
        self.ratio_config = parameters.get("ratio_config")
        self.pattern_config = parameters.get("pattern_config")

        # Maximum concurrent analyses
        self.max_concurrent = parameters.get("max_concurrent", 5)

        # Initialize results
        self.analysis_results = {
            "pair_results": {},
            "summary": {
                "total_pairs": len(self.pair_ids),
                "completed_pairs": 0,
                "failed_pairs": 0
            }
        }

        # Define workflow steps
        await self._define_workflow_steps()

    async def _get_top_pairs(self, count: int) -> List[int]:
        """Get top pairs by trading volume or liquidity."""
        async with UnitOfWork() as uow:
            pairs = await uow.pairs.get_pairs_by_liquidity(limit=count)
            return [pair.id for pair in pairs]

    async def _define_workflow_steps(self) -> None:
        """Define the workflow steps."""
        steps = []

        # Add initialization step
        steps.append(WorkflowStep(
            name="initialize_analysis",
            func=self._initialize_analysis,
            dependencies=[]
        ))

        # Add step to run analyses in parallel
        steps.append(WorkflowStep(
            name="run_parallel_analyses",
            func=self._run_parallel_analyses,
            dependencies=["initialize_analysis"]
        ))

        # Add step to compile results
        steps.append(WorkflowStep(
            name="compile_results",
            func=self._compile_results,
            dependencies=["run_parallel_analyses"]
        ))

        # Set the steps for this workflow
        self.steps = steps

    async def _initialize_analysis(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize the analysis by validating pairs."""
        event_bus = self.context.get_component("event_bus")

        # Publish event for analysis initialization
        await event_bus.publish(
            EventType.MULTI_ANALYSIS_STARTED,
            {
                "pair_count": len(self.pair_ids),
                "analysis_types": [t.value for t in self.analysis_types],
                "start_time": self.start_time,
                "end_time": self.end_time
            }
        )

        # Validate pairs exist
        async with UnitOfWork() as uow:
            valid_pairs = []
            for pair_id in self.pair_ids:
                pair = await uow.pairs.get_by_id(pair_id)
                if pair:
                    valid_pairs.append(pair_id)
                else:
                    logger.warning(f"Pair with ID {pair_id} not found")

            return {
                "valid_pair_ids": valid_pairs
            }

    async def _run_parallel_analyses(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run analyses for all pairs in parallel with concurrency control."""
        valid_pair_ids = context["initialize_analysis"]["valid_pair_ids"]

        # Update summary with validated count
        self.analysis_results["summary"]["total_pairs"] = len(valid_pair_ids)

        # Get resource manager for parallel execution
        resource_manager = get_resource_manager()

        # Prepare tasks for all pairs
        tasks = []
        results = {}

        # Semaphore to limit concurrency
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def analyze_pair(pair_id):
            """Run analysis for a single pair with concurrency control."""
            async with semaphore:
                try:
                    # Create and initialize pair analysis workflow
                    workflow = PairAnalysisWorkflow()

                    # Prepare parameters
                    params = {
                        "pair_id": pair_id,
                        "analysis_types": self.analysis_types,
                        "start_time": self.start_time,
                        "end_time": self.end_time
                    }

                    # Add configs if provided
                    if self.threshold_config:
                        params["threshold_config"] = self.threshold_config
                    if self.wave_config:
                        params["wave_config"] = self.wave_config
                    if self.ratio_config:
                        params["ratio_config"] = self.ratio_config
                    if self.pattern_config:
                        params["pattern_config"] = self.pattern_config

                    # Execute the workflow
                    workflow_id = await workflow.execute(params)

                    # Wait for workflow to complete
                    while True:
                        status = await self.engine.get_workflow_status(workflow_id)
                        if status["status"] in ["completed", "failed", "canceled"]:
                            break
                        await asyncio.sleep(0.5)

                    # Get results
                    if status["status"] == "completed":
                        pair_results = await workflow.get_results()
                        results[pair_id] = {
                            "status": "completed",
                            "results": pair_results
                        }
                        self.analysis_results["summary"]["completed_pairs"] += 1
                    else:
                        results[pair_id] = {
                            "status": "failed",
                            "error": status.get("errors", {"unknown": "Analysis failed"})
                        }
                        self.analysis_results["summary"]["failed_pairs"] += 1

                except Exception as e:
                    logger.exception(f"Error analyzing pair {pair_id}: {str(e)}")
                    results[pair_id] = {
                        "status": "failed",
                        "error": str(e)
                    }
                    self.analysis_results["summary"]["failed_pairs"] += 1

        # Create tasks for all pairs
        for pair_id in valid_pair_ids:
            tasks.append(analyze_pair(pair_id))

        # Run all tasks concurrently
        await asyncio.gather(*tasks)

        return {
            "pair_results": results
        }

    async def _compile_results(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Compile results from all pair analyses."""
        pair_results = context["run_parallel_analyses"]["pair_results"]

        # Store individual pair results
        self.analysis_results["pair_results"] = pair_results

        # Calculate high-level statistics
        significant_patterns = 0
        total_transactions = 0
        active_wallets = set()

        for pair_id, result in pair_results.items():
            if result["status"] == "completed":
                # Extract data from comprehensive results if available
                if AnalysisType.COMPREHENSIVE in result["results"]:
                    comp_result = result["results"][AnalysisType.COMPREHENSIVE]
                    if hasattr(comp_result, "patterns") and comp_result.patterns:
                        significant_patterns += len(comp_result.patterns)

                    if hasattr(comp_result, "total_transactions"):
                        total_transactions += comp_result.total_transactions

                    if hasattr(comp_result, "active_wallets"):
                        active_wallets.update(comp_result.active_wallets)

        # Add summary statistics
        self.analysis_results["summary"]["significant_patterns"] = significant_patterns
        self.analysis_results["summary"]["total_transactions"] = total_transactions
        self.analysis_results["summary"]["unique_active_wallets"] = len(active_wallets)

        # Publish completion event
        event_bus = self.context.get_component("event_bus")
        await event_bus.publish(
            EventType.MULTI_ANALYSIS_COMPLETED,
            {
                "total_pairs": self.analysis_results["summary"]["total_pairs"],
                "completed_pairs": self.analysis_results["summary"]["completed_pairs"],
                "failed_pairs": self.analysis_results["summary"]["failed_pairs"],
                "significant_patterns": significant_patterns,
                "total_transactions": total_transactions,
                "unique_active_wallets": len(active_wallets)
            }
        )

        return self.analysis_results
