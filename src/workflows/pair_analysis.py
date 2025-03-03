# src/workflows/pair_analysis.py
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import asyncio
import logging

from src.core.workflow import WorkflowStep, BaseWorkflow
from src.core.context import get_context
from src.core.events import EventType
from src.analyzers.threshold import ThresholdAnalyzer
from src.analyzers.wave import WaveAnalyzer
from src.analyzers.ratio import RatioAnalyzer
from src.analyzers.pattern import PatternAnalyzer
from src.analyzers.models import (
    AnalysisType, AnalysisStatus, ThresholdConfig,
    WaveConfig, RatioConfig, PatternConfig,
    ThresholdResult, WaveResult, RatioResult, PatternResult, ComprehensiveResult
)
from src.data.unit_of_work import UnitOfWork
from config.settings import get_settings

logger = logging.getLogger(__name__)


class PairAnalysisWorkflow(BaseWorkflow):
    """
    Workflow for analyzing a single trading pair using one or more analysis types.
    """

    def __init__(self, name: str = "pair_analysis"):
        super().__init__(name)
        self.context = get_context()

    async def initialize(self, parameters: Dict[str, Any]) -> None:
        """Initialize the workflow with analysis parameters."""
        self.pair_id = parameters.get("pair_id")
        if not self.pair_id:
            raise ValueError("pair_id is required for pair analysis workflow")

        self.analysis_types = parameters.get("analysis_types", [AnalysisType.COMPREHENSIVE])
        self.start_time = parameters.get("start_time")
        self.end_time = parameters.get("end_time")

        if not self.start_time:
            self.start_time = datetime.utcnow() - timedelta(days=7)  # Default to last 7 days

        if not self.end_time:
            self.end_time = datetime.utcnow()

        # Get configurations from parameters or use defaults
        self.threshold_config = parameters.get("threshold_config", ThresholdConfig())
        self.wave_config = parameters.get("wave_config", WaveConfig())
        self.ratio_config = parameters.get("ratio_config", RatioConfig())
        self.pattern_config = parameters.get("pattern_config", PatternConfig())

        # Initialize results dictionary
        self.analysis_results = {}

        # Define workflow steps based on requested analysis types
        await self._define_workflow_steps()

    async def _define_workflow_steps(self) -> None:
        """Define the workflow steps based on analysis types."""
        steps = []

        # Add fetch pair data step (always required)
        steps.append(WorkflowStep(
            name="fetch_pair_data",
            func=self._fetch_pair_data,
            dependencies=[]
        ))

        # Add analysis steps based on requested types
        if AnalysisType.COMPREHENSIVE in self.analysis_types or AnalysisType.THRESHOLD in self.analysis_types:
            steps.append(WorkflowStep(
                name="threshold_analysis",
                func=self._run_threshold_analysis,
                dependencies=["fetch_pair_data"]
            ))

        if AnalysisType.COMPREHENSIVE in self.analysis_types or AnalysisType.WAVE in self.analysis_types:
            steps.append(WorkflowStep(
                name="wave_analysis",
                func=self._run_wave_analysis,
                dependencies=["fetch_pair_data"]
            ))

        if AnalysisType.COMPREHENSIVE in self.analysis_types or AnalysisType.RATIO in self.analysis_types:
            steps.append(WorkflowStep(
                name="ratio_analysis",
                func=self._run_ratio_analysis,
                dependencies=["fetch_pair_data"]
            ))

        if AnalysisType.COMPREHENSIVE in self.analysis_types or AnalysisType.PATTERN in self.analysis_types:
            steps.append(WorkflowStep(
                name="pattern_analysis",
                func=self._run_pattern_analysis,
                dependencies=["fetch_pair_data"]
            ))

        # If comprehensive analysis, add a step to combine all results
        if AnalysisType.COMPREHENSIVE in self.analysis_types:
            steps.append(WorkflowStep(
                name="combine_results",
                func=self._combine_results,
                dependencies=["threshold_analysis", "wave_analysis", "ratio_analysis", "pattern_analysis"]
            ))

        # Set the steps for this workflow
        self.steps = steps

    async def _fetch_pair_data(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch pair data from the database."""
        async with UnitOfWork() as uow:
            pair = await uow.pairs.get_by_id(self.pair_id)
            if not pair:
                raise ValueError(f"Pair with ID {self.pair_id} not found")

            transactions = await uow.transactions.get_by_time_range(
                start_time=self.start_time,
                end_time=self.end_time,
                pair_id=self.pair_id
            )

            # Publish event for data retrieval
            event_bus = self.context.get_component("event_bus")
            await event_bus.publish(
                EventType.DATA_FETCHED,
                {
                    "pair_id": self.pair_id,
                    "transaction_count": len(transactions),
                    "start_time": self.start_time,
                    "end_time": self.end_time
                }
            )

            logger.info(f"Fetched {len(transactions)} transactions for pair {self.pair_id}")

            return {
                "pair": pair,
                "transactions": transactions
            }

    async def _run_threshold_analysis(self, context: Dict[str, Any]) -> ThresholdResult:
        """Run threshold analysis on the pair."""
        pair = context["fetch_pair_data"]["pair"]
        transactions = context["fetch_pair_data"]["transactions"]

        # Publish analysis started event
        event_bus = self.context.get_component("event_bus")
        await event_bus.publish(
            EventType.ANALYSIS_STARTED,
            {
                "pair_id": self.pair_id,
                "analysis_type": AnalysisType.THRESHOLD,
                "config": self.threshold_config.dict()
            }
        )

        # Create and run analyzer
        analyzer = ThresholdAnalyzer()
        result = await analyzer.analyze(
            pair=pair,
            transactions=transactions,
            config=self.threshold_config
        )

        # Publish analysis completed event
        await event_bus.publish(
            EventType.ANALYSIS_COMPLETED,
            {
                "pair_id": self.pair_id,
                "analysis_type": AnalysisType.THRESHOLD,
                "result_summary": {
                    "threshold": result.threshold,
                    "significant_transactions_count": len(result.significant_transactions),
                    "whale_wallets_count": len(result.whale_wallets)
                }
            }
        )

        # Store result for potential comprehensive analysis
        self.analysis_results[AnalysisType.THRESHOLD] = result
        return result

    async def _run_wave_analysis(self, context: Dict[str, Any]) -> WaveResult:
        """Run wave analysis on the pair."""
        pair = context["fetch_pair_data"]["pair"]
        transactions = context["fetch_pair_data"]["transactions"]

        # Publish analysis started event
        event_bus = self.context.get_component("event_bus")
        await event_bus.publish(
            EventType.ANALYSIS_STARTED,
            {
                "pair_id": self.pair_id,
                "analysis_type": AnalysisType.WAVE,
                "config": self.wave_config.dict()
            }
        )

        # Create and run analyzer
        analyzer = WaveAnalyzer()
        result = await analyzer.analyze(
            pair=pair,
            transactions=transactions,
            config=self.wave_config
        )

        # Publish analysis completed event
        await event_bus.publish(
            EventType.ANALYSIS_COMPLETED,
            {
                "pair_id": self.pair_id,
                "analysis_type": AnalysisType.WAVE,
                "result_summary": {
                    "buy_waves_count": len(result.buy_waves),
                    "sell_waves_count": len(result.sell_waves)
                }
            }
        )

        # Store result for potential comprehensive analysis
        self.analysis_results[AnalysisType.WAVE] = result
        return result

    async def _run_ratio_analysis(self, context: Dict[str, Any]) -> RatioResult:
        """Run ratio analysis on the pair."""
        pair = context["fetch_pair_data"]["pair"]
        transactions = context["fetch_pair_data"]["transactions"]

        # Publish analysis started event
        event_bus = self.context.get_component("event_bus")
        await event_bus.publish(
            EventType.ANALYSIS_STARTED,
            {
                "pair_id": self.pair_id,
                "analysis_type": AnalysisType.RATIO,
                "config": self.ratio_config.dict()
            }
        )

        # Create and run analyzer
        analyzer = RatioAnalyzer()
        result = await analyzer.analyze(
            pair=pair,
            transactions=transactions,
            config=self.ratio_config
        )

        # Publish analysis completed event
        await event_bus.publish(
            EventType.ANALYSIS_COMPLETED,
            {
                "pair_id": self.pair_id,
                "analysis_type": AnalysisType.RATIO,
                "result_summary": {
                    "accumulation_count": len(result.accumulation),
                    "distribution_count": len(result.distribution)
                }
            }
        )

        # Store result for potential comprehensive analysis
        self.analysis_results[AnalysisType.RATIO] = result
        return result

    async def _run_pattern_analysis(self, context: Dict[str, Any]) -> PatternResult:
        """Run pattern analysis on the pair."""
        pair = context["fetch_pair_data"]["pair"]
        transactions = context["fetch_pair_data"]["transactions"]

        # Publish analysis started event
        event_bus = self.context.get_component("event_bus")
        await event_bus.publish(
            EventType.ANALYSIS_STARTED,
            {
                "pair_id": self.pair_id,
                "analysis_type": AnalysisType.PATTERN,
                "config": self.pattern_config.dict()
            }
        )

        # Create and run analyzer
        analyzer = PatternAnalyzer()
        result = await analyzer.analyze(
            pair=pair,
            transactions=transactions,
            config=self.pattern_config
        )

        # Publish analysis completed event
        await event_bus.publish(
            EventType.ANALYSIS_COMPLETED,
            {
                "pair_id": self.pair_id,
                "analysis_type": AnalysisType.PATTERN,
                "result_summary": {
                    "patterns_count": len(result.patterns)
                }
            }
        )

        # Store result for potential comprehensive analysis
        self.analysis_results[AnalysisType.PATTERN] = result
        return result

    async def _combine_results(self, context: Dict[str, Any]) -> ComprehensiveResult:
        """Combine all analysis results into a comprehensive result."""
        # Get all individual analysis results
        threshold_result = self.analysis_results.get(AnalysisType.THRESHOLD)
        wave_result = self.analysis_results.get(AnalysisType.WAVE)
        ratio_result = self.analysis_results.get(AnalysisType.RATIO)
        pattern_result = self.analysis_results.get(AnalysisType.PATTERN)

        # Create comprehensive result
        comprehensive_result = ComprehensiveResult(
            pair_id=self.pair_id,
            threshold_analysis=threshold_result,
            wave_analysis=wave_result,
            ratio_analysis=ratio_result,
            pattern_analysis=pattern_result,
            start_time=self.start_time,
            end_time=self.end_time
        )

        # Publish comprehensive analysis completed event
        event_bus = self.context.get_component("event_bus")
        await event_bus.publish(
            EventType.ANALYSIS_COMPLETED,
            {
                "pair_id": self.pair_id,
                "analysis_type": AnalysisType.COMPREHENSIVE,
                "result_summary": {
                    "threshold_transactions": len(threshold_result.significant_transactions) if threshold_result else 0,
                    "buy_waves": len(wave_result.buy_waves) if wave_result else 0,
                    "sell_waves": len(wave_result.sell_waves) if wave_result else 0,
                    "patterns": len(pattern_result.patterns) if pattern_result else 0
                }
            }
        )

        # Store comprehensive result
        self.analysis_results[AnalysisType.COMPREHENSIVE] = comprehensive_result
        return comprehensive_result

    async def get_results(self) -> Dict[str, Any]:
        """Get the analysis results."""
        return self.analysis_results

