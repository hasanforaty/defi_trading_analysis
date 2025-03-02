# src/analyzers/coordinator.py
from typing import List, Dict, Optional, Tuple, Set, Union, Callable, Any
import asyncio
from datetime import datetime, timedelta
from loguru import logger
from concurrent.futures import ProcessPoolExecutor
from functools import partial

from sqlalchemy.ext.asyncio import AsyncSession

from config.settings import get_settings, TransactionType
from src.models.entities import Pair, Transaction, WalletAnalysis, Wave
from src.data.database import get_db_session
from src.analyzers.threshold import ThresholdAnalyzer
from src.analyzers.wave import WaveDetector
from src.analyzers.ratio import RatioAnalyzer
from src.analyzers.pattern import PatternRecognizer
from src.analyzers.utils import time_window_to_datetime


class AnalysisJob:
    """Represents a single analysis job with status tracking."""

    def __init__(self, job_id: str, job_type: str, pair_id: int, params: Dict = None):
        self.job_id = job_id
        self.job_type = job_type
        self.pair_id = pair_id
        self.params = params or {}
        self.status = "pending"
        self.progress = 0.0
        self.results = None
        self.errors = []
        self.start_time = None
        self.end_time = None

    def start(self):
        """Mark job as started."""
        self.status = "running"
        self.start_time = datetime.utcnow()

    def complete(self, results: Any):
        """Mark job as completed."""
        self.status = "completed"
        self.progress = 100.0
        self.results = results
        self.end_time = datetime.utcnow()

    def fail(self, error: str):
        """Mark job as failed."""
        self.status = "failed"
        self.errors.append(error)
        self.end_time = datetime.utcnow()

    def update_progress(self, progress: float):
        """Update job progress."""
        self.progress = min(max(progress, 0.0), 99.9)  # Cap at 99.9% until complete

    def to_dict(self) -> Dict:
        """Convert job to dictionary."""
        return {
            "job_id": self.job_id,
            "job_type": self.job_type,
            "pair_id": self.pair_id,
            "status": self.status,
            "progress": self.progress,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration": (
                        self.end_time - self.start_time).total_seconds() if self.end_time and self.start_time else None,
            "params": self.params,
            "has_errors": len(self.errors) > 0,
            "error_count": len(self.errors)
        }


class AnalysisCoordinator:
    """
    Orchestrates and coordinates the various analysis algorithms.
    Provides a unified interface for running analysis tasks and tracking their progress.
    """

    def __init__(self, session: AsyncSession = None, max_workers: int = None):
        """Initialize the analysis coordinator with settings and optional session."""
        self.settings = get_settings()
        self._session = session
        self.max_workers = max_workers or self.settings.analysis.max_worker_processes
        self.jobs = {}  # Store active and completed jobs

    async def _get_session(self) -> AsyncSession:
        """Get database session or use provided one."""
        if self._session is not None:
            return self._session

        # Get a new session from the pool
        async for session in get_db_session():
            return session

    async def run_threshold_analysis(
            self,
            pair_id: int,
            start_time: Optional[datetime] = None,
            end_time: Optional[datetime] = None,
            threshold_multiplier: float = 1.0,
            recalculate_threshold: bool = False
    ) -> str:
        """
        Run threshold analysis on a pair.

        Args:
            pair_id: The ID of the trading pair
            start_time: Optional start time for analysis
            end_time: Optional end time for analysis
            threshold_multiplier: Multiplier for threshold
            recalculate_threshold: Whether to force threshold recalculation

        Returns:
            Job ID for tracking
        """
        job_id = f"threshold_{pair_id}_{datetime.utcnow().timestamp()}"
        job = AnalysisJob(job_id, "threshold", pair_id, {
            "start_time": start_time.isoformat() if start_time else None,
            "end_time": end_time.isoformat() if end_time else None,
            "threshold_multiplier": threshold_multiplier,
            "recalculate_threshold": recalculate_threshold
        })
        self.jobs[job_id] = job

        # Start job in background
        asyncio.create_task(self._execute_threshold_analysis(job))

        return job_id

    async def _execute_threshold_analysis(self, job: AnalysisJob):
        """Execute threshold analysis job."""
        session = await self._get_session()
        job.start()

        try:
            analyzer = ThresholdAnalyzer(session)
            pair_id = job.pair_id
            params = job.params

            # Extract parameters
            start_time = datetime.fromisoformat(params["start_time"]) if params.get("start_time") else None
            end_time = datetime.fromisoformat(params["end_time"]) if params.get("end_time") else None
            threshold_multiplier = params.get("threshold_multiplier", 1.0)
            recalculate_threshold = params.get("recalculate_threshold", False)

            # Update progress
            job.update_progress(10.0)

            # Calculate threshold
            threshold = await analyzer.calculate_threshold(pair_id, recalculate_threshold)
            job.update_progress(30.0)

            # Identify significant transactions
            transactions = await analyzer.identify_significant_transactions(
                pair_id, start_time, end_time, threshold_multiplier
            )
            job.update_progress(70.0)

            # Find wallets with multiple threshold transactions
            wallets = await analyzer.find_wallets_with_multiple_threshold_transactions(
                pair_id, min_transaction_count=2, days_lookback=30, threshold_multiplier=threshold_multiplier
            )
            job.update_progress(90.0)

            # Compile results
            results = {
                "threshold": threshold,
                "significant_transactions_count": len(transactions),
                "wallets_with_multiple_transactions": len(wallets),
                "threshold_details": {
                    "value": threshold,
                    "multiplier_applied": threshold_multiplier,
                    "adjusted_value": threshold * threshold_multiplier,
                },
                "wallets": wallets
            }

            job.complete(results)
            logger.info(f"Threshold analysis completed for pair {pair_id}")

        except Exception as e:
            logger.error(f"Error in threshold analysis for pair {job.pair_id}: {str(e)}")
            job.fail(str(e))

    async def run_wave_detection(
            self,
            pair_id: int,
            start_time: Optional[datetime] = None,
            end_time: Optional[datetime] = None,
            min_amount: float = None,
            min_transactions: int = None,
            time_window_minutes: int = None
    ) -> str:
        """
        Run wave detection on a pair.

        Args:
            pair_id: The ID of the trading pair
            start_time: Optional start time for analysis
            end_time: Optional end time for analysis
            min_amount: Minimum amount for wave detection
            min_transactions: Minimum transactions for wave detection
            time_window_minutes: Time window for wave detection

        Returns:
            Job ID for tracking
        """
        job_id = f"wave_{pair_id}_{datetime.utcnow().timestamp()}"
        job = AnalysisJob(job_id, "wave", pair_id, {
            "start_time": start_time.isoformat() if start_time else None,
            "end_time": end_time.isoformat() if end_time else None,
            "min_amount": min_amount,
            "min_transactions": min_transactions,
            "time_window_minutes": time_window_minutes
        })
        self.jobs[job_id] = job

        # Start job in background
        asyncio.create_task(self._execute_wave_detection(job))

        return job_id

    async def _execute_wave_detection(self, job: AnalysisJob):
        """Execute wave detection job."""
        session = await self._get_session()
        job.start()

        try:
            detector = WaveDetector(session)
            pair_id = job.pair_id
            params = job.params

            # Extract parameters
            start_time = datetime.fromisoformat(params["start_time"]) if params.get("start_time") else None
            end_time = datetime.fromisoformat(params["end_time"]) if params.get("end_time") else None
            min_amount = params.get("min_amount")
            min_transactions = params.get("min_transactions")
            time_window_minutes = params.get("time_window_minutes")

            # Update progress
            job.update_progress(10.0)

            # Detect buy waves
            buy_waves = await detector.detect_buy_waves(
                pair_id,
                start_time=start_time,
                end_time=end_time,
                min_amount=min_amount,
                min_transactions=min_transactions,
                time_window_minutes=time_window_minutes
            )
            job.update_progress(50.0)

            # Detect sell waves
            sell_waves = await detector.detect_sell_waves(
                pair_id,
                start_time=start_time,
                end_time=end_time,
                min_amount=min_amount,
                min_transactions=min_transactions,
                time_window_minutes=time_window_minutes
            )
            job.update_progress(90.0)

            # Compile results
            results = {
                "buy_waves": len(buy_waves),
                "sell_waves": len(sell_waves),
                "total_waves": len(buy_waves) + len(sell_waves),
                "buy_wave_details": [
                    {
                        "start_time": wave.start_timestamp.isoformat(),
                        "end_time": wave.end_timestamp.isoformat(),
                        "transaction_count": wave.transaction_count,
                        "total_amount": wave.total_amount,
                        "average_price": wave.average_price
                    } for wave in buy_waves
                ],
                "sell_wave_details": [
                    {
                        "start_time": wave.start_timestamp.isoformat(),
                        "end_time": wave.end_timestamp.isoformat(),
                        "transaction_count": wave.transaction_count,
                        "total_amount": wave.total_amount,
                        "average_price": wave.average_price
                    } for wave in sell_waves
                ]
            }

            job.complete(results)
            logger.info(f"Wave detection completed for pair {pair_id}")

        except Exception as e:
            logger.error(f"Error in wave detection for pair {job.pair_id}: {str(e)}")
            job.fail(str(e))

    async def run_ratio_analysis(
            self,
            pair_id: int,
            min_transaction_count: int = 3,
            min_transaction_amount: float = 1000,
            days_lookback: int = 30
    ) -> str:
        """
        Run ratio analysis on a pair.

        Args:
            pair_id: The ID of the trading pair
            min_transaction_count: Minimum number of transactions required
            min_transaction_amount: Minimum transaction amount required
            days_lookback: Number of days to look back

        Returns:
            Job ID for tracking
        """
        job_id = f"ratio_{pair_id}_{datetime.utcnow().timestamp()}"
        job = AnalysisJob(job_id, "ratio", pair_id, {
            "min_transaction_count": min_transaction_count,
            "min_transaction_amount": min_transaction_amount,
            "days_lookback": days_lookback
        })
        self.jobs[job_id] = job

        # Start job in background
        asyncio.create_task(self._execute_ratio_analysis(job))

        return job_id

    async def _execute_ratio_analysis(self, job: AnalysisJob):
        """Execute ratio analysis job."""
        session = await self._get_session()
        job.start()

        try:
            analyzer = RatioAnalyzer(session)
            pair_id = job.pair_id
            params = job.params

            # Extract parameters
            min_transaction_count = params.get("min_transaction_count", 3)
            min_transaction_amount = params.get("min_transaction_amount", 1000)
            days_lookback = params.get("days_lookback", 30)

            # Set time range
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=days_lookback)

            # Update progress
            job.update_progress(10.0)

            # Find all wallets for the pair
            stmt = "SELECT DISTINCT wallet_address FROM transactions WHERE pair_id = :pair_id AND timestamp BETWEEN :start_time AND :end_time"
            result = await session.execute(stmt, {"pair_id": pair_id, "start_time": start_time, "end_time": end_time})
            wallets = [row[0] for row in result.all()]

            # Analyze each wallet
            results = {
                "all_buys": [],
                "all_sells": [],
                "mostly_buys": [],
                "mostly_sells": [],
                "balanced": [],
                "accumulation": [],
                "distribution": []
            }

            total_wallets = len(wallets)
            for i, wallet in enumerate(wallets):
                # Calculate ratio
                ratio_data = await analyzer.calculate_wallet_ratio(
                    wallet, pair_id, start_time, end_time, min_transaction_count, min_transaction_amount
                )

                # Skip wallets with insufficient data
                if ratio_data.get('insufficient_data', False):
                    continue

                # Store in appropriate category
                pattern = ratio_data.get('pattern')
                if pattern and pattern.value in results:
                    results[pattern.value].append({
                        "wallet_address": wallet,
                        "buy_ratio": ratio_data.get('buy_ratio'),
                        "sell_ratio": ratio_data.get('sell_ratio'),
                        "total_amount": ratio_data.get('total_amount'),
                        "transaction_count": ratio_data.get('total_count')
                    })

                # Track top wallets with significant changes over time
                time_analysis = await analyzer.track_ratio_changes_over_time(wallet, pair_id)
                if time_analysis.get('has_trend'):
                    trend = time_analysis.get('trend')
                    if trend == 'accumulation' and time_analysis.get('trend_strength', 0) > 0.5:
                        results['accumulation'].append({
                            "wallet_address": wallet,
                            "trend_strength": time_analysis.get('trend_strength'),
                            "ratio_changes": time_analysis.get('ratio_changes')
                        })
                    elif trend == 'distribution' and time_analysis.get('trend_strength', 0) > 0.5:
                        results['distribution'].append({
                            "wallet_address": wallet,
                            "trend_strength": time_analysis.get('trend_strength'),
                            "ratio_changes": time_analysis.get('ratio_changes')
                        })

                # Update progress
                progress = 10 + ((i + 1) / total_wallets * 80)
                job.update_progress(progress)

            # Compile summary results
            summary = {
                "total_wallets_analyzed": total_wallets,
                "wallets_with_sufficient_data": sum(len(wallets) for wallets in results.values()),
                "pattern_counts": {pattern: len(wallets) for pattern, wallets in results.items()},
                "patterns": results
            }

            job.complete(summary)
            logger.info(f"Ratio analysis completed for pair {pair_id}")

        except Exception as e:
            logger.error(f"Error in ratio analysis for pair {job.pair_id}: {str(e)}")
            job.fail(str(e))

    async def run_pattern_recognition(
            self,
            pair_id: int,
            start_time: Optional[datetime] = None,
            end_time: Optional[datetime] = None
    ) -> str:
        """
        Run pattern recognition on a pair.

        Args:
            pair_id: The ID of the trading pair
            start_time: Optional start time for analysis
            end_time: Optional end time for analysis

        Returns:
            Job ID for tracking
        """
        job_id = f"pattern_{pair_id}_{datetime.utcnow().timestamp()}"
        job = AnalysisJob(job_id, "pattern", pair_id, {
            "start_time": start_time.isoformat() if start_time else None,
            "end_time": end_time.isoformat() if end_time else None
        })
        self.jobs[job_id] = job

        # Start job in background
        asyncio.create_task(self._execute_pattern_recognition(job))

        return job_id

    async def _execute_pattern_recognition(self, job: AnalysisJob):
        """Execute pattern recognition job."""
        session = await self._get_session()
        job.start()

        try:
            recognizer = PatternRecognizer(session)
            pair_id = job.pair_id
            params = job.params

            # Extract parameters
            start_time = datetime.fromisoformat(params["start_time"]) if params.get("start_time") else None
            end_time = datetime.fromisoformat(params["end_time"]) if params.get("end_time") else None

            # Update progress
            job.update_progress(10.0)

            # Identify patterns
            patterns = await recognizer.identify_patterns(pair_id, start_time, end_time)
            job.update_progress(90.0)

            # Group patterns by type
            patterns_by_type = {}
            for pattern in patterns:
                pattern_type = pattern['type']
                if pattern_type not in patterns_by_type:
                    patterns_by_type[pattern_type] = []
                patterns_by_type[pattern_type].append(pattern)

            # Compile results
            results = {
                "total_patterns": len(patterns),
                "patterns_by_type": {
                    pattern_type: len(patterns_list)
                    for pattern_type, patterns_list in patterns_by_type.items()
                },
                "pattern_details": patterns_by_type
            }

            job.complete(results)
            logger.info(f"Pattern recognition completed for pair {pair_id}")

        except Exception as e:
            logger.error(f"Error in pattern recognition for pair {job.pair_id}: {str(e)}")
            job.fail(str(e))

    async def run_comprehensive_analysis(self, pair_id: int, days_lookback: int = 30) -> str:
        """
        Run all analysis types on a pair.

        Args:
            pair_id: The ID of the trading pair
            days_lookback: Number of days to look back

        Returns:
            Job ID for tracking
        """
        job_id = f"comprehensive_{pair_id}_{datetime.utcnow().timestamp()}"
        job = AnalysisJob(job_id, "comprehensive", pair_id, {
            "days_lookback": days_lookback
        })
        self.jobs[job_id] = job

        # Start job in background
        asyncio.create_task(self._execute_comprehensive_analysis(job))

        return job_id

    async def _execute_comprehensive_analysis(self, job: AnalysisJob):
        """Execute comprehensive analysis job."""
        session = await self._get_session()
        job.start()

        try:
            pair_id = job.pair_id
            days_lookback = job.params.get("days_lookback", 30)

            # Set time range
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=days_lookback)

            # Update progress
            job.update_progress(5.0)

            # Run threshold analysis
            threshold_analyzer = ThresholdAnalyzer(session)
            threshold = await threshold_analyzer.calculate_threshold(pair_id, True)
            significant_txs = await threshold_analyzer.identify_significant_transactions(pair_id, start_time, end_time)
            whale_wallets = await threshold_analyzer.find_wallets_with_multiple_threshold_transactions(pair_id, 2,
                                                                                                       days_lookback)

            job.update_progress(25.0)

            # Run wave detection
            wave_detector = WaveDetector(session)
            buy_waves = await wave_detector.detect_buy_waves(pair_id, start_time, end_time)
            sell_waves = await wave_detector.detect_sell_waves(pair_id, start_time, end_time)

            job.update_progress(50.0)

            # Run ratio analysis
            ratio_analyzer = RatioAnalyzer(session)
            # Get top wallets with transactions
            stmt = """
                SELECT wallet_address 
                FROM transactions 
                WHERE pair_id = :pair_id AND timestamp BETWEEN :start_time AND :end_time 
                GROUP BY wallet_address 
                HAVING COUNT(*) >= 3 
                ORDER BY SUM(amount) DESC 
                LIMIT 100
            """
            result = await session.execute(stmt, {"pair_id": pair_id, "start_time": start_time, "end_time": end_time})
            top_wallets = [row[0] for row in result.all()]

            # Analyze ratios for top wallets
            wallet_ratios = []
            for wallet in top_wallets:
                ratio_data = await ratio_analyzer.calculate_wallet_ratio(
                    wallet, pair_id, start_time, end_time, min_transaction_count=3
                )
                if not ratio_data.get('insufficient_data', False):
                    wallet_ratios.append(ratio_data)

            job.update_progress(75.0)

            # Run pattern recognition
            pattern_recognizer = PatternRecognizer(session)
            patterns = await pattern_recognizer.identify_patterns(pair_id, start_time, end_time)

            job.update_progress(95.0)

            # Compile results
            results = {
                "pair_id": pair_id,
                "analysis_period": {
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat(),
                    "days": days_lookback
                },
                "threshold_analysis": {
                    "threshold": threshold,
                    "significant_transactions": len(significant_txs),
                    "whale_wallets": len(whale_wallets),
                    "top_whales": whale_wallets[:10] if len(whale_wallets) > 10 else whale_wallets
                },
                "wave_analysis": {
                    "buy_waves": len(buy_waves),
                    "sell_waves": len(sell_waves),
                    "largest_buy_wave": max(buy_waves, key=lambda w: w.total_amount).total_amount if buy_waves else 0,
                    "largest_sell_wave": max(sell_waves, key=lambda w: w.total_amount).total_amount if sell_waves else 0
                },
                "ratio_analysis": {
                    "wallets_analyzed": len(wallet_ratios),
                    "buyers": len([w for w in wallet_ratios if w['buy_ratio'] > 0.7]),
                    "sellers": len([w for w in wallet_ratios if w['sell_ratio'] > 0.7]),
                    "balanced_traders": len([w for w in wallet_ratios if 0.4 <= w['buy_ratio'] <= 0.6])
                },
                "pattern_analysis": {
                    "total_patterns": len(patterns),
                    "pattern_types": {pattern['type']: patterns.count(pattern['type']) for pattern in patterns}
                }
            }

            job.complete(results)
            logger.info(f"Comprehensive analysis completed for pair {pair_id}")

        except Exception as e:
            logger.error(f"Error in comprehensive analysis for pair {job.pair_id}: {str(e)}")
            job.fail(str(e))

    async def get_job_status(self, job_id: str) -> Dict:
        """
        Get the status of a job.

        Args:
            job_id: The job ID

        Returns:
            Job status dictionary or None if job not found
        """
        if job_id not in self.jobs:
            return None

        return self.jobs[job_id].to_dict()

    async def get_active_jobs(self) -> List[Dict]:
        """
        Get all active jobs.

        Returns:
            List of active job dictionaries
        """
        return [job.to_dict() for job in self.jobs.values() if job.status == "running"]

    async def get_completed_jobs(self, limit: int = 100) -> List[Dict]:
        """
        Get completed jobs.

        Args:
            limit: Maximum number of jobs to return

        Returns:
            List of completed job dictionaries
        """
        completed = [job.to_dict() for job in self.jobs.values() if job.status == "completed"]
        return sorted(completed, key=lambda j: j["end_time"], reverse=True)[:limit]

    async def get_job_result(self, job_id: str) -> Any:
        """
        Get the result of a completed job.

        Args:
            job_id: The job ID

        Returns:
            Job result or None if job not found or not completed
        """
        if job_id not in self.jobs or self.jobs[job_id].status != "completed":
            return None

        return self.jobs[job_id].results

    async def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a running job.

        Args:
            job_id: The job ID

        Returns:
            True if job was cancelled, False otherwise
        """
        if job_id not in self.jobs or self.jobs[job_id].status != "running":
            return False

        self.jobs[job_id].status = "cancelled"
        self.jobs[job_id].end_time = datetime.utcnow()
        return True
