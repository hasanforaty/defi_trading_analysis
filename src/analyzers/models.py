# src/analyzers/models.py
from typing import List, Dict, Optional, Set, Union, Tuple
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, validator

from config.settings import TransactionType


class AnalysisType(str, Enum):
    """Types of analysis performed."""
    THRESHOLD = "threshold"
    WAVE = "wave"
    RATIO = "ratio"
    PATTERN = "pattern"
    COMPREHENSIVE = "comprehensive"


class AnalysisTimeWindow(str, Enum):
    """Standard time windows for analysis."""
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    HOUR_12 = "12h"
    DAY_1 = "1d"
    DAY_3 = "3d"
    WEEK_1 = "1w"
    MONTH_1 = "1m"


class AnalysisStatus(str, Enum):
    """Analysis job status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class PatternType(str, Enum):
    """Common trading patterns."""
    ACCUMULATION = "accumulation"
    DISTRIBUTION = "distribution"
    PUMP_AND_DUMP = "pump_and_dump"
    BREAKOUT = "breakout"
    REVERSAL = "reversal"
    CONSOLIDATION = "consolidation"
    WHALE_MANIPULATION = "whale_manipulation"
    COORDINATED_BUYS = "coordinated_buys"
    COORDINATED_SELLS = "coordinated_sells"
    LIQUIDITY_GRAB = "liquidity_grab"


class ThresholdConfig(BaseModel):
    """Configuration for threshold analysis."""
    threshold_multiplier: float = 1.0
    recalculate_threshold: bool = False
    min_transaction_amount: float = 500.0
    min_transaction_count: int = 2
    days_lookback: int = 30


class WaveConfig(BaseModel):
    """Configuration for wave detection."""
    min_amount: float = 10000.0
    min_transactions: int = 5
    time_window_minutes: int = 60
    include_wallet_details: bool = True
    min_price_impact: float = 1.0  # Minimum price impact in percentage


class RatioConfig(BaseModel):
    """Configuration for ratio analysis."""
    min_transaction_count: int = 3
    min_transaction_amount: float = 1000.0
    days_lookback: int = 30
    time_windows: List[int] = [1, 7, 30, 90]  # Days


class PatternConfig(BaseModel):
    """Configuration for pattern recognition."""
    time_window_hours: int = 24
    min_confidence: float = 0.7
    min_pattern_strength: float = 0.5
    include_price_data: bool = True
    pattern_types: Optional[List[PatternType]] = None


class AnalysisRequestModel(BaseModel):
    """Base model for analysis requests."""
    pair_id: int
    analysis_type: AnalysisType
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    window: Optional[AnalysisTimeWindow] = None

    # Optional configurations for different analysis types
    threshold_config: Optional[ThresholdConfig] = None
    wave_config: Optional[WaveConfig] = None
    ratio_config: Optional[RatioConfig] = None
    pattern_config: Optional[PatternConfig] = None


class TransactionStats(BaseModel):
    """Statistics for a set of transactions."""
    count: int
    unique_wallets: int
    total_amount: float
    average_amount: float
    min_amount: float
    max_amount: float
    median_amount: float
    first_timestamp: datetime
    last_timestamp: datetime
    buy_count: int = 0
    sell_count: int = 0
    buy_amount: float = 0.0
    sell_amount: float = 0.0
    buy_ratio: float = 0.0

    @validator('buy_ratio', pre=True, always=True)
    def calculate_buy_ratio(cls, v, values):
        if 'buy_count' in values and 'count' in values and values['count'] > 0:
            return values['buy_count'] / values['count']
        return 0.0


class WalletSummary(BaseModel):
    """Summary of a wallet's activity."""
    wallet_address: str
    transaction_count: int
    first_transaction: datetime
    last_transaction: datetime
    total_amount: float
    buy_count: int
    sell_count: int
    buy_amount: float
    sell_amount: float
    largest_transaction: float


class ThresholdResult(BaseModel):
    """Result of threshold analysis."""
    pair_id: int
    threshold: float
    significant_transactions: List[Dict]
    whale_wallets: List[Dict]
    start_time: datetime
    end_time: datetime
    config: ThresholdConfig


class WaveResult(BaseModel):
    """Result of wave detection."""
    pair_id: int
    buy_waves: List[Dict]
    sell_waves: List[Dict]
    start_time: datetime
    end_time: datetime
    config: WaveConfig


class RatioResult(BaseModel):
    """Result of ratio analysis."""
    pair_id: int
    all_buys: List[Dict]
    all_sells: List[Dict]
    mostly_buys: List[Dict]
    mostly_sells: List[Dict]
    balanced: List[Dict]
    accumulation: List[Dict]
    distribution: List[Dict]
    start_time: datetime
    end_time: datetime
    config: RatioConfig


class PatternResult(BaseModel):
    """Result of pattern recognition."""
    pair_id: int
    patterns: List[Dict]
    start_time: datetime
    end_time: datetime
    config: PatternConfig


class ComprehensiveResult(BaseModel):
    """Result of comprehensive analysis."""
    pair_id: int
    threshold_analysis: ThresholdResult
    wave_analysis: WaveResult
    ratio_analysis: RatioResult
    pattern_analysis: PatternResult
    start_time: datetime
    end_time: datetime
