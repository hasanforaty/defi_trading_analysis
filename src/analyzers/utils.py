# src/analyzers/utils.py
from typing import List, Dict, Optional, Tuple, Union
from datetime import datetime, timedelta
import numpy as np
from loguru import logger
import statistics
from collections import defaultdict

from sqlalchemy.ext.asyncio import AsyncSession

from config.settings import TransactionType
from src.models.entities import Transaction
from sqlalchemy.future import select


def time_window_to_datetime(window_minutes: int, reference_time: Optional[datetime] = None) -> Tuple[
    datetime, datetime]:
    """
    Convert a time window in minutes to start and end datetime.

    Args:
        window_minutes: Length of time window in minutes
        reference_time: Reference time (default: now)

    Returns:
        Tuple of (start_time, end_time)
    """
    if reference_time is None:
        reference_time = datetime.utcnow()

    start_time = reference_time - timedelta(minutes=window_minutes)
    return start_time, reference_time


def calculate_moving_average(values: List[float], window: int = 3) -> List[float]:
    """
    Calculate moving average for a list of values.

    Args:
        values: List of values
        window: Moving average window

    Returns:
        List of moving averages (padded with None at the beginning)
    """
    if len(values) < window:
        return [None] * len(values)

    result = [None] * (window - 1)

    for i in range(len(values) - window + 1):
        window_values = values[i:i + window]
        average = sum(window_values) / window
        result.append(average)

    return result


def calculate_zscore(values: List[float]) -> List[float]:
    """
    Calculate Z-score for a list of values.

    Args:
        values: List of values

    Returns:
        List of Z-scores
    """
    if not values:
        return []

    mean = sum(values) / len(values)
    std_dev = statistics.stdev(values) if len(values) > 1 else 0

    if std_dev == 0:
        return [0] * len(values)

    return [(value - mean) / std_dev for value in values]


def detect_outliers(values: List[float], threshold: float = 2.0) -> List[int]:
    """
    Detect outliers in a list of values using Z-score.

    Args:
        values: List of values
        threshold: Z-score threshold for outliers

    Returns:
        List of indices of outlier values
    """
    zscores = calculate_zscore(values)
    return [i for i, z in enumerate(zscores) if abs(z) > threshold]


def group_transactions_by_time(
        transactions: List[Dict],
        interval_minutes: int = 60
) -> Dict[datetime, List[Dict]]:
    """
    Group transactions by time intervals.

    Args:
        transactions: List of transaction dictionaries
        interval_minutes: Interval size in minutes

    Returns:
        Dictionary mapping interval start time to list of transactions
    """
    if not transactions:
        return {}

    # Sort transactions by timestamp
    sorted_txs = sorted(transactions, key=lambda x: x['timestamp'])

    # Determine time range
    start_time = sorted_txs[0]['timestamp']
    end_time = sorted_txs[-1]['timestamp']

    # Create time intervals
    intervals = {}
    current_time = start_time
    while current_time <= end_time:
        interval_end = current_time + timedelta(minutes=interval_minutes)
        intervals[current_time] = []
        current_time = interval_end

    # Group transactions into intervals
    for tx in sorted_txs:
        # Find the right interval
        for interval_start in sorted(intervals.keys()):
            interval_end = interval_start + timedelta(minutes=interval_minutes)
            if interval_start <= tx['timestamp'] < interval_end:
                intervals[interval_start].append(tx)
                break

    # Remove empty intervals
    return {k: v for k, v in intervals.items() if v}


def identify_price_trends(prices: List[float], window_size: int = 3) -> List[str]:
    """
    Identify price trends from a time series of prices.

    Args:
        prices: List of price values
        window_size: Window size for trend detection

    Returns:
        List of trend indicators ('up', 'down', 'sideways')
    """
    if len(prices) < window_size + 1:
        return ['unknown'] * len(prices)

    # Calculate moving average
    ma = calculate_moving_average(prices, window_size)

    # Initialize trends
    trends = ['unknown'] * len(prices)

    # Identify trends based on price vs moving average
    for i in range(window_size, len(prices)):
        if ma[i] is None:
            trends[i] = 'unknown'
            continue

        # Calculate percentage change
        pct_change = (prices[i] - prices[i - 1]) / prices[i - 1] if prices[i - 1] > 0 else 0

        # Determine trend
        if pct_change > 0.01:  # 1% increase
            trends[i] = 'up'
        elif pct_change < -0.01:  # 1% decrease
            trends[i] = 'down'
        else:
            trends[i] = 'sideways'

    return trends


def interpolate_missing_values(values: List[Optional[float]]) -> List[float]:
    """
    Interpolate missing values in a time series.

    Args:
        values: List of values with possible None entries

    Returns:
        List of values with missing values interpolated
    """
    if not values:
        return []

    result = values.copy()

    # Find first non-None value
    start_idx = 0
    while start_idx < len(result) and result[start_idx] is None:
        start_idx += 1

    # If all values are None, return zeros
    if start_idx == len(result):
        return [0.0] * len(result)

    # Fill leading None values with first non-None value
    for i in range(start_idx):
        result[i] = result[start_idx]

    # Interpolate middle None values
    i = start_idx + 1
    while i < len(result):
        if result[i] is None:
            # Find next non-None value
            j = i + 1
            while j < len(result) and result[j] is None:
                j += 1

            if j < len(result):
                # Linear interpolation
                start_val = result[i - 1]
                end_val = result[j]
                step = (end_val - start_val) / (j - i + 1)

                for k in range(i, j):
                    result[k] = start_val + step * (k - i + 1)
            else:
                # Fill remaining None values with last non-None value
                for k in range(i, len(result)):
                    result[k] = result[i - 1]
                break
        i += 1

    return result


def calculate_volatility(prices: List[float], window: int = None) -> float:
    """
    Calculate price volatility as standard deviation of returns.

    Args:
        prices: List of price values
        window: Optional window size (if None, use all data)

    Returns:
        Volatility value
    """
    if not prices or len(prices) < 2:
        return 0.0

    # Use window if specified and enough data is available
    if window and len(prices) > window:
        prices = prices[-window:]

    # Calculate percentage returns
    returns = [(prices[i] - prices[i - 1]) / prices[i - 1] for i in range(1, len(prices))]

    # Calculate volatility as standard deviation of returns
    if not returns:
        return 0.0

    mean_return = sum(returns) / len(returns)
    variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)

    return (variance ** 0.5) * 100  # Return as percentage


def normalize_data(data: List[float]) -> List[float]:
    """
    Normalize data to range [0, 1].

    Args:
        data: List of numeric values

    Returns:
        Normalized data
    """
    if not data:
        return []

    min_val = min(data)
    max_val = max(data)

    if min_val == max_val:
        return [0.5] * len(data)

    return [(x - min_val) / (max_val - min_val) for x in data]


def calculate_price_momentum(prices: List[float], window: int = 14) -> List[float]:
    """
    Calculate price momentum using rate of change (ROC).

    Args:
        prices: List of price values
        window: ROC period

    Returns:
        List of momentum values
    """
    if len(prices) <= window:
        return [0.0] * len(prices)

    momentum = [0.0] * window

    for i in range(window, len(prices)):
        if prices[i - window] > 0:
            momentum.append((prices[i] / prices[i - window] - 1) * 100)
        else:
            momentum.append(0.0)

    return momentum


def detect_divergence(price_data: List[float], indicator_data: List[float]) -> List[bool]:
    """
    Detect divergence between price and indicator.

    Args:
        price_data: List of price values
        indicator_data: List of indicator values

    Returns:
        List of boolean values indicating divergence
    """
    if len(price_data) != len(indicator_data) or len(price_data) < 2:
        return [False] * max(len(price_data), len(indicator_data))

    divergence = [False] * len(price_data)

    for i in range(1, len(price_data)):
        # Price up, indicator down
        if price_data[i] > price_data[i - 1] and indicator_data[i] < indicator_data[i - 1]:
            divergence[i] = True
        # Price down, indicator up
        elif price_data[i] < price_data[i - 1] and indicator_data[i] > indicator_data[i - 1]:
            divergence[i] = True

    return divergence


def identify_transaction_clusters(
        transactions: List[Dict],
        max_gap_minutes: int = 30,
        min_cluster_size: int = 3
) -> List[List[Dict]]:
    """
    Identify clusters of transactions based on time proximity.

    Args:
        transactions: List of transaction dictionaries
        max_gap_minutes: Maximum gap in minutes between transactions in the same cluster
        min_cluster_size: Minimum number of transactions to form a valid cluster

    Returns:
        List of transaction clusters
    """
    if not transactions:
        return []

    # Sort transactions by timestamp
    sorted_txs = sorted(transactions, key=lambda x: x['timestamp'])

    clusters = []
    current_cluster = [sorted_txs[0]]

    for i in range(1, len(sorted_txs)):
        curr_tx = sorted_txs[i]
        prev_tx = sorted_txs[i - 1]

        # Calculate time difference in minutes
        time_diff = (curr_tx['timestamp'] - prev_tx['timestamp']).total_seconds() / 60

        if time_diff <= max_gap_minutes:
            # Add to current cluster
            current_cluster.append(curr_tx)
        else:
            # End current cluster and start a new one
            if len(current_cluster) >= min_cluster_size:
                clusters.append(current_cluster)
            current_cluster = [curr_tx]

    # Add last cluster if it meets minimum size
    if len(current_cluster) >= min_cluster_size:
        clusters.append(current_cluster)

    return clusters


def memoize_async(func):
    """
    Decorator for memoizing async functions.

    Args:
        func: Async function to memoize

    Returns:
        Memoized async function
    """
    cache = {}

    async def wrapper(*args, **kwargs):
        # Create a hashable key from the arguments
        key = str(args) + str(sorted(kwargs.items()))

        if key not in cache:
            cache[key] = await func(*args, **kwargs)

        return cache[key]

    return wrapper


async def get_time_windows_data(
        session: AsyncSession,
        pair_id: int,
        windows: List[int],  # hours
        end_time: Optional[datetime] = None
) -> Dict[str, Dict]:
    """
    Get transaction data for different time windows.

    Args:
        session: SQLAlchemy session
        pair_id: The ID of the trading pair
        windows: List of time windows in hours
        end_time: Optional end time (default: now)

    Returns:
        Dictionary with data for each time window
    """
    if not end_time:
        end_time = datetime.utcnow()

    result = {}

    for hours in windows:
        window_key = f"{hours}h"
        start_time = end_time - timedelta(hours=hours)

        # Get transactions for this window
        stmt = select(Transaction).where(
            Transaction.pair_id == pair_id,
            Transaction.timestamp.between(start_time, end_time)
        ).order_by(Transaction.timestamp)

        transactions = await session.execute(stmt)
        transactions = transactions.scalars().all()

        # Calculate statistics
        buy_count = sum(1 for tx in transactions if tx.transaction_type == TransactionType.BUY)
        sell_count = sum(1 for tx in transactions if tx.transaction_type == TransactionType.SELL)

        buy_amount = sum(tx.amount for tx in transactions if tx.transaction_type == TransactionType.BUY)
        sell_amount = sum(tx.amount for tx in transactions if tx.transaction_type == TransactionType.SELL)

        # Calculate buy/sell ratio
        buy_ratio = buy_count / len(transactions) if transactions else 0
        amount_ratio = buy_amount / (buy_amount + sell_amount) if (buy_amount + sell_amount) > 0 else 0

        # Get unique wallets
        unique_wallets = len(set(tx.wallet_address for tx in transactions))

        result[window_key] = {
            'transactions': len(transactions),
            'buy_count': buy_count,
            'sell_count': sell_count,
            'buy_amount': buy_amount,
            'sell_amount': sell_amount,
            'buy_ratio': buy_ratio,
            'amount_ratio': amount_ratio,
            'unique_wallets': unique_wallets,
            'window_hours': hours
        }

    return result
