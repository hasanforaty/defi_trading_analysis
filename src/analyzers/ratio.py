from typing import List, Dict, Optional, Tuple, Union
import asyncio
from datetime import datetime, timedelta
from enum import Enum
from loguru import logger

from sqlalchemy import select, and_, func, desc
from sqlalchemy.ext.asyncio import AsyncSession

from config.settings import get_settings, TransactionType
from src.models.entities import Pair, Transaction, WalletAnalysis
from src.data.database import get_db_session


class RatioPattern(str, Enum):
    """Patterns of buy/sell ratios."""
    ALL_BUYS = "all_buys"  # 100% buys
    ALL_SELLS = "all_sells"  # 100% sells
    MOSTLY_BUYS = "mostly_buys"  # > 80% buys
    MOSTLY_SELLS = "mostly_sells"  # > 80% sells
    BALANCED = "balanced"  # 40-60% buys
    ACCUMULATION = "accumulation"  # Increasing buy ratio over time
    DISTRIBUTION = "distribution"  # Increasing sell ratio over time


class RatioAnalyzer:
    """
    Analyzes buy/sell ratios for wallets interacting with pairs.
    Identifies wallets with specific ratio patterns and tracks changes over time.
    """

    def __init__(self, session: AsyncSession = None):
        """Initialize the ratio analyzer with settings and optional session."""
        self.settings = get_settings()
        self._session = session

    async def _get_session(self) -> AsyncSession:
        """Get database session or use provided one."""
        if self._session is not None:
            return self._session

        # Get a new session from the pool
        async for session in get_db_session():
            return session

    async def calculate_wallet_ratio(
            self,
            wallet_address: str,
            pair_id: int,
            start_time: Optional[datetime] = None,
            end_time: Optional[datetime] = None,
            min_transaction_count: int = 0,
            min_transaction_amount: float = 0
    ) -> Dict:
        """
        Calculate buy/sell ratios for a wallet on a specific pair.

        Args:
            wallet_address: The wallet address to analyze
            pair_id: The ID of the trading pair
            start_time: Optional start time for filtering transactions
            end_time: Optional end time for filtering transactions
            min_transaction_count: Minimum number of transactions required
            min_transaction_amount: Minimum total transaction amount required

        Returns:
            Dictionary with ratio information
        """
        session = await self._get_session()

        # Set default time range if not provided
        if not end_time:
            end_time = datetime.utcnow()
        if not start_time:
            start_time = end_time - timedelta(days=30)  # Default to 30 days

        # Query buy transactions
        stmt_buys = select(
            func.count().label('count'),
            func.sum(Transaction.amount).label('total_amount')
        ).where(
            Transaction.wallet_address == wallet_address,
            Transaction.pair_id == pair_id,
            Transaction.transaction_type == TransactionType.BUY,
            Transaction.timestamp.between(start_time, end_time)
        )

        # Query sell transactions
        stmt_sells = select(
            func.count().label('count'),
            func.sum(Transaction.amount).label('total_amount')
        ).where(
            Transaction.wallet_address == wallet_address,
            Transaction.pair_id == pair_id,
            Transaction.transaction_type == TransactionType.SELL,
            Transaction.timestamp.between(start_time, end_time)
        )

        # Execute queries
        result_buys = await session.execute(stmt_buys)
        result_sells = await session.execute(stmt_sells)

        buy_count, buy_amount = result_buys.one()
        sell_count, sell_amount = result_sells.one()

        # Handle None values
        buy_count = buy_count or 0
        sell_count = sell_count or 0
        buy_amount = buy_amount or 0
        sell_amount = sell_amount or 0

        # Check minimum requirements
        total_count = buy_count + sell_count
        total_amount = buy_amount + sell_amount

        if total_count < min_transaction_count or total_amount < min_transaction_amount:
            return {
                'wallet_address': wallet_address,
                'pair_id': pair_id,
                'insufficient_data': True,
                'transaction_count': total_count,
                'total_amount': total_amount
            }

        # Calculate ratios
        buy_ratio = buy_count / total_count if total_count > 0 else 0
        sell_ratio = sell_count / total_count if total_count > 0 else 0

        buy_amount_ratio = buy_amount / total_amount if total_amount > 0 else 0
        sell_amount_ratio = sell_amount / total_amount if total_amount > 0 else 0

        # Determine pattern
        pattern = self._determine_ratio_pattern(buy_ratio, buy_amount_ratio, wallet_address, pair_id)

        # Create result
        result = {
            'wallet_address': wallet_address,
            'pair_id': pair_id,
            'buy_count': buy_count,
            'sell_count': sell_count,
            'total_count': total_count,
            'buy_amount': buy_amount,
            'sell_amount': sell_amount,
            'total_amount': total_amount,
            'buy_ratio': buy_ratio,
            'sell_ratio': sell_ratio,
            'buy_amount_ratio': buy_amount_ratio,
            'sell_amount_ratio': sell_amount_ratio,
            'pattern': pattern,
            'start_time': start_time,
            'end_time': end_time
        }

        return result

    def _determine_ratio_pattern(
            self,
            buy_ratio: float,
            buy_amount_ratio: float,
            wallet_address: str,
            pair_id: int
    ) -> RatioPattern:
        """Determine the ratio pattern based on buy/sell ratios."""
        # Average the count ratio and amount ratio for more robust pattern detection
        weighted_buy_ratio = (buy_ratio * 0.4) + (buy_amount_ratio * 0.6)

        if buy_ratio == 1.0:
            return RatioPattern.ALL_BUYS
        elif buy_ratio == 0.0:
            return RatioPattern.ALL_SELLS
        elif weighted_buy_ratio >= 0.8:
            return RatioPattern.MOSTLY_BUYS
        elif weighted_buy_ratio <= 0.2:
            return RatioPattern.MOSTLY_SELLS
        elif 0.4 <= weighted_buy_ratio <= 0.6:
            return RatioPattern.BALANCED

        # For accumulation/distribution patterns, we need historical data
        # This is a placeholder - the actual implementation would require time series analysis
        # which is done in the track_ratio_changes_over_time method
        return RatioPattern.BALANCED

    async def identify_wallets_with_pattern(
            self,
            pair_id: int,
            pattern: RatioPattern,
            min_transaction_count: int = 3,
            min_transaction_amount: float = 1000,
            days_lookback: int = 30
    ) -> List[Dict]:
        """
        Identify wallets exhibiting a specific ratio pattern for a pair.

        Args:
            pair_id: The ID of the trading pair
            pattern: The ratio pattern to look for
            min_transaction_count: Minimum number of transactions required
            min_transaction_amount: Minimum total transaction amount required
            days_lookback: Number of days to look back

        Returns:
            List of wallet analysis results matching the pattern
        """
        session = await self._get_session()

        # Set time range
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=days_lookback)

        # First, get all wallets that interacted with this pair
        stmt = select(Transaction.wallet_address).distinct().where(
            Transaction.pair_id == pair_id,
            Transaction.timestamp.between(start_time, end_time)
        )

        result = await session.execute(stmt)
        wallet_addresses = result.scalars().all()

        # Analyze each wallet
        matching_wallets = []
        for wallet_address in wallet_addresses:
            ratio_data = await self.calculate_wallet_ratio(
                wallet_address,
                pair_id,
                start_time,
                end_time,
                min_transaction_count,
                min_transaction_amount
            )

            # Skip wallets with insufficient data
            if ratio_data.get('insufficient_data', False):
                continue

            # Check if wallet matches the requested pattern
            if ratio_data['pattern'] == pattern:
                matching_wallets.append(ratio_data)

        logger.info(f"Found {len(matching_wallets)} wallets with {pattern.value} pattern for pair {pair_id}")
        return matching_wallets

    async def store_ratio_analysis(self, ratio_data: Dict) -> WalletAnalysis:
        """
        Store ratio analysis results in the WalletAnalysis model.

        Args:
            ratio_data: Dictionary with ratio analysis data

        Returns:
            Created or updated WalletAnalysis object
        """
        session = await self._get_session()

        # Check if analysis already exists
        stmt = select(WalletAnalysis).where(
            WalletAnalysis.wallet_address == ratio_data['wallet_address'],
            WalletAnalysis.pair_id == ratio_data['pair_id']
        )

        result = await session.execute(stmt)
        analysis = result.scalar_one_or_none()

        if not analysis:
            # Create new analysis
            analysis = WalletAnalysis(
                wallet_address=ratio_data['wallet_address'],
                pair_id=ratio_data['pair_id'],
                total_buy_amount=ratio_data['buy_amount'],
                total_sell_amount=ratio_data['sell_amount'],
                buy_sell_ratio=ratio_data['buy_ratio'] / ratio_data['sell_ratio'] if ratio_data[
                                                                                         'sell_ratio'] > 0 else float(
                    'inf'),
                transaction_count=ratio_data['total_count'],
                last_analyzed=datetime.utcnow(),
                notes=f"Pattern: {ratio_data['pattern'].value}"
            )
            session.add(analysis)
        else:
            # Update existing analysis
            analysis.total_buy_amount = ratio_data['buy_amount']
            analysis.total_sell_amount = ratio_data['sell_amount']
            analysis.buy_sell_ratio = ratio_data['buy_ratio'] / ratio_data['sell_ratio'] if ratio_data[
                                                                                                'sell_ratio'] > 0 else float(
                'inf')
            analysis.transaction_count = ratio_data['total_count']
            analysis.last_analyzed = datetime.utcnow()
            analysis.notes = f"Pattern: {ratio_data['pattern'].value}"

        await session.commit()
        await session.refresh(analysis)

        logger.debug(f"Stored ratio analysis for wallet {ratio_data['wallet_address']} on pair {ratio_data['pair_id']}")
        return analysis

    async def track_ratio_changes_over_time(
            self,
            wallet_address: str,
            pair_id: int,
            time_windows: List[int] = [1, 7, 30, 90]  # days
    ) -> Dict:
        """
        Track how a wallet's buy/sell ratio changes over different time periods.

        Args:
            wallet_address: The wallet address to analyze
            pair_id: The ID of the trading pair
            time_windows: List of time windows in days to analyze

        Returns:
            Dictionary with ratio data for each time window
        """
        end_time = datetime.utcnow()
        results = {}

        for days in time_windows:
            start_time = end_time - timedelta(days=days)
            window_key = f"{days}d"

            ratio_data = await self.calculate_wallet_ratio(
                wallet_address,
                pair_id,
                start_time,
                end_time
            )

            # Skip windows with insufficient data
            if ratio_data.get('insufficient_data', False):
                results[window_key] = {
                    'insufficient_data': True,
                    'days': days
                }
                continue

            results[window_key] = {
                'buy_ratio': ratio_data['buy_ratio'],
                'sell_ratio': ratio_data['sell_ratio'],
                'buy_amount_ratio': ratio_data['buy_amount_ratio'],
                'sell_amount_ratio': ratio_data['sell_amount_ratio'],
                'pattern': ratio_data['pattern'].value,
                'days': days
            }

        # Determine trend by comparing oldest and newest windows with data
        valid_windows = [w for w in time_windows if not results.get(f"{w}d", {}).get('insufficient_data', False)]

        if len(valid_windows) >= 2:
            oldest = f"{valid_windows[-1]}d"
            newest = f"{valid_windows[0]}d"

            buy_ratio_change = results[newest]['buy_ratio'] - results[oldest]['buy_ratio']

            if buy_ratio_change > 0.2:
                trend = RatioPattern.ACCUMULATION
            elif buy_ratio_change < -0.2:
                trend = RatioPattern.DISTRIBUTION
            else:
                trend = RatioPattern.BALANCED

            results['trend'] = trend.value
            results['buy_ratio_change'] = buy_ratio_change

        return {
            'wallet_address': wallet_address,
            'pair_id': pair_id,
            'time_windows': results
        }