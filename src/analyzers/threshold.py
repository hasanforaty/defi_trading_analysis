from typing import List, Dict, Optional, Tuple
import asyncio
from datetime import datetime, timedelta
from loguru import logger

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from config.settings import get_settings
from src.models.entities import Pair, Transaction, WalletAnalysis
from src.data.database import get_db_session


class ThresholdAnalyzer:
    """
    Analyzes transactions to identify significant ones based on threshold values.
    Thresholds can be absolute (fixed USD amount) or relative (% of liquidity).
    """

    def __init__(self, session: AsyncSession = None):
        """Initialize the threshold analyzer with settings and optional session."""
        self.settings = get_settings()
        self._session = session
        self.thresholds_cache = {}  # Cache for pair thresholds

    async def _get_session(self) -> AsyncSession:
        """Get database session or use provided one."""
        if self._session is not None:
            return self._session

        # Get a new session from the pool
        async for session in get_db_session():
            return session

    async def calculate_threshold(self, pair_id: int, recalculate: bool = False) -> float:
        """
        Calculate or retrieve the appropriate threshold value for a pair based on its liquidity.

        Args:
            pair_id: The ID of the trading pair
            recalculate: Force recalculation even if cached

        Returns:
            Calculated threshold value in USD
        """
        # Check cache first if not forcing recalculation
        if not recalculate and pair_id in self.thresholds_cache:
            return self.thresholds_cache[pair_id]

        session = await self._get_session()

        # Get the pair data
        pair = await session.get(Pair, pair_id)
        if not pair:
            logger.error(f"Pair with ID {pair_id} not found")
            return self.settings.analysis.significant_transaction_threshold_usd

        # If pair has no liquidity, use default threshold
        if not pair.liquidity or pair.liquidity <= 0:
            # Use transaction volume to estimate appropriate threshold
            stmt = select(func.sum(Transaction.amount)).where(
                Transaction.pair_id == pair_id
            )
            result = await session.execute(stmt)
            total_volume = result.scalar_one_or_none() or 0

            # Get transaction count
            stmt = select(func.count()).where(
                Transaction.pair_id == pair_id
            )
            result = await session.execute(stmt)
            tx_count = result.scalar_one_or_none() or 0

            if tx_count > 0:
                # Use 5% of average transaction as threshold, but minimum of 500 USD
                avg_tx_size = total_volume / tx_count
                threshold = max(avg_tx_size * 0.05, self.settings.analysis.min_transaction_value_usd)
            else:
                threshold = self.settings.analysis.significant_transaction_threshold_usd
        else:
            # Calculate the threshold based on liquidity
            # For high liquidity pairs (>$1M), use 0.5% of liquidity
            # For medium liquidity pairs ($100k-$1M), use 1% of liquidity
            # For low liquidity pairs (<$100k), use 2% of liquidity
            if pair.liquidity >= 1_000_000:
                threshold = pair.liquidity * 0.005  # 0.5% of liquidity
            elif pair.liquidity >= 100_000:
                threshold = pair.liquidity * 0.01  # 1% of liquidity
            else:
                threshold = pair.liquidity * 0.02  # 2% of liquidity

            # Ensure the threshold is at least the minimum transaction value
            threshold = max(threshold, self.settings.analysis.min_transaction_value_usd)

            # Update the pair's threshold in the database
            pair.threshold = threshold
            await session.commit()

        # Cache the result
        self.thresholds_cache[pair_id] = threshold

        logger.debug(f"Calculated threshold for pair {pair_id}: ${threshold:.2f}")
        return threshold

    async def identify_significant_transactions(
            self,
            pair_id: int,
            start_time: Optional[datetime] = None,
            end_time: Optional[datetime] = None,
            threshold_multiplier: float = 1.0
    ) -> List[Transaction]:
        """
        Identify transactions that exceed the threshold for a given pair.

        Args:
            pair_id: The ID of the trading pair
            start_time: Optional start time for filtering transactions
            end_time: Optional end time for filtering transactions
            threshold_multiplier: Multiplier to adjust threshold (e.g., 2.0 for whale transactions)

        Returns:
            List of transactions exceeding the threshold
        """
        session = await self._get_session()

        # Get or calculate threshold
        threshold = await self.calculate_threshold(pair_id)
        adjusted_threshold = threshold * threshold_multiplier

        # Build query
        query = select(Transaction).where(
            Transaction.pair_id == pair_id,
            Transaction.amount >= adjusted_threshold
        )

        # Add time filters if provided
        if start_time:
            query = query.where(Transaction.timestamp >= start_time)
        if end_time:
            query = query.where(Transaction.timestamp <= end_time)

        # Execute query
        result = await session.execute(query)
        transactions = result.scalars().all()

        logger.info(f"Found {len(transactions)} significant transactions for pair {pair_id} "
                    f"with threshold ${adjusted_threshold:.2f}")

        return transactions

    async def update_thresholds_for_changing_liquidity(self) -> Dict[int, float]:
        """
        Update threshold values for pairs with changing liquidity.

        Returns:
            Dictionary mapping pair_id to new threshold value
        """
        session = await self._get_session()

        # Get all pairs
        result = await session.execute(select(Pair))
        pairs = result.scalars().all()

        updated_thresholds = {}
        for pair in pairs:
            old_threshold = pair.threshold
            new_threshold = await self.calculate_threshold(pair.id, recalculate=True)

            if old_threshold is None or abs(old_threshold - new_threshold) / max(old_threshold, 1) > 0.1:
                # Threshold changed by more than 10%, update it
                pair.threshold = new_threshold
                updated_thresholds[pair.id] = new_threshold
                logger.info(f"Updated threshold for pair {pair.id} from "
                            f"${old_threshold or 'None'} to ${new_threshold:.2f}")

        await session.commit()
        return updated_thresholds

    async def find_wallets_with_multiple_threshold_transactions(
            self,
            pair_id: int,
            min_transaction_count: int = 2,
            days_lookback: int = 7,
            threshold_multiplier: float = 1.0
    ) -> List[Dict]:
        """
        Find wallets with multiple transactions exceeding the threshold for a pair.

        Args:
            pair_id: The ID of the trading pair
            min_transaction_count: Minimum number of threshold-exceeding transactions
            days_lookback: Number of days to look back for transactions
            threshold_multiplier: Multiplier to adjust threshold

        Returns:
            List of dictionaries with wallet address and transaction counts
        """
        session = await self._get_session()

        # Calculate time range
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=days_lookback)

        # Get threshold
        threshold = await self.calculate_threshold(pair_id)
        adjusted_threshold = threshold * threshold_multiplier

        # Query for wallets with multiple significant transactions
        stmt = select(
            Transaction.wallet_address,
            func.count().label('tx_count'),
            func.sum(Transaction.amount).label('total_amount')
        ).where(
            Transaction.pair_id == pair_id,
            Transaction.amount >= adjusted_threshold,
            Transaction.timestamp.between(start_time, end_time)
        ).group_by(
            Transaction.wallet_address
        ).having(
            func.count() >= min_transaction_count
        ).order_by(
            func.count().desc()
        )

        result = await session.execute(stmt)
        wallet_stats = [
            {
                'wallet_address': row[0],
                'transaction_count': row[1],
                'total_amount': row[2]
            }
            for row in result.all()
        ]

        logger.info(f"Found {len(wallet_stats)} wallets with {min_transaction_count}+ "
                    f"significant transactions for pair {pair_id}")

        return wallet_stats

    async def is_transaction_significant(self, transaction_id: int) -> bool:
        """
        Check if a specific transaction exceeds the threshold.

        Args:
            transaction_id: The ID of the transaction to check

        Returns:
            True if the transaction is significant, False otherwise
        """
        session = await self._get_session()

        # Get transaction
        transaction = await session.get(Transaction, transaction_id)
        if not transaction:
            logger.error(f"Transaction with ID {transaction_id} not found")
            return False

        # Get threshold for pair
        threshold = await self.calculate_threshold(transaction.pair_id)

        # Check if transaction exceeds threshold
        return transaction.amount >= threshold

