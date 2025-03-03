from typing import List, Dict, Optional, Tuple, Set
import asyncio
from datetime import datetime, timedelta
from collections import defaultdict
from loguru import logger

from sqlalchemy import select, and_, or_, func
from sqlalchemy.ext.asyncio import AsyncSession

from config.settings import get_settings, TransactionType
from src.models.entities import Pair, Transaction, Wave
from src.data.database import get_db_session
from src.analyzers.utils import time_window_to_datetime


class WaveAnalyzer:
    """
    Detects buying or selling waves in transaction data.
    A "wave" is defined as a series of transactions from a single wallet within a timeframe.
    """

    def __init__(self, session: AsyncSession = None):
        """Initialize the wave detector with settings and optional session."""
        self.settings = get_settings()
        self._session = session

    async def _get_session(self) -> AsyncSession:
        """Get database session or use provided one."""
        if self._session is not None:
            return self._session

        # Get a new session from the pool
        async for session in get_db_session():
            return session

    async def detect_waves(
            self,
            pair_id: int,
            transaction_type: TransactionType,
            start_time: Optional[datetime] = None,
            end_time: Optional[datetime] = None,
            min_transactions: Optional[int] = None,
            min_total_amount: Optional[float] = None,
            max_time_between_transactions: Optional[int] = None
    ) -> List[Dict]:
        """
        Detect waves of transactions for a pair.

        Args:
            pair_id: ID of the pair to analyze
            transaction_type: Type of transactions to consider (BUY or SELL)
            start_time: Optional start time for analysis window
            end_time: Optional end time for analysis window
            min_transactions: Minimum number of transactions to consider a wave
            min_total_amount: Minimum total USD amount to consider a wave
            max_time_between_transactions: Maximum minutes between transactions in a wave

        Returns:
            List of dictionaries representing detected waves
        """
        session = await self._get_session()

        # Use default settings if parameters not provided
        min_transactions = min_transactions or self.settings.analysis.min_transactions_for_wave
        min_total_amount = min_total_amount or self.settings.analysis.min_wave_total_value_usd
        max_time_between_transactions = max_time_between_transactions or self.settings.analysis.max_time_between_wave_transactions

        # Set time range if not provided
        if not end_time:
            end_time = datetime.utcnow()
        if not start_time:
            start_time = end_time - timedelta(days=7)  # Default to 1 week

        logger.info(f"Detecting {transaction_type.value} waves for pair {pair_id} "
                    f"from {start_time} to {end_time}")

        # Get all transactions of the specified type for the pair in the time range
        stmt = select(Transaction).where(
            Transaction.pair_id == pair_id,
            Transaction.transaction_type == transaction_type,
            Transaction.timestamp.between(start_time, end_time)
        ).order_by(Transaction.timestamp)

        result = await session.execute(stmt)
        transactions = result.scalars().all()

        if not transactions:
            logger.info(f"No {transaction_type.value} transactions found for pair {pair_id} in time range")
            return []

        # Group transactions by wallet
        wallet_transactions = defaultdict(list)
        for tx in transactions:
            wallet_transactions[tx.wallet_address].append(tx)

        # Detect waves for each wallet
        waves = []

        for wallet, txs in wallet_transactions.items():
            if len(txs) < min_transactions:
                continue

            # Sort transactions by timestamp
            txs.sort(key=lambda tx: tx.timestamp)

            current_wave = []
            for tx in txs:
                if not current_wave:
                    # Start a new wave
                    current_wave.append(tx)
                else:
                    # Check if this transaction belongs to the current wave
                    time_diff = (tx.timestamp - current_wave[-1].timestamp).total_seconds() / 60
                    if time_diff <= max_time_between_transactions:
                        current_wave.append(tx)
                    else:
                        # Process completed wave if it meets criteria
                        await self._process_wave(current_wave, wallet, pair_id, transaction_type, waves,
                                                 min_transactions, min_total_amount)
                        # Start a new wave
                        current_wave = [tx]

            # Process the last wave if it exists
            if current_wave:
                await self._process_wave(current_wave, wallet, pair_id, transaction_type, waves,
                                         min_transactions, min_total_amount)

        logger.info(f"Detected {len(waves)} {transaction_type.value} waves for pair {pair_id}")
        return waves

    async def _process_wave(
            self,
            transactions: List[Transaction],
            wallet_address: str,
            pair_id: int,
            transaction_type: TransactionType,
            waves_list: List[Dict],
            min_transactions: int,
            min_total_amount: float
    ) -> None:
        """Process a potential wave and add it to the list if it meets criteria."""
        if len(transactions) < min_transactions:
            return

        # Calculate wave stats
        total_amount = sum(tx.amount for tx in transactions)
        if total_amount < min_total_amount:
            return

        start_timestamp = transactions[0].timestamp
        end_timestamp = transactions[-1].timestamp
        duration_minutes = (end_timestamp - start_timestamp).total_seconds() / 60

        # Calculate average price
        total_price = sum(tx.price_usd for tx in transactions)
        average_price = total_price / len(transactions)

        # Create wave entry
        wave = {
            'wallet_address': wallet_address,
            'pair_id': pair_id,
            'transaction_type': transaction_type,
            'start_timestamp': start_timestamp,
            'end_timestamp': end_timestamp,
            'duration_minutes': duration_minutes,
            'transaction_count': len(transactions),
            'total_amount': total_amount,
            'average_price': average_price,
            'transaction_ids': [tx.id for tx in transactions]
        }

        waves_list.append(wave)

    async def store_wave(self, wave_data: Dict) -> Wave:
        """
        Store a detected wave in the database.

        Args:
            wave_data: Dictionary with wave information

        Returns:
            Created Wave object
        """
        session = await self._get_session()

        # Create Wave object
        wave = Wave(
            pair_id=wave_data['pair_id'],
            start_timestamp=wave_data['start_timestamp'],
            end_timestamp=wave_data['end_timestamp'],
            total_amount=wave_data['total_amount'],
            transaction_count=wave_data['transaction_count'],
            transaction_type=wave_data['transaction_type'],
            average_price=wave_data['average_price']
        )

        session.add(wave)
        await session.commit()
        await session.refresh(wave)

        logger.debug(f"Stored wave {wave.id} in database")
        return wave

    async def calculate_wave_statistics(self, wave_id: int) -> Dict:
        """
        Calculate detailed statistics for a specific wave.

        Args:
            wave_id: ID of the wave

        Returns:
            Dictionary with wave statistics
        """
        session = await self._get_session()

        # Get wave
        wave = await session.get(Wave, wave_id)
        if not wave:
            logger.error(f"Wave with ID {wave_id} not found")
            return {}

        # Get transactions in the wave time period for the pair
        stmt = select(Transaction).where(
            Transaction.pair_id == wave.pair_id,
            Transaction.transaction_type == wave.transaction_type,
            Transaction.timestamp.between(wave.start_timestamp, wave.end_timestamp)
        )

        result = await session.execute(stmt)
        transactions = result.scalars().all()

        # Calculate statistics
        wallet_counts = defaultdict(int)
        for tx in transactions:
            wallet_counts[tx.wallet_address] += 1

        unique_wallets = len(wallet_counts)
        active_wallets = sum(1 for count in wallet_counts.values() if count >= 2)

        # Get price at start and end
        if transactions:
            sorted_txs = sorted(transactions, key=lambda tx: tx.timestamp)
            start_price = sorted_txs[0].price_usd
            end_price = sorted_txs[-1].price_usd
            price_change = ((end_price - start_price) / start_price) * 100
        else:
            start_price = end_price = price_change = 0

        # Create statistics dictionary
        stats = {
            'wave_id': wave_id,
            'total_transactions': len(transactions),
            'unique_wallets': unique_wallets,
            'active_wallets': active_wallets,
            'duration_minutes': (wave.end_timestamp - wave.start_timestamp).total_seconds() / 60,
            'start_price': start_price,
            'end_price': end_price,
            'price_change_pct': price_change,
            'average_transaction_size': wave.total_amount / wave.transaction_count if wave.transaction_count > 0 else 0
        }

        return stats

    async def identify_wallets_with_multiple_waves(
            self,
            pair_id: int,
            min_waves: int = 2,
            days_lookback: int = 30
    ) -> List[Dict]:
        """
        Identify wallets that have participated in multiple waves.

        Args:
            pair_id: ID of the pair
            min_waves: Minimum number of waves to consider
            days_lookback: Number of days to look back

        Returns:
            List of dictionaries with wallet addresses and wave counts
        """
        session = await self._get_session()

        # Define time range
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=days_lookback)

        # Detect BUY waves
        buy_waves = await self.detect_waves(
            pair_id,
            TransactionType.BUY,
            start_time,
            end_time
        )

        # Detect SELL waves
        sell_waves = await self.detect_waves(
            pair_id,
            TransactionType.SELL,
            start_time,
            end_time
        )

        # Combine all waves
        all_waves = buy_waves + sell_waves

        # Count waves by wallet
        wallet_wave_counts = defaultdict(int)
        wallet_wave_details = defaultdict(lambda: {'buy_waves': 0, 'sell_waves': 0, 'total_amount': 0})

        for wave in buy_waves:
            wallet = wave['wallet_address']
            wallet_wave_counts[wallet] += 1
            wallet_wave_details[wallet]['buy_waves'] += 1
            wallet_wave_details[wallet]['total_amount'] += wave['total_amount']

        for wave in sell_waves:
            wallet = wave['wallet_address']
            wallet_wave_counts[wallet] += 1
            wallet_wave_details[wallet]['sell_waves'] += 1
            wallet_wave_details[wallet]['total_amount'] += wave['total_amount']

        # Filter wallets with multiple waves
        result = []
        for wallet, count in wallet_wave_counts.items():
            if count >= min_waves:
                result.append({
                    'wallet_address': wallet,
                    'total_waves': count,
                    'buy_waves': wallet_wave_details[wallet]['buy_waves'],
                    'sell_waves': wallet_wave_details[wallet]['sell_waves'],
                    'total_amount': wallet_wave_details[wallet]['total_amount']
                })

        # Sort by total waves descending
        result.sort(key=lambda x: x['total_waves'], reverse=True)

        logger.info(f"Identified {len(result)} wallets with {min_waves}+ waves for pair {pair_id}")
        return result

