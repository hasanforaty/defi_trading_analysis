# src/data/integrity.py
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any
from sqlalchemy import select, func, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger

from src.models.entities import Pair, Transaction, WalletAnalysis, Wave


class DataIntegrityService:
    """Utilities for checking and maintaining data integrity."""

    def __init__(self, session: AsyncSession):
        """
        Initialize with a database session.

        Args:
            session: SQLAlchemy AsyncSession
        """
        self.session = session

    async def check_orphaned_transactions(self) -> List[int]:
        """
        Find transactions with invalid pair IDs.

        Returns:
            List[int]: List of orphaned transaction IDs
        """
        # Find transactions with no corresponding pair
        query = select(Transaction.id).outerjoin(
            Pair, Transaction.pair_id == Pair.id
        ).where(Pair.id is None)

        result = await self.session.execute(query)
        return [row[0] for row in result.all()]

    async def check_duplicate_transactions(self) -> List[str]:
        """
        Find duplicate transaction hashes.

        Returns:
            List[str]: List of duplicate transaction hashes
        """
        query = select(
            Transaction.tx_hash,
            func.count(Transaction.id).label('count')
        ).group_by(
            Transaction.tx_hash
        ).having(
            func.count(Transaction.id) > 1
        )

        result = await self.session.execute(query)
        return [row[0] for row in result.all()]

    async def check_inconsistent_wallet_analyses(self) -> List[int]:
        """
        Find wallet analyses with inconsistent transaction counts.

        Returns:
            List[int]: List of inconsistent wallet analysis IDs
        """
        inconsistent_ids = []

        # Get all wallet analyses
        query = select(WalletAnalysis)
        result = await self.session.execute(query)
        wallet_analyses = result.scalars().all()

        for analysis in wallet_analyses:
            # Count actual transactions
            tx_query = select(func.count()).select_from(Transaction).where(
                Transaction.wallet_address == analysis.wallet_address,
                Transaction.pair_id == analysis.pair_id
            )
            tx_result = await self.session.execute(tx_query)
            actual_count = tx_result.scalar() or 0

            # Check if counts match
            if actual_count != analysis.transaction_count:
                inconsistent_ids.append(analysis.id)

        return inconsistent_ids

    async def repair_wallet_analyses(self, wallet_analysis_ids: List[int] = None) -> int:
        """
        Repair inconsistent wallet analyses.

        Args:
            wallet_analysis_ids: Optional list of specific IDs to repair

        Returns:
            int: Number of repaired analyses
        """
        repaired_count = 0

        # Get analyses to repair
        query = select(WalletAnalysis)
        if wallet_analysis_ids:
            query = query.where(WalletAnalysis.id.in_(wallet_analysis_ids))

        result = await self.session.execute(query)
        analyses = result.scalars().all()

        for analysis in analyses:
            # Calculate buy amount
            buy_query = select(func.sum(Transaction.amount)).where(
                Transaction.wallet_address == analysis.wallet_address,
                Transaction.pair_id == analysis.pair_id,
                Transaction.transaction_type == 'BUY'
            )

            buy_result = await self.session.execute(buy_query)
            total_buy = buy_result.scalar() or 0.0

            # Calculate sell amount
            sell_query = select(func.sum(Transaction.amount)).where(
                Transaction.wallet_address == analysis.wallet_address,
                Transaction.pair_id == analysis.pair_id,
                Transaction.transaction_type == 'SELL'
            )

            sell_result = await self.session.execute(sell_query)
            total_sell = sell_result.scalar() or 0.0

            # Count transactions
            count_query = select(func.count()).select_from(Transaction).where(
                Transaction.wallet_address == analysis.wallet_address,
                Transaction.pair_id == analysis.pair_id
            )

            count_result = await self.session.execute(count_query)
            tx_count = count_result.scalar() or 0

            # Calculate ratio
            buy_sell_ratio = None
            if total_sell > 0:
                buy_sell_ratio = total_buy / total_sell

            # Update if different
            if (analysis.total_buy_amount != total_buy or
                    analysis.total_sell_amount != total_sell or
                    analysis.transaction_count != tx_count or
                    analysis.buy_sell_ratio != buy_sell_ratio):
                analysis.total_buy_amount = total_buy
                analysis.total_sell_amount = total_sell
                analysis.transaction_count = tx_count
                analysis.buy_sell_ratio = buy_sell_ratio
                analysis.last_analyzed = datetime.utcnow()

                self.session.add(analysis)
                repaired_count += 1

        await self.session.commit()
        return repaired_count

    async def check_data_completeness(self, pair_id: int,
                                      start_time: datetime,
                                      end_time: datetime) -> Dict[str, Any]:
        """
        Check for potential data gaps in a time range.

        Args:
            pair_id: Pair ID
            start_time: Start of time range
            end_time: End of time range

        Returns:
            Dict[str, Any]: Data completeness statistics
        """
        # Find the longest gap between transactions
        query = select(Transaction.timestamp).where(
            Transaction.pair_id == pair_id,
            Transaction.timestamp >= start_time,
            Transaction.timestamp <= end_time
        ).order_by(Transaction.timestamp)

        result = await self.session.execute(query)
        timestamps = [row[0] for row in result.all()]

        stats = {
            "transaction_count": len(timestamps),
            "time_range_hours": (end_time - start_time).total_seconds() / 3600,
            "average_transactions_per_hour": 0,
            "max_gap_hours": 0,
            "has_potential_gaps": False
        }

        if timestamps:
            # Calculate average transactions per hour
            stats["average_transactions_per_hour"] = (
                    stats["transaction_count"] / stats["time_range_hours"]
            )

            # Find maximum gap
            max_gap = timedelta(seconds=0)

            for i in range(1, len(timestamps)):
                gap = timestamps[i] - timestamps[i - 1]
                if gap > max_gap:
                    max_gap = gap

            stats["max_gap_hours"] = max_gap.total_seconds() / 3600

            # Check if the maximum gap is significantly larger than average
            expected_gap_hours = 1 / stats["average_transactions_per_hour"]
            stats["has_potential_gaps"] = (
                    stats["max_gap_hours"] > expected_gap_hours * 3
            )

        return stats

    async def verify_wave_integrity(self) -> List[int]:
        """
        Verify that waves contain accurate transaction information.

        Returns:
            List[int]: List of wave IDs with integrity issues
        """
        inconsistent_wave_ids = []

        # Get all waves
        wave_query = select(Wave)
        wave_result = await self.session.execute(wave_query)
        waves = wave_result.scalars().all()

        for wave in waves:
            # Count actual transactions in the wave time range
            tx_query = select(func.count(), func.sum(Transaction.amount)).select_from(
                Transaction
            ).where(
                Transaction.pair_id == wave.pair_id,
                Transaction.transaction_type == wave.transaction_type,
                Transaction.timestamp >= wave.start_timestamp,
                Transaction.timestamp <= wave.end_timestamp
            )

            tx_result = await self.session.execute(tx_query)
            actual_count, actual_amount = tx_result.one()
            actual_count = actual_count or 0
            actual_amount = actual_amount or 0.0

            # Check if counts and amounts match
            if (actual_count != wave.transaction_count or
                    abs(actual_amount - wave.total_amount) > 0.01 * wave.total_amount):
                inconsistent_wave_ids.append(wave.id)

        return inconsistent_wave_ids
