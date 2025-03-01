from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set, Union

from sqlalchemy import and_, or_, func, desc, asc, case, select, join, outerjoin
from sqlalchemy.sql import Select
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger

from src.models.entities import Pair, Transaction, WalletAnalysis, Wave
from config.settings import ChainType, TransactionType


class QueryOptimizer:
    """
    Specialized query builders for complex analysis operations with optimized SQL.
    """

    def __init__(self, session: AsyncSession):
        """
        Initialize the query optimizer with a database session.

        Args:
            session: SQLAlchemy AsyncSession
        """
        self.session = session

    async def get_wallet_transaction_history(self, wallet_address: str,
                                             limit: int = 100,
                                             include_pairs: bool = True) -> List[Dict[str, Any]]:
        """
        Get transaction history for a wallet with pair details.

        Args:
            wallet_address: Wallet address
            limit: Maximum number of transactions to return
            include_pairs: Whether to include pair details

        Returns:
            List[Dict[str, Any]]: Transaction history
        """
        if include_pairs:
            query = select(
                Transaction.id,
                Transaction.tx_hash,
                Transaction.timestamp,
                Transaction.amount,
                Transaction.price_usd,
                Transaction.transaction_type,
                Pair.address.label('pair_address'),
                Pair.token0_symbol,
                Pair.token1_symbol,
                Pair.chain
            ).join(
                Pair, Transaction.pair_id == Pair.id
            ).where(
                Transaction.wallet_address == wallet_address
            ).order_by(
                desc(Transaction.timestamp)
            ).limit(limit)
        else:
            query = select(
                Transaction
            ).where(
                Transaction.wallet_address == wallet_address
            ).order_by(
                desc(Transaction.timestamp)
            ).limit(limit)

        result = await self.session.execute(query)

        if include_pairs:
            return [dict(row._mapping) for row in result.all()]
        else:
            return [row[0].__dict__ for row in result.all()]

    async def find_active_wallets_by_pair(self, pair_id: int,
                                          min_transactions: int = 3,
                                          days: int = 7) -> List[str]:
        """
        Find active wallet addresses for a specific pair.

        Args:
            pair_id: Pair ID
            min_transactions: Minimum number of transactions
            days: Number of days to look back

        Returns:
            List[str]: List of active wallet addresses
        """
        start_date = datetime.utcnow() - timedelta(days=days)

        query = select(
            Transaction.wallet_address,
            func.count(Transaction.id).label('tx_count')
        ).where(
            Transaction.pair_id == pair_id,
            Transaction.timestamp >= start_date
        ).group_by(
            Transaction.wallet_address
        ).having(
            func.count(Transaction.id) >= min_transactions
        ).order_by(
            desc('tx_count')
        )

        result = await self.session.execute(query)
        return [row[0] for row in result.all()]

    async def calculate_price_impact(self, pair_id: int,
                                     amount_usd: float) -> Optional[float]:
        """
        Calculate estimated price impact for a transaction amount.

        Args:
            pair_id: Pair ID
            amount_usd: Transaction amount in USD

        Returns:
            Optional[float]: Estimated price impact percentage
        """
        # Get pair liquidity
        pair_query = select(Pair.liquidity).where(Pair.id == pair_id)
        pair_result = await self.session.execute(pair_query)
        liquidity = pair_result.scalar()

        if not liquidity or liquidity == 0:
            return None

        # Simple price impact estimation
        impact = (amount_usd / liquidity) * 100
        return impact

    async def get_top_traded_pairs(self, chain: ChainType = None,
                                   days: int = 1,
                                   limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get top traded pairs by volume.

        Args:
            chain: Optional blockchain filter
            days: Number of days to look back
            limit: Maximum number of pairs to return

        Returns:
            List[Dict[str, Any]]: List of top traded pairs with volume
        """
        start_date = datetime.utcnow() - timedelta(days=days)

        query = select(
            Pair.id,
            Pair.address,
            Pair.token0_symbol,
            Pair.token1_symbol,
            Pair.chain,
            Pair.liquidity,
            func.sum(Transaction.amount).label('volume')
        ).join(
            Transaction, Pair.id == Transaction.pair_id
        ).where(
            Transaction.timestamp >= start_date
        )

        if chain:
            query = query.where(Pair.chain == chain)

        query = query.group_by(
            Pair.id,
            Pair.address,
            Pair.token0_symbol,
            Pair.token1_symbol,
            Pair.chain,
            Pair.liquidity
        ).order_by(
            desc('volume')
        ).limit(limit)

        result = await self.session.execute(query)
        return [dict(row._mapping) for row in result.all()]

    async def get_consecutive_transactions(self, pair_id: int,
                                           transaction_type: TransactionType,
                                           min_consecutive: int = 3,
                                           max_minutes_between: int = 15,
                                           limit: int = 100) -> List[List[Transaction]]:
        """
        Find sequences of consecutive transactions of the same type.

        Args:
            pair_id: Pair ID
            transaction_type: Transaction type (BUY or SELL)
            min_consecutive: Minimum number of consecutive transactions
            max_minutes_between: Maximum minutes between transactions
            limit: Maximum number of sequences to return

        Returns:
            List[List[Transaction]]: List of transaction sequences
        """
        # First get all transactions of the specified type for the pair
        query = select(Transaction).where(
            Transaction.pair_id == pair_id,
            Transaction.transaction_type == transaction_type
        ).order_by(Transaction.timestamp)

        result = await self.session.execute(query)
        transactions = result.scalars().all()

        # Group transactions into consecutive sequences
        sequences = []
        current_sequence = []
        max_time_diff = timedelta(minutes=max_minutes_between)

        for i, tx in enumerate(transactions):
            if not current_sequence:
                current_sequence.append(tx)
            else:
                last_tx = current_sequence[-1]
                if tx.timestamp - last_tx.timestamp <= max_time_diff:
                    current_sequence.append(tx)
                else:
                    if len(current_sequence) >= min_consecutive:
                        sequences.append(current_sequence)
                    current_sequence = [tx]

            # Check if we're at the end or if we've found enough sequences
            if i == len(transactions) - 1:
                if len(current_sequence) >= min_consecutive:
                    sequences.append(current_sequence)

            if len(sequences) >= limit:
                break

        return sequences

    async def paginate_results(self, query: Select, page: int = 1,
                               page_size: int = 50) -> Tuple[List[Any], int, int]:
        """
        Paginate query results with count information.

        Args:
            query: SQLAlchemy select query
            page: Page number (1-indexed)
            page_size: Page size

        Returns:
            Tuple[List[Any], int, int]: (items, total_count, total_pages)
        """
        # Get total count
        count_query = select(func.count()).select_from(query.subquery())
        total_count = await self.session.execute(count_query)
        total_count = total_count.scalar() or 0

        # Calculate pagination values
        total_pages = max(1, (total_count + page_size - 1) // page_size)
        page = min(max(1, page), total_pages)
        offset = (page - 1) * page_size

        # Apply pagination to query
        query = query.limit(page_size).offset(offset)

        # Execute query
        result = await self.session.execute(query)
        items = result.all()

        return items, total_count, total_pages
