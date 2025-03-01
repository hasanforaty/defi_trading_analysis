from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any, Dict, Generic, List, Optional, Type, TypeVar, Union, Tuple
from sqlalchemy import select, update, delete, and_, or_, func, desc, asc
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import Select
from sqlalchemy.exc import SQLAlchemyError
from loguru import logger

from src.models.entities import Base, Pair, Transaction, WalletAnalysis, Wave
from config.settings import ChainType, TransactionType

# Generic type variable for models
T = TypeVar('T', bound=Base)


class BaseRepository(Generic[T], ABC):
    """
    Abstract base repository implementing common CRUD operations.
    """

    def __init__(self, session: AsyncSession, model_class: Type[T]):
        """
        Initialize the repository with a session and model class.

        Args:
            session: SQLAlchemy AsyncSession
            model_class: SQLAlchemy model class
        """
        self.session = session
        self.model_class = model_class

    async def create(self, **kwargs) -> T:
        """
        Create a new entity.

        Args:
            **kwargs: Entity attributes

        Returns:
            T: Created entity
        """
        entity = self.model_class(**kwargs)
        self.session.add(entity)
        await self.session.flush()
        return entity

    async def get_by_id(self, entity_id: int) -> Optional[T]:
        """
        Get entity by ID.

        Args:
            entity_id: Entity ID

        Returns:
            Optional[T]: Entity if found, None otherwise
        """
        query = select(self.model_class).where(self.model_class.id == entity_id)
        result = await self.session.execute(query)
        return result.scalars().first()

    async def get_all(self, limit: int = None, offset: int = None) -> List[T]:
        """
        Get all entities with optional pagination.

        Args:
            limit: Maximum number of entities to return
            offset: Number of entities to skip

        Returns:
            List[T]: List of entities
        """
        query = select(self.model_class)

        if limit is not None:
            query = query.limit(limit)

        if offset is not None:
            query = query.offset(offset)

        result = await self.session.execute(query)
        return result.scalars().all()

    async def update(self, entity_id: int, **kwargs) -> Optional[T]:
        """
        Update an entity by ID.

        Args:
            entity_id: Entity ID
            **kwargs: Entity attributes to update

        Returns:
            Optional[T]: Updated entity if found, None otherwise
        """
        entity = await self.get_by_id(entity_id)
        if entity:
            for key, value in kwargs.items():
                if hasattr(entity, key):
                    setattr(entity, key, value)

            await self.session.flush()
            return entity
        return None

    async def delete(self, entity_id: int) -> bool:
        """
        Delete an entity by ID.

        Args:
            entity_id: Entity ID

        Returns:
            bool: True if entity was deleted, False otherwise
        """
        entity = await self.get_by_id(entity_id)
        if entity:
            await self.session.delete(entity)
            await self.session.flush()
            return True
        return False

    async def count(self) -> int:
        """
        Count total number of entities.

        Returns:
            int: Entity count
        """
        query = select(func.count()).select_from(self.model_class)
        result = await self.session.execute(query)
        return result.scalar()

    async def exists(self, entity_id: int) -> bool:
        """
        Check if entity with given ID exists.

        Args:
            entity_id: Entity ID

        Returns:
            bool: True if entity exists, False otherwise
        """
        query = select(func.count()).select_from(self.model_class).where(
            self.model_class.id == entity_id
        )
        result = await self.session.execute(query)
        return result.scalar() > 0

    async def bulk_create(self, entities: List[Dict[str, Any]]) -> List[T]:
        """
        Create multiple entities in a single batch.

        Args:
            entities: List of entity attribute dictionaries

        Returns:
            List[T]: List of created entities
        """
        db_entities = [self.model_class(**entity_dict) for entity_dict in entities]
        self.session.add_all(db_entities)
        await self.session.flush()
        return db_entities

    async def bulk_update(self, update_dicts: List[Dict[str, Any]], id_field: str = "id") -> int:
        """
        Update multiple entities in a single batch.

        Args:
            update_dicts: List of dictionaries containing ID and attributes to update
            id_field: Field name for entity ID

        Returns:
            int: Number of updated entities
        """
        update_count = 0
        for update_dict in update_dicts:
            entity_id = update_dict.pop(id_field)
            if await self.update(entity_id, **update_dict):
                update_count += 1
        return update_count


class PairRepository(BaseRepository[Pair]):
    """Repository for Pair model operations."""

    def __init__(self, session: AsyncSession):
        super().__init__(session, Pair)

    async def get_by_address(self, address: str) -> Optional[Pair]:
        """
        Get pair by contract address.

        Args:
            address: Pair contract address

        Returns:
            Optional[Pair]: Pair if found, None otherwise
        """
        query = select(Pair).where(Pair.address == address)
        result = await self.session.execute(query)
        return result.scalars().first()

    async def get_by_chain(self, chain: ChainType, limit: int = 100, offset: int = 0) -> List[Pair]:
        """
        Get pairs by blockchain.

        Args:
            chain: Blockchain type
            limit: Maximum number of pairs to return
            offset: Number of pairs to skip

        Returns:
            List[Pair]: List of pairs
        """
        query = select(Pair).where(Pair.chain == chain).limit(limit).offset(offset)
        result = await self.session.execute(query)
        return result.scalars().all()

    async def get_pairs_by_liquidity(self, min_liquidity: float = None,
                                     max_liquidity: float = None,
                                     chain: ChainType = None,
                                     limit: int = 100) -> List[Pair]:
        """
        Get pairs ordered by liquidity with optional filters.

        Args:
            min_liquidity: Minimum liquidity value
            max_liquidity: Maximum liquidity value
            chain: Optional blockchain filter
            limit: Maximum number of pairs to return

        Returns:
            List[Pair]: List of pairs ordered by liquidity
        """
        query = select(Pair).order_by(desc(Pair.liquidity))

        if min_liquidity is not None:
            query = query.where(Pair.liquidity >= min_liquidity)

        if max_liquidity is not None:
            query = query.where(Pair.liquidity <= max_liquidity)

        if chain is not None:
            query = query.where(Pair.chain == chain)

        query = query.limit(limit)

        result = await self.session.execute(query)
        return result.scalars().all()

    async def update_liquidity(self, pair_id: int, liquidity: float) -> Optional[Pair]:
        """
        Update pair liquidity.

        Args:
            pair_id: Pair ID
            liquidity: New liquidity value

        Returns:
            Optional[Pair]: Updated pair if found, None otherwise
        """
        return await self.update(pair_id, liquidity=liquidity, last_updated=datetime.utcnow())

    async def update_threshold(self, pair_id: int, threshold: float) -> Optional[Pair]:
        """
        Update pair transaction threshold.

        Args:
            pair_id: Pair ID
            threshold: New threshold value

        Returns:
            Optional[Pair]: Updated pair if found, None otherwise
        """
        return await self.update(pair_id, threshold=threshold, last_updated=datetime.utcnow())

    async def get_stale_pairs(self, hours: int = 24) -> List[Pair]:
        """
        Get pairs that haven't been updated in the specified time.

        Args:
            hours: Hours since last update

        Returns:
            List[Pair]: List of stale pairs
        """
        stale_time = datetime.utcnow() - timedelta(hours=hours)
        query = select(Pair).where(Pair.last_updated < stale_time)
        result = await self.session.execute(query)
        return result.scalars().all()


class TransactionRepository(BaseRepository[Transaction]):
    """Repository for Transaction model operations."""

    def __init__(self, session: AsyncSession):
        super().__init__(session, Transaction)

    async def get_by_tx_hash(self, tx_hash: str) -> Optional[Transaction]:
        """
        Get transaction by hash.

        Args:
            tx_hash: Transaction hash

        Returns:
            Optional[Transaction]: Transaction if found, None otherwise
        """
        query = select(Transaction).where(Transaction.tx_hash == tx_hash)
        result = await self.session.execute(query)
        return result.scalars().first()

    async def get_by_wallet(self, wallet_address: str, limit: int = 100) -> List[Transaction]:
        """
        Get transactions by wallet address.

        Args:
            wallet_address: Wallet address
            limit: Maximum number of transactions to return

        Returns:
            List[Transaction]: List of transactions
        """
        query = select(Transaction).where(
            Transaction.wallet_address == wallet_address
        ).order_by(desc(Transaction.timestamp)).limit(limit)

        result = await self.session.execute(query)
        return result.scalars().all()

    async def get_by_pair(self, pair_id: int, limit: int = 100) -> List[Transaction]:
        """
        Get transactions by pair ID.

        Args:
            pair_id: Pair ID
            limit: Maximum number of transactions to return

        Returns:
            List[Transaction]: List of transactions
        """
        query = select(Transaction).where(
            Transaction.pair_id == pair_id
        ).order_by(desc(Transaction.timestamp)).limit(limit)

        result = await self.session.execute(query)
        return result.scalars().all()

    async def get_by_time_range(self, start_time: datetime, end_time: datetime,
                                pair_id: int = None,
                                tx_type: TransactionType = None) -> List[Transaction]:
        """
        Get transactions within a time range with optional filters.

        Args:
            start_time: Start of time range
            end_time: End of time range
            pair_id: Optional pair ID filter
            tx_type: Optional transaction type filter

        Returns:
            List[Transaction]: List of transactions
        """
        query = select(Transaction).where(
            Transaction.timestamp >= start_time,
            Transaction.timestamp <= end_time
        )

        if pair_id is not None:
            query = query.where(Transaction.pair_id == pair_id)

        if tx_type is not None:
            query = query.where(Transaction.transaction_type == tx_type)

        query = query.order_by(Transaction.timestamp)

        result = await self.session.execute(query)
        return result.scalars().all()

    async def get_large_transactions(self, threshold: float, pair_id: int = None,
                                     limit: int = 100) -> List[Transaction]:
        """
        Get transactions exceeding the amount threshold.

        Args:
            threshold: Minimum transaction amount
            pair_id: Optional pair ID filter
            limit: Maximum number of transactions to return

        Returns:
            List[Transaction]: List of large transactions
        """
        query = select(Transaction).where(Transaction.amount >= threshold)

        if pair_id is not None:
            query = query.where(Transaction.pair_id == pair_id)

        query = query.order_by(desc(Transaction.amount)).limit(limit)

        result = await self.session.execute(query)
        return result.scalars().all()

    async def calculate_volume(self, pair_id: int, start_time: datetime,
                               end_time: datetime,
                               tx_type: TransactionType = None) -> float:
        """
        Calculate transaction volume for a pair in a time range.

        Args:
            pair_id: Pair ID
            start_time: Start of time range
            end_time: End of time range
            tx_type: Optional transaction type filter

        Returns:
            float: Total transaction volume
        """
        query = select(func.sum(Transaction.amount)).where(
            Transaction.pair_id == pair_id,
            Transaction.timestamp >= start_time,
            Transaction.timestamp <= end_time
        )

        if tx_type is not None:
            query = query.where(Transaction.transaction_type == tx_type)

        result = await self.session.execute(query)
        return result.scalar() or 0.0

    async def batch_insert(self, transactions: List[Dict[str, Any]]) -> List[Transaction]:
        """
        Insert multiple transactions efficiently.

        Args:
            transactions: List of transaction dictionaries

        Returns:
            List[Transaction]: List of inserted transactions
        """
        return await self.bulk_create(transactions)


class WalletAnalysisRepository(BaseRepository[WalletAnalysis]):
    """Repository for WalletAnalysis model operations."""

    def __init__(self, session: AsyncSession):
        super().__init__(session, WalletAnalysis)

    async def get_by_wallet_and_pair(self, wallet_address: str,
                                     pair_id: int) -> Optional[WalletAnalysis]:
        """
        Get wallet analysis for a specific wallet and pair.

        Args:
            wallet_address: Wallet address
            pair_id: Pair ID

        Returns:
            Optional[WalletAnalysis]: Wallet analysis if found, None otherwise
        """
        query = select(WalletAnalysis).where(
            WalletAnalysis.wallet_address == wallet_address,
            WalletAnalysis.pair_id == pair_id
        )

        result = await self.session.execute(query)
        return result.scalars().first()

    async def get_by_ratio(self, min_ratio: float, pair_id: int = None,
                           min_transaction_count: int = 2,
                           limit: int = 100) -> List[WalletAnalysis]:
        """
        Get wallet analyses with buy/sell ratio above threshold.

        Args:
            min_ratio: Minimum buy/sell ratio
            pair_id: Optional pair ID filter
            min_transaction_count: Minimum number of transactions
            limit: Maximum number of analyses to return

        Returns:
            List[WalletAnalysis]: List of wallet analyses
        """
        query = select(WalletAnalysis).where(
            WalletAnalysis.buy_sell_ratio >= min_ratio,
            WalletAnalysis.transaction_count >= min_transaction_count
        )

        if pair_id is not None:
            query = query.where(WalletAnalysis.pair_id == pair_id)

        query = query.order_by(desc(WalletAnalysis.buy_sell_ratio)).limit(limit)

        result = await self.session.execute(query)
        return result.scalars().all()

    async def update_analysis_from_transactions(self, wallet_address: str,
                                                pair_id: int) -> Optional[WalletAnalysis]:
        """
        Update wallet analysis based on latest transactions.

        Args:
            wallet_address: Wallet address
            pair_id: Pair ID

        Returns:
            Optional[WalletAnalysis]: Updated wallet analysis
        """
        # Get transaction totals
        buy_query = select(func.sum(Transaction.amount)).where(
            Transaction.wallet_address == wallet_address,
            Transaction.pair_id == pair_id,
            Transaction.transaction_type == TransactionType.BUY
        )

        sell_query = select(func.sum(Transaction.amount)).where(
            Transaction.wallet_address == wallet_address,
            Transaction.pair_id == pair_id,
            Transaction.transaction_type == TransactionType.SELL
        )

        count_query = select(func.count()).where(
            Transaction.wallet_address == wallet_address,
            Transaction.pair_id == pair_id
        )

        buy_result = await self.session.execute(buy_query)
        sell_result = await self.session.execute(sell_query)
        count_result = await self.session.execute(count_query)

        total_buy = buy_result.scalar() or 0.0
        total_sell = sell_result.scalar() or 0.0
        tx_count = count_result.scalar() or 0

        # Calculate buy/sell ratio, handling division by zero
        buy_sell_ratio = None
        if total_sell > 0:
            buy_sell_ratio = total_buy / total_sell

        # Update or create wallet analysis
        analysis = await self.get_by_wallet_and_pair(wallet_address, pair_id)

        if analysis:
            return await self.update(
                analysis.id,
                total_buy_amount=total_buy,
                total_sell_amount=total_sell,
                buy_sell_ratio=buy_sell_ratio,
                transaction_count=tx_count,
                last_analyzed=datetime.utcnow()
            )
        else:
            return await self.create(
                wallet_address=wallet_address,
                pair_id=pair_id,
                total_buy_amount=total_buy,
                total_sell_amount=total_sell,
                buy_sell_ratio=buy_sell_ratio,
                transaction_count=tx_count,
                last_analyzed=datetime.utcnow()
            )

    async def get_active_wallets(self, min_transactions: int = 5,
                                 days: int = 7) -> List[WalletAnalysis]:
        """
        Get actively trading wallets with minimum transaction count.

        Args:
            min_transactions: Minimum number of transactions
            days: Number of days to look back

        Returns:
            List[WalletAnalysis]: List of active wallet analyses
        """
        start_date = datetime.utcnow() - timedelta(days=days)

        query = select(WalletAnalysis).where(
            WalletAnalysis.transaction_count >= min_transactions,
            WalletAnalysis.last_analyzed >= start_date
        ).order_by(desc(WalletAnalysis.transaction_count))

        result = await self.session.execute(query)
        return result.scalars().all()


class WaveRepository(BaseRepository[Wave]):
    """Repository for Wave model operations."""

    def __init__(self, session: AsyncSession):
        super().__init__(session, Wave)

    async def get_by_pair(self, pair_id: int,
                          start_time: datetime = None,
                          end_time: datetime = None,
                          wave_type: TransactionType = None,
                          limit: int = 100) -> List[Wave]:
        """
        Get waves for a specific pair with optional filters.

        Args:
            pair_id: Pair ID
            start_time: Optional start time filter
            end_time: Optional end time filter
            wave_type: Optional wave type filter
            limit: Maximum number of waves to return

        Returns:
            List[Wave]: List of waves
        """
        query = select(Wave).where(Wave.pair_id == pair_id)

        if start_time is not None:
            query = query.where(Wave.end_timestamp >= start_time)

        if end_time is not None:
            query = query.where(Wave.start_timestamp <= end_time)

        if wave_type is not None:
            query = query.where(Wave.transaction_type == wave_type)

        query = query.order_by(desc(Wave.start_timestamp)).limit(limit)

        result = await self.session.execute(query)
        return result.scalars().all()

    async def get_by_size(self, min_amount: float,
                          min_transactions: int = None,
                          pair_id: int = None,
                          wave_type: TransactionType = None,
                          limit: int = 100) -> List[Wave]:
        """
        Get waves by minimum total amount and optional filters.

        Args:
            min_amount: Minimum total amount
            min_transactions: Optional minimum transaction count
            pair_id: Optional pair ID filter
            wave_type: Optional wave type filter
            limit: Maximum number of waves to return

        Returns:
            List[Wave]: List of waves
        """
        query = select(Wave).where(Wave.total_amount >= min_amount)

        if min_transactions is not None:
            query = query.where(Wave.transaction_count >= min_transactions)

        if pair_id is not None:
            query = query.where(Wave.pair_id == pair_id)

        if wave_type is not None:
            query = query.where(Wave.transaction_type == wave_type)

        query = query.order_by(desc(Wave.total_amount)).limit(limit)

        result = await self.session.execute(query)
        return result.scalars().all()

    async def get_by_wallet(self, wallet_address: str,
                            limit: int = 100) -> List[Wave]:
        """
        Get waves for a specific wallet.

        Args:
            wallet_address: Wallet address
            limit: Maximum number of waves to return

        Returns:
            List[Wave]: List of waves
        """
        query = select(Wave).where(Wave.wallet_address == wallet_address)
        query = query.order_by(desc(Wave.start_timestamp)).limit(limit)

        result = await self.session.execute(query)
        return result.scalars().all()

    async def get_by_duration(self, min_duration_minutes: int = None,
                              max_duration_minutes: int = None,
                              pair_id: int = None,
                              limit: int = 100) -> List[Wave]:
        """
        Get waves by duration constraints.

        Args:
            min_duration_minutes: Minimum duration in minutes
            max_duration_minutes: Maximum duration in minutes
            pair_id: Optional pair ID filter
            limit: Maximum number of waves to return

        Returns:
            List[Wave]: List of waves
        """
        query = select(Wave)

        if min_duration_minutes is not None:
            min_duration_seconds = min_duration_minutes * 60
            query = query.where(
                func.extract('epoch', Wave.end_timestamp) -
                func.extract('epoch', Wave.start_timestamp) >= min_duration_seconds
            )

        if max_duration_minutes is not None:
            max_duration_seconds = max_duration_minutes * 60
            query = query.where(
                func.extract('epoch', Wave.end_timestamp) -
                func.extract('epoch', Wave.start_timestamp) <= max_duration_seconds
            )

        if pair_id is not None:
            query = query.where(Wave.pair_id == pair_id)

        query = query.limit(limit)

        result = await self.session.execute(query)
        return result.scalars().all()
