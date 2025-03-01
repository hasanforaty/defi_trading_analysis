from typing import Optional, Type, TypeVar, Generic, Any, Dict
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger

from src.data.repository import (
    BaseRepository, PairRepository, TransactionRepository,
    WalletAnalysisRepository, WaveRepository
)


class UnitOfWork:
    """
    Implements the Unit of Work pattern for managing transaction boundaries.
    """

    def __init__(self, session: AsyncSession):
        """
        Initialize with a database session.

        Args:
            session: SQLAlchemy AsyncSession
        """
        self.session = session
        self._active = False

    @property
    def pairs(self) -> PairRepository:
        """Get the Pair repository."""
        return PairRepository(self.session)

    @property
    def transactions(self) -> TransactionRepository:
        """Get the Transaction repository."""
        return TransactionRepository(self.session)

    @property
    def wallet_analyses(self) -> WalletAnalysisRepository:
        """Get the WalletAnalysis repository."""
        return WalletAnalysisRepository(self.session)

    @property
    def waves(self) -> WaveRepository:
        """Get the Wave repository."""
        return WaveRepository(self.session)

    async def __aenter__(self):
        """Begin a transaction when entering the context."""
        self._active = True
        self.transaction = await self.session.begin()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """End the transaction when exiting the context."""
        if exc_type is not None:
            # An exception occurred, roll back
            await self.rollback()
            logger.error(f"Transaction rolled back due to error: {exc_val}")
        else:
            # No exception, commit
            await self.commit()

        await self.session.close()
        self._active = False

    async def commit(self):
        """Commit the transaction."""
        if self._active:
            await self.session.commit()
            logger.debug("Transaction committed")

    async def rollback(self):
        """Roll back the transaction."""
        if self._active:
            await self.session.rollback()
            logger.debug("Transaction rolled back")

    async def refresh(self, obj):
        """Refresh the object from the database."""
        await self.session.refresh(obj)


@asynccontextmanager
async def get_unit_of_work(session: AsyncSession):
    """
    Context manager for creating a unit of work.

    Args:
        session: SQLAlchemy AsyncSession

    Yields:
        UnitOfWork: Unit of work instance
    """
    uow = UnitOfWork(session)
    async with uow:
        yield uow
