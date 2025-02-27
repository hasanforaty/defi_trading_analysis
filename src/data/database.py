# src/data/database.py
from typing import AsyncGenerator, Any, Dict, Optional
import logging
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy import MetaData, inspect
from sqlalchemy.future import select

from config.settings import get_settings
from src.models.entities import Base

logger = logging.getLogger(__name__)


class Database:
    """Database connection management and session factory."""

    def __init__(self) -> None:
        """Initialize database with settings from configuration."""
        self.settings = get_settings().database
        self.engine = create_async_engine(
            self.settings.get_connection_string(),
            echo=self.settings.echo,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
        )
        self.async_session_factory = async_sessionmaker(
            bind=self.engine,
            expire_on_commit=False,
            autoflush=False,
            autocommit=False,
        )
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize database by creating tables."""
        if not self._initialized:
            async with self.engine.begin() as conn:
                # Create tables
                await conn.run_sync(Base.metadata.create_all)
            self._initialized = True
            logger.info("Database tables initialized")

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        """Provide a session context manager."""
        session = self.async_session_factory()
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            logger.error(f"Database error: {str(e)}")
            raise
        finally:
            await session.close()

    async def close(self) -> None:
        """Close database connections."""
        if self.engine:
            await self.engine.dispose()
            logger.info("Database connections closed")


# Global database instance
db = Database()


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for getting a database session."""
    async with db.session() as session:
        yield session
