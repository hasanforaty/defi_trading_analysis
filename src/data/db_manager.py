from typing import AsyncGenerator, Dict, Optional, Any, List, Tuple
import time
import asyncio
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession
from sqlalchemy.sql import text
from sqlalchemy.exc import SQLAlchemyError, DBAPIError
from loguru import logger

from src.data.database import Database
from config.settings import get_settings


class DatabaseManager(Database):
    """
    Enhanced database manager extending the basic Database class with additional
    connection management features, health checks, and advanced session handling.
    """

    def __init__(self, pool_size: int = None, max_overflow: int = None,
                 pool_timeout: int = None, pool_recycle: int = None) -> None:
        """
        Initialize the database manager with configurable connection pool parameters.

        Args:
            pool_size: Maximum number of connections to keep open
            max_overflow: Maximum number of connections above pool_size
            pool_timeout: Seconds to wait for a connection before timing out
            pool_recycle: Seconds after which a connection is recycled
        """
        # Initialize parent class
        super().__init__()

        # Override engine with customized pool settings
        settings = get_settings().database
        pool_settings = {
            "pool_size": pool_size or 5,
            "max_overflow": max_overflow or 10,
            "pool_timeout": pool_timeout or 30,
            "pool_recycle": pool_recycle or 1800,  # 30 minutes
            "pool_pre_ping": True,
            "echo": settings.echo
        }

        # Create engine with custom pool settings
        self.engine = AsyncEngine.create(
            settings.get_connection_string(),
            **pool_settings
        )

        # Track connection issues
        self._connection_failures = 0
        self._last_health_check = 0
        self._reconnect_timeout = 5  # seconds
        self._health_check_interval = 60  # seconds
        self._max_connection_failures = 3

        logger.info(f"DatabaseManager initialized with pool_size={pool_settings['pool_size']}, "
                    f"max_overflow={pool_settings['max_overflow']}")

    @asynccontextmanager
    async def transaction_session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Provide a session context manager with explicit transaction management.
        Use this when you need explicit transaction control.

        Yields:
            AsyncSession: SQLAlchemy async session
        """
        session = self.async_session_factory()
        try:
            async with session.begin():
                yield session
            # Transaction is automatically committed if no exceptions
        except Exception as e:
            # Transaction is automatically rolled back on exception
            logger.error(f"Transaction error: {str(e)}")
            raise
        finally:
            await session.close()

    @asynccontextmanager
    async def session_local_transaction(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Provide a session with transaction control at individual operation level.
        This allows more granular control within the session.

        Yields:
            AsyncSession: SQLAlchemy async session
        """
        session = self.async_session_factory()
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            logger.error(f"Session error: {str(e)}")
            raise
        finally:
            await session.close()

    async def health_check(self) -> Tuple[bool, Optional[str]]:
        """
        Perform a health check on the database connection.

        Returns:
            Tuple[bool, Optional[str]]: (is_healthy, error_message)
        """
        current_time = time.time()

        # Don't check too frequently
        if current_time - self._last_health_check < self._health_check_interval:
            return True, None

        self._last_health_check = current_time

        try:
            async with self.session() as session:
                # Simple query to check database connection
                result = await session.execute(text("SELECT 1"))
                await result.fetchone()

                # Reset connection failure counter on success
                self._connection_failures = 0
                return True, None

        except SQLAlchemyError as e:
            self._connection_failures += 1
            error_msg = f"Database health check failed: {str(e)}"
            logger.error(error_msg)
            return False, error_msg

    async def reconnect(self) -> bool:
        """
        Attempt to reconnect to the database after connection failures.

        Returns:
            bool: True if reconnection was successful, False otherwise
        """
        if self._connection_failures < self._max_connection_failures:
            logger.info(f"Attempting to reconnect to database (attempt {self._connection_failures})")

            try:
                # Dispose the current engine
                await self.engine.dispose()

                # Wait before reconnecting
                await asyncio.sleep(self._reconnect_timeout)

                # Create a new engine
                settings = get_settings().database
                self.engine = AsyncEngine.create(
                    settings.get_connection_string(),
                    pool_pre_ping=True,
                    pool_size=5,
                    max_overflow=10,
                )

                # Test the connection
                is_healthy, _ = await self.health_check()
                if is_healthy:
                    logger.info("Database reconnection successful")
                    self._connection_failures = 0
                    return True

            except Exception as e:
                logger.error(f"Reconnection attempt failed: {str(e)}")

        logger.critical(f"Maximum database reconnection attempts reached ({self._max_connection_failures})")
        return False

    async def initialize_db(self, drop_all: bool = False) -> None:
        """
        Initialize database schema with option to drop existing tables.

        Args:
            drop_all: If True, drops all existing tables before creating new ones
        """
        try:
            async with self.engine.begin() as conn:
                if drop_all:
                    logger.warning("Dropping all database tables")
                    await conn.run_sync(lambda sync_conn: sync_conn.execute(text("SET FOREIGN_KEY_CHECKS=0")))
                    await conn.run_sync(lambda sync_conn: self.metadata.drop_all(sync_conn))
                    await conn.run_sync(lambda sync_conn: sync_conn.execute(text("SET FOREIGN_KEY_CHECKS=1")))

                # Create tables
                logger.info("Creating database tables")
                await conn.run_sync(lambda sync_conn: self.metadata.create_all(sync_conn))

            self._initialized = True
            logger.info("Database schema initialization complete")

        except SQLAlchemyError as e:
            logger.error(f"Database initialization error: {str(e)}")
            raise

    async def get_connection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the database connection pool.

        Returns:
            Dict[str, Any]: Connection pool statistics
        """
        stats = {
            "pool_size": self.engine.pool.size(),
            "checkedin_connections": self.engine.pool.checkedin(),
            "checkedout_connections": self.engine.pool.checkedout(),
            "overflow": self.engine.pool.overflow(),
            "connection_failures": self._connection_failures,
            "last_health_check": self._last_health_check
        }
        return stats

    async def execute_raw_query(self, query: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Execute a raw SQL query directly.
        Use with caution - primarily for diagnostics and admin operations.

        Args:
            query: SQL query string
            params: Query parameters

        Returns:
            List[Dict[str, Any]]: Query results as list of dictionaries
        """
        async with self.session() as session:
            result = await session.execute(text(query), params or {})
            if result.returns_rows:
                column_names = result.keys()
                rows = result.fetchall()
                return [dict(zip(column_names, row)) for row in rows]
            return []

    async def vacuum_analyze(self, table_name: Optional[str] = None) -> None:
        """
        Run VACUUM ANALYZE to reclaim storage and update statistics.

        Args:
            table_name: Optional specific table name to vacuum
        """
        query = f"VACUUM ANALYZE {table_name or ''}"
        try:
            async with self.engine.begin() as conn:
                await conn.execute(text(query))
            logger.info(f"VACUUM ANALYZE completed successfully for {table_name or 'all tables'}")
        except SQLAlchemyError as e:
            logger.error(f"VACUUM ANALYZE failed: {str(e)}")

