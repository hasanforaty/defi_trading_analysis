# tests/data/test_database.py
import pytest
from sqlalchemy import text
from src.data.database import Database


@pytest.mark.asyncio
async def test_database_connection(test_db):
    """Test database connection."""
    async with test_db.session() as session:
        # Execute a simple query to test connection
        result = await session.execute(text("SELECT 1"))
        value = result.scalar()
        assert value == 1


@pytest.mark.asyncio
async def test_database_session_context_manager(test_db):
    """Test database session context manager."""
    # Test successful transaction
    async with test_db.session() as session:
        await session.execute(text("SELECT 1"))
        # Session should commit successfully

    # Test transaction rollback on exception
    try:
        async with test_db.session() as session:
            await session.execute(text("SELECT 1"))
            raise ValueError("Test exception")
    except ValueError:
        # Session should have rolled back
        pass
