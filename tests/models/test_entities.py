# tests/models/test_entities.py
import pytest
import datetime
from sqlalchemy.future import select

from src.models.entities import Pair, Transaction, WalletAnalysis, Wave
from config.settings import ChainType, TransactionType


@pytest.mark.asyncio
async def test_pair_create_and_query(test_db_session):
    """Test creating and querying a Pair record."""
    # Create a pair
    pair = Pair(
        address="0x1234567890123456789012345678901234567890",
        chain=ChainType.ETHEREUM,
        token0_address="0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        token0_symbol="TOKEN0",
        token1_address="0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        token1_symbol="TOKEN1",
        liquidity=1000000.0,
        threshold=5000.0,
        last_updated=datetime.datetime.utcnow()
    )

    test_db_session.add(pair)
    await test_db_session.commit()

    # Query the pair
    result = await test_db_session.execute(
        select(Pair).where(Pair.address == "0x1234567890123456789012345678901234567890")
    )
    db_pair = result.scalars().first()

    # Assert
    assert db_pair is not None
    assert db_pair.address == "0x1234567890123456789012345678901234567890"
    assert db_pair.chain == ChainType.ETHEREUM
    assert db_pair.token0_symbol == "TOKEN0"
    assert db_pair.token1_symbol == "TOKEN1"
    assert db_pair.liquidity == 1000000.0
    assert db_pair.threshold == 5000.0


@pytest.mark.asyncio
async def test_relationship_between_pair_and_transaction(test_db_session):
    """Test relationship between Pair and Transaction."""
    # Create a pair
    pair = Pair(
        address="0x2234567890123456789012345678901234567890",
        chain=ChainType.BSC,
        token0_address="0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        token0_symbol="TOKEN0",
        token1_address="0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        token1_symbol="TOKEN1",
        liquidity=2000000.0,
        threshold=10000.0,
        last_updated=datetime.datetime.utcnow()
    )

    test_db_session.add(pair)
    await test_db_session.flush()

    # Create a transaction linked to the pair
    transaction = Transaction(
        pair_id=pair.id,
        tx_hash="0xabcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
        wallet_address="0xcccccccccccccccccccccccccccccccccccccccc",
        timestamp=datetime.datetime.utcnow(),
        amount=5000.0,
        price_usd=1.25,
        transaction_type=TransactionType.BUY
    )

    test_db_session.add(transaction)
    await test_db_session.commit()

    # Query the pair with its transaction
    result = await test_db_session.execute(
        select(Pair).where(Pair.address == "0x2234567890123456789012345678901234567890")
    )
    db_pair = result.scalars().first()

    # Assert
    assert db_pair is not None
    assert len(db_pair.transactions) == 1
    assert db_pair.transactions[0].tx_hash == "0xabcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890"
    assert db_pair.transactions[0].transaction_type == TransactionType.BUY
