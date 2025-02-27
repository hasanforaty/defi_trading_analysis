# src/models/entities.py
from datetime import datetime
from enum import Enum
from typing import List, Optional
from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime, Enum as SQLAEnum, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from config.settings import ChainType, TransactionType

Base = declarative_base()


class Pair(Base):
    """Token pair model representing a trading pair on a DEX."""
    __tablename__ = "pairs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    address = Column(String(42), nullable=False, index=True, unique=True)
    chain = Column(SQLAEnum(ChainType), nullable=False)
    token0_address = Column(String(42), nullable=False)
    token0_symbol = Column(String(10), nullable=False)
    token1_address = Column(String(42), nullable=False)
    token1_symbol = Column(String(10), nullable=False)
    liquidity = Column(Float, nullable=True)
    threshold = Column(Float, nullable=True)
    last_updated = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Relationships
    transactions = relationship("Transaction", back_populates="pair", cascade="all, delete-orphan")
    wallet_analyses = relationship("WalletAnalysis", back_populates="pair", cascade="all, delete-orphan")
    waves = relationship("Wave", back_populates="pair", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Pair(id={self.id}, address='{self.address}', chain='{self.chain}', token0='{self.token0_symbol}', token1='{self.token1_symbol}')>"


class Transaction(Base):
    """Model representing a DEX transaction."""
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    pair_id = Column(Integer, ForeignKey("pairs.id"), nullable=False, index=True)
    tx_hash = Column(String(66), nullable=False, unique=True, index=True)
    wallet_address = Column(String(42), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    amount = Column(Float, nullable=False)  # Amount in USD
    price_usd = Column(Float, nullable=False)
    transaction_type = Column(SQLAEnum(TransactionType), nullable=False)

    # Relationships
    pair = relationship("Pair", back_populates="transactions")

    def __repr__(self) -> str:
        return f"<Transaction(id={self.id}, tx_hash='{self.tx_hash}', type='{self.transaction_type}', amount={self.amount})>"


class WalletAnalysis(Base):
    """Model for wallet transaction analysis results."""
    __tablename__ = "wallet_analyses"

    id = Column(Integer, primary_key=True, autoincrement=True)
    wallet_address = Column(String(42), nullable=False, index=True)
    pair_id = Column(Integer, ForeignKey("pairs.id"), nullable=False, index=True)
    total_buy_amount = Column(Float, nullable=False, default=0.0)
    total_sell_amount = Column(Float, nullable=False, default=0.0)
    buy_sell_ratio = Column(Float, nullable=True)
    transaction_count = Column(Integer, nullable=False, default=0)
    last_analyzed = Column(DateTime, nullable=False, default=datetime.utcnow)
    notes = Column(Text, nullable=True)

    # Relationships
    pair = relationship("Pair", back_populates="wallet_analyses")

    def __repr__(self) -> str:
        return f"<WalletAnalysis(id={self.id}, wallet='{self.wallet_address}', ratio={self.buy_sell_ratio}, tx_count={self.transaction_count})>"


class Wave(Base):
    """Model representing a detected buying/selling wave."""
    __tablename__ = "waves"

    id = Column(Integer, primary_key=True, autoincrement=True)
    pair_id = Column(Integer, ForeignKey("pairs.id"), nullable=False, index=True)
    start_timestamp = Column(DateTime, nullable=False)
    end_timestamp = Column(DateTime, nullable=False)
    total_amount = Column(Float, nullable=False)  # Total amount in USD
    transaction_count = Column(Integer, nullable=False)
    transaction_type = Column(SQLAEnum(TransactionType), nullable=False)
    average_price = Column(Float, nullable=True)

    # Relationships
    pair = relationship("Pair", back_populates="waves")

    def __repr__(self) -> str:
        return f"<Wave(id={self.id}, type='{self.transaction_type}', tx_count={self.transaction_count}, total=${self.total_amount:.2f})>"
