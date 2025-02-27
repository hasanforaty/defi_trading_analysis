# src/api/transaction_fetcher.py
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from loguru import logger

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, insert, func

from src.api.dextools import DexToolsApiClient
from src.api.cache import CacheManager
from src.data.database import Database
from src.models.entities import Pair
from src.models.entities import Token
from src.models.entities import Transaction
from src.api.data_transformer import (
    transform_pair_info,
    transform_transaction,
    filter_transactions,
)


class TransactionFetcher:
    """
    Handles fetching transaction data from DexTools API.
    Combines API client and caching, handles pagination, and stores results in the database.
    """

    def __init__(
            self,
            api_client: Optional[DexToolsApiClient] = None,
            cache_manager: Optional[CacheManager] = None,
            database: Optional[Database] = None,
            max_workers: int = 5,
            page_size: int = 100
    ):
        """
        Initialize the transaction fetcher.

        Args:
            api_client: DexTools API client instance
            cache_manager: Cache manager instance
            database: Database interface instance
            max_workers: Maximum number of concurrent workers for fetching data
            page_size: Number of transactions to fetch per page
        """
        self.api_client = api_client or DexToolsApiClient()
        self.cache_manager = cache_manager or CacheManager()
        self.database = database or Database()
        self.max_workers = max_workers
        self.page_size = page_size

    async def ensure_pair_exists(
            self,
            session: AsyncSession,
            chain: str,
            pair_address: str
    ) -> Tuple[int, Tuple[int, int]]:
        """
        Ensure that the pair and its tokens exist in the database.

        Args:
            session: Database session
            chain: Blockchain ID
            pair_address: Address of the trading pair

        Returns:
            Tuple of (pair_id, (token1_id, token2_id))
        """
        # Check if pair already exists
        pair_query = select(Pair).where(
            (Pair.chain == chain) &
            (func.lower(Pair.address) == func.lower(pair_address))
        )
        result = await session.execute(pair_query)
        pair = result.scalars().first()

        if pair is not None:
            # Pair exists, return its ID and token IDs
            return pair.id, (pair.token1_id, pair.token2_id)

        # Pair doesn't exist, fetch from API and create
        pair_info = await self.api_client.get_pair_info(chain, pair_address)
        pair_data, token1_data, token2_data = transform_pair_info(pair_info, chain)

        # Check if tokens exist
        token1_query = select(Token).where(
            (Token.chain == chain) &
            (func.lower(Token.address) == func.lower(token1_data["address"]))
        )
        token2_query = select(Token).where(
            (Token.chain == chain) &
            (func.lower(Token.address) == func.lower(token2_data["address"]))
        )

        token1_result = await session.execute(token1_query)
        token2_result = await session.execute(token2_query)

        token1 = token1_result.scalars().first()
        token2 = token2_result.scalars().first()

        # Create tokens if they don't exist
        if token1 is None:
            token1_stmt = insert(Token).values(**token1_data).returning(Token.id)
            token1_result = await session.execute(token1_stmt)
            token1_id = token1_result.scalar_one()
        else:
            token1_id = token1.id

        if token2 is None:
            token2_stmt = insert(Token).values(**token2_data).returning(Token.id)
            token2_result = await session.execute(token2_stmt)
            token2_id = token2_result.scalar_one()
        else:
            token2_id = token2.id

        # Create the pair
        pair_data["token1_id"] = token1_id
        pair_data["token2_id"] = token2_id
        pair_stmt = insert(Pair).values(**pair_data).returning(Pair.id)
        pair_result = await session.execute(pair_stmt)
        pair_id = pair_result.scalar_one()

        # Commit the changes
        await session.commit()

        return pair_id, (token1_id, token2_id)

    async def fetch_transactions_page(
            self,
            chain: str,
            pair_address: str,
            from_timestamp: Optional[int] = None,
            to_timestamp: Optional[int] = None,
            page: int = 0
    ) -> Tuple[List[Dict[str, Any]], bool]:
        """
        Fetch a single page of transactions.

        Args:
            chain: Blockchain ID
            pair_address: Address of the trading pair
            from_timestamp: Start timestamp (Unix time in seconds)
            to_timestamp: End timestamp (Unix time in seconds)
            page: Page number (0-indexed)

        Returns:
            Tuple of (list of transactions, has_more)
        """
        response = await self.api_client.get_transactions(
            chain=chain,
            pair_address=pair_address,
            from_timestamp=from_timestamp,
            to_timestamp=to_timestamp,
            page=page,
            page_size=self.page_size
        )

        # Extract transactions from the response
        transactions = response.get("results", [])

        # Check if there are more pages
        total_pages = response.get("totalPages", 0)
        has_more = (page + 1) < total_pages

        logger.info(
            f"Fetched {len(transactions)} transactions from {chain}/{pair_address}, page {page + 1}/{total_pages}")
        return transactions, has_more

    async def fetch_all_transactions(
            self,
            chain: str,
            pair_address: str,
            from_timestamp: Optional[int] = None,
            to_timestamp: Optional[int] = None,
            max_pages: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch all transactions for a pair within a time range, handling pagination.

        Args:
            chain: Blockchain ID
            pair_address: Address of the trading pair
            from_timestamp: Start timestamp (Unix time in seconds)
            to_timestamp: End timestamp (Unix time in seconds)
            max_pages: Maximum number of pages to fetch (None for all)

        Returns:
            List of transaction data
        """
        all_transactions = []
        page = 0
        has_more = True

        # Set up concurrent workers
        semaphore = asyncio.Semaphore(self.max_workers)

        async def fetch_page_with_semaphore(page_num):
            async with semaphore:
                return await self.fetch_transactions_page(
                    chain, pair_address, from_timestamp, to_timestamp, page_num
                )

        # Initialize first page to check total
        first_page_result, has_more = await fetch_page_with_semaphore(0)
        all_transactions.extend(first_page_result)

        if not has_more or (max_pages is not None and page >= max_pages - 1):
            return all_transactions

        # Prepare to fetch remaining pages concurrently
        page = 1
        tasks = []

        while has_more and (max_pages is None or page < max_pages):
            tasks.append(fetch_page_with_semaphore(page))
            page += 1

            # If we've reached max_workers, wait for some tasks to complete
            if len(tasks) >= self.max_workers:
                results = await asyncio.gather(*tasks)
                for transactions, page_has_more in results:
                    all_transactions.extend(transactions)
                has_more = any(page_has_more for _, page_has_more in results)
                tasks = []

        # Fetch any remaining pages
        if tasks:
            results = await asyncio.gather(*tasks)
            for transactions, _ in results:
                all_transactions.extend(transactions)

        return all_transactions

    async def fetch_and_store_transactions(
            self,
            chain: str,
            pair_address: str,
            from_timestamp: Optional[int] = None,
            to_timestamp: Optional[int] = None,
            max_pages: Optional[int] = None,
            min_value_usd: float = 0.0,
            max_transactions: Optional[int] = None
    ) -> int:
        """
        Fetch all transactions for a pair and store them in the database.

        Args:
            chain: Blockchain ID
            pair_address: Address of the trading pair
            from_timestamp: Start timestamp (Unix time in seconds)
            to_timestamp: End timestamp (Unix time in seconds)
            max_pages: Maximum number of pages to fetch (None for all)
            min_value_usd: Minimum transaction value in USD to include
            max_transactions: Maximum number of transactions to store (None for all)

        Returns:
            Number of transactions stored
        """
        # Fetch all transactions
        raw_transactions = await self.fetch_all_transactions(
            chain, pair_address, from_timestamp, to_timestamp, max_pages
        )

        # Filter transactions if needed
        if min_value_usd > 0:
            raw_transactions = filter_transactions(raw_transactions, min_value_usd=min_value_usd)

        # Limit the number of transactions if specified
        if max_transactions is not None and len(raw_transactions) > max_transactions:
            raw_transactions = raw_transactions[:max_transactions]

        # Process and store transactions
        async with self.database.session() as session:
            # Ensure pair exists
            pair_id, token_ids = await self.ensure_pair_exists(session, chain, pair_address)

            # Transform and store transactions
            tx_count = 0
            for raw_tx in raw_transactions:
                tx_data = transform_transaction(raw_tx, pair_id, token_ids)

                # Check if transaction already exists
                tx_query = select(Transaction).where(
                    (Transaction.chain == chain) &
                    (Transaction.tx_hash == tx_data["tx_hash"])
                )
                result = await session.execute(tx_query)
                existing_tx = result.scalars().first()

                if existing_tx is None:
                    # Transaction doesn't exist, create it
                    await session.execute(insert(Transaction).values(**tx_data))
                    tx_count += 1

            if tx_count > 0:
                await session.commit()
                logger.info(f"Stored {tx_count} new transactions for {chain}/{pair_address}")

            return tx_count

    async def update_pair_liquidity(
            self,
            chain: str,
            pair_address: str
    ) -> Dict[str, Any]:
        """
        Update liquidity information for a pair.

        Args:
            chain: Blockchain ID
            pair_address: Address of the trading pair

        Returns:
            Updated liquidity data
        """
        # Fetch liquidity data
        liquidity_data = await self.api_client.get_pair_liquidity(chain, pair_address)

        # Update pair in database
        async with self.database.session() as session:
            # Find the pair
            pair_query = select(Pair).where(
                (Pair.chain == chain) &
                (func.lower(Pair.address) == func.lower(pair_address))
            )
            result = await session.execute(pair_query)
            pair = result.scalars().first()

            if pair is not None:
                # Update liquidity
                pair.liquidity_usd = liquidity_data.get("liquidity", 0)
                pair.token1_reserve = liquidity_data.get("reserves", {}).get("mainToken", 0)
                pair.token2_reserve = liquidity_data.get("reserves", {}).get("sideToken", 0)
                pair.updated_at = datetime.utcnow()

                await session.commit()
                logger.info(f"Updated liquidity for {chain}/{pair_address}")

            return liquidity_data

    async def update_pair_price(
            self,
            chain: str,
            pair_address: str
    ) -> Dict[str, Any]:
        """
        Update price information for a pair.

        Args:
            chain: Blockchain ID
            pair_address: Address of the trading pair

        Returns:
            Updated price data
        """
        # Fetch price data
        price_data = await self.api_client.get_pair_price(chain, pair_address)

        # Update pair in database
        async with self.database.session() as session:
            # Find the pair
            pair_query = select(Pair).where(
                (Pair.chain == chain) &
                (func.lower(Pair.address) == func.lower(pair_address))
            )
            result = await session.execute(pair_query)
            pair = result.scalars().first()

            if pair is not None:
                # Update price information
                pair.price_usd = price_data.get("priceUsd", 0)
                pair.price_native = price_data.get("priceNative", 0)
                pair.volume_24h = price_data.get("volume24h", 0)
                pair.tx_count_24h = price_data.get("txCount24h", 0)
                pair.updated_at = datetime.utcnow()

                await session.commit()
                logger.info(f"Updated price for {chain}/{pair_address}")

            return price_data

    async def close(self):
        """Close the fetcher and its resources."""
        await self.api_client.close()
        await self.cache_manager.close()
