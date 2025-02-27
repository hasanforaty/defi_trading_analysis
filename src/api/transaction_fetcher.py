"""Transaction fetcher for DexTools API."""
import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta

from src.api.dextools import DexToolsApiClient
from src.api.cache import CacheManager
from src.api.exceptions import APIError
from src.models.entities import Transaction
from src.data.database import get_session, AsyncSession

logger = logging.getLogger(__name__)


class TransactionFetcher:
    """
    Fetches transactions from DexTools API and stores them in the database.

    This class combines the API client and caching system to efficiently
    fetch transaction data for a given pair and time range.
    """

    def __init__(
            self,
            api_client: Optional[DexToolsApiClient] = None,
            cache_manager: Optional[CacheManager] = None,
            max_concurrent_requests: int = 5,
            cache_ttl: int = 300  # 5 minutes
    ):
        """
        Initialize the transaction fetcher.

        Args:
            api_client: DexTools API client
            cache_manager: Cache manager
            max_concurrent_requests: Maximum number of concurrent requests
            cache_ttl: Time-to-live for cached data in seconds
        """
        self.api_client = api_client or DexToolsApiClient()
        self.cache_manager = cache_manager or CacheManager(prefix="transactions")
        self.max_concurrent_requests = max_concurrent_requests
        self.cache_ttl = cache_ttl

    async def fetch_pair_info(self, chain: str, pair_address: str) -> Dict[str, Any]:
        """
        Fetch information about a trading pair with caching.

        Args:
            chain: Blockchain identifier
            pair_address: Trading pair contract address

        Returns:
            Dictionary containing pair information
        """
        cache_key = f"pair_info:{chain}:{pair_address}"

        # Try to get from cache
        cached_info = await self.cache_manager.get(cache_key)
        if cached_info:
            logger.debug(f"Using cached pair info for {chain}:{pair_address}")
            return cached_info

        # Fetch from API
        logger.debug(f"Fetching pair info for {chain}:{pair_address}")
        pair_info = await self.api_client.get_pair_info(chain, pair_address)

        # Cache the result
        await self.cache_manager.set(cache_key, pair_info, ttl=self.cache_ttl)

        return pair_info

    async def fetch_pair_liquidity(self, chain: str, pair_address: str) -> Dict[str, Any]:
        """
        Fetch liquidity information for a trading pair with caching.

        Args:
            chain: Blockchain identifier
            pair_address: Trading pair contract address

        Returns:
            Dictionary containing liquidity information
        """
        cache_key = f"pair_liquidity:{chain}:{pair_address}"

        # Try to get from cache
        cached_info = await self.cache_manager.get(cache_key)
        if cached_info:
            logger.debug(f"Using cached liquidity info for {chain}:{pair_address}")
            return cached_info

        # Fetch from API
        logger.debug(f"Fetching liquidity info for {chain}:{pair_address}")
        liquidity_info = await self.api_client.get_pair_liquidity(chain, pair_address)

        # Cache the result
        await self.cache_manager.set(cache_key, liquidity_info, ttl=self.cache_ttl)

        return liquidity_info

    async def fetch_transactions_page(
            self,
            chain: str,
            pair_address: str,
            page: int,
            page_size: int = 100,
            from_timestamp: Optional[Union[int, datetime]] = None,
            to_timestamp: Optional[Union[int, datetime]] = None,
            use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Fetch a single page of transactions for a trading pair.

        Args:
            chain: Blockchain identifier
            pair_address: Trading pair contract address
            page: Page number
            page_size: Number of results per page
            from_timestamp: Start timestamp
            to_timestamp: End timestamp
            use_cache: Whether to use cache

        Returns:
            Dictionary containing transaction data for the page
        """
        # Generate cache key
        from_str = str(from_timestamp) if from_timestamp else "none"
        to_str = str(to_timestamp) if to_timestamp else "none"
        cache_key = f"transactions:{chain}:{pair_address}:{from_str}:{to_str}:{page}:{page_size}"

        # Try to get from cache if enabled
        if use_cache:
            cached_data = await self.cache_manager.get(cache_key)
            if cached_data:
                logger.debug(f"Using cached transactions for {cache_key}")
                return cached_data

        # Fetch from API
        logger.debug(f"Fetching transactions page {page} for {chain}:{pair_address}")
        transactions = await self.api_client.get_transactions(
            chain=chain,
            pair_address=pair_address,
            from_timestamp=from_timestamp,
            to_timestamp=to_timestamp,
            page=page,
            page_size=page_size
        )

        # Cache the result if cache is enabled
        if use_cache:
            await self.cache_manager.set(cache_key, transactions, ttl=self.cache_ttl)

        return transactions

    async def fetch_all_transactions(
            self,
            chain: str,
            pair_address: str,
            from_timestamp: Optional[Union[int, datetime]] = None,
            to_timestamp: Optional[Union[int, datetime]] = None,
            max_pages: int = 10,
            page_size: int = 100,
            use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Fetch all transactions for a trading pair across multiple pages.

        Args:
            chain: Blockchain identifier
            pair_address: Trading pair contract address
            from_timestamp: Start timestamp
            to_timestamp: End timestamp
            max_pages: Maximum number of pages to fetch
            page_size: Number of results per page
            use_cache: Whether to use cache

        Returns:
            List of transaction dictionaries
        """
        all_transactions = []
        tasks = []

        # Fetch first page to get total pages
        first_page = await self.fetch_transactions_page(
            chain=chain,
            pair_address=pair_address,
            page=0,
            page_size=page_size,
            from_timestamp=from_timestamp,
            to_timestamp=to_timestamp,
            use_cache=use_cache
        )

        # Add transactions from first page
        if 'results' in first_page:
            all_transactions.extend(first_page['results'])

        # Determine total pages (limited by max_pages)
        total_pages = min(
            first_page.get('totalPages', 1),
            max_pages
        )

        # Skip the first page as we already fetched it
        for page in range(1, total_pages):
            # Limit concurrent requests
            if len(tasks) >= self.max_concurrent_requests:
                # Wait for some tasks to complete
                done, pending = await asyncio.wait(
                    tasks, return_when=asyncio.FIRST_COMPLETED
                )
                tasks = list(pending)

            # Create task for fetching this page
            task = asyncio.create_task(
                self.fetch_transactions_page(
                    chain=chain,
                    pair_address=pair_address,
                    page=page,
                    page_size=page_size,
                    from_timestamp=from_timestamp,
                    to_timestamp=to_timestamp,
                    use_cache=use_cache
                )
            )
            tasks.append(task)

        # Wait for all remaining tasks to complete
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Error fetching transactions: {result}")
                    continue

                if 'results' in result:
                    all_transactions.extend(result['results'])

        return all_transactions

    def _normalize_transaction(self, chain: str, pair_address: str, raw_tx: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize a raw transaction from the API into our internal format.

        Args:
            chain: Blockchain identifier
            pair_address: Trading pair contract address
            raw_tx: Raw transaction data from API

        Returns:
            Normalized transaction dictionary
        """
        # Extract relevant fields from raw transaction
        tx_hash = raw_tx.get('txHash')
        block_number = raw_tx.get('blockNumber')
        timestamp = raw_tx.get('timestamp')

        if isinstance(timestamp, str):
            # Convert ISO string to datetime
            timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))

        # Determine transaction type
        tx_type = raw_tx.get('type', 'unknown')

        # Extract token amounts
        token0_amount = raw_tx.get('token0Amount', 0)
        token1_amount = raw_tx.get('token1Amount', 0)

        # Extract wallet address
        wallet_address = raw_tx.get('walletAddress', '')

        # Extract price impact if available
        price_impact = raw_tx.get('priceImpact', 0)

        # Format the normalized transaction
        return {
            'tx_hash': tx_hash,
            'block_number': block_number,
            'timestamp': timestamp,
            'chain': chain,
            'pair_address': pair_address,
            'type': tx_type,
            'token0_amount': token0_amount,
            'token1_amount': token1_amount,
            'wallet_address': wallet_address,
            'price_impact': price_impact,
        }

    async def store_transactions(
            self,
            chain: str,
            pair_address: str,
            transactions: List[Dict[str, Any]],
            db_session: Optional[AsyncSession] = None
    ) -> List[Transaction]:
        """
        Store transactions in the database.

        Args:
            chain: Blockchain identifier
            pair_address: Trading pair contract address
            transactions: List of normalized transaction dictionaries
            db_session: Database session to use

        Returns:
            List of created Transaction objects
        """
        # Get a database session if not provided
        close_session = False
        if db_session is None:
            db_session = get_session()
            close_session = True

        try:
            # Create Transaction objects from normalized data
            transaction_objects = []
            for tx_data in transactions:
                normalized_tx = self._normalize_transaction(chain, pair_address, tx_data)

                tx_obj = Transaction(
                    tx_hash=normalized_tx['tx_hash'],
                    block_number=normalized_tx['block_number'],
                    timestamp=normalized_tx['timestamp'],
                    chain=chain,
                    pair_address=pair_address,
                    tx_type=normalized_tx['type'],
                    token0_amount=normalized_tx['token0_amount'],
                    token1_amount=normalized_tx['token1_amount'],
                    wallet_address=normalized_tx['wallet_address'],
                    price_impact=normalized_tx['price_impact']
                )

                transaction_objects.append(tx_obj)

            # Add all transactions to the session
            db_session.add_all(transaction_objects)

            # Commit the session
            await db_session.commit()

            return transaction_objects

        except Exception as e:
            # Rollback on error
            await db_session.rollback()
            logger.error(f"Error storing transactions: {e}")
            raise

        finally:
            # Close the session if we created it
            if close_session:
                await db_session.close()

    async def fetch_and_store_transactions(
            self,
            chain: str,
            pair_address: str,
            from_timestamp: Optional[Union[int, datetime]] = None,
            to_timestamp: Optional[Union[int, datetime]] = None,
            max_pages: int = 10,
            page_size: int = 100,
            use_cache: bool = True,
            db_session: Optional[AsyncSession] = None
    ) -> List[Transaction]:
        """
        Fetch transactions for a trading pair and store them in the database.

        Args:
            chain: Blockchain identifier
            pair_address: Trading pair contract address
            from_timestamp: Start timestamp
            to_timestamp: End timestamp
            max_pages: Maximum number of pages to fetch
            page_size: Number of results per page
            use_cache: Whether to use cache
            db_session: Database session to use

        Returns:
            List of Transaction objects created or retrieved
        """
        # Fetch transactions from API
        raw_transactions = await self.fetch_all_transactions(
            chain=chain,
            pair_address=pair_address,
            from_timestamp=from_timestamp,
            to_timestamp=to_timestamp,
            max_pages=max_pages,
            page_size=page_size,
            use_cache=use_cache
        )

        # Store transactions in the database
        if raw_transactions:
            return await self.store_transactions(
                chain=chain,
                pair_address=pair_address,
                transactions=raw_transactions,
                db_session=db_session
            )

        return []

    async def get_price_history(
            self,
            chain: str,
            pair_address: str,
            resolution: str = "1D",
            limit: int = 30,
            use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Get price history for a trading pair.

        Args:
            chain: Blockchain identifier
            pair_address: Trading pair contract address
            resolution: Time resolution (e.g., "1D" for daily)
            limit: Number of data points to retrieve
            use_cache: Whether to use cache

        Returns:
            List of price data points
        """
        # Use the API client to get price information
        price_data = await self.api_client.get_pair_price(chain, pair_address)

        # Extract the price history from the response
        # Note: This is a simplified implementation, as the actual
        # structure depends on the DEXTools API response format
        if 'priceHistory' in price_data and isinstance(price_data['priceHistory'], list):
            return price_data['priceHistory']

        # Return an empty list if no price history is available
        return []

    async def analyze_transaction_patterns(
            self,
            chain: str,
            pair_address: str,
            from_timestamp: Optional[Union[int, datetime]] = None,
            to_timestamp: Optional[Union[int, datetime]] = None,
            use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze transaction patterns for a trading pair.

        Args:
            chain: Blockchain identifier
            pair_address: Trading pair contract address
            from_timestamp: Start timestamp
            to_timestamp: End timestamp
            use_cache: Whether to use cache

        Returns:
            Dictionary containing transaction pattern analysis
        """
        # Fetch transactions
        transactions = await self.fetch_all_transactions(
            chain=chain,
            pair_address=pair_address,
            from_timestamp=from_timestamp,
            to_timestamp=to_timestamp,
            use_cache=use_cache
        )

        if not transactions:
            return {
                "total_transactions": 0,
                "buy_count": 0,
                "sell_count": 0,
                "buy_volume": 0,
                "sell_volume": 0,
                "average_transaction_size": 0,
                "largest_transaction": None,
                "whale_transactions": [],
                "transaction_time_distribution": {}
            }

        # Initialize counters and accumulators
        buy_count = 0
        sell_count = 0
        buy_volume = 0
        sell_volume = 0
        largest_tx = None
        largest_tx_amount = 0
        whale_transactions = []
        time_distribution = {}

        # Process each transaction
        for tx in transactions:
            # Determine transaction size (this is simplified and depends on the actual data structure)
            tx_amount = tx.get('token1Amount', 0)  # Assuming token1 is the base currency

            # Determine transaction type
            tx_type = tx.get('type', 'unknown')

            if tx_type == 'buy':
                buy_count += 1
                buy_volume += tx_amount
            elif tx_type == 'sell':
                sell_count += 1
                sell_volume += tx_amount

            # Track largest transaction
            if tx_amount > largest_tx_amount:
                largest_tx_amount = tx_amount
                largest_tx = tx

            # Identify whale transactions (arbitrary threshold, should be configurable)
            if tx_amount > 10000:  # Example threshold
                whale_transactions.append(tx)

            # Track time distribution
            if isinstance(tx.get('timestamp'), datetime):
                hour = tx['timestamp'].hour
                time_distribution[hour] = time_distribution.get(hour, 0) + 1

        # Calculate average transaction size
        total_volume = buy_volume + sell_volume
        total_count = buy_count + sell_count
        avg_tx_size = total_volume / total_count if total_count > 0 else 0

        # Return the analysis results
        return {
            "total_transactions": total_count,
            "buy_count": buy_count,
            "sell_count": sell_count,
            "buy_volume": buy_volume,
            "sell_volume": sell_volume,
            "buy_sell_ratio": buy_count / sell_count if sell_count > 0 else float('inf'),
            "average_transaction_size": avg_tx_size,
            "largest_transaction": largest_tx,
            "whale_transactions": whale_transactions,
            "transaction_time_distribution": time_distribution
        }
