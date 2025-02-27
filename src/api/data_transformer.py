"""Utilities for transforming API data to internal models."""
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def parse_datetime(datetime_str: str) -> datetime:
    """
    Parse ISO datetime string to datetime object.

    Args:
        datetime_str: ISO format datetime string

    Returns:
        datetime object
    """
    return datetime.fromisoformat(datetime_str.replace('Z', '+00:00'))


def normalize_transaction_data(raw_transaction: Dict[str, Any], chain: str, pair_address: str) -> Dict[str, Any]:
    """
    Normalize transaction data from the API.

    Args:
        raw_transaction: Raw transaction data from API
        chain: Blockchain identifier
        pair_address: Trading pair contract address

    Returns:
        Normalized transaction dictionary
    """
    # Extract relevant fields
    tx_hash = raw_transaction.get('txHash')
    block_number = raw_transaction.get('blockNumber')
    timestamp = raw_transaction.get('timestamp')

    if isinstance(timestamp, str):
        timestamp = parse_datetime(timestamp)

    # Transaction type (buy, sell, etc.)
    tx_type = raw_transaction.get('type', 'unknown')

    # Token amounts
    token0_amount = raw_transaction.get('token0Amount', 0)
    token1_amount = raw_transaction.get('token1Amount', 0)

    # Wallet address
    wallet_address = raw_transaction.get('walletAddress', '')

    # Price impact and other fields
    price_impact = raw_transaction.get('priceImpact', 0)
    price = raw_transaction.get('price', 0)

    return {
        'tx_hash': tx_hash,
        'block_number': block_number,
        'timestamp': timestamp,
        'chain': chain,
        'pair_address': pair_address,
        'tx_type': tx_type,
        'token0_amount': token0_amount,
        'token1_amount': token1_amount,
        'wallet_address': wallet_address,
        'price_impact': price_impact,
        'price': price
    }


def normalize_pair_info(raw_pair_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize pair information from the API.

    Args:
        raw_pair_info: Raw pair information from API

    Returns:
        Normalized pair information dictionary
    """
    # Basic pair info
    exchange_name = raw_pair_info.get('exchangeName', '')
    exchange_factory = raw_pair_info.get('exchangeFactory', '')

    # Creation info
    creation_time = raw_pair_info.get('creationTime')
    if isinstance(creation_time, str):
        creation_time = parse_datetime(creation_time)

    creation_block = raw_pair_info.get('creationBlock', 0)

    # Token info
    main_token = raw_pair_info.get('mainToken', {})
    side_token = raw_pair_info.get('sideToken', {})

    # Fee
    fee = raw_pair_info.get('fee', 0)

    return {
        'exchange_name': exchange_name,
        'exchange_factory': exchange_factory,
        'creation_time': creation_time,
        'creation_block': creation_block,
        'main_token_address': main_token.get('address', ''),
        'main_token_symbol': main_token.get('symbol', ''),
        'main_token_name': main_token.get('name', ''),
        'side_token_address': side_token.get('address', ''),
        'side_token_symbol': side_token.get('symbol', ''),
        'side_token_name': side_token.get('name', ''),
        'fee': fee
    }


def normalize_liquidity_data(raw_liquidity: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize liquidity data from the API.

    Args:
        raw_liquidity: Raw liquidity data from API

    Returns:
        Normalized liquidity dictionary
    """
    # Extract reserves
    reserves = raw_liquidity.get('reserves', {})

    main_token_reserves = reserves.get('mainToken', 0)
    side_token_reserves = reserves.get('sideToken', 0)

    # Extract liquidity
    liquidity = raw_liquidity.get('liquidity', 0)

    return {
        'main_token_reserves': main_token_reserves,
        'side_token_reserves': side_token_reserves,
        'liquidity': liquidity
    }


def normalize_price_data(raw_price: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize price data from the API.

    Args:
        raw_price: Raw price data from API

    Returns:
        Normalized price dictionary
    """
    # Current price
    price_usd = raw_price.get('priceUsd', 0)
    price_native = raw_price.get('priceNative', 0)

    # Volume
    volume_usd = raw_price.get('volumeUsd', 0)

    # Buy/sell metrics
    buy_count = raw_price.get('buyCount', 0)
    sell_count = raw_price.get('sellCount', 0)
    buy_volume = raw_price.get('buyVolume', 0)
    sell_volume = raw_price.get('sellVolume', 0)

    # Price change percentages
    price_change_5m = raw_price.get('priceChange5m', 0)
    price_change_1h = raw_price.get('priceChange1h', 0)
    price_change_6h = raw_price.get('priceChange6h', 0)
    price_change_24h = raw_price.get('priceChange24h', 0)

    return {
        'price_usd': price_usd,
        'price_native': price_native,
        'volume_usd': volume_usd,
        'buy_count': buy_count,
        'sell_count': sell_count,
        'buy_volume': buy_volume,
        'sell_volume': sell_volume,
        'price_change_5m': price_change_5m,
        'price_change_1h': price_change_1h,
        'price_change_6h': price_change_6h,
        'price_change_24h': price_change_24h
    }
