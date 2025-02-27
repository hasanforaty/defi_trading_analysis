# src/api/data_transformer.py
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional


def transform_pair_info(
        pair_info: Dict[str, Any],
        chain: str
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Transform raw pair info from the API into database models.

    Args:
        pair_info: Raw pair info from the API
        chain: Blockchain ID

    Returns:
        Tuple of (pair_data, token1_data, token2_data)
    """
    # Extract token data
    main_token = pair_info.get("mainToken", {})
    side_token = pair_info.get("sideToken", {})

    # Create token1 data (main token)
    token1_data = {
        "chain": chain,
        "address": main_token.get("address", ""),
        "name": main_token.get("name", ""),
        "symbol": main_token.get("symbol", ""),
        "decimals": main_token.get("decimals", 18),
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
    }

    # Create token2 data (side token)
    token2_data = {
        "chain": chain,
        "address": side_token.get("address", ""),
        "name": side_token.get("name", ""),
        "symbol": side_token.get("symbol", ""),
        "decimals": side_token.get("decimals", 18),
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
    }

    # Parse creation time
    creation_time_str = pair_info.get("creationTime", "")
    try:
        creation_time = datetime.fromisoformat(creation_time_str.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        creation_time = datetime.utcnow()

    # Create pair data
    pair_data = {
        "chain": chain,
        "address": pair_info.get("address", ""),
        "dex_name": pair_info.get("exchangeName", ""),
        "dex_factory": pair_info.get("exchangeFactory", ""),
        "fee": pair_info.get("fee", 0.0),
        "creation_time": creation_time,
        "creation_block": pair_info.get("creationBlock", 0),
        "token1_id": None,  # To be filled later
        "token2_id": None,  # To be filled later
        "token1_reserve": 0.0,
        "token2_reserve": 0.0,
        "liquidity_usd": 0.0,
        "price_usd": 0.0,
        "price_native": 0.0,
        "volume_24h": 0.0,
        "tx_count_24h": 0,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
    }

    return pair_data, token1_data, token2_data


def transform_transaction(
        tx_data: Dict[str, Any],
        pair_id: int,
        token_ids: Tuple[int, int]
) -> Dict[str, Any]:
    """
    Transform raw transaction data from the API into database model.

    Args:
        tx_data: Raw transaction data from the API
        pair_id: ID of the pair in the database
        token_ids: Tuple of (token1_id, token2_id)

    Returns:
        Transformed transaction data
    """
    # Extract basic transaction info
    tx_hash = tx_data.get("txHash", "")
    block_number = tx_data.get("blockNumber", 0)
    chain = tx_data.get("chain", "")

    # Parse timestamp
    timestamp_str = tx_data.get("timestamp", "")
    try:
        timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        timestamp = datetime.utcnow()

    # Determine transaction type (buy/sell)
    # In DexTools API, 'type' field has values like 'buy', 'sell', 'unknown'
    tx_type = tx_data.get("type", "unknown").lower()

    # Extract amounts and prices
    amount_in_token = tx_data.get("amountIn", 0.0)
    amount_out_token = tx_data.get("amountOut", 0.0)
    price_usd = tx_data.get("priceUsd", 0.0)
    price_native = tx_data.get("priceNative", 0.0)

    # Determine which token is being bought/sold
    token1_id, token2_id = token_ids
    if tx_type == "buy":
        # Buying main token, selling side token
        from_token_id = token2_id
        to_token_id = token1_id
    else:
        # Selling main token, buying side token
        from_token_id = token1_id
        to_token_id = token2_id

    # Create transformed transaction data
    transformed_data = {
        "pair_id": pair_id,
        "chain": chain,
        "tx_hash": tx_hash,
        "block_number": block_number,
        "timestamp": timestamp,
        "tx_type": tx_type,
        "from_token_id": from_token_id,
        "to_token_id": to_token_id,
        "from_address": tx_data.get("from", ""),
        "to_address": tx_data.get("to", ""),
        "amount_in": amount_in_token,
        "amount_out": amount_out_token,
        "price_usd": price_usd,
        "price_native": price_native,
        "value_usd": tx_data.get("valueUsd", 0.0),
        "created_at": datetime.utcnow(),
    }

    return transformed_data


def filter_transactions(
        transactions: List[Dict[str, Any]],
        min_value_usd: float = 0.0,
        tx_types: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Filter transactions based on criteria.

    Args:
        transactions: List of raw transaction data
        min_value_usd: Minimum transaction value in USD
        tx_types: List of transaction types to include (e.g., ["buy", "sell"])

    Returns:
        Filtered list of transactions
    """
    result = []

    for tx in transactions:
        # Check value threshold
        if min_value_usd > 0 and tx.get("valueUsd", 0.0) < min_value_usd:
            continue

        # Check transaction type
        if tx_types is not None and tx.get("type", "").lower() not in [t.lower() for t in tx_types]:
            continue

        result.append(tx)

    return result


def normalize_timestamps(
        transactions: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Normalize timestamps in transaction data.

    Args:
        transactions: List of raw transaction data

    Returns:
        Transactions with normalized timestamps
    """
    result = []

    for tx in transactions:
        # Make a copy of the transaction
        normalized_tx = tx.copy()

        # Parse timestamp
        timestamp_str = tx.get("timestamp", "")
        try:
            timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            # Convert to UTC timestamp in seconds
            normalized_tx["timestamp_unix"] = int(timestamp.timestamp())
            # Format as ISO string
            normalized_tx["timestamp_iso"] = timestamp.isoformat()
        except (ValueError, TypeError):
            # Keep original if parsing fails
            pass

        result.append(normalized_tx)

    return result
