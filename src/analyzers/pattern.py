from typing import List, Dict, Optional, Tuple, Set, Union
import asyncio
from datetime import datetime, timedelta
from enum import Enum
from loguru import logger

from sqlalchemy import select, and_, or_, func, desc
from sqlalchemy.ext.asyncio import AsyncSession

from config.settings import get_settings, TransactionType
from src.models.entities import Pair, Transaction, WalletAnalysis, Wave
from src.data.database import get_db_session
from src.analyzers.utils import time_window_to_datetime


class PatternType(str, Enum):
    """Common trading patterns."""
    ACCUMULATION = "accumulation"  # Gradual accumulation over time
    DISTRIBUTION = "distribution"  # Gradual distribution over time
    PUMP_PREPARATION = "pump_preparation"  # Preparing for a pump event
    DUMP_PREPARATION = "dump_preparation"  # Preparing for a dump event
    WHALE_ACTIVITY = "whale_activity"  # Large single-wallet activity
    BOT_TRADING = "bot_trading"  # Consistent small trades with precise timing
    WASH_TRADING = "wash_trading"  # Same wallet buying and selling


class PatternConfidence(float, Enum):
    """Confidence levels for pattern detection."""
    LOW = 0.3
    MEDIUM = 0.6
    HIGH = 0.9


class PatternAnalyzer:
    """
    Identifies common trading patterns in transaction data.
    Uses time-series analysis and multiple pattern matching techniques.
    """

    def __init__(self, session: AsyncSession = None):
        """Initialize the pattern recognizer with settings and optional session."""
        self.settings = get_settings()
        self._session = session

    async def _get_session(self) -> AsyncSession:
        """Get database session or use provided one."""
        if self._session is not None:
            return self._session

        # Get a new session from the pool
        async for session in get_db_session():
            return session

    async def identify_patterns(
            self,
            pair_id: int,
            start_time: Optional[datetime] = None,
            end_time: Optional[datetime] = None
    ) -> List[Dict]:
        """
        Identify patterns in transaction data for a pair.

        Args:
            pair_id: ID of the pair to analyze
            start_time: Optional start time for analysis window
            end_time: Optional end time for analysis window

        Returns:
            List of identified patterns with metadata
        """
        session = await self._get_session()
        logger.info(f"Analyzing patterns for pair {pair_id}")

        # Set time range if not provided
        if not end_time:
            end_time = datetime.utcnow()
        if not start_time:
            start_time = end_time - timedelta(days=14)  # Default to 2 weeks

        # Get pair information
        pair = await session.get(Pair, pair_id)
        if not pair:
            logger.error(f"Pair with ID {pair_id} not found")
            return []

        # Identify each pattern type
        patterns = []

        # Add patterns from each recognizer function
        accumulation_patterns = await self._identify_accumulation_patterns(pair_id, start_time, end_time)
        patterns.extend(accumulation_patterns)

        distribution_patterns = await self._identify_distribution_patterns(pair_id, start_time, end_time)
        patterns.extend(distribution_patterns)

        pump_prep_patterns = await self._identify_pump_preparation(pair_id, start_time, end_time)
        patterns.extend(pump_prep_patterns)

        whale_patterns = await self._identify_whale_activity(pair_id, start_time, end_time)
        patterns.extend(whale_patterns)

        bot_patterns = await self._identify_bot_trading(pair_id, start_time, end_time)
        patterns.extend(bot_patterns)

        wash_patterns = await self._identify_wash_trading(pair_id, start_time, end_time)
        patterns.extend(wash_patterns)

        # Calculate price movements for context
        price_context = await self._calculate_price_movements(pair_id, start_time, end_time)

        # Add price context to each pattern
        for pattern in patterns:
            pattern_start = pattern['start_time']
            pattern_end = pattern['end_time']

            # Find closest price context timeframes
            relevant_price_movements = []
            for movement in price_context:
                # Check if there's overlap between pattern and price movement
                if (movement['start_time'] <= pattern_end and
                        movement['end_time'] >= pattern_start):
                    relevant_price_movements.append(movement)

            pattern['price_context'] = relevant_price_movements

        logger.info(f"Identified {len(patterns)} patterns for pair {pair_id}")
        return patterns

    async def _identify_accumulation_patterns(
            self,
            pair_id: int,
            start_time: datetime,
            end_time: datetime
    ) -> List[Dict]:
        """Identify accumulation patterns."""
        session = await self._get_session()
        patterns = []

        # Get all wallet analyses for this pair
        stmt = select(WalletAnalysis).where(
            WalletAnalysis.pair_id == pair_id,
            WalletAnalysis.last_analyzed >= start_time,
            WalletAnalysis.last_analyzed <= end_time
        )
        result = await session.execute(stmt)
        analyses = result.scalars().all()

        # Identify wallets with high buy/sell ratio and significant amounts
        accumulation_wallets = []
        for analysis in analyses:
            if (analysis.buy_sell_ratio is not None and
                    analysis.buy_sell_ratio > 5.0 and  # 5x more buys than sells
                    analysis.total_buy_amount > self.settings.analysis.significant_transaction_threshold_usd and
                    analysis.transaction_count >= 5):
                accumulation_wallets.append(analysis.wallet_address)

        if not accumulation_wallets:
            return []

        # Get transactions for accumulation wallets
        stmt = select(Transaction).where(
            Transaction.pair_id == pair_id,
            Transaction.wallet_address.in_(accumulation_wallets),
            Transaction.timestamp.between(start_time, end_time)
        ).order_by(Transaction.timestamp)

        result = await session.execute(stmt)
        transactions = result.scalars().all()

        # Group transactions by wallet
        wallet_txs = {}
        for tx in transactions:
            if tx.wallet_address not in wallet_txs:
                wallet_txs[tx.wallet_address] = []
            wallet_txs[tx.wallet_address].append(tx)

        # Analyze accumulation patterns for each wallet
        for wallet, txs in wallet_txs.items():
            # Sort by timestamp
            txs.sort(key=lambda x: x.timestamp)

            # Check for gradual accumulation (buys spaced out over time)
            buy_txs = [tx for tx in txs if tx.transaction_type == TransactionType.BUY]
            if len(buy_txs) < 5:
                continue

            # Calculate time spans between buys
            time_spans = []
            for i in range(1, len(buy_txs)):
                span = (buy_txs[i].timestamp - buy_txs[i - 1].timestamp).total_seconds() / 3600  # hours
                time_spans.append(span)

            avg_time_span = sum(time_spans) / len(time_spans) if time_spans else 0

            # Check if buys are reasonably spaced (not all at once, not too far apart)
            if 1 <= avg_time_span <= 48:  # Between 1 and 48 hours average spacing
                total_buy_amount = sum(tx.amount for tx in buy_txs)
                avg_buy_amount = total_buy_amount / len(buy_txs)

                # Calculate variance in buy amounts
                variance = sum((tx.amount - avg_buy_amount) ** 2 for tx in buy_txs) / len(buy_txs)
                std_dev = variance ** 0.5
                coefficient_of_variation = std_dev / avg_buy_amount if avg_buy_amount > 0 else 0

                # Lower coefficient of variation means more consistent buys
                confidence = PatternConfidence.MEDIUM
                if coefficient_of_variation < 0.3:
                    confidence = PatternConfidence.HIGH
                elif coefficient_of_variation > 0.7:
                    confidence = PatternConfidence.LOW

                pattern = {
                    'type': PatternType.ACCUMULATION,
                    'wallet_address': wallet,
                    'pair_id': pair_id,
                    'strength': total_buy_amount / self.settings.analysis.significant_transaction_threshold_usd,
                    'confidence': confidence,
                    'start_time': buy_txs[0].timestamp,
                    'end_time': buy_txs[-1].timestamp,
                    'transaction_count': len(buy_txs),
                    'total_amount': total_buy_amount,
                    'avg_time_between_buys_hours': avg_time_span,
                    'buy_amount_variance': coefficient_of_variation
                }

                patterns.append(pattern)

        return patterns

    async def _identify_distribution_patterns(
            self,
            pair_id: int,
            start_time: datetime,
            end_time: datetime
    ) -> List[Dict]:
        """Identify distribution patterns (similar to accumulation but for sells)."""
        session = await self._get_session()
        patterns = []

        # Similar to accumulation but looking for sells instead of buys
        # Get all wallet analyses for this pair
        stmt = select(WalletAnalysis).where(
            WalletAnalysis.pair_id == pair_id,
            WalletAnalysis.last_analyzed >= start_time,
            WalletAnalysis.last_analyzed <= end_time
        )
        result = await session.execute(stmt)
        analyses = result.scalars().all()

        # Identify wallets with high sell/buy ratio and significant amounts
        distribution_wallets = []
        for analysis in analyses:
            # A buy_sell_ratio close to 0 means mostly sells
            if (analysis.buy_sell_ratio is not None and
                    analysis.buy_sell_ratio < 0.2 and  # 5x more sells than buys
                    analysis.total_sell_amount > self.settings.analysis.significant_transaction_threshold_usd and
                    analysis.transaction_count >= 5):
                distribution_wallets.append(analysis.wallet_address)

        if not distribution_wallets:
            return []

        # Get transactions for distribution wallets
        stmt = select(Transaction).where(
            Transaction.pair_id == pair_id,
            Transaction.wallet_address.in_(distribution_wallets),
            Transaction.timestamp.between(start_time, end_time)
        ).order_by(Transaction.timestamp)

        result = await session.execute(stmt)
        transactions = result.scalars().all()

        # Group transactions by wallet
        wallet_txs = {}
        for tx in transactions:
            if tx.wallet_address not in wallet_txs:
                wallet_txs[tx.wallet_address] = []
            wallet_txs[tx.wallet_address].append(tx)

        # Analyze distribution patterns for each wallet
        for wallet, txs in wallet_txs.items():
            # Sort by timestamp
            txs.sort(key=lambda x: x.timestamp)

            # Check for gradual distribution (sells spaced out over time)
            sell_txs = [tx for tx in txs if tx.transaction_type == TransactionType.SELL]
            if len(sell_txs) < 5:
                continue

            # Calculate time spans between sells
            time_spans = []
            for i in range(1, len(sell_txs)):
                span = (sell_txs[i].timestamp - sell_txs[i - 1].timestamp).total_seconds() / 3600  # hours
                time_spans.append(span)

            avg_time_span = sum(time_spans) / len(time_spans) if time_spans else 0

            # Check if sells are reasonably spaced
            if 1 <= avg_time_span <= 48:  # Between 1 and 48 hours average spacing
                total_sell_amount = sum(tx.amount for tx in sell_txs)
                avg_sell_amount = total_sell_amount / len(sell_txs)

                # Calculate variance in sell amounts
                variance = sum((tx.amount - avg_sell_amount) ** 2 for tx in sell_txs) / len(sell_txs)
                std_dev = variance ** 0.5
                coefficient_of_variation = std_dev / avg_sell_amount if avg_sell_amount > 0 else 0

                # Lower coefficient of variation means more consistent sells
                confidence = PatternConfidence.MEDIUM
                if coefficient_of_variation < 0.3:
                    confidence = PatternConfidence.HIGH
                elif coefficient_of_variation > 0.7:
                    confidence = PatternConfidence.LOW

                pattern = {
                    'type': PatternType.DISTRIBUTION,
                    'wallet_address': wallet,
                    'pair_id': pair_id,
                    'strength': total_sell_amount / self.settings.analysis.significant_transaction_threshold_usd,
                    'confidence': confidence,
                    'start_time': sell_txs[0].timestamp,
                    'end_time': sell_txs[-1].timestamp,
                    'transaction_count': len(sell_txs),
                    'total_amount': total_sell_amount,
                    'avg_time_between_sells_hours': avg_time_span,
                    'sell_amount_variance': coefficient_of_variation
                }

                patterns.append(pattern)

        return patterns

    async def _identify_pump_preparation(
            self,
            pair_id: int,
            start_time: datetime,
            end_time: datetime
    ) -> List[Dict]:
        """Identify patterns that may indicate preparation for a pump event."""
        session = await self._get_session()
        patterns = []

        # Look for increased buy activity from multiple wallets in a short time
        # First, get daily buy volumes
        daily_volumes = []
        current_date = start_time.date()
        end_date = end_time.date()

        while current_date <= end_date:
            day_start = datetime.combine(current_date, datetime.min.time())
            day_end = datetime.combine(current_date, datetime.max.time())

            # Get buy volume for this day
            stmt = select(func.sum(Transaction.amount)).where(
                Transaction.pair_id == pair_id,
                Transaction.transaction_type == TransactionType.BUY,
                Transaction.timestamp.between(day_start, day_end)
            )
            result = await session.execute(stmt)
            day_volume = result.scalar_one_or_none() or 0

            # Get unique buyer count for this day
            stmt = select(func.count(Transaction.wallet_address.distinct())).where(
                Transaction.pair_id == pair_id,
                Transaction.transaction_type == TransactionType.BUY,
                Transaction.timestamp.between(day_start, day_end)
            )
            result = await session.execute(stmt)
            unique_buyers = result.scalar_one_or_none() or 0

            daily_volumes.append({
                'date': current_date,
                'volume': day_volume,
                'unique_buyers': unique_buyers
            })
            current_date += timedelta(days=1)

        # Look for significant volume increases
        for i in range(3, len(daily_volumes)):
            # Calculate average of previous 3 days
            prev_avg_volume = sum(day['volume'] for day in daily_volumes[i - 3:i]) / 3
            prev_avg_buyers = sum(day['unique_buyers'] for day in daily_volumes[i - 3:i]) / 3

            current_volume = daily_volumes[i]['volume']
            current_buyers = daily_volumes[i]['unique_buyers']

            # Check if there's a significant increase
            if (current_volume > prev_avg_volume * 2 and  # Volume doubled
                    current_buyers > prev_avg_buyers * 1.5):  # 50% more buyers

                # This could indicate pump preparation
                day_start = datetime.combine(daily_volumes[i]['date'], datetime.min.time())
                day_end = datetime.combine(daily_volumes[i]['date'], datetime.max.time())

                # Calculate confidence based on how extreme the increase is
                volume_increase_factor = current_volume / prev_avg_volume if prev_avg_volume > 0 else 2
                buyer_increase_factor = current_buyers / prev_avg_buyers if prev_avg_buyers > 0 else 1.5

                combined_factor = (volume_increase_factor + buyer_increase_factor) / 2

                confidence = PatternConfidence.MEDIUM
                if combined_factor > 3:
                    confidence = PatternConfidence.HIGH
                elif combined_factor < 2:
                    confidence = PatternConfidence.LOW

                pattern = {
                    'type': PatternType.PUMP_PREPARATION,
                    'pair_id': pair_id,
                    'strength': volume_increase_factor,
                    'confidence': confidence,
                    'start_time': day_start,
                    'end_time': day_end,
                    'volume': current_volume,
                    'unique_buyers': current_buyers,
                    'previous_avg_volume': prev_avg_volume,
                    'previous_avg_buyers': prev_avg_buyers,
                    'volume_increase_pct': (
                                                       current_volume - prev_avg_volume) / prev_avg_volume * 100 if prev_avg_volume > 0 else 0,
                    'buyers_increase_pct': (
                                                       current_buyers - prev_avg_buyers) / prev_avg_buyers * 100 if prev_avg_buyers > 0 else 0
                }

                patterns.append(pattern)

        return patterns

    async def _identify_whale_activity(
            self,
            pair_id: int,
            start_time: datetime,
            end_time: datetime
    ) -> List[Dict]:
        """Identify patterns of large wallet activity (whales)."""
        session = await self._get_session()
        patterns = []

        # Get threshold for the pair
        pair = await session.get(Pair, pair_id)
        if not pair or not pair.threshold:
            return []

        # We'll define whale activity as transactions at least 5x the threshold
        whale_threshold = pair.threshold * 5

        # Get all large transactions
        stmt = select(Transaction).where(
            Transaction.pair_id == pair_id,
            Transaction.amount >= whale_threshold,
            Transaction.timestamp.between(start_time, end_time)
        ).order_by(Transaction.timestamp)

        result = await session.execute(stmt)
        large_txs = result.scalars().all()

        # Group by wallet and transaction type
        whale_activity = {}
        for tx in large_txs:
            key = (tx.wallet_address, tx.transaction_type)
            if key not in whale_activity:
                whale_activity[key] = []
            whale_activity[key].append(tx)

        # For each whale, create a pattern
        for (wallet, tx_type), txs in whale_activity.items():
            if len(txs) < 1:  # Even a single large transaction is noteworthy
                continue

            total_amount = sum(tx.amount for tx in txs)

            # Calculate market impact (this is simplified)
            # We'd need to know the actual liquidity at transaction time for more accurate impact
            market_impact = total_amount / (pair.liquidity or 1) * 100

            # Determine confidence based on amount relative to threshold and market impact
            threshold_multiple = total_amount / (pair.threshold or 1)

            confidence = PatternConfidence.MEDIUM
            if threshold_multiple > 10 or market_impact > 5:
                confidence = PatternConfidence.HIGH
            elif threshold_multiple < 7 and market_impact < 2:
                confidence = PatternConfidence.LOW

            pattern = {
                'type': PatternType.WHALE_ACTIVITY,
                'wallet_address': wallet,
                'pair_id': pair_id,
                'transaction_type': tx_type,
                'strength': threshold_multiple,
                'confidence': confidence,
                'start_time': txs[0].timestamp,
                'end_time': txs[-1].timestamp,
                'transaction_count': len(txs),
                'total_amount': total_amount,
                'avg_transaction_size': total_amount / len(txs),
                'estimated_market_impact_pct': market_impact
            }

            patterns.append(pattern)

        return patterns

    async def _identify_bot_trading(
            self,
            pair_id: int,
            start_time: datetime,
            end_time: datetime
    ) -> List[Dict]:
        """Identify patterns of bot trading activity."""
        session = await self._get_session()
        patterns = []

        # Get all transactions for the pair
        stmt = select(Transaction).where(
            Transaction.pair_id == pair_id,
            Transaction.timestamp.between(start_time, end_time)
        ).order_by(Transaction.wallet_address, Transaction.timestamp)

        result = await session.execute(stmt)
        transactions = result.scalars().all()

        # Group by wallet
        wallet_txs = {}
        for tx in transactions:
            if tx.wallet_address not in wallet_txs:
                wallet_txs[tx.wallet_address] = []
            wallet_txs[tx.wallet_address].append(tx)

        # Look for wallets with consistent small trades or precise timing
        for wallet, txs in wallet_txs.items():
            # Need sufficient transactions to detect a pattern
            if len(txs) < 10:
                continue

            # Sort by timestamp
            txs.sort(key=lambda x: x.timestamp)

            # Calculate time differences between consecutive transactions
            time_diffs = []
            for i in range(1, len(txs)):
                time_diff = (txs[i].timestamp - txs[i - 1].timestamp).total_seconds() / 60  # minutes
                time_diffs.append(time_diff)

            # Calculate statistics on time differences
            avg_time_diff = sum(time_diffs) / len(time_diffs) if time_diffs else 0

            # Calculate variance in time differences
            variance = sum((diff - avg_time_diff) ** 2 for diff in time_diffs) / len(time_diffs) if time_diffs else 0
            std_dev = variance ** 0.5
            coefficient_of_variation = std_dev / avg_time_diff if avg_time_diff > 0 else float('inf')

            # Also look at transaction amounts
            amounts = [tx.amount for tx in txs]
            avg_amount = sum(amounts) / len(amounts)

            # Calculate variance in amounts
            amount_variance = sum((amount - avg_amount) ** 2 for amount in amounts) / len(amounts)
            amount_std_dev = amount_variance ** 0.5
            amount_coefficient_of_variation = amount_std_dev / avg_amount if avg_amount > 0 else float('inf')

            # Bot trading often has low variance in timing and/or amounts
            is_likely_bot = False
            confidence = PatternConfidence.LOW

            # Very regular timing indicates a bot
            if coefficient_of_variation < 0.2:
                is_likely_bot = True
                confidence = PatternConfidence.HIGH
            elif coefficient_of_variation < 0.5:
                is_likely_bot = True
                confidence = PatternConfidence.MEDIUM

            # Very regular amounts also indicates a bot
            if amount_coefficient_of_variation < 0.1:
                is_likely_bot = True
                if confidence != PatternConfidence.HIGH:
                    confidence = PatternConfidence.MEDIUM

            # Another bot pattern: alternating buy/sell with similar amounts
            buy_sell_alternating = True
            for i in range(1, len(txs)):
                if txs[i].transaction_type == txs[i - 1].transaction_type:
                    buy_sell_alternating = False
                    break

            if buy_sell_alternating and len(txs) > 5:
                is_likely_bot = True
                confidence = PatternConfidence.HIGH

            if is_likely_bot:
                pattern = {
                    'type': PatternType.BOT_TRADING,
                    'wallet_address': wallet,
                    'pair_id': pair_id,
                    'strength': 1 / coefficient_of_variation if coefficient_of_variation > 0 else 10,
                    # Higher for more regular timing
                    'confidence': confidence,
                    'start_time': txs[0].timestamp,
                    'end_time': txs[-1].timestamp,
                    'transaction_count': len(txs),
                    'total_amount': sum(amounts),
                    'avg_transaction_size': avg_amount,
                    'time_regularity': 1 / coefficient_of_variation if coefficient_of_variation > 0 else 10,
                    'amount_regularity': 1 / amount_coefficient_of_variation if amount_coefficient_of_variation > 0 else 10,
                    'buy_sell_alternating': buy_sell_alternating
                }

                patterns.append(pattern)

        return patterns

    async def _identify_wash_trading(
            self,
            pair_id: int,
            start_time: datetime,
            end_time: datetime
    ) -> List[Dict]:
        """Identify patterns of wash trading (same wallet buying and selling)."""
        session = await self._get_session()
        patterns = []

        # Get all transactions for the pair
        stmt = select(Transaction).where(
            Transaction.pair_id == pair_id,
            Transaction.timestamp.between(start_time, end_time)
        ).order_by(Transaction.wallet_address, Transaction.timestamp)

        result = await session.execute(stmt)
        transactions = result.scalars().all()

        # Group by wallet
        wallet_txs = {}
        for tx in transactions:
            if tx.wallet_address not in wallet_txs:
                wallet_txs[tx.wallet_address] = []
            wallet_txs[tx.wallet_address].append(tx)

        # Look for wallets with both buys and sells in similar amounts
        for wallet, txs in wallet_txs.items():
            # Need sufficient transactions to detect a pattern
            if len(txs) < 6:  # At least 3 buys and 3 sells
                continue

            # Group by transaction type
            buys = [tx for tx in txs if tx.transaction_type == TransactionType.BUY]
            sells = [tx for tx in txs if tx.transaction_type == TransactionType.SELL]

            # Need both buys and sells
            if len(buys) < 3 or len(sells) < 3:
                continue

            # Sort by timestamp
            buys.sort(key=lambda x: x.timestamp)
            sells.sort(key=lambda x: x.timestamp)

            # Calculate total amounts
            total_buy_amount = sum(tx.amount for tx in buys)
            total_sell_amount = sum(tx.amount for tx in sells)

            # Check if buy and sell amounts are similar (potential wash trading)
            min_amount = min(total_buy_amount, total_sell_amount)
            max_amount = max(total_buy_amount, total_sell_amount)

            amount_ratio = min_amount / max_amount if max_amount > 0 else 0

            # Wash trading typically has similar buy and sell amounts
            if amount_ratio > 0.7:  # At least 70% similar
                # Look for time patterns - wash trading often alternates buy/sell
                all_txs = buys + sells
                all_txs.sort(key=lambda x: x.timestamp)

                # Count buy/sell alternations
                alternations = 0
                for i in range(1, len(all_txs)):
                    if all_txs[i].transaction_type != all_txs[i - 1].transaction_type:
                        alternations += 1

                alternation_ratio = alternations / (len(all_txs) - 1) if len(all_txs) > 1 else 0

                # Determine confidence based on amount similarity and alternation pattern
                confidence = PatternConfidence.MEDIUM
                if amount_ratio > 0.9 and alternation_ratio > 0.7:
                    confidence = PatternConfidence.HIGH
                elif amount_ratio < 0.8 or alternation_ratio < 0.4:
                    confidence = PatternConfidence.LOW

                pattern = {
                    'type': PatternType.WASH_TRADING,
                    'wallet_address': wallet,
                    'pair_id': pair_id,
                    'strength': amount_ratio * alternation_ratio,
                    'confidence': confidence,
                    'start_time': min(buys[0].timestamp, sells[0].timestamp),
                    'end_time': max(buys[-1].timestamp, sells[-1].timestamp),
                    'transaction_count': len(all_txs),
                    'buy_count': len(buys),
                    'sell_count': len(sells),
                    'total_buy_amount': total_buy_amount,
                    'total_sell_amount': total_sell_amount,
                    'amount_similarity': amount_ratio,
                    'alternation_ratio': alternation_ratio
                }

                patterns.append(pattern)

        return patterns

        async def _calculate_price_movements(
                self,
                pair_id: int,
                start_time: datetime,
                end_time: datetime
        ) -> List[Dict]:
            """Calculate price movements for context."""
            session = await self._get_session()

            # Get transactions ordered by time
            stmt = select(Transaction).where(
                Transaction.pair_id == pair_id,
                Transaction.timestamp.between(start_time, end_time)
            ).order_by(Transaction.timestamp)

            result = await session.execute(stmt)
            transactions = result.scalars().all()

            if not transactions:
                return []

            # Split the time range into periods (e.g., days)
            periods = []
            current_period_start = start_time
            period_length = timedelta(days=1)

            while current_period_start < end_time:
                current_period_end = min(current_period_start + period_length, end_time)

                # Get transactions in this period
                period_txs = [tx for tx in transactions if current_period_start <= tx.timestamp <= current_period_end]

                if period_txs:
                    # Get first and last price in period
                    start_price = period_txs[0].price_usd
                    end_price = period_txs[-1].price_usd

                    # Calculate price change
                    price_change_pct = ((end_price - start_price) / start_price) * 100 if start_price > 0 else 0

                    # Calculate volume in period
                    volume = sum(tx.amount for tx in period_txs)

                    periods.append({
                        'start_time': current_period_start,
                        'end_time': current_period_end,
                        'start_price': start_price,
                        'end_price': end_price,
                        'price_change_pct': price_change_pct,
                        'volume': volume,
                        'transaction_count': len(period_txs)
                    })

                current_period_start = current_period_end

            # Identify significant price movements
            significant_movements = []
            for period in periods:
                # Consider movements significant if price changed by more than 5%
                # or volume was particularly high
                if abs(period['price_change_pct']) >= 5.0:
                    movement_type = "increase" if period['price_change_pct'] > 0 else "decrease"

                    significant_movements.append({
                        'start_time': period['start_time'],
                        'end_time': period['end_time'],
                        'price_change_pct': period['price_change_pct'],
                        'volume': period['volume'],
                        'movement_type': movement_type,
                        'significance': abs(period['price_change_pct']) / 5.0  # Normalized significance
                    })

            return significant_movements

        async def _identify_dump_preparation(
                self,
                pair_id: int,
                start_time: datetime,
                end_time: datetime
        ) -> List[Dict]:
            """Identify patterns that may indicate preparation for a dump event."""
            session = await self._get_session()
            patterns = []

            # Similar to pump preparation but for sells
            # Look for increased sell activity from multiple wallets in a short time
            daily_volumes = []
            current_date = start_time.date()
            end_date = end_time.date()

            while current_date <= end_date:
                day_start = datetime.combine(current_date, datetime.min.time())
                day_end = datetime.combine(current_date, datetime.max.time())

                # Get sell volume for this day
                stmt = select(func.sum(Transaction.amount)).where(
                    Transaction.pair_id == pair_id,
                    Transaction.transaction_type == TransactionType.SELL,
                    Transaction.timestamp.between(day_start, day_end)
                )
                result = await session.execute(stmt)
                day_volume = result.scalar_one_or_none() or 0

                # Get unique seller count for this day
                stmt = select(func.count(Transaction.wallet_address.distinct())).where(
                    Transaction.pair_id == pair_id,
                    Transaction.transaction_type == TransactionType.SELL,
                    Transaction.timestamp.between(day_start, day_end)
                )
                result = await session.execute(stmt)
                unique_sellers = result.scalar_one_or_none() or 0

                daily_volumes.append({
                    'date': current_date,
                    'volume': day_volume,
                    'unique_sellers': unique_sellers
                })
                current_date += timedelta(days=1)

            # Look for significant volume increases
            for i in range(3, len(daily_volumes)):
                # Calculate average of previous 3 days
                prev_avg_volume = sum(day['volume'] for day in daily_volumes[i - 3:i]) / 3
                prev_avg_sellers = sum(day['unique_sellers'] for day in daily_volumes[i - 3:i]) / 3

                current_volume = daily_volumes[i]['volume']
                current_sellers = daily_volumes[i]['unique_sellers']

                # Check if there's a significant increase
                if (current_volume > prev_avg_volume * 2 and  # Volume doubled
                        current_sellers > prev_avg_sellers * 1.5):  # 50% more sellers

                    # This could indicate dump preparation
                    day_start = datetime.combine(daily_volumes[i]['date'], datetime.min.time())
                    day_end = datetime.combine(daily_volumes[i]['date'], datetime.max.time())

                    # Calculate confidence based on how extreme the increase is
                    volume_increase_factor = current_volume / prev_avg_volume if prev_avg_volume > 0 else 2
                    seller_increase_factor = current_sellers / prev_avg_sellers if prev_avg_sellers > 0 else 1.5

                    combined_factor = (volume_increase_factor + seller_increase_factor) / 2

                    confidence = PatternConfidence.MEDIUM
                    if combined_factor > 3:
                        confidence = PatternConfidence.HIGH
                    elif combined_factor < 2:
                        confidence = PatternConfidence.LOW

                    pattern = {
                        'type': PatternType.DUMP_PREPARATION,
                        'pair_id': pair_id,
                        'strength': volume_increase_factor,
                        'confidence': confidence,
                        'start_time': day_start,
                        'end_time': day_end,
                        'volume': current_volume,
                        'unique_sellers': current_sellers,
                        'previous_avg_volume': prev_avg_volume,
                        'previous_avg_sellers': prev_avg_sellers,
                        'volume_increase_pct': (
                                                           current_volume - prev_avg_volume) / prev_avg_volume * 100 if prev_avg_volume > 0 else 0,
                        'sellers_increase_pct': (
                                                            current_sellers - prev_avg_sellers) / prev_avg_sellers * 100 if prev_avg_sellers > 0 else 0
                    }

                    patterns.append(pattern)

            return patterns

        async def correlate_patterns_with_price(
                self,
                patterns: List[Dict],
                pair_id: int
        ) -> List[Dict]:
            """
            Correlate identified patterns with price movements.

            Args:
                patterns: List of identified patterns
                pair_id: The ID of the trading pair

            Returns:
                Enhanced patterns with price correlation information
            """
            if not patterns:
                return []

            # Get all start and end times
            all_start_times = [p['start_time'] for p in patterns]
            all_end_times = [p['end_time'] for p in patterns]

            # Get overall time range
            global_start = min(all_start_times)
            global_end = max(all_end_times)

            # Get price movements for the entire period
            price_movements = await self._calculate_price_movements(pair_id, global_start, global_end)

            # Correlate each pattern with price movements
            for pattern in patterns:
                pattern_start = pattern['start_time']
                pattern_end = pattern['end_time']

                # Look for price movements that overlap with this pattern
                relevant_movements = []
                subsequent_movements = []

                for movement in price_movements:
                    # Check for overlapping time periods
                    if (movement['start_time'] <= pattern_end and
                            movement['end_time'] >= pattern_start):
                        relevant_movements.append(movement)
                    # Check for subsequent movements (within 3 days after pattern ends)
                    elif (movement['start_time'] > pattern_end and
                          movement['start_time'] <= pattern_end + timedelta(days=3)):
                        subsequent_movements.append(movement)

                # Calculate average price movement during the pattern
                if relevant_movements:
                    avg_price_change = sum(m['price_change_pct'] for m in relevant_movements) / len(relevant_movements)
                    max_price_change = max(m['price_change_pct'] for m in relevant_movements)
                    min_price_change = min(m['price_change_pct'] for m in relevant_movements)
                else:
                    avg_price_change = 0
                    max_price_change = 0
                    min_price_change = 0

                # Determine if subsequent price movements align with pattern type expectations
                pattern_aligns_with_price = False
                expected_direction = None

                # Define expected price directions for different pattern types
                if pattern['type'] == PatternType.ACCUMULATION:
                    expected_direction = "increase"
                elif pattern['type'] == PatternType.DISTRIBUTION:
                    expected_direction = "decrease"
                elif pattern['type'] == PatternType.PUMP_PREPARATION:
                    expected_direction = "increase"
                elif pattern['type'] == PatternType.DUMP_PREPARATION:
                    expected_direction = "decrease"

                # Check if subsequent movements match expectations
                if expected_direction and subsequent_movements:
                    subsequent_change = sum(m['price_change_pct'] for m in subsequent_movements) / len(
                        subsequent_movements)

                    if (expected_direction == "increase" and subsequent_change > 3) or \
                            (expected_direction == "decrease" and subsequent_change < -3):
                        pattern_aligns_with_price = True

                # Add price correlation to pattern
                pattern['price_correlation'] = {
                    'concurrent_movements': relevant_movements,
                    'subsequent_movements': subsequent_movements,
                    'avg_price_change_during': avg_price_change,
                    'max_price_change_during': max_price_change,
                    'min_price_change_during': min_price_change,
                    'aligns_with_expectations': pattern_aligns_with_price
                }

                # Adjust confidence if price movements align with pattern
                if pattern_aligns_with_price and pattern['confidence'] != PatternConfidence.HIGH:
                    # Upgrade confidence one level if price aligns with pattern
                    if pattern['confidence'] == PatternConfidence.MEDIUM:
                        pattern['confidence'] = PatternConfidence.HIGH
                    elif pattern['confidence'] == PatternConfidence.LOW:
                        pattern['confidence'] = PatternConfidence.MEDIUM

            return patterns

        async def find_related_wallet_patterns(
                self,
                pair_id: int,
                min_wallet_overlap: float = 0.3,
                days_lookback: int = 30
        ) -> List[Dict]:
            """
            Find groups of wallets with similar trading patterns, which might indicate coordination.

            Args:
                pair_id: The ID of the trading pair
                min_wallet_overlap: Minimum overlap ratio between pattern timestamps
                days_lookback: Number of days to look back

            Returns:
                List of related wallet groups with their patterns
            """
            session = await self._get_session()

            # Set time range
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=days_lookback)

            # Get active wallets for this pair
            stmt = select(WalletAnalysis).where(
                WalletAnalysis.pair_id == pair_id,
                WalletAnalysis.transaction_count >= 5,
                WalletAnalysis.last_analyzed >= start_time
            )

            result = await session.execute(stmt)
            wallet_analyses = result.scalars().all()

            if not wallet_analyses:
                return []

            # Get transactions for each wallet
            wallet_transactions = {}
            for analysis in wallet_analyses:
                stmt = select(Transaction).where(
                    Transaction.pair_id == pair_id,
                    Transaction.wallet_address == analysis.wallet_address,
                    Transaction.timestamp.between(start_time, end_time)
                ).order_by(Transaction.timestamp)

                result = await session.execute(stmt)
                transactions = result.scalars().all()

                if transactions:
                    wallet_transactions[analysis.wallet_address] = transactions

            # Group wallets by similar transaction timing
            related_groups = []
            processed_wallets = set()

            for wallet_a, txs_a in wallet_transactions.items():
                if wallet_a in processed_wallets:
                    continue

                # Create a new group with this wallet
                current_group = {
                    'wallets': [wallet_a],
                    'core_wallet': wallet_a,
                    'transaction_count': len(txs_a),
                    'buy_transactions': sum(1 for tx in txs_a if tx.transaction_type == TransactionType.BUY),
                    'sell_transactions': sum(1 for tx in txs_a if tx.transaction_type == TransactionType.SELL),
                    'first_transaction': txs_a[0].timestamp,
                    'last_transaction': txs_a[-1].timestamp,
                }

                # Compare with other wallets
                for wallet_b, txs_b in wallet_transactions.items():
                    if wallet_a == wallet_b or wallet_b in processed_wallets:
                        continue

                    # Check for timing similarity
                    timing_similarity = self._calculate_transaction_timing_similarity(txs_a, txs_b)

                    if timing_similarity >= min_wallet_overlap:
                        current_group['wallets'].append(wallet_b)
                        current_group['transaction_count'] += len(txs_b)
                        current_group['buy_transactions'] += sum(
                            1 for tx in txs_b if tx.transaction_type == TransactionType.BUY)
                        current_group['sell_transactions'] += sum(
                            1 for tx in txs_b if tx.transaction_type == TransactionType.SELL)
                        current_group['first_transaction'] = min(current_group['first_transaction'], txs_b[0].timestamp)
                        current_group['last_transaction'] = max(current_group['last_transaction'], txs_b[-1].timestamp)

                        processed_wallets.add(wallet_b)

                # Only add groups with multiple wallets
                if len(current_group['wallets']) > 1:
                    # Calculate the dominant transaction type
                    if current_group['buy_transactions'] > current_group['sell_transactions'] * 2:
                        pattern_type = PatternType.COORDINATED_BUYS
                    elif current_group['sell_transactions'] > current_group['buy_transactions'] * 2:
                        pattern_type = PatternType.COORDINATED_SELLS
                    else:
                        pattern_type = None

                    if pattern_type:
                        # Calculate confidence based on number of wallets and transactions
                        wallet_count = len(current_group['wallets'])
                        tx_ratio = max(current_group['buy_transactions'], current_group['sell_transactions']) / \
                                   current_group['transaction_count']

                        confidence = PatternConfidence.LOW
                        if wallet_count >= 5 and tx_ratio >= 0.8:
                            confidence = PatternConfidence.HIGH
                        elif wallet_count >= 3 and tx_ratio >= 0.7:
                            confidence = PatternConfidence.MEDIUM

                        related_groups.append({
                            'type': pattern_type,
                            'wallets': current_group['wallets'],
                            'core_wallet': current_group['core_wallet'],
                            'wallet_count': wallet_count,
                            'transaction_count': current_group['transaction_count'],
                            'buy_transactions': current_group['buy_transactions'],
                            'sell_transactions': current_group['sell_transactions'],
                            'start_time': current_group['first_transaction'],
                            'end_time': current_group['last_transaction'],
                            'confidence': confidence,
                            'strength': wallet_count * tx_ratio
                        })

                processed_wallets.add(wallet_a)

            return related_groups

        def _calculate_transaction_timing_similarity(self, txs_a: List[Transaction], txs_b: List[Transaction]) -> float:
            """Calculate similarity in transaction timing between two wallets."""
            if not txs_a or not txs_b:
                return 0.0

            # Create time windows (e.g., 1-hour buckets)
            window_size = timedelta(hours=1)

            # Find overall time range
            all_txs = txs_a + txs_b
            start_time = min(tx.timestamp for tx in all_txs)
            end_time = max(tx.timestamp for tx in all_txs)

            # Create buckets
            current_time = start_time
            buckets = []

            while current_time <= end_time:
                bucket_end = current_time + window_size
                buckets.append((current_time, bucket_end))
                current_time = bucket_end

            # Count transactions in each bucket
            bucket_counts_a = [0] * len(buckets)
            bucket_counts_b = [0] * len(buckets)

            for tx in txs_a:
                for i, (start, end) in enumerate(buckets):
                    if start <= tx.timestamp < end:
                        bucket_counts_a[i] += 1
                        break

            for tx in txs_b:
                for i, (start, end) in enumerate(buckets):
                    if start <= tx.timestamp < end:
                        bucket_counts_b[i] += 1
                        break

            # Calculate overlap
            common_buckets = sum(1 for a, b in zip(bucket_counts_a, bucket_counts_b) if a > 0 and b > 0)
            total_active_buckets = sum(1 for a, b in zip(bucket_counts_a, bucket_counts_b) if a > 0 or b > 0)

            if total_active_buckets == 0:
                return 0.0

            return common_buckets / total_active_buckets
