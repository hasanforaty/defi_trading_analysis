# DeFi Trading Pattern Analysis Module - Analyzers Documentation

## Table of Contents
1. [Overview](#overview)
2. [Threshold Analyzer](#threshold-analyzer)
3. [Wave Detector](#wave-detector)
4. [Ratio Analyzer](#ratio-analyzer)
5. [Pattern Recognizer](#pattern-recognizer)
6. [Analysis Coordinator](#analysis-coordinator)
7. [Utility Functions](#utility-functions)
8. [Integration Guide](#integration-guide)

## Overview

This document provides detailed information about the analysis components that form the core of the DeFi Trading Pattern Analysis module. These components work together to detect patterns, anomalies, and significant activities in DeFi trading data.

The analyzers should be placed in the following directory structure:

```
src/
└── analyzers/
    ├── threshold.py       # Threshold analysis for significant transactions
    ├── wave.py            # Wave detection for coordinated buying/selling
    ├── ratio.py           # Analysis of buy/sell ratios for wallets
    ├── pattern.py         # Recognition of trading patterns
    ├── coordinator.py     # Orchestration of multiple analysis types
    ├── utils.py           # Utility functions for analysis
    └── models.py          # Data models for analysis requests/results
```

## Threshold Analyzer

**File: `src/analyzers/threshold.py`**

The Threshold Analyzer identifies significant transactions based on configurable thresholds, which can be absolute (fixed USD amount) or relative (percentage of liquidity).

### Key Features

- **Dynamic threshold calculation** based on pair liquidity
- **Identification of significant transactions** exceeding thresholds
- **Wallet analysis** to find addresses with multiple significant transactions

### Key Methods

#### `calculate_threshold(pair_id, recalculate=False)`
Calculates the appropriate threshold value for a pair based on its liquidity.

- Higher liquidity pairs (>$1M): 0.5% of liquidity
- Medium liquidity pairs ($100k-$1M): 1% of liquidity
- Lower liquidity pairs (<$100k): 2% of liquidity

#### `identify_significant_transactions(pair_id, start_time, end_time, threshold_multiplier)`
Identifies transactions exceeding the threshold for a given pair within a specified time range.

#### `find_wallets_with_multiple_threshold_transactions(pair_id, min_transaction_count, days_lookback, threshold_multiplier)`
Finds wallets that have made multiple transactions exceeding the threshold.

### Example Usage

```python
# Initialize analyzer
threshold_analyzer = ThresholdAnalyzer(session)

# Find significant transactions
significant_txs = await threshold_analyzer.identify_significant_transactions(
    pair_id=123,
    start_time=datetime(2024, 1, 1),
    end_time=datetime(2024, 2, 1),
    threshold_multiplier=1.0  # Use default threshold
)

# Find wallets with multiple significant transactions
wallets = await threshold_analyzer.find_wallets_with_multiple_threshold_transactions(
    pair_id=123,
    min_transaction_count=2,  # At least 2 transactions
    days_lookback=30,  # Look back 30 days
    threshold_multiplier=1.0
)
```

## Wave Detector

**File: `src/analyzers/wave.py`**

The Wave Detector identifies coordinated buying or selling waves by detecting sequences of transactions within specified time windows.

### Key Features

- **Detection of buy waves** with configurable parameters
- **Detection of sell waves** with configurable parameters
- **Wave statistics calculation** for detailed analysis
- **Identification of wallets** participating in multiple waves

### Key Methods

#### `detect_waves(pair_id, transaction_type, start_time, end_time, min_transactions, min_total_amount, max_time_between_transactions)`
Detects waves of transactions for a pair based on specified criteria.

#### `calculate_wave_statistics(wave_id)`
Calculates detailed statistics for a specific wave.

#### `identify_wallets_with_multiple_waves(pair_id, min_waves, days_lookback)`
Identifies wallets that have participated in multiple waves.

### Example Usage

```python
# Initialize detector
wave_detector = WaveDetector(session)

# Detect buy waves
buy_waves = await wave_detector.detect_waves(
    pair_id=123,
    transaction_type=TransactionType.BUY,
    start_time=datetime(2024, 1, 1),
    end_time=datetime(2024, 2, 1),
    min_transactions=3,  # At least 3 transactions in a wave
    min_total_amount=10000,  # Minimum $10,000 total in wave
    max_time_between_transactions=15  # Max 15 minutes between transactions
)

# Find wallets participating in multiple waves
active_wallets = await wave_detector.identify_wallets_with_multiple_waves(
    pair_id=123,
    min_waves=2,  # At least 2 waves
    days_lookback=30  # Look back 30 days
)
```

## Ratio Analyzer

**File: `src/analyzers/ratio.py`**

The Ratio Analyzer examines buy/sell ratios for wallets, identifying specific trading patterns and tracking ratio changes over time.

### Key Features

- **Buy/sell ratio calculation** for individual wallets
- **Pattern identification** based on ratio analysis
- **Tracking of ratio changes** over time

### Ratio Patterns

- `ALL_BUYS`: 100% buys
- `ALL_SELLS`: 100% sells
- `MOSTLY_BUYS`: > 80% buys
- `MOSTLY_SELLS`: > 80% sells
- `BALANCED`: 40-60% buys
- `ACCUMULATION`: Increasing buy ratio over time
- `DISTRIBUTION`: Increasing sell ratio over time

### Key Methods

#### `calculate_wallet_ratio(wallet_address, pair_id, start_time, end_time, min_transaction_count, min_transaction_amount)`
Calculates buy/sell ratios for a specific wallet on a pair.

#### `identify_wallets_with_pattern(pair_id, pattern, min_transaction_count, min_transaction_amount, days_lookback)`
Identifies wallets exhibiting a specific ratio pattern for a pair.

#### `track_ratio_changes_over_time(wallet_address, pair_id, days_lookback)`
Tracks how a wallet's buy/sell ratio has changed over multiple time windows.

### Example Usage

```python
# Initialize analyzer
ratio_analyzer = RatioAnalyzer(session)

# Calculate wallet ratio
ratio_data = await ratio_analyzer.calculate_wallet_ratio(
    wallet_address="0x1234...",
    pair_id=123,
    start_time=datetime(2024, 1, 1),
    end_time=datetime(2024, 2, 1),
    min_transaction_count=3,  # At least 3 transactions
    min_transaction_amount=1000  # At least $1,000 in transactions
)

# Find wallets with mostly buys
buyers = await ratio_analyzer.identify_wallets_with_pattern(
    pair_id=123,
    pattern=RatioPattern.MOSTLY_BUYS,
    min_transaction_count=3,
    min_transaction_amount=1000,
    days_lookback=30
)
```

## Pattern Recognizer

**File: `src/analyzers/pattern.py`**

The Pattern Recognizer identifies common trading patterns in transaction data using time-series analysis and pattern matching techniques.

### Key Features

- **Multiple pattern types** detection
- **Confidence levels** for pattern detection
- **Context-aware analysis** with price movement correlation

### Pattern Types

- `ACCUMULATION`: Gradual accumulation over time
- `DISTRIBUTION`: Gradual distribution over time
- `PUMP_PREPARATION`: Preparing for a pump event
- `DUMP_PREPARATION`: Preparing for a dump event
- `WHALE_ACTIVITY`: Large single-wallet activity
- `BOT_TRADING`: Consistent small trades with precise timing
- `WASH_TRADING`: Same wallet buying and selling

### Key Methods

#### `identify_patterns(pair_id, start_time, end_time)`
Identifies all patterns in transaction data for a pair.

Pattern-specific methods:
- `_identify_accumulation_patterns(pair_id, start_time, end_time)`
- `_identify_distribution_patterns(pair_id, start_time, end_time)`
- `_identify_pump_preparation(pair_id, start_time, end_time)`
- `_identify_whale_activity(pair_id, start_time, end_time)`
- `_identify_bot_trading(pair_id, start_time, end_time)`
- `_identify_wash_trading(pair_id, start_time, end_time)`

### Example Usage

```python
# Initialize recognizer
pattern_recognizer = PatternRecognizer(session)

# Identify all patterns
patterns = await pattern_recognizer.identify_patterns(
    pair_id=123,
    start_time=datetime(2024, 1, 1),
    end_time=datetime(2024, 2, 1)
)

# Check for specific pattern types
accumulation_patterns = [p for p in patterns if p['type'] == PatternType.ACCUMULATION]
whale_activity = [p for p in patterns if p['type'] == PatternType.WHALE_ACTIVITY]
```

## Analysis Coordinator

**File: `src/analyzers/coordinator.py`**

The Analysis Coordinator orchestrates multiple analysis types, providing a unified interface for running complex analysis workflows.

### Key Features

- **Job management** with status tracking
- **Parallel execution** of analysis tasks
- **Comprehensive analysis** combining multiple analyzer results

### Key Methods

#### `run_threshold_analysis(pair_id, start_time, end_time, threshold_multiplier, recalculate_threshold)`
Runs threshold analysis on a pair.

#### `run_wave_detection(pair_id, start_time, end_time, min_amount, min_transactions, time_window_minutes)`
Runs wave detection on a pair.

#### `run_ratio_analysis(pair_id, days_lookback)`
Runs ratio analysis on a pair.

#### `run_pattern_recognition(pair_id, start_time, end_time)`
Runs pattern recognition on a pair.

#### `run_comprehensive_analysis(pair_id, days_lookback, config)`
Runs all analysis types on a pair and combines the results.

#### `get_job_status(job_id)`
Gets the status of a job.

#### `get_job_result(job_id)`
Gets the result of a completed job.

### Example Usage

```python
# Initialize coordinator
coordinator = AnalysisCoordinator(session)

# Run comprehensive analysis
job_id = await coordinator.run_comprehensive_analysis(
    pair_id=123,
    days_lookback=30,
    config={
        "threshold": {
            "min_amount": 5000,  # $5,000 threshold
            "days_lookback": 30
        },
        "wave": {
            "min_amount": 10000,  # $10,000 minimum wave amount
            "time_window_minutes": 1440,  # 24 hours
            "min_transactions": 2  # At least 2 transactions
        },
        "ratio": {
            "buy_percentage": 90,  # 90% buys
            "sell_percentage": 10   # 10% sells
        }
    }
)

# Check job status
while True:
    status = await coordinator.get_job_status(job_id)
    if status["status"] == "completed":
        break
    await asyncio.sleep(2)

# Get results
results = await coordinator.get_job_result(job_id)
```

## Utility Functions

**File: `src/analyzers/utils.py`**

Utility functions for data analysis operations used across the different analyzers.

### Key Functions

#### `normalize_data(data)`
Normalizes a list of numeric values to a 0-1 scale.

#### `calculate_price_momentum(prices, window)`
Calculates price momentum using the rate of change (ROC) over a defined window.

#### `detect_divergence(price_data, indicator_data)`
Identifies divergence between price and indicator data.

#### `identify_transaction_clusters(transactions, max_gap_minutes, min_cluster_size)`
Detects clusters of transactions based on time proximity.

#### `memoize_async(func)`
Decorator for caching results of asynchronous functions.

#### `get_time_windows_data(session, pair_id, windows, end_time)`
Retrieves transaction data over specified time windows for a given trading pair.

#### `time_window_to_datetime(window_minutes, end_time)`
Converts a time window in minutes to a start datetime based on an end datetime.

### Example Usage

```python
# Identify transaction clusters
clusters = identify_transaction_clusters(
    transactions=transactions,
    max_gap_minutes=15,  # Maximum 15 minutes between transactions in a cluster
    min_cluster_size=3   # Minimum 3 transactions to form a cluster
)

# Get transaction data for multiple time windows
windows_data = await get_time_windows_data(
    session=session,
    pair_id=123,
    windows=[1, 7, 30],  # 1-day, 7-day, and 30-day windows
    end_time=datetime.utcnow()
)
```

## Integration Guide

To integrate these analyzers for your custom workflow:

1. **Initialize each analyzer** with a database session
2. **Configure analysis parameters** based on your requirements
3. **Use the AnalysisCoordinator** for complex workflows

### Example: Finding Wallets Meeting Multiple Criteria

For your specific workflow requirement of finding wallets with:
- Buys exceeding amount X
- Amount Y bought within 24 hours (wave detection)
- Buy/sell ratio of 90% or higher

```python
async def find_target_wallets(
    pair_id: int, 
    amount_x: float,  # Minimum transaction amount
    amount_y: float,  # Minimum wave amount
    session: AsyncSession
):
    # Create the analysis coordinator
    coordinator = AnalysisCoordinator(session)
    
    # Define your custom configuration
    config = {
        "threshold": {
            "min_amount": amount_x,
            "days_lookback": 180  # 6 months
        },
        "wave": {
            "min_amount": amount_y,
            "time_window_minutes": 1440,  # 24 hours
            "min_transactions": 2  # At least 2 transactions
        },
        "ratio": {
            "buy_percentage": 90,  # 90% buys
            "sell_percentage": 10   # 10% sells
        }
    }
    
    # Run comprehensive analysis
    job_id = await coordinator.run_comprehensive_analysis(
        pair_id=pair_id, 
        days_lookback=180,  # 6 months
        config=config
    )
    
    # Wait for analysis to complete
    while True:
        status = await coordinator.get_job_status(job_id)
        if status["status"] == "completed":
            break
        await asyncio.sleep(2)
    
    # Get results
    results = await coordinator.get_job_result(job_id)
    
    # Extract wallets meeting all criteria
    threshold_wallets = set(w['wallet_address'] for w in results['threshold_analysis']['wallets'])
    wave_wallets = set(w['wallet_address'] for w in results['wave_analysis']['wallets_with_waves'])
    ratio_wallets = set(w['wallet_address'] for w in results['ratio_analysis']['mostly_buys'])
    
    # Find intersection of all sets
    target_wallets = threshold_wallets.intersection(wave_wallets).intersection(ratio_wallets)
    
    return list(target_wallets)
```

This integration combines all three analysis types to find wallets that meet your specific criteria. The coordinator handles the execution and result aggregation, making it easy to implement complex workflows.
