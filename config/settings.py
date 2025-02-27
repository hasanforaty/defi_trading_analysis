# config/settings.py
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, validator, PostgresDsn
from pydantic_settings import BaseSettings
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class ChainType(str, Enum):
    """Supported blockchain networks."""
    ETHEREUM = "ethereum"
    BSC = "bsc"
    POLYGON = "polygon"
    ARBITRUM = "arbitrum"
    OPTIMISM = "optimism"


class LogLevel(str, Enum):
    """Log levels supported by loguru."""
    TRACE = "TRACE"
    DEBUG = "DEBUG"
    INFO = "INFO"
    SUCCESS = "SUCCESS"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class TransactionType(str, Enum):
    """Types of transactions."""
    BUY = "BUY"
    SELL = "SELL"


class DatabaseSettings(BaseModel):
    """Database connection settings."""
    driver: str = "postgresql+asyncpg"
    username: str
    password: str
    host: str
    port: int = 5432
    database: str
    echo: bool = False

    def get_connection_string(self) -> str:
        """Generate database connection string."""
        return f"{self.driver}://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"


class DexToolsApiSettings(BaseModel):
    """DexTools API configuration."""
    api_key: str
    base_url: str = "https://api.dextools.io/v1"
    rate_limit: int = 10  # requests per second
    timeout: int = 30  # seconds
    retry_attempts: int = 3


class AnalysisSettings(BaseModel):
    """Analysis parameters for trading patterns."""
    # Transaction thresholds
    min_transaction_value_usd: float = 500.0
    significant_transaction_threshold_usd: float = 5000.0
    whale_transaction_threshold_usd: float = 50000.0

    # Time frames (in minutes)
    short_timeframe: int = 15
    medium_timeframe: int = 60
    long_timeframe: int = 240

    # Wave detection parameters
    min_transactions_for_wave: int = 3
    max_time_between_wave_transactions: int = 15  # minutes
    min_wave_total_value_usd: float = 10000.0

    # Wallet analysis
    min_wallet_transaction_count: int = 2
    significant_buy_sell_ratio: float = 2.0


class LoggingSettings(BaseModel):
    """Logging configuration."""
    level: LogLevel = LogLevel.INFO
    log_to_file: bool = True
    log_file_path: str = "logs/defi_analyzer.log"
    log_rotation: str = "10 MB"
    log_retention: str = "1 week"
    log_compression: str = "zip"


class Settings(BaseSettings):
    """Main application settings."""
    # Application metadata
    app_name: str = "DeFi Trading Pattern Analysis Tool"
    app_version: str = "0.1.0"
    debug: bool = False

    # Component settings
    database: DatabaseSettings
    dextools_api: DexToolsApiSettings
    analysis: AnalysisSettings = AnalysisSettings()
    logging: LoggingSettings = LoggingSettings()

    # Default chains to monitor
    enabled_chains: List[ChainType] = [ChainType.ETHEREUM, ChainType.BSC]

    class Config:
        env_file = ".env"
        env_nested_delimiter = "__"


def get_settings() -> Settings:
    """Load and return application settings."""
    return Settings(
        database=DatabaseSettings(
            username=os.getenv("DB_USERNAME", "postgres"),
            password=os.getenv("DB_PASSWORD", ""),
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", "5432")),
            database=os.getenv("DB_NAME", "defi_analyzer"),
            echo=os.getenv("DB_ECHO", "False").lower() == "true"
        ),
        dextools_api=DexToolsApiSettings(
            api_key=os.getenv("DEXTOOLS_API_KEY", ""),
            base_url=os.getenv("DEXTOOLS_BASE_URL", "https://api.dextools.io/v1"),
            rate_limit=int(os.getenv("DEXTOOLS_RATE_LIMIT", "10")),
            timeout=int(os.getenv("DEXTOOLS_TIMEOUT", "30")),
            retry_attempts=int(os.getenv("DEXTOOLS_RETRY_ATTEMPTS", "3"))
        ),
        debug=os.getenv("APP_DEBUG", "False").lower() == "true"
    )
