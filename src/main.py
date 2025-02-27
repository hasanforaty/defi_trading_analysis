# src/main.py
import asyncio
import logging
from typing import List, Dict, Any

from loguru import logger
from config.settings import get_settings
from src.data.logging import setup_logging
from src.data.database import db


async def startup() -> None:
    """Initialize application components."""
    # Load settings
    settings = get_settings()

    # Setup logging
    setup_logging()
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")

    # Initialize database
    await db.initialize()
    logger.info("Database initialized")

    logger.info("Application startup complete")


async def shutdown() -> None:
    """Clean up application resources."""
    logger.info("Shutting down application")

    # Close database connections
    await db.close()

    logger.info("Shutdown complete")


async def main() -> None:
    """Main application entrypoint."""
    try:
        await startup()
        # Here you would start your application logic
        logger.info("Application running. Press Ctrl+C to exit.")

        # Keep application running until interrupted
        while True:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        logger.info("Application interrupted")
    finally:
        await shutdown()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logger.exception(f"Unhandled exception: {str(e)}")
        exit(1)
