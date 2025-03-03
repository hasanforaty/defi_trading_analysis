from typing import Dict, Any, Optional, Type, TypeVar, Generic, Callable, Set
import inspect
from functools import wraps
import asyncio
from contextlib import asynccontextmanager
from loguru import logger

from src.data.db_manager import DatabaseManager
from src.data.unit_of_work import UnitOfWork, get_unit_of_work
from config.settings import get_settings

T = TypeVar('T')


class ApplicationContext:
    """
    Central dependency injection container for the application.
    Manages component lifecycle and provides service location capabilities.
    """

    def __init__(self):
        """Initialize the application context."""
        self._components: Dict[str, Any] = {}
        self._factories: Dict[str, Callable[..., Any]] = {}
        self._singletons: Set[str] = set()
        self._initializing: Set[str] = set()
        self._initialized = False
        self._settings = get_settings()

        # Register core components
        self.register_singleton("settings", lambda: self._settings)
        self.register_singleton("db_manager", self._create_db_manager)

        logger.info("ApplicationContext initialized")

    def _create_db_manager(self) -> DatabaseManager:
        """Factory method to create a database manager instance."""
        settings = self.get_component("settings")
        db_settings = settings.database

        # Create database manager with pool settings from config
        db_manager = DatabaseManager(
            pool_size=db_settings.pool_size,
            max_overflow=db_settings.max_overflow,
            pool_timeout=db_settings.pool_timeout,
            pool_recycle=db_settings.pool_recycle
        )

        logger.debug("DatabaseManager created")
        return db_manager

    async def initialize(self):
        """Initialize the application context and its components."""
        if self._initialized:
            logger.warning("ApplicationContext already initialized")
            return

        logger.info("Initializing ApplicationContext")

        # Initialize database
        db_manager = self.get_component("db_manager")
        healthy, error = await db_manager.health_check()

        if not healthy:
            logger.error(f"Database health check failed: {error}")
            raise RuntimeError(f"Failed to initialize database: {error}")

        logger.info("ApplicationContext initialization complete")
        self._initialized = True

    def register_component(self, name: str, factory: Callable[..., Any], singleton: bool = False):
        """
        Register a component with the application context.

        Args:
            name: Name for the component
            factory: Factory function to create the component
            singleton: If True, the component will be created once and reused
        """
        if name in self._components or name in self._factories:
            raise ValueError(f"Component '{name}' is already registered")

        self._factories[name] = factory
        if singleton:
            self._singletons.add(name)

        logger.debug(f"Component '{name}' registered (singleton={singleton})")

    def register_singleton(self, name: str, factory: Callable[..., Any]):
        """Register a singleton component."""
        self.register_component(name, factory, singleton=True)

    def get_component(self, name: str) -> Any:
        """
        Get a component by name, creating it if necessary.

        Args:
            name: Name of the component to retrieve

        Returns:
            The requested component

        Raises:
            ValueError: If the component is not registered or a circular dependency is detected
        """
        # Check if component exists
        if name in self._components:
            return self._components[name]

        # Check if factory exists
        if name not in self._factories:
            raise ValueError(f"Component '{name}' is not registered")

        # Check for circular dependencies
        if name in self._initializing:
            raise ValueError(f"Circular dependency detected for component '{name}'")

        # Create the component
        self._initializing.add(name)
        try:
            factory = self._factories[name]
            component = factory()

            # Store if singleton
            if name in self._singletons:
                self._components[name] = component

            logger.debug(f"Component '{name}' created")
            return component
        finally:
            self._initializing.remove(name)

    @asynccontextmanager
    async def get_unit_of_work(self):
        """
        Get a unit of work for database operations.

        Yields:
            UnitOfWork: Unit of work instance
        """
        db_manager = self.get_component("db_manager")
        async with db_manager.session() as session:
            async with get_unit_of_work(session) as uow:
                yield uow

    def has_component(self, name: str) -> bool:
        """Check if a component is registered."""
        return name in self._components or name in self._factories

    async def dispose(self):
        """Dispose of the application context and its components."""
        if not self._initialized:
            return

        logger.info("Disposing ApplicationContext")

        # Dispose of components that need cleanup
        if "db_manager" in self._components:
            db_manager = self._components["db_manager"]
            await db_manager.engine.dispose()

        self._components.clear()
        self._initialized = False
        logger.info("ApplicationContext disposed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        # Create and run a coroutine to dispose the context
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.dispose())


# Global context instance for application-wide use
_global_context: Optional[ApplicationContext] = None


def get_context() -> ApplicationContext:
    """Get the global application context instance."""
    global _global_context
    if _global_context is None:
        _global_context = ApplicationContext()
    return _global_context


def inject(component_name: str):
    """
    Decorator to inject components into function parameters.

    Args:
        component_name: Name of the component to inject

    Returns:
        Decorator function
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # If component is not already provided, inject it
            if component_name not in kwargs:
                context = get_context()
                kwargs[component_name] = context.get_component(component_name)
            return func(*args, **kwargs)

        return wrapper

    return decorator

