"""Base node class for all graph nodes."""

import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseNode(ABC):
    """Base class for all graph nodes."""

    def __init__(self, name: str):
        """Initialize the node with a name."""
        self.name = name

    @abstractmethod
    def __call__(self, state: dict) -> dict:
        """Execute the node logic.

        Args:
            state: The current agent state

        Returns:
            Updated state dictionary
        """
        pass

    def log_node_start(self):
        """Log the start of node execution."""
        logger.info("=" * 60)
        logger.info(f"NODE: {self.name}")

    def log_node_end(self):
        """Log the end of node execution."""
        logger.info("=" * 60)
