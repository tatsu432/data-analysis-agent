"""Confluence API client for infrastructure layer."""

import logging
import os

logger = logging.getLogger(__name__)

# Global Confluence client (will be initialized on first use)
_confluence_client = None


class ConfluenceClient:
    """Client for interacting with Confluence API."""

    def __init__(self) -> None:
        """Initialize the Confluence client."""
        self._client = None

    def get_client(self):
        """Get or create Confluence client."""
        global _confluence_client
        if _confluence_client is None:
            try:
                from atlassian import Confluence
            except ImportError:
                raise ImportError(
                    "atlassian-python-api is required. Install with: pip install atlassian-python-api"
                )

            url = os.getenv("CONFLUENCE_URL", "")
            username = os.getenv("CONFLUENCE_USERNAME", "")
            api_token = os.getenv("CONFLUENCE_API_TOKEN", "")

            if not all([url, username, api_token]):
                raise ValueError(
                    "Missing Confluence credentials. Set CONFLUENCE_URL, CONFLUENCE_USERNAME, and CONFLUENCE_API_TOKEN"
                )

            _confluence_client = Confluence(
                url=url,
                username=username,
                password=api_token,  # API token is used as password
                cloud=True,  # Set to False for Confluence Server/Data Center
            )
            logger.info(f"Initialized Confluence client for {url}")

        return _confluence_client
