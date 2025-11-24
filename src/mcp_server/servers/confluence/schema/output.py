"""Output schemas for Confluence domain."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ConfluencePageInfo(BaseModel):
    """Information about a Confluence page."""

    id: str = Field(description="Page ID")
    page_id: str = Field(description="Page ID (alias)")
    title: str = Field(description="Page title")
    name: str = Field(description="Page name (alias for title)")
    url: str = Field(description="Page URL")
    page_url: str = Field(description="Page URL (alias)")
    pageUrl: str = Field(description="Page URL (alias)")
    space_key: Optional[str] = Field(default=None, description="Space key")
    excerpt: Optional[str] = Field(default=None, description="Page excerpt")
    snippet: Optional[str] = Field(default=None, description="Page snippet (alias)")
    body: Optional[str] = Field(default=None, description="Page body (alias)")


class ConfluenceSearchPagesOutput(BaseModel):
    """Output schema for Confluence page search."""

    pages: List[Dict[str, Any]] = Field(description="List of page information")
    results: List[Dict[str, Any]] = Field(description="Results (alias)")
    items: List[Dict[str, Any]] = Field(description="Items (alias)")
    error: Optional[str] = Field(default=None, description="Error message if any")


class ConfluenceGetPageOutput(BaseModel):
    """Output schema for getting a Confluence page."""

    page_id: str = Field(description="Page ID")
    id: str = Field(description="Page ID (alias)")
    title: str = Field(description="Page title")
    content: str = Field(description="Page content")
    body: str = Field(description="Page body (alias)")
    text: str = Field(description="Page text (alias)")
    url: str = Field(description="Page URL")
    page_url: str = Field(description="Page URL (alias)")
    pageUrl: str = Field(description="Page URL (alias)")
    error: Optional[str] = Field(default=None, description="Error message if any")


class ConfluenceCreatePageOutput(BaseModel):
    """Output schema for creating a Confluence page."""

    page_id: str = Field(description="Created page ID")
    id: str = Field(description="Page ID (alias)")
    url: str = Field(description="Page URL")
    page_url: str = Field(description="Page URL (alias)")
    pageUrl: str = Field(description="Page URL (alias)")
    title: str = Field(description="Page title")
    error: Optional[str] = Field(default=None, description="Error message if any")


class ConfluenceUpdatePageOutput(BaseModel):
    """Output schema for updating a Confluence page."""

    page_id: str = Field(description="Updated page ID")
    id: str = Field(description="Page ID (alias)")
    url: str = Field(description="Page URL")
    page_url: str = Field(description="Page URL (alias)")
    pageUrl: str = Field(description="Page URL (alias)")
    title: str = Field(description="Page title")
    error: Optional[str] = Field(default=None, description="Error message if any")
