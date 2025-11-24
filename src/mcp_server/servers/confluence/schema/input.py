"""Input schemas for Confluence domain."""

from typing import Optional

from pydantic import BaseModel, Field


class ConfluenceSearchPagesInput(BaseModel):
    """Input schema for searching Confluence pages."""

    query: str = Field(..., description="Search query string")
    space_key: Optional[str] = Field(
        default=None, description="Optional space key to limit search"
    )
    limit: int = Field(
        default=10, ge=1, le=100, description="Maximum number of results to return"
    )


class ConfluenceGetPageInput(BaseModel):
    """Input schema for getting a Confluence page."""

    page_id: str = Field(..., description="The ID of the page to retrieve")


class ConfluenceCreatePageInput(BaseModel):
    """Input schema for creating a Confluence page."""

    space_key: str = Field(
        ...,
        description="The space key where the page should be created. Use 'ANALYTICS' for the Analytics space.",
    )
    title: str = Field(..., description="Page title")
    body: str = Field(
        ...,
        description="Page body content (markdown, HTML, or Confluence Storage Format)",
    )
    parent_id: Optional[str] = Field(
        default=None, description="Optional parent page ID"
    )


class ConfluenceUpdatePageInput(BaseModel):
    """Input schema for updating a Confluence page."""

    page_id: str = Field(..., description="The ID of the page to update")
    title: Optional[str] = Field(default=None, description="Optional new title")
    body: Optional[str] = Field(default=None, description="Optional new body content")
