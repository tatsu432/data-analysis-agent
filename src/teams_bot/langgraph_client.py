"""LangGraph client wrapper for the Teams bot."""

import json
import logging
import uuid

import aiohttp

from .config import get_settings

logger = logging.getLogger(__name__)

# Namespace UUID for generating deterministic UUIDs from conversation IDs
# This is a fixed namespace UUID (UUID for "data-analysis-agent-teams-bot")
CONVERSATION_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")


def conversation_id_to_uuid(conversation_id: str) -> str:
    """Convert a Teams conversation ID to a deterministic UUID.

    LangGraph requires thread IDs to be UUIDs, but Teams conversation IDs
    are not UUIDs. This function generates a deterministic UUID from the
    conversation ID using UUID5, ensuring the same conversation ID always
    maps to the same UUID.

    Args:
        conversation_id: The Teams conversation ID

    Returns:
        A UUID string derived from the conversation ID
    """
    return str(uuid.uuid5(CONVERSATION_NAMESPACE, conversation_id))


def extract_content_text(content):
    """Extract text from content, handling both string and list formats.

    Args:
        content: Can be a string, list of strings, or list of dicts with 'text' key

    Returns:
        str: Extracted text content
    """
    if not content:
        return ""

    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        # Handle list of strings or list of dicts
        text_parts = []
        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, dict):
                # Try common keys for text content
                text = item.get("text") or item.get("content") or str(item)
                if text:
                    text_parts.append(str(text))
        return " ".join(text_parts)
    else:
        # Fallback: convert to string
        return str(content)


class LangGraphClient:
    """Client for interacting with the LangGraph server."""

    def __init__(self) -> None:
        """Initialize the LangGraph client with settings."""
        self.settings = get_settings()
        self.server_url = self.settings.langgraph_server_url
        self.assistant_id = self.settings.langgraph_assistant_id
        self.graph_name = self.settings.langgraph_graph_name

    async def _get_or_create_thread(self, thread_id: str) -> str:
        """Get or create a thread in LangGraph Server."""
        try:
            async with aiohttp.ClientSession() as session:
                # Try to get existing thread
                async with session.get(
                    f"{self.server_url}/threads/{thread_id}",
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as response:
                    if response.status == 200:
                        return thread_id

                # Create new thread if it doesn't exist
                async with session.post(
                    f"{self.server_url}/threads",
                    json={"thread_id": thread_id},
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as response:
                    if response.status in [200, 201]:
                        return thread_id
                    else:
                        error_text = await response.text()
                        raise Exception(
                            f"Failed to create thread: {response.status} - {error_text}"
                        )
        except Exception as e:
            logger.error(f"Error getting/creating thread {thread_id}: {e}")
            raise

    async def _get_recursion_limit(self) -> int:
        """Get recursion limit from graph configuration."""
        # Import here to avoid circular imports
        try:
            from src.langgraph_server.graph import get_recursion_limit

            return get_recursion_limit()
        except ImportError:
            # Default fallback
            return 50

    async def _create_run(
        self,
        thread_id: str,
        input_data: dict,
        assistant_id: str,
    ) -> dict:
        """Create a run in LangGraph Server."""
        # Verify assistant exists
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.server_url}/assistants/{assistant_id}",
                timeout=aiohttp.ClientTimeout(total=5),
            ) as response:
                if response.status != 200:
                    raise Exception(f"Assistant {assistant_id} not found")

            # Get recursion limit
            recursion_limit = await self._get_recursion_limit()

            # Create run with config
            config = {
                "configurable": {
                    "thread_id": thread_id,
                },
                "recursion_limit": recursion_limit,
            }

            async with session.post(
                f"{self.server_url}/threads/{thread_id}/runs",
                json={
                    "assistant_id": assistant_id,
                    "input": input_data,
                    "config": config,
                },
                timeout=aiohttp.ClientTimeout(total=30),
            ) as response:
                if response.status in [200, 201]:
                    return await response.json()
                else:
                    error_text = await response.text()
                    try:
                        error_json = await response.json()
                        error_text = str(error_json)
                    except Exception:
                        pass
                    raise Exception(
                        f"Failed to create run: {response.status} - {error_text}"
                    )

    async def _stream_run_events(self, thread_id: str, run_id: str) -> str:
        """Stream events from a run and extract the final response."""
        accumulated_content = ""
        current_event_type = None
        buffer = ""

        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.server_url}/threads/{thread_id}/runs/{run_id}/stream",
                timeout=aiohttp.ClientTimeout(total=300),
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(
                        f"Failed to stream events: {response.status} - {error_text}"
                    )

                # Read SSE stream chunk by chunk and parse line by line
                async for chunk in response.content.iter_chunked(8192):
                    if not chunk:
                        continue

                    buffer += chunk.decode("utf-8", errors="ignore")

                    # Process complete lines
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        line = line.strip()

                        if not line:
                            continue

                        if line.startswith("event: "):
                            current_event_type = line[7:].strip()
                        elif line.startswith("data: "):
                            data_str = line[6:]
                            try:
                                event_data = json.loads(data_str)

                                if current_event_type == "values":
                                    if "messages" in event_data:
                                        # Extract AI message content
                                        for msg in event_data["messages"]:
                                            if msg.get("type") == "ai":
                                                raw_content = msg.get("content", "")
                                                content = extract_content_text(
                                                    raw_content
                                                )
                                                if content and content.strip():
                                                    accumulated_content = content

                                current_event_type = None
                            except json.JSONDecodeError:
                                pass

        return accumulated_content.strip()

    async def run_query(
        self,
        user_message: str,
        user_id: str,
        conversation_id: str,
    ) -> str:
        """Run a query through the LangGraph assistant and return the response.

        Args:
            user_message: The user's message text
            user_id: Teams user ID (for logging/tracking)
            conversation_id: Teams conversation ID (converted to UUID for thread_id)

        Returns:
            The assistant's response text

        Raises:
            Exception: If the query fails
        """
        logger.info(
            f"Processing query from user {user_id} in conversation {conversation_id}"
        )

        try:
            # Convert conversation_id to UUID format (LangGraph requires UUID thread IDs)
            thread_id = conversation_id_to_uuid(conversation_id)
            logger.debug(
                f"Converted conversation ID {conversation_id} to thread ID {thread_id}"
            )

            # Ensure thread exists
            await self._get_or_create_thread(thread_id)

            # Create run
            run_data = await self._create_run(
                thread_id,
                {"messages": [{"role": "human", "content": user_message}]},
                self.assistant_id,
            )

            run_id = run_data.get("run_id") or run_data.get("id")
            if not run_id:
                raise Exception(f"Failed to get run ID from response: {run_data}")

            logger.info(f"Created run {run_id} for thread {thread_id}")

            # Stream events and get final response
            response_text = await self._stream_run_events(thread_id, run_id)

            if not response_text:
                response_text = (
                    "I'm working on your request, but didn't receive a response yet."
                )

            logger.info(
                f"Completed query for user {user_id}, response length: {len(response_text)}"
            )

            return response_text

        except Exception as e:
            logger.error(
                f"Error processing query for user {user_id}: {e}", exc_info=True
            )
            raise
