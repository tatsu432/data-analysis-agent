"""Microsoft Teams bot implementation using Bot Framework SDK."""

import logging

from botbuilder.core import ActivityHandler, MessageFactory, TurnContext
from botbuilder.schema import Activity, ChannelAccount
from botbuilder.schema._models_py3 import ErrorResponseException

from .config import get_settings
from .langgraph_client import LangGraphClient

logger = logging.getLogger(__name__)


class DataAnalysisTeamsBot(ActivityHandler):
    """Bot handler for the Data Analysis Agent on Microsoft Teams."""

    def __init__(self) -> None:
        """Initialize the bot with a LangGraph client."""
        super().__init__()
        self.langgraph_client = LangGraphClient()
        self.settings = get_settings()
        self.is_local_dev = not self.settings.microsoft_app_id

    async def _safe_send_activity(
        self, turn_context: TurnContext, activity_or_text, critical: bool = True
    ) -> bool:
        """Safely send an activity, handling authentication errors gracefully.

        Args:
            turn_context: The turn context
            activity_or_text: Activity or text to send
            critical: If True, log error; if False, silently fail (for non-critical activities)

        Returns:
            True if sent successfully, False otherwise
        """
        try:
            await turn_context.send_activity(activity_or_text)
            return True
        except ErrorResponseException as e:
            if "Unauthorized" in str(e) and self.is_local_dev:
                if critical:
                    # Get channel info for better error message
                    channel_id = (
                        turn_context.activity.channel_id
                        if turn_context.activity.channel_id
                        else "unknown"
                    )
                    if channel_id == "webchat":
                        logger.warning(
                            f"Failed to send activity (authentication required for Web Chat): {e}. "
                            "Web Chat requires Azure credentials. Use Bot Framework Emulator for local testing."
                        )
                    else:
                        logger.warning(
                            f"Failed to send activity (authentication required): {e}. "
                            "This is expected in local dev mode without Azure credentials."
                        )
                else:
                    logger.debug(
                        f"Non-critical activity send failed (local dev mode): {e}"
                    )
                return False
            else:
                # Re-raise if it's not an auth error or we're in production
                if critical:
                    logger.error(f"Failed to send activity: {e}")
                raise
        except Exception as e:
            if critical:
                logger.error(f"Failed to send activity: {e}")
            else:
                logger.debug(f"Non-critical activity send failed: {e}")
            return False

    async def on_members_added_activity(
        self,
        members_added: list[ChannelAccount],
        turn_context: TurnContext,
    ) -> None:
        """Send a welcome message when members are added to the conversation."""
        for member in members_added:
            if member.id != turn_context.activity.recipient.id:
                welcome_message = (
                    "Hi, I'm the Data Analysis Agent.\n\n"
                    "Ask me analytical questions about your datasets or domain knowledge, "
                    "and I'll analyze them."
                )
                sent = await self._safe_send_activity(
                    turn_context, MessageFactory.text(welcome_message), critical=False
                )
                if sent:
                    logger.info(f"Sent welcome message to {member.id}")
                else:
                    logger.debug(
                        "Welcome message not sent (local dev mode without credentials)"
                    )

    async def on_message_activity(self, turn_context: TurnContext) -> None:
        """Handle incoming messages from users."""
        # Extract message text
        text = turn_context.activity.text
        if not text or not text.strip():
            logger.warning("Received empty message")
            await self._safe_send_activity(
                turn_context,
                MessageFactory.text("Please send me a question or request."),
                critical=True,
            )
            return

        # Extract user and conversation IDs
        user_id = (
            turn_context.activity.from_property.id
            if turn_context.activity.from_property
            else "unknown"
        )
        conversation_id = (
            turn_context.activity.conversation.id
            if turn_context.activity.conversation
            else f"conv_{user_id}"
        )

        logger.info(
            f"Received message from user {user_id} in conversation {conversation_id}: {text[:100]}"
        )

        # Send typing indicator (non-critical, can fail in local dev)
        typing_activity = Activity(type="typing")
        await self._safe_send_activity(turn_context, typing_activity, critical=False)

        try:
            # Process query through LangGraph
            response_text = await self.langgraph_client.run_query(
                user_message=text,
                user_id=user_id,
                conversation_id=conversation_id,
            )

            # Send response back to user (critical - must succeed)
            sent = await self._safe_send_activity(
                turn_context, MessageFactory.text(response_text), critical=True
            )
            if sent:
                logger.info(f"Sent response to user {user_id}")
            else:
                # Log the response so user can see what the bot would have said
                channel_id = (
                    turn_context.activity.channel_id
                    if turn_context.activity.channel_id
                    else "unknown"
                )
                logger.error(
                    f"Failed to send response to user {user_id} (authentication required). "
                    f"Channel: {channel_id}. "
                    f"Response that would have been sent ({len(response_text)} chars):\n"
                    f"{response_text[:500]}{'...' if len(response_text) > 500 else ''}"
                )
                if channel_id == "webchat":
                    logger.error(
                        "⚠️  Bot Framework Web Chat requires Azure credentials to send messages. "
                        "Please set MICROSOFT_APP_ID and MICROSOFT_APP_PASSWORD in your .env file, "
                        "or use the Bot Framework Emulator (channelId='emulator') for local testing without credentials."
                    )

        except Exception as e:
            logger.error(
                f"Error processing message from user {user_id}: {e}",
                exc_info=True,
            )
            # Send friendly error message (critical - try to send)
            error_message = (
                "I encountered an error while processing your request. "
                "Please try again later or contact support if the issue persists."
            )
            await self._safe_send_activity(
                turn_context, MessageFactory.text(error_message), critical=True
            )
