"""HTTP server for the Microsoft Teams bot."""

import asyncio
import json
import logging

from aiohttp import web
from botbuilder.core import BotFrameworkAdapter, BotFrameworkAdapterSettings
from botbuilder.schema import Activity

from .bot import DataAnalysisTeamsBot
from .config import get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("teams_bot")


class TeamsBotServer:
    """HTTP server for the Teams bot."""

    def __init__(self) -> None:
        """Initialize the server with settings and bot."""
        self.settings = get_settings()
        self.bot = DataAnalysisTeamsBot()

        # Create Bot Framework adapter
        # Use empty strings if not provided (allows local dev with emulator)
        app_id = self.settings.microsoft_app_id or ""
        app_password = self.settings.microsoft_app_password or ""

        if not app_id or not app_password:
            logger.warning(
                "MICROSOFT_APP_ID or MICROSOFT_APP_PASSWORD not set. "
                "Authentication will be bypassed (suitable for local development with emulator)."
            )

        adapter_settings = BotFrameworkAdapterSettings(
            app_id=app_id,
            app_password=app_password,
        )
        self.adapter = BotFrameworkAdapter(adapter_settings)

        # Set up error handler
        self.adapter.on_turn_error = self._on_turn_error

    async def _on_turn_error(self, turn_context, error: Exception) -> None:
        """Handle errors in bot turns."""
        logger.error(f"Unhandled error in bot turn: {error}", exc_info=True)

        # Send a message to the user (try to send, but don't fail if auth is required)
        try:
            from botbuilder.schema._models_py3 import ErrorResponseException

            await turn_context.send_activity(
                "I encountered an error. Please try again later."
            )
        except ErrorResponseException as e:
            if "Unauthorized" in str(e) and not self.settings.microsoft_app_id:
                logger.debug(
                    "Could not send error message (local dev mode without credentials)"
                )
            else:
                logger.error(f"Failed to send error message: {e}")
        except Exception as e:
            logger.error(f"Failed to send error message: {e}")

    def _is_emulator(self, activity: Activity) -> bool:
        """Check if the activity is from the Bot Framework Emulator."""
        # Emulator uses "emulator" as channel_id, or "webchat" in some cases
        # Also check serviceUrl for localhost
        channel_id = activity.channel_id if activity.channel_id else ""
        service_url = activity.service_url if activity.service_url else ""

        is_emulator_channel = channel_id in ["emulator", "webchat"]
        is_localhost = "localhost" in service_url or "127.0.0.1" in service_url

        return is_emulator_channel or is_localhost

    async def _handle_messages(self, request: web.Request) -> web.Response:
        """Handle POST requests to /api/messages."""
        try:
            # Read request body
            body = await request.read()
            body_str = body.decode("utf-8")

            logger.info(f"Received request: {body_str[:200]}")

            # Get auth header
            auth_header = request.headers.get("Authorization", "")

            # Parse JSON
            try:
                activity_dict = json.loads(body_str)
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in request: {e}")
                return web.Response(
                    status=400,
                    text="Invalid JSON in request body",
                    content_type="text/plain",
                )

            # Deserialize Activity
            try:
                activity = Activity().deserialize(activity_dict)
            except Exception as e:
                logger.error(f"Failed to deserialize Activity: {e}")
                return web.Response(
                    status=400,
                    text="Invalid Activity format",
                    content_type="text/plain",
                )

            # For local development, skip authentication if app_id is not set
            # This allows testing with Bot Framework Emulator without Azure credentials
            if not self.settings.microsoft_app_id:
                logger.debug(
                    "Skipping authentication (local dev mode - no app_id configured)"
                )
                # Clear auth header to bypass authentication
                auth_header = ""
            else:
                # Get auth header from request
                auth_header = request.headers.get("Authorization", "")

            # Process activity through adapter
            async def process_activity(turn_context):
                await self.bot.on_turn(turn_context)

            # Try to process the activity
            try:
                invoke_response = await self.adapter.process_activity(
                    activity, auth_header, process_activity
                )
            except PermissionError as auth_error:
                # If authentication fails and we're in local dev mode without credentials
                if not self.settings.microsoft_app_id:
                    channel_id = (
                        activity.channel_id if activity.channel_id else "unknown"
                    )
                    logger.error(
                        f"❌ Authentication failed for {channel_id} channel: {auth_error}\n"
                        f"   Web Chat requires valid Azure credentials to authenticate requests.\n"
                        f"   Solutions:\n"
                        f"   1. Use Bot Framework Emulator (channelId='emulator') for local testing without credentials\n"
                        f"   2. Set MICROSOFT_APP_ID and MICROSOFT_APP_PASSWORD in your .env file for Web Chat\n"
                        f"   Current request cannot be processed without authentication."
                    )
                    # Return 200 OK to prevent Web Chat from retrying the same request
                    # Note: The activity was not processed, but we return success to stop retries
                    return web.Response(
                        status=200, text="OK", content_type="text/plain"
                    )
                else:
                    # In production with credentials, this is a real auth failure
                    logger.error(f"Authentication failed: {auth_error}")
                    raise
            except Exception as e:
                # Other errors should be re-raised
                logger.error(f"Error processing activity: {e}", exc_info=True)
                raise

            # Return appropriate response
            if invoke_response and invoke_response.status == 200:
                return web.Response(
                    status=200,
                    text=json.dumps(invoke_response.body)
                    if invoke_response.body
                    else "OK",
                    content_type="application/json",
                )
            else:
                return web.Response(status=201, text="OK", content_type="text/plain")

        except Exception as e:
            logger.error(f"Error handling request: {e}", exc_info=True)
            return web.Response(
                status=500,
                text=f"Internal server error: {str(e)}",
                content_type="text/plain",
            )

    def create_app(self) -> web.Application:
        """Create the aiohttp web application."""
        app = web.Application()

        # Handle CORS preflight requests
        async def handle_options(request: web.Request) -> web.Response:
            return web.Response(
                status=200,
                headers={
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "POST, OPTIONS",
                    "Access-Control-Allow-Headers": "Content-Type, Authorization",
                },
            )

        app.router.add_post("/api/messages", self._handle_messages)
        app.router.add_options("/api/messages", handle_options)

        # Health check endpoint
        async def health_check(request: web.Request) -> web.Response:
            return web.Response(
                text="Teams bot is running",
                content_type="text/plain",
            )

        app.router.add_get("/health", health_check)
        return app

    async def run(self) -> None:
        """Run the HTTP server."""
        app = self.create_app()
        runner = web.AppRunner(app)
        await runner.setup()

        site = web.TCPSite(
            runner,
            host="0.0.0.0",
            port=self.settings.port,
        )
        await site.start()

        logger.info(f"Teams bot server started on http://0.0.0.0:{self.settings.port}")
        logger.info(f"Endpoint: http://localhost:{self.settings.port}/api/messages")
        logger.info(f"Health check: http://localhost:{self.settings.port}/health")

        # Keep running
        try:
            await asyncio.Event().wait()
        except KeyboardInterrupt:
            logger.info("Shutting down server...")
        finally:
            await runner.cleanup()


def main() -> None:
    """Main entrypoint for running the server."""
    server = TeamsBotServer()
    try:
        asyncio.run(server.run())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")


if __name__ == "__main__":
    main()
