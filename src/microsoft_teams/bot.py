from botbuilder.core import TurnContext, ActivityHandler

class EchoBot(ActivityHandler):
    async def on_message_activity(self, turn_context: TurnContext):
        text = turn_context.activity.text
        await turn_context.send_activity(f"You said: {text}")
