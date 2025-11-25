import os

import dotenv
from bot import EchoBot
from botbuilder.core import BotFrameworkAdapter, BotFrameworkAdapterSettings
from botbuilder.schema import Activity
from fastapi import FastAPI, Request

# Load credentials (from Azure App Registration)
dotenv.load_dotenv()
APP_ID = os.environ["MICROSOFT_APP_ID"]
APP_PASSWORD = os.environ["MICROSOFT_APP_PASSWORD"]

settings = BotFrameworkAdapterSettings(APP_ID, APP_PASSWORD)
adapter = BotFrameworkAdapter(settings)

bot = EchoBot()
app = FastAPI()


@app.post("/api/messages")
async def messages(req: Request):
    body = await req.json()
    activity = Activity().deserialize(body)
    auth_header = req.headers.get("Authorization", "")

    response = await adapter.process_activity(activity, auth_header, bot.on_turn)
    return response
