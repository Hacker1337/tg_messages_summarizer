import logging
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler
from telegram.ext import filters, MessageHandler, ApplicationBuilder, CommandHandler, ContextTypes

import os
from dotenv import load_dotenv

from summarizing_script import summarize_messages

project_dir = os.path.join(os.path.dirname(__file__))
dotenv_path = os.path.join(project_dir, '.env')
load_dotenv(dotenv_path)

# Get the bot token from the environment variables
bot_token = os.environ["TG_bot_token"]

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id,
                                   text="I'm a summarizing bot. Forward me messages from the chat you want to summarize.\n"
                                   "Type /summarize to summarize forwarded messages.\n"
                                   "Supported languages are Russian and English.")


async def summarize(update: Update, context: ContextTypes.DEFAULT_TYPE):
    messages = context.chat_data.setdefault("messages", [])
    summary = summarize_messages(messages)
    await context.bot.send_message(chat_id=update.effective_chat.id, text=summary)
    context.chat_data["messages"] = []


async def save_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.chat_data.setdefault("messages", []).append(update.message)


if __name__ == '__main__':
    application = ApplicationBuilder().token(bot_token).build()

    start_handler = CommandHandler('start', start)
    summarize_handler = CommandHandler('summarize', summarize)
    message_handler = MessageHandler(
        filters.TEXT & filters.FORWARDED, save_message)

    application.add_handler(start_handler)
    application.add_handler(summarize_handler)
    application.add_handler(message_handler)

    application.run_polling()
