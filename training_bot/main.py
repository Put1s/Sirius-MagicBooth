import asyncio
import os

import torch
from aiogram import Bot, Dispatcher
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.fsm.strategy import FSMStrategy
from aiogram.types import BotCommand

import training_bot.handlers.commands as commands
import training_bot.handlers.photo as photo
import training_bot.handlers.train as train
from logs import get_train_bot_logger

logger = get_train_bot_logger()


# Run bot
async def main():
    bot = Bot(token=os.environ["TOKEN"], parse_mode="HTML")
    dp = Dispatcher(storage=MemoryStorage(), fsm_strategy=FSMStrategy.GLOBAL_USER)

    dp.include_router(commands.router)
    dp.include_router(photo.router)
    dp.include_router(train.router)

    await bot.delete_webhook(drop_pending_updates=True)
    await bot.set_my_commands(
        [
            BotCommand(command="start", description="start"),
            BotCommand(command="clear", description="clear"),
            BotCommand(command="train", description="train"),
        ]
    )
    logger.info("Starting train-bot")
    await dp.start_polling(bot)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn', force=True)
    asyncio.run(main())
