import asyncio
import os

import torch
from aiogram import Bot, Dispatcher
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.fsm.strategy import FSMStrategy
from aiogram.types import BotCommand

import tgbot.handlers.commands as commands
import tgbot.handlers.photo as photo
import tgbot.handlers.result as result
import tgbot.handlers.train as train
from logs import get_train_bot_logger

logger = get_train_bot_logger()


# Run bot
async def main():
    bot = Bot(token=os.environ["TNT"], parse_mode="HTML")
    dp = Dispatcher(storage=MemoryStorage(), fsm_strategy=FSMStrategy.GLOBAL_USER)

    dp.include_router(commands.router)
    dp.include_router(photo.router)
    dp.include_router(train.router)
    dp.include_router(result.router)

    await bot.delete_webhook(drop_pending_updates=True)
    await bot.set_my_commands(
        [
            BotCommand(command="info", description="Краткая информация о боте"),
            BotCommand(command="clear", description="Очистка набора фотографий"),
            BotCommand(command="train", description="Тренировка модели"),
            BotCommand(command="generate", description="Генерация изображений"),
        ]
    )
    logger.info("Starting train-bot")
    await dp.start_polling(bot)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn', force=True)
    asyncio.run(main())
