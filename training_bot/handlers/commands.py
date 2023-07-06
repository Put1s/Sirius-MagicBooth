import os
import shutil
from random import choice

from aiogram import Router
from aiogram.filters import Command, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.types import Message

from logs import get_train_bot_logger

logger = get_train_bot_logger()

router = Router()

images_path = "./images/"


@router.message(Command(commands=["start"]), StateFilter(None))
async def hello_message(message: Message, state: FSMContext):
    await message.answer(
        f"Привет!\nКидай мне фотки, а я буду добавлять их в твой датасет. Если захочешь очистить датасет - отправь "
        f"/clear. Чтобы обучить модель - /train [object name]"
    )


@router.message(Command(commands=["clear"]), StateFilter(None))
async def hello_message(message: Message, state: FSMContext):
    user_id = message.from_user.id
    user_folder = os.path.join(images_path, f"{user_id}")

    if not os.path.exists(user_folder):
        await message.answer_sticker(
            sticker="CAACAgIAAxkBAAEG3wZjnkJ5UCim7xKBi59u3BMvyBrV7QACZwADIYxvMNe64KrOePLELAQ"
        )
        return

    shutil.rmtree(user_folder)

    logger.info(f"Deleted {user_folder}")

    stickers = [
        "CAACAgIAAxkBAAEG3wABY55B2v_pSXqUDxgZt1IYOuRNJ38AAgMRAAJVYNBIZs_M199xGossBA",
        "CAACAgIAAxkBAAEG3v5jnkGoX0qf5uuSw1GRzJ4LiLOXVgACUhgAAkc6yEhlQ6VprwwZrSwE",
    ]
    await message.answer_sticker(sticker=choice(stickers))
