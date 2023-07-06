import os
import shutil

from aiogram import Router
from aiogram.filters import Command, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.types import Message

from logs import get_train_bot_logger

logger = get_train_bot_logger()

router = Router()

images_path = "./images/"


@router.message(Command(commands=["start", "info"]), StateFilter(None))
async def info(message: Message, state: FSMContext):
    await message.answer(
        f"<b>Magic Booth</b> - это нейросеть, которая создает невероятные изображения, обучаясь на ваших фотографиях.\n"
        f"Обязательно прочитайте этот гайд, чтобы результат был намного лучше!\n"
        f"1. Загрузите ваши фотографии в бота. Лицо на фотографии должно четко выраженно. Чем фотографии разнообразнее, тем лучше!  Минамальное количество фотографий: 15.\n"
        f"Учтите! Если вы до этого уже загружали и тренировали модель, то перед новой загрузкой используйте команду /clear.\n"
        f"2. Выберите ваш пол.\n"
        f"3. Запустите тренировку модели, используя команду /train\n"
        f"4. Ожидайте завершения обучения. Этот процесс может быть долгим из-за большой очереди.\n"
        f"5. Используйте команду /generate, чтобы сгенерировать изображения\n"
        f"6. Выберите стиль, в котором вы хотите получить изображение.\n"
        f"7. Ожидайте получение результата.\n"
        f"Если вам будет необходимо перечитать гайд, то используйте команду /info.\n"
        f"C любовью, команда Magic Booth!"
    )


@router.message(Command(commands=["clear"]), StateFilter(None))
async def clear(message: Message, state: FSMContext):
    user_id = message.from_user.id
    user_folder = os.path.join(images_path, f"{user_id}")

    if not os.path.exists(user_folder):
        await message.answer("В наборе нет ни одной фотографии")
        return

    shutil.rmtree(user_folder)

    logger.info(f"Deleted {user_folder}")

    await message.answer("Очистка...")
    await message.answer("Очистка завершена!")
